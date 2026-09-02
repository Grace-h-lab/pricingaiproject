# -*- coding: utf-8 -*-
"""The reproduction comparator must fail on every difference it can see.

Each test below is a difference the comparator must report: a duplicate key, a column
present on one side only, a missing cell, a NaN facing a number (`abs(nan - x) > tol` is
False, so a subtraction-only test would pass it), an infinity mismatch, and a row present
in only one of the two files.
"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reproduce


def _csv(rows, header):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in header) + "\n")
    return path


BASE = [{"seed": "0", "arm": "a", "nv": "1.5", "note": "ok"},
        {"seed": "1", "arm": "a", "nv": "2.5", "note": "ok"}]
HEAD = ["seed", "arm", "nv", "note"]
KEYS = ["seed", "arm"]


def _cmp(a_rows, b_rows, a_head=HEAD, b_head=HEAD):
    return reproduce.compare(_csv(a_rows, a_head), _csv(b_rows, b_head), KEYS)


def test_identical_files_pass():
    c = _cmp(BASE, BASE)
    assert not c.get("faults"), c
    assert c["max_abs_diff"] == 0.0


def test_duplicate_key_is_caught():
    dup = BASE + [{"seed": "0", "arm": "a", "nv": "9.9", "note": "ok"}]
    c = _cmp(BASE, dup)
    assert any("duplicate key" in f for f in c["faults"]), c["faults"]


def test_missing_column_is_caught():
    b = [{k: v for k, v in r.items() if k != "nv"} for r in BASE]
    c = _cmp(BASE, b, b_head=["seed", "arm", "note"])
    assert any("only in archive" in f for f in c["faults"]), c["faults"]


def test_extra_column_is_caught():
    b = [dict(r, extra="1") for r in BASE]
    c = _cmp(BASE, b, b_head=HEAD + ["extra"])
    assert any("only in repro" in f for f in c["faults"]), c["faults"]


def test_nan_against_a_number_is_caught():
    b = [dict(BASE[0], nv="nan"), BASE[1]]
    c = _cmp(BASE, b)
    assert any("NaN" in f for f in c["faults"]), c["faults"]


def test_nan_on_both_sides_is_equal():
    a = [dict(BASE[0], nv="nan"), BASE[1]]
    c = _cmp(a, a)
    assert not c.get("faults"), c["faults"]


def test_infinity_mismatch_is_caught():
    a = [dict(BASE[0], nv="inf"), BASE[1]]
    b = [dict(BASE[0], nv="-inf"), BASE[1]]
    c = _cmp(a, b)
    assert any("infinity" in f for f in c["faults"]), c["faults"]


def test_row_lost_from_the_reproduction_is_a_fault():
    """The code has stopped producing something the evidence contains."""
    c = _cmp(BASE, BASE[:1])
    assert any("absent from the reproduction" in f for f in c["faults"]), c["faults"]


def test_row_gained_by_the_reproduction_is_an_extension_not_a_fault():
    """A script that has grown an arm since the archive was written. Reported, never
    silent, and never called a regression."""
    b = BASE + [{"seed": "2", "arm": "a", "nv": "3.5", "note": "ok"}]
    c = _cmp(BASE, b)
    assert not c["faults"], c["faults"]
    assert any("archive does not contain" in e for e in c["extensions"]), c["extensions"]


def test_text_column_difference_is_caught():
    b = [dict(BASE[0], note="changed"), BASE[1]]
    c = _cmp(BASE, b)
    assert any("text differs" in f for f in c["faults"]), c["faults"]


def test_numeric_difference_above_tolerance_is_caught():
    b = [dict(BASE[0], nv="1.6"), BASE[1]]
    c = _cmp(BASE, b)
    assert any("differs by" in f for f in c["faults"]), c["faults"]


def test_difference_within_tolerance_passes():
    b = [dict(BASE[0], nv="1.5000000000001"), BASE[1]]
    c = _cmp(BASE, b)
    assert not c.get("faults"), c["faults"]


def test_known_column_is_exempt_but_still_measured():
    b = [dict(BASE[0], nv="9.9"), BASE[1]]
    c = reproduce.compare(_csv(BASE, HEAD), _csv(b, HEAD), KEYS, known={"nv": "documented"})
    assert not c.get("faults"), c["faults"]
    assert c["known_diff"]["nv"] > 8.0


# --------------------------------------------------------------------------- provenance
def test_provenance_records_the_device_actually_used():
    """The device field must follow the override, not merely report what hardware exists.

    Written as a subprocess because the override is read at import time.
    """
    import json
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prog = ("import sys, json; sys.path.insert(0, %r);"
            "from pricing_dt.core import provenance;"
            "print(json.dumps(provenance.collect()['env']))" % root)
    env = dict(os.environ, PRICING_DT_DEVICE="cpu", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         env=env, cwd=root)
    assert out.returncode == 0, out.stderr[-500:]
    rec = json.loads(out.stdout.strip().splitlines()[-1])
    assert rec["device"] == "cpu", rec
    assert rec["device_forced_by_env"] is True, rec


def test_provenance_refuses_to_replace_a_record():
    """A record describes the run that produced the files beside it, so a later command
    pointed at the same directory must not silently replace it."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pricing_dt.core import provenance
    d = tempfile.mkdtemp()
    provenance.stamp(d)
    try:
        provenance.stamp(d)
        raise AssertionError("a second stamp was allowed to overwrite the first")
    except RuntimeError:
        pass
    provenance.stamp(d, replace=True)          # explicit rebuild is still allowed
    assert provenance.read(d) is not None



def test_provenance_describe_handles_both_schemas():
    """A record that is not a machine stamp must not crash the reader.

    repro_20260827's machine record was destroyed by an earlier --verify and replaced by an
    account of the loss, whose recovered fields sit under `recovered_from_document`. Indexing
    the machine schema at the top level made `python reproduce.py --verify`, the command the
    README opens with, die on the directory it defaults to.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from pricing_dt.core import provenance

    machine = {"commit_short": "abc1234", "dirty_files": 3,
               "env": {"device": "cpu", "torch": "2.11.0+cpu"}}
    assert "abc1234" in provenance.describe(machine)

    recovered = {"record_status": "OVERWRITTEN -- this is not a machine-collected record",
                 "recovered_from_document": {"commit_short": "dbe3796", "dirty_files": 82,
                                             "device": "GPU", "torch": "2.11.0+cu128",
                                             "source": "REPRODUCE.md"}}
    line = provenance.describe(recovered)
    assert "dbe3796" in line
    assert "not a machine stamp" in line

    assert provenance.describe(None) == "no provenance record"
    assert "unrecognised" in provenance.describe({"something": "else"})

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print("%d tests in this file passed" % len(fns))
