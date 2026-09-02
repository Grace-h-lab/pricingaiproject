"""Reproduce every result the dissertation's headline figures are computed from.

The dissertation states 31 headline figures. `verify_claims.py` recomputes
each one from a named raw file, which proves the text matches the archived results; it does
not prove the archived results can be produced again. This script closes that gap:

    python reproduce.py            # run all five, diff against the archive, verify claims
    python reproduce.py --verify   # skip the runs, diff and verify what is already there
    python reproduce.py --only logger_value context
    python reproduce.py --verify --no-figures   # numbers only

Each entry below carries the command as it was actually issued. Those commands were
recovered from the `protocol.json` each run wrote beside its outputs; where a directory had
no protocol file the entry says so, because an unrecorded command is a reproducibility gap
whatever the numbers turn out to be.

Comparison is by key columns, not by row order, and reports three things separately:
rows only in the archive, rows only in the reproduction, and the largest absolute
difference over the rows both contain. A row-count difference is not automatically a
failure -- a script that has since grown an extra arm or mode emits more rows without
changing any of the old ones -- but it is never silent.
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

REPRO = "repro_20260827"

# name -> how to run it, where the archive is, and how to line the two up.
RUNS = [
    dict(name="logger_value",
         module="pricing_dt.diagnostics.diag_logger_value",
         args=["--seeds", "10"],
         csv="logger_value.csv",
         archive="results_logger_value_20260825",
         keys=["N", "noise", "seed"],
         protocol="summary.json only -- parameters inferred from the default seeds=10"),
    dict(name="context",
         module="pricing_dt.diagnostics.diag_commercial_context",
         args=[],
         csv="commercial_context.csv",
         archive="results_gpu_context_20260819",
         keys=["mode", "seed"],
         protocol="none -- command reconstructed from the CLI defaults"),
    dict(name="expertq_sweep",
         module="pricing_dt.diagnostics.diag_expertq_sweep",
         args=["--expert-q", "0.0,0.25,0.5,0.75,1.0", "--sizes", "100,400,1600",
               "--noise", "0.5", "--seeds", "5", "--topk", "3"],
         csv="expertq_sweep.csv",
         archive="results_expertq_sweep_20260825",
         keys=["expert_q", "N", "seed"],
         protocol="protocol.json",
         # This archive was written at 16:29 on 2026-08-25; the sibling expertq_channel run
         # at 19:13 the same afternoon already agrees with today's code, so astar_in_mask was
         # redefined in between. What changed cannot be recovered -- it predates the
         # provenance record and the tree carries uncommitted edits -- so the discrepancy is
         # reported rather than explained. No dissertation figure reads this column: the 96.6%
         # coverage statistic in 4.6.2 comes from results_bandit_20260821 -- so the file is
         # left as the archive of record and the discrepancy is reported, not suppressed.
         known_diff={"astar_in_mask": "archive predates a same-day correction"}),
    dict(name="expertq_channel",
         module="pricing_dt.diagnostics.diag_expertq_sweep",
         args=["--expert-q", "0.0,0.25,0.5,0.75,1.0", "--sizes", "100,400,1600",
               "--noise", "0.5", "--seeds", "3", "--topk", "3", "--with-dt"],
         csv="expertq_sweep.csv",
         archive="results_expertq_channel_20260825",
         keys=["expert_q", "N", "seed"],
         protocol="protocol.json"),
    dict(name="family_table",
         module="pricing_dt.diagnostics.diag_family_table",
         args=["--support-topk", "3"],
         csv="four_family_raw.csv",
         archive="results_family_table_20260825",
         keys=["cell_id", "seed", "method"],
         protocol="protocol.json",
         # This run computes 14 of the 32 arms. The other 18 come from
         # results_bandit_20260821/combined_raw.csv, and until this pass the concatenation
         # of the two -- four_family_raw.csv, the source for eleven headline figures -- was
         # not produced by any code. diag_family_table now writes it via --merge-with.
         note="four_family_raw.csv = this run's 14 arms + 18 arms from "
              "results_bandit_20260821/combined_raw.csv"),
]

# The five runs above produce the numbers. Three dissertation figures are composed rather
# than plotted by the reporting package, so until now nothing checked that they could be
# redrawn at all -- and Appendix H.2 claims REPRODUCE.md records the contract between
# figures, files and commands. These are that contract for the composed figures. `reads`
# names the run files a figure takes its curves from, so a figure that could silently drift
# from the tables is the one with a non-empty list.
FIGURES = [
    dict(script="pricing_dt/reporting/make_env_dynamics_figure.py",
         png="results/figures/env_dynamics.png",
         reads=[],
         note="environment dynamics; every constant comes from SimConfig, InvConfig and"
              " InventoryMDP at draw time, so it cannot drift from the code"),
    dict(script="pricing_dt/reporting/make_pipeline_figure.py",
         png="results/figures/pipeline_overview.png",
         reads=[],
         note="schematic of the comparison; carries no measured quantity"),
    dict(script="pricing_dt/reporting/make_decision_figure.py",
         png="results/figures/deployment_procedure.png",
         reads=[],
         note="quantities are quoted from Table 5.1, each of which verify_claims.py checks"),
    dict(script="pricing_dt/reporting/make_glance_figure.py",
         png="results/figures/study_at_a_glance.png",
         reads=["results/trust_region_summary.csv",
                "results/env2_trust.csv",
                "results_expertq_sweep_20260825/expertq_sweep.csv"],
         note="both trust-region sweeps and the competence sweep are read, not transcribed"),
    dict(script="pricing_dt/reporting/make_architecture_figure.py",
         png="results/figures/architecture_pipeline.png",
         reads=[],
         note="every constant is read from the shipped config at draw time, so a renamed"
              " field fails the draw instead of printing a stale number"),
    dict(script="pricing_dt/reporting/make_architecture_figure.py",
         png="results/figures/architecture_protocol.png",
         reads=[],
         note="drawn by the same script as architecture_pipeline.png"),
    dict(script="pricing_dt/reporting/make_practitioner_figure.py",
         png="results/figures/practitioner_checks.png",
         reads=["results_family_table_20260825/four_family_raw.csv"],
         note="the file:line pointers it prints are the checks a reader is meant to run,"
              " so they are verified by hand; the ten masked arms are read, not typed"),
]


def redraw(figs):
    """Redraw each composed figure and report what it read. Returns the ones that failed.

    One script may emit more than one page, so each is run once and its report reused for
    every page it produces; a script that fails fails for all of them.
    """
    bad, ran = [], {}
    print(f"\n{'page':40s}{'reads':>7s}   status")
    for f in figs:
        name = os.path.basename(f["png"])
        missing = [p for p in f["reads"] if not os.path.exists(p)]
        if missing:
            print(f"{name:40s}{len(f['reads']):>7d}   missing {missing[0]}")
            bad.append(name)
            continue
        if f["script"] not in ran:
            ran[f["script"]] = subprocess.run(
                [sys.executable, f["script"]], capture_output=True, text=True)
        p = ran[f["script"]]
        if p.returncode != 0 or not os.path.exists(f["png"]):
            print(f"{name:40s}{len(f['reads']):>7d}   FAILED")
            print((p.stderr or p.stdout)[-800:])
            bad.append(name)
            continue
        # A composed figure places its text on a fixed grid, so an overrun is silent in the
        # PNG but not in the script's own report; treat it as a failure rather than a note.
        # Match the failure header exactly: the clean report reads "no text overruns", which
        # a substring search for "overrun" would score as a failure.
        overran = "TEXT OVERRUNS (" in p.stdout
        kb = os.path.getsize(f["png"]) // 1024
        print(f"{name:40s}{len(f['reads']):>7d}   "
              + ("text overruns" if overran else f"ok, {kb} kB"))
        if overran:
            bad.append(name)
    return bad


def read(path):
    import csv
    with io.open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(s):
    """(is_number, value). Empty and non-numeric text return (False, None)."""
    if s is None:
        return False, None
    try:
        return True, float(s)
    except (TypeError, ValueError):
        return False, None


def _cell_equal(va, vb, tol):
    """Compare one cell. Returns (equal, numeric_gap_or_None, why_if_unequal).

    NaN and infinity are decided before subtraction. `abs(nan - x)` is nan and `nan > tol`
    is False, so a subtraction-only test silently passes every NaN it meets, which is the
    one difference a reproduction check most needs to catch.
    """
    na, fa = _num(va)
    nb, fb = _num(vb)
    if na != nb:
        return False, None, f"{'number' if na else 'text'} vs {'number' if nb else 'text'}"
    if not na:                                   # both text: exact match required
        return (va == vb), None, None if va == vb else "text differs"
    a_nan, b_nan = fa != fa, fb != fb            # NaN is the only value unequal to itself
    if a_nan or b_nan:
        return (a_nan and b_nan), None, None if a_nan and b_nan else "NaN on one side only"
    a_inf, b_inf = fa in (float("inf"), float("-inf")), fb in (float("inf"), float("-inf"))
    if a_inf or b_inf:
        return (fa == fb), None, None if fa == fb else "infinity mismatch"
    d = abs(fa - fb)
    return (d <= tol), d, None if d <= tol else f"differs by {d:.3e}"


def compare(a_path, b_path, keys, known=(), tol=1e-9):
    # `tol` implements the SAME-DEVICE criterion of RUN_GUIDE Step 1: a re-run on the
    # device that produced the archive must return every cell unchanged. It is not a
    # cross-device criterion. CPU against a GPU archive moves learned-policy values by
    # up to 2.1%, which this will report as a failure, correctly.
    """Align two result files on `keys` and check them completely.

    The check is exhaustive by construction: key columns must be unique on both sides, the
    two column sets must match, every cell of every shared column is compared as a number
    or as text, and rows present on one side only enter the verdict rather than being
    merely counted, so `max diff = 0` means the two files agree and not only that the
    cells that happened to be examined agreed.

    Columns named in `known` are measured but kept out of the verdict: a column the archive
    is known to carry from superseded code is a documented fact about the archive, not a
    failure of this pass, and burying it in the same number as a real regression would hide
    both.
    """
    A, B = read(a_path), read(b_path)
    if not A or not B:
        return dict(error="one side is empty")
    cols_a, cols_b = list(A[0]), list(B[0])
    absent = [k for k in keys if k not in cols_a or k not in cols_b]
    if absent:
        return dict(error=f"key column(s) absent: {absent}")

    faults = []

    # ---- schema ----
    only_a = [c for c in cols_a if c not in cols_b]
    only_b = [c for c in cols_b if c not in cols_a]
    if only_a:
        faults.append(f"columns only in archive: {only_a}")
    if only_b:
        faults.append(f"columns only in repro: {only_b}")

    # ---- key uniqueness ----
    def index(rowset):
        out, dupes = {}, []
        for r in rowset:
            k = tuple(r[c] for c in keys)
            if k in out:
                dupes.append(k)
            out[k] = r
        return out, dupes

    IA, dup_a = index(A)
    IB, dup_b = index(B)
    if dup_a:
        faults.append(f"{len(dup_a)} duplicate key(s) in archive, e.g. {dup_a[0]}")
    if dup_b:
        faults.append(f"{len(dup_b)} duplicate key(s) in repro, e.g. {dup_b[0]}")

    # ---- key sets and row counts ----
    only_ka, only_kb = set(IA) - set(IB), set(IB) - set(IA)
    # Asymmetric on purpose. Losing an archived row means the code no longer produces
    # something the evidence contains, which is a failure. Gaining one means the script
    # has grown an arm or a mode since, which is not, provided every archived row still
    # reproduces -- but it is never silent, and it is never called "identical".
    if only_ka:
        faults.append(f"{len(only_ka)} archived key(s) absent from the reproduction, "
                      f"e.g. {sorted(only_ka)[0]}")
    extensions = []
    if only_kb:
        extensions.append(f"{len(only_kb)} key(s) the archive does not contain, e.g. "
                          f"{sorted(only_kb)[0]}; {len(A)} archived rows vs {len(B)}")

    # ---- every cell of every shared column, numeric and text alike ----
    shared_cols = [c for c in cols_a if c in cols_b]
    shared = sorted(set(IA) & set(IB))
    worst, worst_col, flagged, mismatches = 0.0, None, {}, []
    for k in shared:
        for col in shared_cols:
            eq, gap, why = _cell_equal(IA[k].get(col), IB[k].get(col), tol)
            if col in known:
                if gap is not None:
                    flagged[col] = max(flagged.get(col, 0.0), gap)
                continue
            if gap is not None and gap > worst:
                worst, worst_col = gap, col
            if not eq and len(mismatches) < 5:
                mismatches.append(f"{col} at {k}: {why}")
    if mismatches:
        faults.append("cell mismatch: " + "; ".join(mismatches))

    return dict(rows_archive=len(A), rows_repro=len(B), shared=len(shared),
                only_archive=len(only_ka), only_repro=len(only_kb),
                max_abs_diff=worst, worst_col=worst_col, known_diff=flagged,
                faults=faults, extensions=extensions)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="skip the runs; diff and verify whatever is already in " + REPRO)
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="restrict to these runs (default: all five)")
    ap.add_argument("--outroot", default=REPRO)
    ap.add_argument("--no-figures", action="store_true",
                    help="skip redrawing the composed figures")
    args = ap.parse_args()

    runs = [r for r in RUNS if not args.only or r["name"] in args.only]
    if args.only and len(runs) != len(args.only):
        sys.exit(f"unknown run name in {args.only}; known: {[r['name'] for r in RUNS]}")

    from pricing_dt.core import provenance
    # --verify reads what is already on disk and must leave it exactly as it found it.
    # Stamping before this branch is what replaced the first pass's record with a later
    # check's, so the stamp now happens only on the branch that writes files.
    if args.verify:
        prov = provenance.read(args.outroot)
        if prov is None:
            sys.exit(f"{args.outroot} carries no provenance.json; nothing to verify")
        print(f"verifying {args.outroot}, produced by "
              f"{provenance.describe(prov)}")
        print(f"this check runs on {provenance.collect()['env']['device']} and writes "
              f"nothing into {args.outroot}\n")
    else:
        prov = provenance.stamp(args.outroot, replace=True,
                                extra={"reproduces": [r["name"] for r in runs]})
        print(provenance.describe(prov) + "\n")

    # ---- run ----
    if not args.verify:
        for r in runs:
            out = os.path.join(args.outroot, r["name"])
            print(f"[run ] {r['name']:16s} -> {out}")
            t0 = time.time()
            cmd = [sys.executable, "-m", r["module"], "--outdir", out] + r["args"]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                print(p.stdout[-2000:], p.stderr[-2000:])
                sys.exit(f"{r['name']} exited {p.returncode}")
            print(f"       done in {time.time() - t0:7.1f}s")
        print()

    # ---- diff ----
    print(f"{'run':18s}{'archive':>9s}{'repro':>8s}{'shared':>8s}{'only-A':>8s}"
          f"{'only-R':>8s}{'max |diff|':>13s}")
    bad, grown = [], []
    for r in runs:
        a = os.path.join(r["archive"], r["csv"])
        b = os.path.join(args.outroot, r["name"], r["csv"])
        if not os.path.exists(b):
            print(f"{r['name']:18s}  not produced"); bad.append(r["name"]); continue
        c = compare(a, b, r["keys"], known=r.get("known_diff", {}))
        if "error" in c:
            print(f"{r['name']:18s}  {c['error']}"); bad.append(r["name"]); continue
        print(f"{r['name']:18s}{c['rows_archive']:>9d}{c['rows_repro']:>8d}"
              f"{c['shared']:>8d}{c['only_archive']:>8d}{c['only_repro']:>8d}"
              f"{c['max_abs_diff']:>13.3e}")
        for col, dv in sorted(c.get("known_diff", {}).items()):
            print(f"{'':18s}  known: {col} differs by {dv:.3e} "
                  f"-- {r['known_diff'][col]}")
        for f in c.get("faults", []):
            print(f"{'':18s}  FAULT: {f}")
        for e in c.get("extensions", []):
            print(f"{'':18s}  EXTENDED: {e}")
        if c["shared"] == 0 or c.get("faults"):
            bad.append(r["name"])
        elif c.get("extensions"):
            grown.append(r["name"])

    # ---- the composed figures, redrawn from whatever is now on disk ----
    figbad = [] if args.no_figures else redraw(FIGURES)

    # ---- the 31 claims, recomputed from the reproduction rather than the archive ----
    remaps = [f"{os.path.join(r['archive'], r['csv'])}="
              f"{os.path.join(args.outroot, r['name'], r['csv'])}"
              for r in runs
              if os.path.exists(os.path.join(args.outroot, r["name"], r["csv"]))]
    print(f"\nrecomputing the dissertation's 31 headline figures from {len(remaps)} "
          f"reproduced file(s):")
    v = subprocess.run([sys.executable, "verify_claims.py"] + remaps,
                       capture_output=True, text=True)
    print("  " + (v.stdout.strip().splitlines() or ["no output"])[-1])

    summary = {"diff_failures": bad, "figure_failures": figbad,
               "verify_returncode": v.returncode}
    if not args.verify:
        json.dump(summary, io.open(os.path.join(args.outroot, "reproduce_summary.json"),
                                   "w", encoding="utf-8"), indent=2)
    else:
        print(f"\n(read-only check: {args.outroot} was not modified)")
    if bad or figbad or v.returncode != 0:
        sys.exit(f"\nNOT reproduced cleanly: {bad or 'diffs ok'}, "
                 f"figures {figbad or 'ok'}, claims exit {v.returncode}")
    if grown:
        print("\nreproduced: every archived row identical and all 31 claims recompute; "
              f"{', '.join(grown)} also produced rows the archive does not contain, "
              "listed above")
    else:
        print("\nreproduced: every shared row identical, and all 31 claims recompute")


if __name__ == "__main__":
    main()
