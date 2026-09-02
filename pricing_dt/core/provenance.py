"""Record what produced a set of results, so a run can be traced back to its code.

Every diagnostic already writes a `protocol.json` naming the parameters it was given.
That is enough to re-issue the command and not enough to reproduce it: the same command
against a different commit, a different device or a different torch build can return
different numbers, and nothing in the archived directories said which of those applied.

`stamp(outdir, extra=...)` writes `provenance.json` alongside the outputs with the commit,
whether the tree was dirty at the time, the exact argv, the device the run actually used
and the versions of the libraries that decide numerical behaviour.

The dirty flag matters more than it looks. A result produced against uncommitted edits
cannot be reproduced from any commit, and this project's own archived runs predate that
record -- `REPRODUCE.md` says which ones and what was done about it.
"""
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

__all__ = ["stamp", "collect"]


def _git(*args):
    """Return git output, or None when git is unavailable or this is not a checkout."""
    try:
        out = subprocess.run(("git",) + args, capture_output=True, text=True, timeout=15,
                             cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                                 os.path.abspath(__file__)))))
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions():
    """Versions of the libraries that can move a number, plus the device actually used."""
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "torch", "sklearn"):
        try:
            v[name] = __import__(name).__version__
        except Exception:
            v[name] = None
    try:
        import torch
        v["cuda_available"] = bool(torch.cuda.is_available())
        # The device the code will ACTUALLY use, not merely the one present. Asking
        # torch.cuda directly meant that a run forced onto CPU with PRICING_DT_DEVICE=cpu
        # computed on CPU and then recorded the GPU's name, so the one field that exists
        # to say where a number came from said the opposite of where it came from.
        from pricing_dt.core.torch_utils import default_device
        dev = default_device()
        if dev.type == "cuda":
            idx = dev.index if dev.index is not None else 0
            v["device"] = torch.cuda.get_device_name(idx)
        else:
            v["device"] = dev.type
        v["device_forced_by_env"] = bool(os.environ.get("PRICING_DT_DEVICE"))
        v["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    except Exception:
        v["cuda_available"], v["device"] = None, None
    return v


def collect(extra=None):
    """Build the provenance record without writing it."""
    dirty = _git("status", "--porcelain")
    rec = {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "commit": _git("rev-parse", "HEAD"),
        "commit_short": _git("rev-parse", "--short", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # None means git could not be consulted, which is not the same as a clean tree
        "dirty_files": None if dirty is None else len([x for x in dirty.splitlines() if x]),
        "env": _versions(),
    }
    if extra:
        rec.update(extra)
    return rec


def stamp(outdir, extra=None, replace=False):
    """Write `provenance.json` into `outdir` and return the record.

    Refuses to overwrite an existing record unless `replace=True`. A provenance file is
    the only machine account of how the numbers beside it were produced, so a later
    command pointed at the same directory must not silently overwrite it.
    Pass `replace=True` only when the directory's contents are genuinely being rebuilt.
    """
    path = os.path.join(outdir, "provenance.json")
    if os.path.exists(path) and not replace:
        raise RuntimeError(
            "%s already exists. A provenance record describes the run that produced the "
            "files beside it, so it is not replaced by a later command. Write to a new "
            "--outdir, or pass replace=True if this directory really is being rebuilt."
            % path)
    rec = collect(extra)
    os.makedirs(outdir, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return rec


def describe(rec):
    """One line saying where a record's files came from, whatever schema it uses.

    A record is normally the machine stamp `collect()` builds. It can also be a note
    written after the machine record was lost, whose recovered fields sit under
    `recovered_from_document` and are explicitly a written account rather than
    collected evidence. Callers must not assume the first shape.
    """
    if not rec:
        return "no provenance record"
    if "commit_short" in rec:
        env = rec.get("env") or {}
        return ("commit %s (%s uncommitted) | %s | torch %s"
                % (rec["commit_short"], rec.get("dirty_files", "?"),
                   env.get("device", "?"), env.get("torch", "?")))
    r = rec.get("recovered_from_document")
    if r:
        return ("commit %s (%s uncommitted) | %s | torch %s  [%s: transcribed from %s, not a machine stamp]"
                % (r.get("commit_short", "?"), r.get("dirty_files", "?"),
                   r.get("device", "?"), r.get("torch", "?"),
                   rec.get("record_status", "recovered"), r.get("source", "the document")))
    return "provenance record in an unrecognised schema"


def read(outdir):
    """The record already in `outdir`, or None. For read-only callers such as --verify."""
    path = os.path.join(outdir, "provenance.json")
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)
