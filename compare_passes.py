"""Compare several independent reproduction passes against each other and against the archive.

`reproduce.py` answers "does one fresh run agree with the archive?". Repeating it answers a
different and harder question: **is the agreement a property of the code, or of one lucky
run?** A single pass cannot tell those apart. A file that matches the archive because both
were produced by the same non-deterministic path that happened to land twice looks exactly
like a file that matches because the computation is deterministic.

So this reports two things separately, because they can fail for different reasons:

  ACROSS REPRODUCTIONS   every pass against every other pass. Anything but zero here means
                         the code is not deterministic on this machine, and no amount of
                         agreement with the archive would rescue that.

  VERSUS THE ARCHIVE     the passes against the stored results. A difference here with zero
                         above means the archive is stale, not that the code is unstable --
                         which is a statement about a file, and fixable by regenerating it.

Usage:
    python compare_passes.py repro_20260827 repro_pass2 repro_pass3 repro_pass4
"""
import argparse
import csv
import io
import os
import sys

from reproduce import RUNS


def load(path, keys):
    if not os.path.exists(path):
        return None
    with io.open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {tuple(r[k] for k in keys): r for r in rows} if rows else None


def spread(datasets, keys):
    """Largest disagreement per column over the cells every dataset shares."""
    present = [d for d in datasets if d]
    if len(present) < 2:
        return None, 0
    shared = set(present[0])
    for d in present[1:]:
        shared &= set(d)
    worst = {}
    for k in shared:
        for col in present[0][k]:
            vals = []
            for d in present:
                try:
                    vals.append(float(d[k][col]))
                except (KeyError, TypeError, ValueError):
                    vals = []
                    break
            if len(vals) == len(present):
                g = max(vals) - min(vals)
                if g > worst.get(col, -1.0):
                    worst[col] = g
    return worst, len(shared)


def report(title, per_run):
    print(f"\n{title}")
    # `exact` counts columns whose spread is literally 0.0. Columns that merely agree to
    # 1e-9 are counted as passing but are not called identical -- CSV round-trip noise is a
    # real difference, just an irrelevant one, and the two should not be printed as the same.
    print(f"  {'run':18s}{'cells':>8s}{'cols':>6s}{'exact':>9s}{'<=1e-9':>9s}"
          f"{'max spread':>13s}  worst column")
    clean = True
    for name, worst, ncell, known in per_run:
        if worst is None:
            # A run whose files are missing was not compared. Reporting that as clean is
            # how a truncated pass came to look like a passing one, so it fails instead.
            print(f"  {name:18s}  NOT CHECKED: fewer than two datasets available")
            clean = False
            continue
        plain = {c: v for c, v in worst.items() if c not in known}
        bad = {c: v for c, v in plain.items() if v > 1e-9}
        exact = sum(1 for v in plain.values() if v == 0.0)
        top = max(plain.items(), key=lambda kv: kv[1]) if plain else ("-", 0.0)
        print(f"  {name:18s}{ncell:>8d}{len(worst):>6d}"
              f"{f'{exact}/{len(plain)}':>9s}"
              f"{f'{len(plain) - len(bad)}/{len(plain)}':>9s}"
              f"{top[1]:>13.3e}  {top[0] if top[1] > 1e-9 else '-'}")
        for c, v in sorted(worst.items()):
            if c in known:
                print(f"  {'':18s}  known: {c} spreads {v:.3e} -- {known[c]}")
        if bad:
            clean = False
    return clean


def main():
    # exit 0: reproductions agree with each other and with the archive
    #      1: reproductions disagree with each other, or a run could not be checked
    #      2: reproductions agree, but the archive differs outside its documented columns
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="reproduction output roots, in order")
    args = ap.parse_args()

    print(f"{len(args.roots)} reproduction pass(es) + 1 archive = "
          f"{len(args.roots) + 1} independent datasets")
    for r in args.roots:
        print(f"  pass: {r}")

    across, versus = [], []
    for run in RUNS:
        keys = run["keys"]
        known = run.get("known_diff", {})
        reps = [load(os.path.join(root, run["name"], run["csv"]), keys) for root in args.roots]
        arch = load(os.path.join(run["archive"], run["csv"]), keys)
        # reproductions are held to bit-equality with no exemptions: a known-stale archive
        # column is a fact about the archive, and says nothing about run-to-run determinism.
        w, n = spread(reps, keys)
        across.append((run["name"], w, n, {}))
        w2, n2 = spread(reps + [arch], keys)
        versus.append((run["name"], w2, n2, known))

    ok1 = report("ACROSS REPRODUCTIONS  (every pass vs every other; expected: all zero)", across)
    ok2 = report("VERSUS THE ARCHIVE    (passes + archive together)", versus)

    print()
    if ok1:
        print("deterministic: no reproduction disagrees with another beyond 1e-9; "
              "the `exact` column says how many are identical to the last bit")
    else:
        print("NOT deterministic: reproductions disagree with each other -- see above")
    if not ok2:
        print("the archive differs from the reproductions outside the columns named above")
    # Both comparisons decide the exit status. `ok2` used to be printed and discarded, so a
    # run that reproduced itself perfectly but disagreed with the archive still exited 0,
    # and any caller checking only the status was told the archive had been verified.
    if ok1 and ok2:
        sys.exit(0)
    sys.exit(1 if not ok1 else 2)


if __name__ == "__main__":
    main()
