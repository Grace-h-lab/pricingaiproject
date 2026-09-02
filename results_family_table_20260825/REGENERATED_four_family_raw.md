# `four_family_raw.csv` was regenerated on 2026-08-27

## What was wrong

The copy archived on 2026-08-25 had been assembled by hand from this run's `family_raw.csv`
and the eighteen arms in `results_bandit_20260821/combined_raw.csv`. Nothing in the
repository produced it, and it did not agree with the `family_raw.csv` stored beside it:
13 of 1,260 rows differed in the last bits, every one an estimate-then-optimise arm, across
five columns — `v_policy`, `v_behaviour_expected`, `v_optimal_same_start`, `nv` and
`mean_behavior_prob`, by at most 5.7e-14.

The hand-assembled file had therefore been built from a *different execution* of
`diag_family_table` than the one whose outputs sit in this directory — a different machine,
thread count or library build, differing only in floating-point summation order. Four
independent reproductions on one machine agree with each other to the last bit, so the code
is deterministic; it was the archived file that could not be derived from its own siblings.

## What was done

`diag_family_table` now writes this file itself, via `--merge-with`, refusing to proceed if
the two sides share a `(cell, seed, method)`. This copy is the output of that code path.

    python -m pricing_dt.diagnostics.diag_family_table --support-topk 3 --outdir <out>

## What changed in the numbers

Nothing that is read. Over the 32 method arms the largest change in mean normalised value is
**1.1e-16**, against a claim tolerance of 1e-3. Every figure the dissertation states from
this file is identical to six decimal places, and `dissertation/_verify_claims.py` (since moved to `verify_claims.py`) returns
all 26 claims matching against either copy. **No citation changes: the path is the same.**

## Provenance

| | |
|---|---|
| Regenerated | 2026-08-27T06:05:22+00:00 |
| Commit | `dbe3796` (83 files uncommitted) |
| Device | NVIDIA GeForce RTX 5060 Laptop GPU |
| Libraries | torch 2.11.0+cu128, numpy 2.4.4, scipy 1.15.3 |
| New file sha256 (first 16) | `fc6f84767ec079c4` |
| Superseded file sha256 (first 16) | `1ee02f0c8ce12f75` |

The superseded copy is kept beside this note rather than deleted; it is the file every result
reported before 2026-08-27 was computed from, and the two agree to fourteen significant
figures.
