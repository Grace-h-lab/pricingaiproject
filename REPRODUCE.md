# Reproducing the dissertation's headline figures

The dissertation states 31 headline figures. Two different questions can be asked about them,
and this project answers them with two different scripts:

| Question | Script | What it proves |
|---|---|---|
| Does the text match the results on disk? | `python verify_claims.py` | The 31 figures recompute from named raw files, not from a summary anyone could have typed |
| Can those results be produced again? | `python reproduce.py` | Re-runs the five producing commands and diffs the new output against the archive, then recomputes the 31 figures **from the new files** |

The second is the stronger claim, and it had not been made before the reproduction
reported here: the run of 2026-08-27 archived as `repro_20260827/`, repeated three more
times as `repro_pass2/` to `repro_pass4/`. **This pass** below means that reproduction.

---

## The contract: 31 figures ← 5 raw files ← 5 commands

Only five files feed the verification. Everything else in the 66 `results*` directories is
supporting evidence for the text, not a source of a headline number.

| Run | Command | Output |
|---|---|---|
| `logger_value` | `python -m pricing_dt.diagnostics.diag_logger_value --seeds 10` | `logger_value.csv` |
| `context` | `python -m pricing_dt.diagnostics.diag_commercial_context` | `commercial_context.csv` |
| `expertq_sweep` | `python -m pricing_dt.diagnostics.diag_expertq_sweep --expert-q 0.0,0.25,0.5,0.75,1.0 --sizes 100,400,1600 --noise 0.5 --seeds 5 --topk 3` | `expertq_sweep.csv` |
| `expertq_channel` | as above, `--seeds 3 --with-dt` | `expertq_sweep.csv` |
| `family_table` | `python -m pricing_dt.diagnostics.diag_family_table --support-topk 3` | `four_family_raw.csv` |

`reproduce.py` holds this table in executable form; the prose here and the `RUNS` list there
are the same fact written twice, and the script is the one that is checked.

## Running it

```bash
python reproduce.py                       # run all five, diff, verify           (~50 min, GPU)
python reproduce.py --verify              # diff and verify what is already there  (seconds)
python reproduce.py --only logger_value   # one run
```

The comparison aligns rows by key columns rather than by position, and reports rows present
only in the archive, rows present only in the reproduction, and the largest absolute
difference over the rows both contain. A row-count difference is not automatically a failure,
since a script that has since grown an extra arm emits more rows without changing the old ones,
but it is never silent.

## The composed figures

Most figures are plotted from a results CSV by `pricing_dt/reporting/`. Four scripts are
*composed*, producing six pages between them: they place boxes and text on a fixed grid
rather than plotting a frame, so nothing about them is checked by the numeric contract
above, and until this pass nothing checked that they could be redrawn at all.
`reproduce.py` now redraws every page and reports what each one read.

| Script | Output | Reads |
|---|---|---|
| `pricing_dt/reporting/make_env_dynamics_figure.py` | `env_dynamics.png` (Figure 3.1) | nothing; it reads `SimConfig`, `InvConfig` and `InventoryMDP` |
| `pricing_dt/reporting/make_pipeline_figure.py` | `pipeline_overview.png` (Figure 3.2) | nothing; it is a schematic and carries no measured quantity |
| `pricing_dt/reporting/make_decision_figure.py` | `deployment_procedure.png` (Figure 5.1) | nothing; its quantities are quoted from Table 5.1, each of which `verify_claims.py` checks |
| `pricing_dt/reporting/make_glance_figure.py` | `study_at_a_glance.png` (Figure B.1) | `results/trust_region_summary.csv`, `results/env2_trust.csv`, `results_expertq_sweep_20260825/expertq_sweep.csv` |
| `pricing_dt/reporting/make_architecture_figure.py` | `architecture_pipeline.png` (Figure C.1) and `architecture_protocol.png` (Figure C.2) | no run file; every constant it prints is read from the shipped configuration at draw time, so a renamed field fails the draw instead of printing a stale number |
| `pricing_dt/reporting/make_practitioner_figure.py` | `practitioner_checks.png` (Figure D.1) | `results_family_table_20260825/four_family_raw.csv` for the value axis; the `file:line` pointers it prints were checked against the source by hand |

Two of the four read run files, so their curves are the ones that could
silently drift from the tables; that is why they read them rather than carrying transcribed
values, and the architecture pages take the same precaution against the configuration.
Both also report their own text overruns, because text that overflows a hand-placed box is
invisible to the script and obvious to a reader; `reproduce.py` treats an overrun as a
failure rather than a note.

`python reproduce.py --verify --no-figures` skips this step when only the numbers matter.

---

## What this pass found

**Three of the five archived directories recorded the parameters they were run with; two did
not.** `protocol.json` names the arguments for `expertq_sweep`, `expertq_channel` and
`family_table`. `results_logger_value_20260825` holds only a `summary.json` of the numbers,
and `results_gpu_context_20260819` holds no record at all. For those two the command in the
table above was reconstructed from the CLI defaults, and the reproduction confirms the
reconstruction is right, because the numbers come back identical.

**The file eleven headline figures are computed from was not produced by any code.**
`four_family_raw.csv` holds 32 arms: 14 that `diag_family_table` computes and 18 that come
from the bandit/IQL/Q-DT runs in `results_bandit_20260821/combined_raw.csv`. The archived
copy had been assembled by hand, so the documented command did not in fact reproduce the
source of those figures. The concatenation was verified against the archive (same 2,880
keys, agreeing to 5.7e-14, which is the CSV round-trip tolerance this project already
records in its own anchor check), and `diag_family_table` now writes the merged file itself
under `--merge-with`, refusing to proceed if the two sides share a `(cell, seed, method)`.

**One archived column was produced by superseded code, and the archive itself proves it.**
Every column of `results_expertq_sweep_20260825/expertq_sweep.csv` reproduces bit-for-bit
except `astar_in_mask`, which differs in all 75 rows. The explanation is in the timestamps:
that file was written at 16:29 on 2026-08-25, and `results_expertq_channel_20260825`, the
same script run at 19:13 the same afternoon, already agrees with today's code to the last
bit. The column's definition therefore changed between those two runs. What the change was
cannot be recovered: it predates the provenance record this pass added, and the working tree
carries uncommitted edits. Both versions rise monotonically with logger quality, so the
archived column is a different measurement rather than an obviously broken one: the
archived value at `expert_q = 0` is identical across every N and seed, which the corrected
column is not.

No dissertation figure reads this column: the 96.6% coverage statistic in §4.6.2 comes from
`results_bandit_20260821`, and `verify_claims.py` never touches `astar_in_mask`. The file is
therefore left as the archive of record, and `reproduce.py` carries the discrepancy as a
named `known_diff`, measured and printed with its reason on every pass, kept out of the
verdict, with every other column still held to bit-equality. Folding it into the same number
as a real regression would hide both.

**No archived run recorded the commit, the device or the library versions.** The same command
against a different torch build can return different numbers, and nothing on disk said which
build produced these. `pricing_dt/core/provenance.py` now writes a `provenance.json` carrying
the commit, whether the tree was dirty, the exact `argv`, the device actually used and the
versions of numpy, scipy, torch and scikit-learn. `reproduce.py` stamps one for every pass.

**The working tree was dirty when this reproduction ran**, and the provenance record says by how
many files. Results produced against uncommitted edits cannot be reproduced from any commit,
which is a limitation of this pass.

**`commercial_context.csv` has gained a mode since it was archived.** The archived file holds
only the `season_product` rows; the current script emits `baseline` as well. The three
Chapter 5 checks that read this file averaged over *every* row, which was correct only
because of a property of the archived file that nothing stated. Regenerating it made three
correct figures appear wrong (`−3.454` read as `−3.957`). `verify_claims.py` now filters on
`mode == season_product` explicitly and asserts the rows exist, so it is robust to
regeneration. This was a latent fault, not a broken result: no dissertation figure was wrong.

---

## Determinism

Every run is seeded from the seed list in its command, and repeated runs of the same command
on the same machine return bit-identical CSVs; that is what the `max |diff| = 0` column
below is asserting. Two caveats bound how far that carries:

- **Across devices.** The reproduction ran on the same GPU as the archive. cuDNN kernel
  selection can differ between devices and builds, so an identical command on different
  hardware is expected to agree to floating-point tolerance rather than bit-for-bit.
- **Across commits.** Determinism is a property of a command *and* a code state. The
  provenance record is what makes the pair recoverable; before this pass only the command
  was recorded.

## Results of this pass

2026-08-27, commit `dbe3796` with 82 files uncommitted, RTX 5060 Laptop GPU,
torch 2.11.0+cu128, numpy 2.4.4, scipy 1.15.3.

| Run | Archive | Repro | Shared | Only archive | Only repro | max abs diff |
|---|--:|--:|--:|--:|--:|--:|
| `logger_value` | 90 | 90 | 90 | 0 | 0 | 0 |
| `context` | 10 | 20 | 10 | 0 | 10 | 0 |
| `expertq_sweep` | 75 | 75 | 75 | 0 | 0 | 0 |
| `expertq_channel` | 45 | 45 | 45 | 0 | 0 | 0 |
| `family_table` | 2880 | 2880 | 2880 | 0 | 0 | 5.7e-14 |

Every shared row is identical, and **all 31 headline figures recompute from the newly
generated files** (the count at the time of these passes; two checks were added
afterwards, over a run these passes do not regenerate). Two entries need their number read correctly rather than at face value.
The ten rows present only in the reproduction of `context` are the `baseline` mode the script
gained after that archive was written; no shared row moved. The `5.7e-14` on `family_table`
is not a formatting artefact and not instability in the code; it is traced below to the
archived merged file having been assembled from a different execution than the one whose
`family_raw.csv` sits beside it.

One column is excluded from the verdict by name and printed on every pass:
`astar_in_mask` in `expertq_sweep`, for the reason given above.

**Determinism was also checked without reference to the archive.** `diag_family_table` was
run twice from the same command on the same machine; the two outputs agree on every one of
1,260 rows to the last bit. That is a stronger statement than agreement with an archive,
because it cannot be satisfied by a stale file that happens to match.

Total wall-clock: about 50 minutes with three runs sharing one laptop GPU.
`logger_value` takes 12 seconds, `context` 5 minutes, `family_table` roughly 35.

## Four independent passes

One pass cannot separate a deterministic computation from a non-deterministic one that
happened to land on the archived answer twice. The pass above was therefore repeated three
more times, into separate output roots, and `compare_passes.py` compares all four against
each other and against the archive:

```
python compare_passes.py repro_20260827 repro_pass2 repro_pass3 repro_pass4
```

**Across reproductions, every column of every run has a spread of exactly zero**: 3,110 rows
and 79 columns over the five runs, which have different widths, so 69,350 values per pass and
277,400 over the four, with no disagreement anywhere, not even at 1e-14. Each pass independently recomputed all 31 headline figures from its own output.

| Run | Cells | Cols | Exact across passes | vs archive |
|---|--:|--:|--:|--:|
| `logger_value` | 90 | 11 | 11/11 | 11/11 exact |
| `context` | 20 | 13 | 13/13 | 13/13 exact |
| `expertq_sweep` | 75 | 14 | 14/14 | 13/13 exact + 1 known |
| `expertq_channel` | 45 | 18 | 18/18 | 18/18 exact |
| `family_table` | 2880 | 23 | 23/23 | 18/23 exact, 23/23 within 1e-9 |

That last row is the reason for running more than once, and it turned out to say something
about the archive that a single pass could not have established.

Five of `family_table`'s columns differ from the archive by up to 5.7e-14:
`v_policy`, `v_behaviour_expected`, `v_optimal_same_start`, `nv`, `mean_behavior_prob`.
Across the four reproductions those same five columns are identical to the last bit, so the
code is not the unstable party. Locating the disagreement shows why: **the archived
`four_family_raw.csv` disagrees with the archived `family_raw.csv` in the same directory**, on
13 of 1,260 rows, every one of them an estimate-then-optimise arm. The hand-assembled merged
file was therefore built from a different execution of `diag_family_table` than the one stored
beside it: a different machine, thread count or library build, differing only in
floating-point summation order. Fourteen significant figures agree, the claim tolerance is
1e-3, and no dissertation figure moves; but the directory is not internally derivable, which
is the thing this pass exists to detect.

**Resolved on 2026-08-27.** `diag_family_table` now produces this file, so the archived copy
was replaced with the output of that code path. The hand-assembled original is kept beside it
as `four_family_raw.SUPERSEDED-20260825-handassembled.csv`, with
`REGENERATED_four_family_raw.md` recording what differed, the sha256 of both copies and the
provenance of the replacement. Over the 32 method arms the largest change in mean normalised
value is 1.1e-16 against a claim tolerance of 1e-3; every figure the dissertation states from
this file is identical to six decimal places, all 31 claims verify against either copy, and
**no citation changes because the path is unchanged**. `family_table` now agrees with the
archive on 23 of 23 columns exactly, and the directory is internally derivable.

**A note for anyone auditing this with pandas.** `pd.read_csv` defaults to
`float_precision="high"`, which is not round-trip exact: it reports only two of those five
columns as differing and silently rounds the other three into agreement. `compare_passes.py`
reads through the `csv` module and Python's own `float()`, which is correctly rounded. A
last-bit audit performed with pandas defaults will understate what it finds.

The three extra passes were launched within seconds of each other against the same commit and
each took about 123 minutes of accumulated run time while sharing one GPU; their
`provenance.json` files record the timestamps.
