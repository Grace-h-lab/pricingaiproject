# Start here

MSc Applied Artificial Intelligence dissertation, with the code that produced every number
in it and the result files those numbers are computed from.

**What a Pricing Log Can Support: How Action Coverage and Off-Support Freedom Shape the
Reliability of Offline Pricing Methods**

---

## Check it before reading it

Three commands, in increasing order of strength. Appendix H.2 states the same three.

```bash
pip install -r requirements.txt

python -m pytest tests/ -q            # 24 invariant tests                    seconds
python verify_claims.py               # 31 headline numbers, from raw files   seconds
python reproduce.py --verify          # diffs the shipped runs without re-running   minutes
python reproduce.py                   # re-runs 5 commands, redraws 7 figures  ~50 min, GPU
```

`--verify` is the short path: it skips the runs and diffs, row by row, whatever is already
in `repro_20260827/` against the archive, then recomputes the thirty-one figures from it.
It establishes that the shipped reproduction agrees with the shipped results. Only the
full command establishes that the runs can be produced again.

The first says the environments and metrics behave as specified. The second recomputes each
of the thirty-one headline figures from the named raw run files rather than from a summary.
The third re-runs the commands those files descend from, compares the new output with the
archive row by row, and recomputes the thirty-one figures from the new files.

`REPRODUCE.md` records the contract between figures, files and commands, what the
reproduction found, and the two limits on determinism. It was run four times;
`compare_passes.py` compares the passes against each other and against the archive.

## From the dissertation to the code

Every table and figure, the script that produces it, and the directory its numbers come
from. Batches are never merged, so the directory is part of the identity of a number
(Appendix G.1). `results/` is the CPU batch of 2026-07.

| Dissertation | Script | Results |
|---|---|---|
| Table 4.1 | `pricing_dt/diagnostics/diag_trust_region.py`, `diag_env2_channels.py` | `results/` |
| Table 4.2 | `pricing_dt/diagnostics/diag_expertq_sweep.py --with-dt` | `results_expertq_channel_20260825/` |
| Table 4.3, Figure 4.1 | `pricing_dt/diagnostics/diag_trust_region.py` | `results/` |
| Table 4.4, Figure 4.2 | `pricing_dt/diagnostics/diag_env2_channels.py` | `results/` |
| Table 4.5, Figure 4.3 | `pricing_dt/diagnostics/diag_data_channel.py` | `results/` |
| Table 4.6, Table 4.10 (E1 column) | `pricing_dt/diagnostics/diag_target_stats.py` | `results/` |
| Table 4.10 (E2 column) | `pricing_dt/diagnostics/diag_target_stats.py --env2` | `results_target_stats_env2_20260831/` |
| Table 4.7, Figure 4.4 | `pricing_dt/diagnostics/diag_target_decomp.py` | `results/` |
| Tables 4.8, 4.9 | `pricing_dt/diagnostics/diag_qstar_confirm.py` | `results/` |
| Table 4.11, Figure D.1 | `pricing_dt/diagnostics/diag_family_table.py` | `results_family_table_20260825/` |
| Table 4.12 | `pricing_dt/diagnostics/diag_gate2_pricing.py` | `results_gate2_masked_20260821/` |
| Table 4.13 | `pricing_dt/diagnostics/diag_env2_support.py` | `results_env2_support_n30_20260824/` |
| Table 4.14, Figure 4.7 | `pricing_dt/diagnostics/diag_bandit_baseline.py`, `diag_appendix_llm.py` | `results_bandit_20260821/`, `results_appendix_llm_*` |
| Figure 4.5 | `pricing_dt/reporting/make_figures.py` | `results_bandit_20260821/`, `results_gate2_masked_20260821/` |
| Figure 4.6 | `pricing_dt/reporting/fig_ceiling_floor.py` | `results_expertq_sweep_20260825/` |
| Table E.1, Figure E.1 | `pricing_dt/diagnostics/diag_c2_fixed.py` | `results/` |
| Table E.2, Figure E.2 | `pricing_dt/diagnostics/diag_heldout_protocol.py`, `diag_conditioning.py` | `results/` |
| Figure E.3 | `pricing_dt/reporting/fig_optimiser_curse.py` | `results_demand_curves_20260818/` |
| Table F.1 | `pricing_dt/diagnostics/diag_bandit_baseline.py` | `results_bandit_20260821/` |
| Table F.3 | `benchmark_minigrid/run_support_crossing.py` | `results_minigrid_support_20260825/` |
| Table F.4 | `pricing_dt/diagnostics/diag_bprime_xgb.py`, `diag_bprime_lightgbm.py`, `diag_bprime_catboost.py` | `results/bprime_xgb.csv`, `results_lightgbm_20260815/`, `results_catboost_20260815/` |
| Figure G.1 | `pricing_dt/reporting/make_figures.py` | `results/` |
| §5.7 | `pricing_dt/diagnostics/diag_commercial_context.py` | `results_gpu_context_20260819/` |
| The logging policy's own value | `pricing_dt/diagnostics/diag_logger_value.py` | `results_logger_value_20260825/` |

Figures 3.1, 3.2, 5.1, B.1, C.1, C.2 and D.1 are composed rather than plotted: their scripts are
in `pricing_dt/reporting/` and are listed with their outputs in `REPRODUCE.md`. Tables 2.1,
E.3, F.5 and G.1 to G.5 summarise material stated elsewhere and have no single producing run.

`python verify_claims.py` checks thirty-one of these numbers by recomputing
them from the raw per-run rows in the directories named above.

## What to read, in order

| | | |
|---|---|---|
| 1 | `the dissertation, submitted separately` | **The submission.** Not included in this package. Where it and a shipped file disagree, the archived run files govern: `verify_claims.py` recomputes thirty-one headline numbers from them. |
| 2 | `REPRODUCE.md` | What was re-run, what agreed, and what did not. |
| 3 | `SUBMISSION_MANIFEST.md` | Every path in the package, including the module map. |
| 4 | `RUN_GUIDE.md` | How to run anything not covered by the three commands above. |

The two pre-registrations, `PREREGISTRATION_QSTAR.md` and
`PREREGISTRATION_SUPPORT_D4RL.md`, were written before the seeds they govern were evaluated.
The second one records a prediction that **failed**; Appendix F.3.1 reports the failure as a
result rather than as a caveat.

## Things worth knowing before the numbers

- **Two normalisation anchors, not one.** `nv = 1` is the optimal sequential policy in both
  environments, but `nv = 0` is the oracle myopic policy in E1 and the logging policy in E2.
  Values are not comparable across environments.
- **Two inference-time operators, easily confused.** The trust region `Top_m` ranks by the
  policy's own probability (Equation 3.1); the logged-support mask `Supp_k` ranks by how
  often the log played each action (Equation 3.2). They are never interchanged.
- **The no-learner floor is a reference level, not a lower bound.** Three of the ten arms in
  Table 4.11 score below it.
- **The project's own original claim does not survive.** Structured relabelling beating Q-DT
  was produced by three measurement asymmetries; under a fair comparator it becomes a loss of
  0.119. Appendix E reports each failure with its corrected number, and the pre-fix results
  tree is retained so the correction is auditable.
- **Batches are never merged.** Results were produced on two devices over several weeks, and
  the earlier runs predate the provenance record, so device and code differences cannot be
  separated after the fact. Appendix G.1 lists which batch every quoted figure comes from.
- **The real retail data is descriptive calibration**, not causal validation. It fixes a
  plausible elasticity operating point; it does not identify counterfactual policy value.

## What is not in this package

The raw `online_retail_II.xlsx` (44 MB) is excluded for size; Appendix H.3 gives the DOI and
the check that distinguishes the right edition from a widely mirrored extract. Credentials,
caches, run logs, smoke outputs and superseded result batches are excluded; no number in the
dissertation traces to any of them.
