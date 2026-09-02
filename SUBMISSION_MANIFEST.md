# Submission manifest

Every path in the package, the rule that put it there, and what the rule leaves out.

## The rule

A file ships if a reviewer needs it to answer one of three questions: **what was claimed**
(the dissertation), **what produced it** (the code), or **can it be checked** (the result
lineages and the verification path). Everything else stays out, and the sections below say
which reason applies to each exclusion.

Applying it: **43 MB in, 155 MB out**, from a working tree of about 198 MB.

## The tree

```
pricing_dt_submission/                                              43 MB
│
├── README.md                      entry point; Appendix H.2 names this file
├── REPRODUCE.md                   the figure/file/command contract, and what the passes found
├── RUN_GUIDE.md                   environment setup and step-by-step commands
├── SUBMISSION_MANIFEST.md         every path, and why it is here
├── requirements.txt               pinned to the versions the reported runs used
├── reproduce.py                   full reproduction
├── compare_passes.py              cross-pass comparison
├── run.py                         thin CLI wrapper
│
├── tests/                         fast invariant checks
│
├── pricing_dt/                    research implementation      66 .py, 0.7 MB
│   ├── core/           14  simulator, data, demand models, dt, qdt, relabel,
│   │                       baselines, ope, metrics, realdata
│   ├── envs/            2  inventory.py — environment E2
│   ├── diagnostics/    41  one script per reported diagnostic
│   ├── experiments/     2  E0–E3 and real-data runners
│   ├── reporting/       4  figure generation from the result CSVs
│   └── cli/             2  run.py implementation
│
├── benchmark_minigrid/            the pre-registered external replication, with its own
│                                  README and requirements file. Its outputs are the
│                                  immutable results_minigrid_support_20260825/ below.
│
└── ── evidence: immutable batches, never merged (Appendix G.1) ──         36 MB
    │
    ├── results/                                   CPU, 2026-07
    │     Figures 4.1–4.5, 4.7, E.1–E.3, G.1, and Figure B.1's trust-region panels
    ├── results_gpu_context_20260819/              GPU, 2026-08-19 — §5.7
    │
    ├── results_gate2_masked_20260821/          ┐
    ├── results_bandit_20260821/                ├ 2026-08-21 — §4.6.1, §4.6.2, Appendix E.4
    ├── results_appendix_llm_controls_20260821/ ┘
    ├── results_appendix_llm_deepseek_20260824/    the language-model arm
    ├── results_env2_support_n30_20260824/         the E2 crossing, 30 seeds
    │
    ├── results_logger_value_20260825/          ┐
    ├── results_expertq_sweep_20260825/         ├ 2026-08-25 — the logging policy's own
    ├── results_expertq_channel_20260825/       │ value, both competence sweeps, the
    ├── results_family_table_20260825/          ┘ cross-family table
    │
    ├── results_demand_curves_20260818/            Figure E.3
    ├── results_lightgbm_20260815/               ┐ the LightGBM and CatBoost rows of
    ├── results_catboost_20260815/               ┘ Table F.4
    ├── results_minigrid_support_20260825/         Appendix F.3.1
    ├── results_real_ope_masked_20260825/          §5.5
    ├── results_target_stats_env2_20260831/        Table 4.10's E2 column
    │
    ├── repro_20260827/                            reproduction pass 1
    └── repro_pass2/, repro_pass3/, repro_pass4/   passes 2 to 4
```

Supporting documents that ship alongside: `PREREGISTRATION_QSTAR.md`,
`PREREGISTRATION_SUPPORT_D4RL.md`, `APPENDIX_LLM_SANITY.md`, `D4RL_FEASIBILITY.md`
and `audit/CHANNEL_RESULTS.md`.

## Entry and verification

| Path | Purpose |
|---|---|
| `README.md` | **Start here.** The three verification commands and their order, a table from every dissertation table and figure to the script and results directory behind it, and what to know before the numbers. |
| `requirements.txt` | Python dependencies, pinned to the versions the reported runs used. `torch` is pinned without its CUDA build tag so the file installs on a CPU-only machine. Every result can be produced on CPU, only slower, but a CPU run checked against the GPU archive is expected to fail `reproduce.py`: RUN_GUIDE Step 1 sets out the two reproducibility criteria and only the same-device one is a cell-level check. The commands in the table above, `run.py` and each diagnostic behind a reproduced run stamp a `provenance.json` with the commit, the device actually used and the library versions. |
| `tests/test_smoke.py` | Eight invariant tests over the environments and metrics. |
| `tests/test_reproduce_compare.py` | Sixteen tests over the strict comparator in `reproduce.py` and over `provenance.describe`: schema drift, key collisions, missing and extra keys, NaN and infinity, and the read-only guarantee of `--verify`. |
| `verify_claims.py` | Recomputes each of the thirty-one headline figures from the named raw run files. |
| `reproduce.py` | Re-runs the five commands the cited files descend from, diffs the new output against the archive, and redraws the seven composed figures. |
| `compare_passes.py` | Compares four independent reproductions against each other and against the archive. |
| `REPRODUCE.md` | The contract between figures, files and commands; the environment; what the passes found; the two limits on determinism. |
| `RUN_GUIDE.md` | How to run anything the three commands do not cover. |

## The dissertation

| Path | Purpose |
|---|---|
| `verify_claims.py` | Recomputes each of the thirty-one headline figures from the named raw run files. Moved to the root when the document was taken out of the package: it reads result CSVs, not prose. |
| `pricing_dt/reporting/make_env_dynamics_figure.py` | Figure 3.1, the two environments. |
| `pricing_dt/reporting/make_pipeline_figure.py` | Figure 3.2, the three channels. |
| `pricing_dt/reporting/make_decision_figure.py` | Figure 5.1, the pre-deployment sequence. |
| `pricing_dt/reporting/make_glance_figure.py` | Figure B.1, the study on one page. |
| `pricing_dt/reporting/make_architecture_figure.py` | Figures C.1 and C.2, the architecture. |
| `pricing_dt/reporting/make_practitioner_figure.py` | Figure D.1, the four diagnostics. |

## The code

| Path | Purpose |
|---|---|
| `run.py`, `pricing_dt/cli/run.py` | Root CLI wrapper and its implementation. |
| `pricing_dt/core/simulator.py` | Environment E1, the reference-price pricing MDP, and its exact dynamic programming. |
| `pricing_dt/envs/inventory.py` | Environment E2, lost-sales inventory with censored demand. |
| `pricing_dt/core/data.py` | The logging policies and trajectory generation. |
| `pricing_dt/core/demand_model.py` | The structured demand prior and its unconstrained comparator. |
| `pricing_dt/core/dt.py` | The sequence model and the logged-support mask. |
| `pricing_dt/core/qdt.py` | The conservative critic and its five relabelling read-outs. |
| `pricing_dt/core/relabel.py` | The structured return relabelling of panel D. |
| `pricing_dt/core/baselines.py` | Behaviour cloning, IQL, the offline bandits, estimate-then-optimise. |
| `pricing_dt/core/ope.py` | The off-policy estimators of the secondary strand. |
| `pricing_dt/core/metrics.py` | Normalised value, in-model gap, off-support rate, the ceilings and the floor. |
| `pricing_dt/core/realdata.py` | Online Retail II loading, with the check that identifies the edition. |
| `pricing_dt/diagnostics/*.py` | One script per reported diagnostic, forty in all. |
| `pricing_dt/experiments/experiments.py` | The E0–E3 and real-data runners. |
| `pricing_dt/reporting/make_figures.py` | The nineteen figures drawn from the result CSVs. |
| `pricing_dt/reporting/fig_ceiling_floor.py` | Figure 4.6. |
| `pricing_dt/reporting/fig_optimiser_curse.py` | Figure E.3. |
| `benchmark_minigrid/` | The implementation of the pre-registered external replication on the D4RL MiniGrid logs, with its own README and requirements file. Its outputs are `results_minigrid_support_20260825/`; there is one MiniGrid track, not two. |
| `check_llm_pilot.py` | The go/no-go gates for the language-model appendix run. |
| `pyproject.toml` | Package metadata, the tested Python range, the `pricing-dt` entry point, and the dependency extras (`test`, `realdata`, `boosting`, `llm`, `all`). `requirements.txt` installs the same core set and stays because it is the command the guides document. |
| `LICENSE` | MIT for the code, with a scope note: the dissertation text is not covered, no copy of the real data is distributed, and the benchmark carries its own licence. |
| `CITATION.cff` | How to cite the software, the dissertation and the dataset. |
| `.github/workflows/ci.yml` | The checks run on every push: compile, the twenty-four tests, the cross-reference and float-numbering checks, the thirty-one claim checks, and a CPU smoke through `run.py`. It is not a reproduction; only the same-device criterion of RUN_GUIDE Step 1 is a cell-level check. |

## The evidence

Four batches, produced on two devices over several weeks. They are never merged: the earlier
runs predate the provenance record, so device and code differences cannot be separated after
the fact. Appendix G.1 records which batch every quoted figure comes from.

| Path | Batch | Used for |
|---|---|---|
| `results/` | CPU, 2026-07 | Figures 4.1–4.5, 4.7, E.1–E.3 and G.1; the trust-region panels of Figure B.1; `results/figures/` holds all thirty-two drawn figures |
| `results_gpu_context_20260819/` | GPU, 2026-08-19 | The season-and-product context diagnostic of §5.7 |
| `results_gate2_masked_20260821/` | 2026-08-21 | The support constraint crossed with all five target families |
| `results_bandit_20260821/` | 2026-08-21 | The offline bandit reference point; `combined_raw.csv` merges it with the masked run |
| `results_appendix_llm_controls_20260821/` | 2026-08-21 | The no-learner floor and chance baseline, with every prompt archived |
| `results_appendix_llm_deepseek_20260824/` | 2026-08-24 | The language-model arm of Appendix E.4, ten seeds, prompts archived |
| `results_env2_support_n30_20260824/` | 2026-08-24 | The support crossing repeated in E2 over thirty seeds |
| `results_logger_value_20260825/` | 2026-08-25 | The logging policy's own value |
| `results_expertq_sweep_20260825/` | 2026-08-25 | The ceiling and floor across logging-policy competence |
| `results_expertq_channel_20260825/` | 2026-08-25 | The two channels across the same sweep |
| `results_family_table_20260825/` | 2026-08-25 | The cross-family table under one mask and one protocol |
| `results_demand_curves_20260818/` | 2026-08-18 | The state-level demand-curve diagnostic behind Figure E.3 |
| `results_lightgbm_20260815/`, `results_catboost_20260815/` | 2026-08-15 | The LightGBM and CatBoost rows of Table F.4. The structured, neural and XGBoost rows of the same table come from `results/bprime_xgb.csv`. |
| `results_minigrid_support_20260825/` | 2026-08-25 | The output of `benchmark_minigrid/`, read by Appendix F.3.1 |
| `results_real_ope_masked_20260825/` | 2026-08-25 | The real-log boundary of §5.5 |
| `results_target_stats_env2_20260831/` | 2026-08-31 | Table 4.10's E2 column, ten seeds. Replaces a five-seed exploratory measurement whose exact-$Q^*$ value did not reproduce. |
| `repro_20260827/` | 2026-08-27 | Reproduction pass 1. Named for its date because it was the first, before there were others to number. |
| `repro_pass2/`, `repro_pass3/`, `repro_pass4/` | 2026-08-27 | Passes 2 to 4. `compare_passes.py` reads all four. |

## Supporting record

| Path | Purpose |
|---|---|
| `PREREGISTRATION_QSTAR.md` | The confirmatory study's design, fixed before its seeds were evaluated. |
| `PREREGISTRATION_SUPPORT_D4RL.md` | The external replication's predictions and decision rule, fixed before any policy was trained. Its main prediction failed. |
| `audit/CHANNEL_RESULTS.md` | The exploratory results of 2026-08-19. `PREREGISTRATION_QSTAR.md` dates itself against its §4 and §4.2, citing it by bare filename. |
| `APPENDIX_LLM_SANITY.md` | The language-model appendix's sanity checks. |
| `D4RL_FEASIBILITY.md` | Why D4RL's continuous-action tasks were out of scope. |

## What stays out, and why

| Excluded | Size | Reason |
|---|--:|---|
| `pdf/`, `canvas*/` | 63 MB | Poster and canvas prototypes. `pdf/en/2_technical_architecture.pdf` was the prototype for Appendix C; the appendix supersedes it. |
| `online_retail_II.xlsx` | 44 MB | Gitignored. Appendix H.3 gives the DOI and the check that identifies the right edition. |
| `SUBMISSION_FINAL/`, `Dissertation_Submission_Final_20260827.zip` | 31 MB | Earlier packages of this same material. |
| 49 further `results*` directories | 18 MB | Smoke runs, pilots, and batches superseded by the four above. No dissertation number traces to any of them. |
| `logs/`, `__pycache__/`, `.pytest_cache/` | — | Transient. |
| `.deepseek_key` | — | Credential. Gitignored, read at run time, and a scan of every text file in the tree finds no copy of it anywhere. |
| `dissertation/` | 6 MB | The document, not the code submission: the chapter and appendix sources, the `.docx` and `.tex` they build, and the eleven scripts that render or check the prose. The dissertation is submitted separately, so shipping its sources here would duplicate it and invite the two copies to disagree. The seven scripts in that folder that work on results rather than on prose were moved out and do ship: `verify_claims.py` at the root, and the six `make_*_figure.py` under `pricing_dt/reporting/`. |
| Eight dated audit and review notes | — | Process records of how the project got here, not evidence of what it found. Deleted; nothing that ships referenced any of them. |
| `PACKAGE_README.md`, `00_START_HERE.md` | — | Two further entry points competing with `README.md`, both stale. Deleted. |

## Two choices worth stating

**The language-model prompt dumps stay.** `appendix_llm_state_prompts.jsonl` and
`appendix_llm_actions.csv` are 10.5 MB of the 36 MB of evidence, and only the 40 KB
`appendix_llm_raw.csv` is read by anything. They stay because they are what makes the
Appendix E.4 arm auditable by someone who wants to see what the model was actually asked.

**All forty diagnostics ship.** Searching the documents for script names finds only
fourteen, but that test is too weak: the dissertation cites findings rather than filenames,
and `diag_trust_region.py` (Figures 4.1 and 4.2), `diag_sparse_fixed.py` (F.3.2) and
`diag_target_decomp.py` (Table 4.7) all fall in the unnamed group.

**Licensed MIT, with a stated scope.** `LICENSE` covers the source: `pricing_dt/`,
`tests/`, `benchmark_minigrid/` and the top-level scripts, together with the build and
check scripts that stay with the dissertation sources and are not in this package. It does not cover the dissertation text, and it does not cover
the real transaction data, of which no copy is distributed: that is downloaded at run
time from the UCI repository under CC BY 4.0, and the aggregate quantities derived
from it carry that attribution requirement into the result CSVs. See Appendix H.3.

## Still open

- **The title page has no student number or supervisor.** With the guard gone the build
  prints a notice and omits those two lines rather than refusing. They should be filled in
  at the top of `build_docx.py`, which lives with the dissertation sources and is not
  part of this package.
- **The benchmark logs' licence and access date were never recorded.** Nothing in this
  repository holds either, so Appendix H.3 says where they are carried rather than stating a
  licence on assumption. If they are obtained, that one table row is the only place to edit.
- **The word count is 16,498 against a 16,500 threshold**, a margin of two words. Any
  addition to the body must be paid for by a cut of the same size.
