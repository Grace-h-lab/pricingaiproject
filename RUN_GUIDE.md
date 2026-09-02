# RUN GUIDE — getting real results, step by step

This walks from a clean machine to full results, for both tracks:
- **Track A (simulator)** — produces the simulator evidence, with exact ground truth. No download.
- **Track B (real data, Online Retail II)** — calibrates the simulator to a realistic elasticity and runs a realism check on real logs.

Run Track A to get your headline results; run Track B to anchor them to reality and answer "is this realistic?".

---

## Which runner to use

`run.py` provides the broad E0-E3 pipeline. For the results the dissertation reports, use
the diagnostics below: the Q-DT comparator there relabels with the action-dependent
`mode="td"` target, and Appendix E records why the action-independent one it replaced
invalidated every earlier comparison against it.

Use these diagnostics for the current paper framing:

```bash
python -m pricing_dt.diagnostics.diag_channel_ladder   --outdir results
python -m pricing_dt.diagnostics.diag_conditioning     --outdir results
python -m pricing_dt.diagnostics.diag_c2_fixed         --outdir results
python -m pricing_dt.diagnostics.diag_heldout_protocol --outdir results
python -m pricing_dt.diagnostics.diag_demand_curve_amplification --outdir results_demand_curves_20260818
python -m pricing_dt.diagnostics.diag_env2_channels    --outdir results
python -m pricing_dt.diagnostics.diag_gate2_pricing --smoke --seeds 0 --cells 60:0.4 --outdir results_gate2_pricing_smoke --overwrite
python -m pricing_dt.diagnostics.diag_gate2_pricing --outdir results_gate2_pricing_full_20260818 --overwrite
```

The defensible main claim is the channel/support mechanism: a model is risky as
an action selector off logged support, but safer when confined to
return-conditioning or support-filtered imitation.

Commercial-pricing reading: the simulator is not a full commercial pricing engine. It is
the controlled core used to identify the channel mechanism with exact counterfactual
values. A fuller pipeline with customer/product/time features, demand forecasting,
expected demand, and a revenue/profit optimiser is the right external-validity extension;
it should be used to ask whether the mechanism survives deployment realism, not to replace
the controlled evidence.

---

## Step 0 — Get the code
Unzip `pricing_dt_submission.zip` and `cd pricing_dt_submission`. All commands below run
from the root of that folder, not from the `pricing_dt/` package directory inside it:
that directory holds the library only, and has no `run.py`, `requirements.txt` or `tests/`.

## Step 1 — Python environment
The reported runs used Python 3.11.9. `requirements.txt` pins every library to the
version they used, so a fresh environment reproduces the same build rather than
whatever is current.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` is pinned without its CUDA build tag, so this installs on a CPU-only machine.
For the CUDA build the reported runs used, install `torch==2.11.0` from the index at
<https://pytorch.org/get-started/locally/>.

**Two reproducibility claims, and only one of them is what `reproduce.py` tests.**
Every result here can be produced on CPU, only slower, but *reproduced* means
something different depending on whether the device changes:

| | What must hold | Tolerance | Tested by |
|---|---|---|---|
| **Same device, exact** | every cell of every result file returns unchanged | `1e-9`, and in practice `0.0` | `reproduce.py`, and `compare_passes.py` across the four archived passes |
| **Across devices, statistical** | anchors and action-channel values unchanged; learned-policy values within their measured device spread; seed-averaged conclusions unchanged | not a cell-level tolerance; see the measured spread below | judged by hand against the figures, not by `reproduce.py` |

The anchors and the action-channel values are exact either way, because they are
dynamic-programming quantities rather than learned ones. Learned-policy values are not:
"Device changes the numbers" below gives the measured size of the difference.
**A CPU run checked against the GPU archive is therefore expected to fail
`reproduce.py`**, and that failure is the tool working, not a defect in either run.

Verify the environment before trusting anything it produces:

```bash
python -m pytest tests/ -q                 # 24 invariant tests
python verify_claims.py                    # 31 headline numbers
python reproduce.py --verify               # diffs the shipped runs, no re-running
```

### Device selection (GPU by default)

The code **auto-selects CUDA when a GPU is visible** and falls back to CPU otherwise.
Nothing needs to be passed on the command line; every runner prints the device it
chose on its first line. To confirm what you have:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
Override with the `PRICING_DT_DEVICE` environment variable:
```bash
PRICING_DT_DEVICE=cpu    python run.py --exp all --preset full --outdir results   # force CPU
PRICING_DT_DEVICE=cuda:1 python run.py --exp all --preset full --outdir results   # pick a GPU
```
TF32 is disabled on CUDA so GPU matmuls keep the full float32 mantissa.

**Device changes the numbers, like a seed does.** Every *anchor* (optimal value,
behaviour value, per-start DP ceilings) is an exact dynamic-programming quantity and
is bit-identical on any device. Learned-policy values are not: float32 reduction
order differs between CPU and CUDA kernels, and the resulting ~1e-7 weight
differences occasionally flip a discrete argmax. Measured on E1 (20 cells), CPU vs
GPU moves `v_vanillaDT` by up to 2.1% and flips one borderline per-cell verdict in
twenty. Seed-averaged conclusions are unaffected, but **record the device next to
the seed** when a specific CSV has to be reproduced exactly. This is the measured
content of the second row of the table in Step 1: it is why a cross-device check is
judged on conclusions rather than on cells, and why `reproduce.py`'s `1e-9` belongs
to the first row only.

**Do not mix devices within one results directory.** Re-run a whole track on one
device rather than merging CPU and GPU CSVs.

## Step 2 — Verify the environment

`README.md` gives the three commands that check the installation, in increasing order of
strength; run those first. The command below is different in kind: it exercises every
experiment path end to end at a scale too small to mean anything numerically, which is
how to find out that a runner is broken before spending an hour on it.

```bash
python run.py --exp all --preset smoke --outdir results_smoke   # runs all experiments, writes 10 CSVs
python tests/test_smoke.py                                      # asserts the scientific invariants
```
The first command must finish and write its CSVs; numbers at smoke scale are **not** meaningful — this only proves every code path runs. The second runs a fast test suite that checks properties the claims depend on (optimum recovery, **action-dependent relabelling**, stitching-necessity of the data, the misspecification monotonicity flip, and pooled==segmented OPE at zero drift). It prints PASS/FAIL per test and exits non-zero on any failure. (Runs under plain `python`; `pytest tests/test_smoke.py` also works if installed.)

---

## Track A — Simulator results (your evidential core)

### Step 3 — Run the full experiments
Each can run alone; `all` runs everything.

```bash
python run.py --exp e0     --preset full --outdir results        # testbed sanity
python run.py --exp seqnec --preset full --outdir results        # RQ1 / C0 (is it really sequential?)
python run.py --exp e1     --preset full --outdir results        # RQ1 / C1 (vanilla DT fails to stitch)
python run.py --exp e2   --preset full --outdir results        # RQ2 broad factorial diagnostic
python run.py --exp e2ab --preset full --outdir results        # RQ2 prior-isolation diagnostic
python run.py --exp mis  --preset full --outdir results        # RQ2 robustness (misspecification scan, formal)
python run.py --exp e3   --preset full --outdir results        # RQ3 / C3 (OPE bias under drift)
```
Rough time for the whole `--exp all --preset full` sweep on one RTX 5060 Laptop GPU:
**~70 min** (E0/C0 ~20 s, E1 ~2 min, the rest is E2 + E2-AB + misspecification scan + E3).
To go bigger, edit `ExpConfig` in `pricing_dt/core/config.py` (`seeds`, `data_sizes`,
`noise_levels`, `ref_strengths`, `misspec_levels`).

**Run several jobs at once.** The networks are small, so a single process leaves the
GPU roughly 85% idle — it is bound by kernel-launch latency, not arithmetic. Running
4–6 independent experiments or diagnostics concurrently (each with its own
`--outdir`) pushes utilisation to ~80% and cuts wall-clock nearly proportionally.
Peak GPU memory is about 300 MB per process.

### Step 4 — Read the outputs and check the claims
CSVs land in `results/`.

- **C0** — `c0_sequential_necessity.csv`: `sequential_gap` should be ~0 at `ref_strength_delta=0` (a disguised bandit) and **grow** with delta; `sequential_structure_material` becomes **True** once the gap is non-trivial. This justifies the RL framing over a static demand-model-plus-optimiser.
- **C1** — `e1_vanilla_failure.csv`: `vanilla_stitches` is **False** and `stitch_avg_margin_vanillaDT` negative (vanilla DT cannot beat the best logged trajectory from the same start). Use per-start stitching margins, not a global best-logged trajectory ceiling.
- **C2** — use `audit/CHANNEL_RESULTS.md`, `pricing_dt/diagnostics/diag_c2_fixed.py`,
  `pricing_dt/diagnostics/diag_conditioning.py`, `pricing_dt/diagnostics/diag_heldout_protocol.py`, and
  `pricing_dt/diagnostics/diag_gate2_pricing.py`: after the Q-DT target is made action-dependent and
  conditioning is treated fairly, the structured-vs-Q-DT headline no longer
  supports an unqualified win claim. The Gate 2 evidence in this package is
  `results_gate2_masked_20260821/`, the source for Table 4.12; the batch in
  `results_gate2_pricing_full_20260818/` is superseded and not included, and can be
  regenerated with the full command above. The smoke command only verifies that the
  Gate 2 path runs. `run.py --exp e2` remains a broad
  factorial diagnostic if regenerated, but it is not the claim-critical source.
- **C1 magnitude / well-posedness** — `stitch_margin_*` columns = learned policy value minus the best **de-noised** logged return **from the same start state**, averaged over starts (positive = stitching achieved). The per-start framing is deliberate: comparing a single luckiest trajectory against a start-averaged policy value would be ill-posed. `v_optimal` is the optimum from the logged starts and always sits above `v_behaviour_expected`.
- **Mechanism** — use `pricing_dt/diagnostics/diag_e2ab_fixed.py`, `pricing_dt/diagnostics/diag_channel_ladder.py`,
  `pricing_dt/diagnostics/diag_target_decomp.py`, and `pricing_dt/diagnostics/diag_trust_region.py`. The current mechanism
  evidence is the channel/support axis in `audit/CHANNEL_RESULTS.md`: model error is
  damaging in the action channel and bounded in the goal/support channel.
- **Misspecification (formal)** — `mis_scan_summary.csv`: `prior_monotone` is computed on raw `log_demand`, while `prior_degenerate_after_clamp` flags cases where `forward()` clipping makes the curve constant. The key quantity is `mean_advantage` over the Q-DT floor: gross misspecification erodes and can reverse the structured advantage, bounding the result on prior correctness. Do not read the scan as strictly monotone in severity.
- **Mechanism diagnostics + a pricing baseline** — five standalone diagnostics on the hardest cell (run directly, not via `run.py`):
  ```bash
  python -m pricing_dt.diagnostics.diag_optimism_verdict   --outdir results   # decisive: shape vs magnitude, + lam frontier
  python -m pricing_dt.diagnostics.diag_action_shape       --outdir results   # is the gain ranking accuracy?  (no)
  python -m pricing_dt.diagnostics.diag_denoise_verdict    --outdir results   # is the gain denoising/smoothness?  (no)
  python -m pricing_dt.diagnostics.diag_estimate_optimize  --outdir results   # canonical estimate-then-optimize pricing baseline
  python -m pricing_dt.diagnostics.diag_demand_curve_amplification --outdir results_demand_curves_20260818  # state-level demand/Q curves
  ```
  `optimism_verdict.csv`: the structured prior beats a magnitude-**matched** unstructured optimism (B's calibrated target scaled to A's inflated level) by **+0.26, paired Wilcoxon p=0.004**, and `optimism_frontier.csv` traces a clean **monotone** value-vs-`lam` knob (0.44→0.67 as structure-shaping goes 0→1). This supports target **shape**, not only target level, as the active factor. `action_shape_scan.csv`: the structured model ranks the true per-state action values **worse** than the unconstrained one (Spearman A=−0.19 vs B=+0.60, and its argmax-agreement is seed-invariant = prior-dominated), so ranking accuracy is not the explanation in this diagnostic. `denoise_verdict.csv`: A's targets are **higher**-variance, not smoother (std 211 vs 133), and smoothing B recovers only 17% of the gap, so denoising is not sufficient to explain the result. Together these diagnostics are most consistent with structured, prior-determined optimistic **shaping**. `estimate_optimize.csv`: every DT variant beats the canonical **estimate-then-optimize** pipeline in this hardest-cell diagnostic (best plan `EtO_unconstrained` 0.19 < weakest DT `B_calibrated` 0.33), while the same structured demand model performs very poorly as a direct planner (`EtO_structured` −4.5; in-model value ≈3.75× the true optimum, optimizer's-curse gap +1631). The project therefore treats the prior's optimism as harmful for direct planning but useful when contained by the DT's imitation constraint. The corrected account is Appendix E of the dissertation.
  `demand_curve_amplification`: this is the state-level explanation of the previous sentence. It writes `demand_curve_state_summary.csv`, `demand_curve_points.csv`, `demand_curve_seed_summary.csv`, plus `demand_curve_summary.png` and `demand_curve_top_states.png`. Current full-run means: structured action-channel value `-4.543`, in-model gap `+1631.0`, wrong-action share `0.73`, mean state policy regret `175.3`, unsupported-choice share `0.90`, outside top-3 support share `0.62`; unconstrained action-channel value `0.193`, in-model gap `+95.2`, mean state policy regret `22.2`. The top-state curves show the optimizer selecting `p=0.50` in high-reference states where the true dynamic optimum is about `p=1.40`, because fitted demand is most optimistic exactly where the argmax searches. Use this as a mechanism figure, not as a new model leaderboard.
- **Gate 2 corrected-pricing full rerun** — a superseded batch, NOT included in this
  package; regenerate it with the `--outdir results_gate2_pricing_full_20260818`
  command above. It runs 9 pricing cells x 10 seeds x 8 methods = 720 policy
  evaluations. Overall
  mean normalised values: Structured DT with support mask `0.745`, corrected Q-DT
  `td` denoised `0.711`, unmasked Structured DT `0.688`, Q-DT `q_sa` `0.626`,
  IQL `0.569`, Vanilla DT with support mask `0.552`, Vanilla DT `0.490`.
  Support containment improves Structured DT by `+0.0567` (raw p=`3.7e-11`) and
  Vanilla DT by `+0.0620` (raw p=`3.5e-09`). Within those eight methods the
  support-masked Structured DT is the best overall mean and beats corrected Q-DT `td`
  by `+0.0339` (raw p=`0.0075`) — but note that the shipped
  `results_gate2_masked_20260821/` batch masks eleven methods rather than eight, and
  there support-masked IQL (`0.893`) and support-masked Q-DT (`0.859`) both outrank
  support-masked Structured DT (`0.740`), which is why the caveat below is the
  operative reading. Corrected Q-DT `td` is also
  corrected Q-DT `td` is strongest at `N=1600` (`0.771` vs support-masked
  Structured DT `0.707`). Treat this as support/channel evidence, not a universal
  Structured-DT leaderboard claim.
- **Strong tabular B' diagnostics** — optional hardest-cell checks asking whether a
  modern tabular forecaster without economic structure can replace the structured
  demand prior as a relabelling signal:
  ```bash
  python -m pricing_dt.diagnostics.diag_bprime_xgb       --outdir results
  python -m pricing_dt.diagnostics.diag_bprime_catboost  --outdir results_catboost_20260815
  python -m pricing_dt.diagnostics.diag_bprime_lightgbm  --outdir results_lightgbm_20260815
  ```
  Current full-run means under the shared 80/20 split: A structured `0.561`,
  XGBoost `0.458`, CatBoost `0.427`, LightGBM `0.376`; LightGBM gives the
  clearest paired gap (`+0.185`, p=`0.002`).
- **Profit-objective sensitivity** — optional simulator-only check replacing
  revenue with `(price - unit_cost) * demand` while preserving exact DP:
  ```bash
  python -m pricing_dt.diagnostics.diag_profit_sensitivity --outdir results_profit_20260815 --costs 0.0,0.2,0.4
  ```
  Current full-run means: at unit costs `0.0/0.2/0.4`, structured-DT profit
  relabel gives `0.670/0.570/0.522`, while direct structured
  estimate-then-optimise remains far below behaviour at `-4.543/-3.515/-2.599`.
  The channel warning survives, but the structured relabel advantage is
  cost-sensitive.
- **Commercial constraints** — optional simulator-only check adding a business
  constraints layer to estimate-then-optimise:
  ```bash
  python -m pricing_dt.diagnostics.diag_pricing_constraints --outdir results_constraints_20260815
  ```
  Current full-run means for the structured direct optimiser: unconstrained
  `-4.543`, margin floor `-3.805`, price-change limit `-0.940`, logged-support
  top-3 `0.472`, all constraints `-0.806`. Support constraints strongly reduce
  optimizer's-curse damage (`support - none` median `+4.921`, p=`0.001953`),
  but tight combined constraints also hurt the oracle (`all - none` median
  `-1.753`, p=`0.001953`). Read this as a commercial safety/feasibility
  trade-off, not as a replacement for the channel mechanism.
- **Commercial-context robustness** — optional simulator-only check adding
  observed season and product-type context while preserving exact DP:
  ```bash
  python -m pricing_dt.diagnostics.diag_commercial_context --outdir results_context_20260815 --modes season_product
  ```
  Current full-run means with three product types and four seasons:
  vanilla DT `0.426`, structured-DT context relabel `0.361`, direct structured
  estimate-then-optimise `-3.454`, direct unconstrained estimate-then-optimise
  `0.814`. The structured action-channel warning survives strongly
  (`structuredDT_context - EtO_structured` median `+3.857`, p=`0.001953`), but
  the structured relabel is not better than vanilla DT (median `-0.038`,
  p=`0.164`). Treat this as external-validity boundary evidence, not a new
  headline win.
- **C3** — `e3_ope_summary.csv`: the comparative quantity `segmentation_benefit` = `mean_abs_bias_pooled` − `mean_abs_bias_segmented` is **0 at `drift=0`** (identical by construction — a correctness check) and **grows monotonically with drift** (and with `logger_tv`), reaching ≈40 at the highest drift in the full run. Crucially this is the **weak (state-independent) q̂** headline; the `*_strong` columns (a capable q̂) show the effect is **largely masked** — a property of doubly robustness, demonstrated not just asserted. Honest scope: absolute "segmented ≈ 0" is **not** achieved (the weak q̂ leaves a large baseline bias); the claim is the drift-growing **reduction** in bias from segmenting.

---

## Track B — Real dataset: Online Retail II

**What it is:** ~1.07M real UK online-retail transactions, 2009–2011 (UCI id=502, DOI 10.24432/C5CG6D, CC BY 4.0). Columns: `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`.

### Step 5 — Get the data

The raw Online Retail II file is not bundled in the review folder to keep the
submission small. Download it manually when re-running the real-data calibration
or realism checks.

- UCI: https://archive.ics.uci.edu/dataset/502/online+retail+ii  → an `.xlsx` (two sheets, 2009–2010 and 2010–2011).
- Or Kaggle mirror `mashlyn/online-retail-ii-uci` → `online_retail_II.csv` (single file, easiest).
Place the file in the project folder and pass it with `--real-data`.

`ucimlrepo` remains an optional best-effort fallback, but Online Retail II is not
reliably exposed through UCI's Python import API; the local file path is the
reproducible route.

### Step 6 — Calibrate the simulator to real elasticity
```bash
python run.py --exp calib --real-data online_retail_II.csv --outdir results
```
Reads `results/calibration.csv`: `median_elasticity` plus `suggested_beta`, `suggested_elasticity_lo/hi`. Put those into `pricing_dt/core/config.py` (`SimConfig.beta`, `ModelConfig.elasticity_lo/hi`) and **re-run Track A** so the ground-truth experiments sit at a realistic elasticity. This is the main, defensible use of the real data — it keeps every C1/C2/C3 result exact while anchoring the regime to reality.

Additional real-data evidence:
```bash
python -m pricing_dt.diagnostics.diag_real_pricing_evidence --real-data online_retail_II.xlsx --outdir results_real_pricing_evidence_20260815
```
Current run: 1,041,670 cleaned transactions, 4,917 products, transaction-level
median absolute elasticity `1.372` (IQR `0.953`–`1.826`), weekly product
fixed-effect absolute elasticity `1.914` (descriptive, not causal), median
historical price-bin coverage `6/11` per product, and at each product-time
decision `10/11 = 0.909` of the discrete price grid is counterfactual.
This supports calibration and overlap discussion; it does not identify a
ground-truth optimal price.

### Step 7 — Realism check on real logs (OPE only)
```bash
python run.py --exp realism --preset full --real-data online_retail_II.csv --outdir results
```
Reads `results/realism.csv`: episode counts, whether the fitted demand curve is monotone in price (the interpretable artefact), and OPE value estimates for the logged policy vs vanilla-DT vs BC on a **time-based held-out split**. Real data has no ground-truth policy value, so these are OPE estimates only — reported as a realism check, not as the headline evidence.

---

## Step 8 — From CSVs to figures
A plotting script turns the CSVs into PNGs automatically:
```bash
python make_figures.py --indir results            # writes results/figures/*.png
```
It produces, for whichever current CSVs are present: the C0 sequential-gap curve,
the E1 vanilla-failure bars, the misspecification-scan curve, the C3 bias-vs-drift
lines, and the corrected fixed-QDT/channel figures (`c2_fixed_baseline.png`,
`e2ab_fixed.png`, `elasticity_fixed.png`, `channel_ladder.png`, `target_decomp.png`,
`trust_region_scissors.png`, etc.). Each figure is skipped gracefully if its CSV
is absent, so it works after a single experiment or all of them. The CSVs are tidy
if you prefer to plot your own (matplotlib/seaborn).

The demand-curve amplification diagnostic writes its own figures directly to its
`--outdir`, because it needs state-level point curves rather than only aggregate
CSV plots:
```bash
python -m pricing_dt.diagnostics.diag_demand_curve_amplification --outdir results_demand_curves_20260818
```

## Notes / honest limits
- `smoke` is for pipeline checks only; the separations appear at `full` scale (and scale `ExpConfig` further if you have GPU time).
- The real-data episode definition (weekly aggregation, per-product price scaling, window length) in `pricing_dt/core/realdata.py` involves modelling choices — review and adjust them for your products; they are documented in that file.
- Calendar seasonality is not explicitly modelled in the current real-data track:
  the split preserves time order, but there are no week-of-year/month/holiday
  covariates. That is acceptable for the controlled channel-mechanism claim; a
  deployment-facing pricing study should add seasonal context and rerun the
  realism/OPE diagnostics.
- `train_iql` is implemented in `pricing_dt/core/baselines.py`. ContextFormer is deliberately not
  included in the cleaned mainline; add it later as a separate explicit baseline
  if it becomes load-bearing. The claim-critical comparators (vanilla DT,
  corrected Q-DT, structured DT, EDT-style inference, and discrete IQL) are
  implemented.
