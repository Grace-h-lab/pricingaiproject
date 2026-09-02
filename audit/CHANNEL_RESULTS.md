# Results: fixing the baseline, and what the goal channel actually consumes

*Exploratory results of 2026-08-19. `QDT_legacy` and the value 0.438 are the defective
comparator, corrected in Appendix E.1.*

Everything below is at the project's standard hardest cell (`N=100`, `noise=0.5`,
`delta=3`, 10 seeds unless stated), same anchors and protocol as
`pricing_dt/diagnostics/diag_optimism_verdict.py` / `pricing_dt/diagnostics/diag_estimate_optimize.py`, so the numbers are
directly comparable with the published tables. Paired Wilcoxon across seeds.

Provenance check: the new harness reproduces the published numbers exactly —
`A_structured` 0.670 (published 0.670), `QDT_legacy` 0.439 (published 0.438),
trust-region endpoints 0.670 and −4.543 with in-model 1800.3 and curse gap +1631
(published −4.543, ~1800, +1631). Differences below are therefore attributable to
the changes made, not to harness drift.

---

## 1. The blocking defect, confirmed and quantified

`pricing_dt/core/qdt.py` relabelled with the STATE value `V(s_t) = max_a Q(s_t,a)`, which does not
depend on the logged action — the exact defect Appendix E.1 documents and corrects
for the structured relabeller, left in the comparator.

It is now measured directly. `pricing_dt/diagnostics/diag_target_stats.py` computes the within-state
spread of each target column (the variation that lets a DT prefer one logged
continuation over another at the same state):

| arm | spread_within | disc_ratio | value |
|---|---:|---:|---:|
| A_structured | 51.2 | 0.334 | 0.670 |
| QDT_qsa (fixed) | 452.9 | 0.380 | 0.604 |
| QDT_td (fixed) | 455.9 | 0.589 | 0.581 |
| vanilla | 83.0 | 0.878 | 0.441 |
| **QDT_legacy (broken)** | **0.000** | **0.000** | 0.439 |
| oracle_Qstar | 4.1 | 0.080 | 0.359 |

`spread_within = 0.000` exactly: the broken target carried **zero** information
distinguishing trajectories from the same state. That is the defect, measured.

**Cost of the fix** (`results/channel_ladder.csv`):

| comparison | value |
|---|---|
| advantage over the BROKEN baseline | **+0.231** |
| advantage over the best FIXED baseline (`QDT_qsa` 0.604) | **+0.066**, p = 0.049 |
| advantage over the CQL-α-tuned fixed baseline (0.592) | **+0.078**, p = 0.23 (n.s.) |

The α sweep {0.1, 1.0, 5.0} barely moves the baseline (0.582 / 0.581 / 0.573), so
conservatism was never the issue — action-dependence was.

---

## 2. The evaluation protocol is a second confound

Every arm is conditioned on the 0.95 quantile of **its own** RTG column
(`experiments._eval_dt`). Different relabellers produce different target
distributions, so this is not one protocol — it is a different ask per arm.

`pricing_dt/diagnostics/diag_conditioning.py` gives every arm the same treatment (sweep the conditioning
target, report each arm's best):

| arm | at default q0.95 | at its own best | best rule |
|---|---:|---:|---|
| QDT_td (fixed) | 0.581 | **0.686** | q1.0 |
| A_structured | 0.670 | 0.684 | **q0.95** |
| vanilla | 0.441 | 0.598 | max ×1.25 |
| oracle_Qstar | 0.359 | 0.568 | max ×1.25 |

At equal generosity: **A − QDT_td = −0.002, p = 0.73** (indistinguishable);
A − vanilla = +0.086, p = 0.13 (n.s.).

The structured arm's optimum sits exactly on the default rule and falls away
sharply on both sides (−0.74 at q0.5, −0.30 at max×2), while every comparator peaks
elsewhere and vanilla holds a broad plateau. The published protocol is thus optimal
for the proposed method and suboptimal for all three comparators
(`results/figures/conditioning_sweep.png`).

Caveat, stated plainly: "best over a swept grid" is oracle-tuned on the test
objective for *every* arm, so all these numbers are upward-biased. The fair reading
is not "0.686 is achievable in practice" but "under equal treatment the arms are
indistinguishable". The clean fix for the paper is to select the conditioning target
on a held-out split, for all arms, and report that.

**Net of §1 and §2: the C2 headline does not survive.** It required both a broken
comparator and a comparator-unfavourable evaluation rule.

### 2.1 The robustness table re-measured against the fair baseline

`run.py --exp mis` re-run into a `results_fixedqdt/` directory that was not kept. The structured column reproduces
to the digit (0.670 / 0.636 / 0.306 / 0.421 / 0.410), which is the control — it
should not depend on the Q-DT fix — while the floor rises 0.438 → 0.581:

| severity | 0.0 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---:|---:|---:|---:|---:|
| structured | 0.670 | 0.636 | 0.306 | 0.421 | 0.410 |
| Q-DT floor, broken | 0.438 | 0.438 | 0.438 | 0.438 | 0.438 |
| Q-DT floor, **fixed** | 0.581 | 0.581 | 0.581 | 0.581 | 0.581 |
| advantage, broken | +0.231 | +0.198 | −0.133 | −0.017 | −0.028 |
| advantage, **fixed** | **+0.089** | +0.055 | **−0.276** | −0.160 | −0.171 |

The published reading ("correct-prior advantage ≈ 0.23, eroding as severity rises")
becomes: a slim advantage only at a near-correct prior, and a clear *deficit* from
severity 1.0 onward. The method is worse than a correct Q-DT across most of the
misspecification range.

Two things to carry into the discussion:

- The curve is **non-monotone** (0.670 → 0.636 → 0.306 → 0.421 → 0.410): severity
  1.0 is the minimum and the grossly-wrong priors at 2.0/4.0 *recover*. There is no
  graceful-degradation curve to plot, so that claim should not be made.
- What is true, and is the better claim, is a **floor**: across every severity the
  goal channel stays in [0.306, 0.670] — always above the behaviour policy (0.0) and
  never anywhere near the action channel's −4.543. The imitation constraint bounds
  the downside; it does not guarantee target quality.

### 2.2 C2 along the data-size axis — the claim fails everywhere, not just at one cell

`pricing_dt/diagnostics/diag_c2_fixed.py`, `noise=0.5`, 10 seeds per cell. (This replaces a full
`run.py --exp e2` re-run, which was started and abandoned as infeasible here:
`e2_core` performs ~277 rollouts per arm per cell across 9 cells. The memoised
evaluator gives an identical number from ~21 rollouts, since the rollout is
deterministic per start bin; the trade is the noise axis, which the published grid
already shows to be the less informative one.)

| N | vanilla | Q-DT broken | Q-DT td | Q-DT q_sa | structured | adv vs broken | adv vs fair | p | Holm p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.441 | 0.438 | 0.581 | 0.604 | 0.670 | +0.231 (p=.002) | **+0.066** | .049 | **.146** |
| 400 | 0.531 | 0.218 | 0.766 | 0.767 | 0.720 | +0.503 (p=.002) | **−0.047** | .084 | **.168** |
| 1600 | 0.548 | 0.241 | 0.721 | 0.642 | 0.679 | +0.438 (p=.002) | **−0.042** | .084 | **.168** |

- **No cell survives Holm correction** against a fair baseline, and the advantage
  **reverses sign** at N ≥ 400.
- The published "advantage grows with N" direction note is fully explained: the
  broken baseline *degrades* with data (0.438 → 0.218 → 0.241) while a correct one
  *improves* (0.581 → 0.766 → 0.721) — which is what a value-based method should do
  with more data, and the original result was the reverse.
- Sanity: against the broken baseline this harness reproduces advantages of
  +0.231/+0.503/+0.438, bracketing the published grid mean of +0.391. So the effect
  reproduces, and it is attributable to the comparator.

Figure: `results/figures/c2_fixed_baseline.png`.

### 2.3 C1 (stitching) — the third headline claim, contaminated by the same two problems

C1 was never re-checked after the defect was found, and it inherits both confounds:
the published margins compare against `Q-DT -25.314` (the zero-signal target), and
every arm was read at its own q0.95. `pricing_dt/diagnostics/diag_c1_fixed.py` re-measures the same metric
(`metrics.stitching_score`: per-start policy value minus the best de-noised logged
return from the same start) on test bins under held-out target selection.

| arm | margin (held-out) | beat fraction | margin (default rule) |
|---|---:|---:|---:|
| structuredDT | **−2.11** | 0.398 | −2.72 |
| QDT_td (fixed) | −2.98 | 0.330 | −7.41 |
| QDT_qsa (fixed) | −5.44 | 0.370 | −6.23 |
| vanillaDT | −8.20 | 0.257 | −15.14 |
| oracle_Qstar | −8.96 | 0.270 | −19.30 |
| QDT_legacy (broken) | −11.07 | 0.270 | −16.57 |

Paired vs structured: QDT_legacy +8.96 (p=.0039, **holm=.0195**), oracle_Qstar +6.85
(holm=.148), vanillaDT +6.09 (holm=.316), QDT_qsa +3.33 (holm=.316),
**QDT_td +0.87 (p=.85)**.

- "The structured DT is the significantly best stitcher of the three DT variants
  (p < 1e-16, best in 95.6% of cells)" — **survives only against the broken
  comparator**. Against a correct Q-DT it is a tie.
- "No variant clears the strict per-start majority bar" — **survives**; the best
  beat-fraction is 0.398 < 0.5. The published intrinsic-ceiling reading stands.
- The protocol effect is again large and asymmetric: honest selection improves
  QDT_td by 4.4 margin units, vanilla by 6.9, oracle by 10.3, and structured by 0.6.

**E1 is settled by the same run.** The claim "vanilla DT does not reliably stitch"
holds — beat fraction 0.257, well under 0.5 — but the published strict-success figure
of 2.2% becomes ~26% under honest conditioning, an order of magnitude, so the number
must be restated even though the conclusion stands.

### 2.4 E2-AB prior-isolation ablation, re-run fairly

`pricing_dt/diagnostics/diag_e2ab_fixed.py`, held-out protocol. Arm D was the broken Q-DT, so it is split.

| arm | held-out | default rule | published |
|---|---:|---:|---:|
| C_misspecified | **0.685** | 0.677 | ~0.67 |
| A_full_prior | 0.681 | 0.668 | ~0.67 |
| D_bootstrap_fixed | 0.665 | 0.584 | — |
| D_bootstrap_broken | 0.521 | 0.428 | ~0.44 |
| B_no_constraints | 0.518 | 0.327 | ~0.33 |

The three claims the ablation was built to make now separate:

- **A > D, "a structured relabel beats the bootstrapped value": DEAD.** +0.016,
  p = 0.77. The published 0.67 ≫ 0.44 was the defect: `D_fixed − D_broken = +0.144`,
  p = 0.002.
- **A > B, "the economic prior beats an unconstrained demand model": SURVIVES.**
  +0.163, p = 0.0098 (Holm over the three pre-specified comparisons: 0.029).
- **A ≈ C, "prior correctness is not what matters": SURVIVES.** −0.004, p = 0.56.

So the surviving residue is narrow but real: a **bounded/monotone parameterisation**
yields a better conditioning target than an unconstrained MLP — but no better than a
bootstrapped value function, and whether the bounds are *correct* is irrelevant. That
is consistent with §4.1 and §4: what the channel needs is a non-degenerate aspiration
field, which both the bounded prior and a bootstrapped Q supply, and which the
unconstrained MLP (0.518) and the exact `Q*` (0.556) do not.

### 2.5 Elasticity band, re-run fairly — overturned

`pricing_dt/diagnostics/diag_elasticity_fixed.py`, held-out protocol, over the Online Retail II band.

| β | structured | Q-DT fixed | advantage | p | Holm | published |
|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 0.704 | 0.742 | −0.038 | .30 | .60 | +0.00 |
| **1.37** (data median) | 0.528 | 0.638 | **−0.110** | **.014** | .055 | +0.23 |
| 1.83 | 0.612 | 0.656 | −0.044 | .084 | .25 | +0.30 |
| 2.00 | 0.681 | 0.665 | +0.016 | .77 | .77 | +0.51 |

The published reading — advantage growing monotonically with elasticity, significant
at and above the median — does not survive in any part. The advantage is negative at
three of four points, there is no monotone trend, and at **the elasticity the real
data actually calibrates to (β = 1.37) the method is behind by 0.110** (raw p = .014).
That is the economically relevant operating point, so this is the least favourable
place for the table to fail.

### 2.6 Sparse coverage — the surviving boundary, re-checked and strengthened

`pricing_dt/diagnostics/diag_sparse_fixed.py`, held-out protocol, 10 seeds (the published probe used 5), and
with the fixed Q-DT added — the original compared only A against B, so it could not
distinguish "thin support hurts the structured relabeller" from "thin support hurts
return-conditioned methods as a class". The axis account predicts the latter.

| K | A_structured | B_uncon | QDT_fixed | vanilla | A−B | p | **A−QDT** | p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0.665 | 0.526 | 0.663 | 0.564 | +0.139 | .020 | +0.002 | 1.00 |
| 5 | 0.176 | 0.294 | 0.457 | 0.377 | −0.118 | .065 | **−0.281** | **.002** |
| 3 | −0.230 | −0.441 | 0.046 | −0.081 | +0.211 | .049 | **−0.277** | **.002** |
| 2 | −1.911 | −1.882 | −1.882 | −1.912 | −0.028 | .50 | −0.028 | .50 |

**The boundary claim holds, and is now properly established.** Drops from K=11 to K=2
are 2.576 / 2.408 / 2.545 / 2.476 — a spread of 0.168 across a ~2.5-unit collapse, with
all four arms converging to ≈ −1.9 (published: ≈ −1.96). Thin action support degrades
every return-conditioned method essentially equally, which is what the support /
trust-region account predicts and what the A-vs-B-only design could not show.

**A new adverse finding at intermediate coverage.** At K=5 and K=3 the structured
relabeller is *significantly worse* than a fair Q-DT (−0.281 and −0.277, both p=0.002).
Mechanically this is coherent: the structured target's greedy roll-forward extrapolates
the demand curve into price regions the log never visits, which is exactly where thin
coverage bites, whereas a bootstrapped Q only ever scores actions present in the data.

Note this **inverts the original motivation for the structure**. The published rationale
was that under sparse coverage an unconstrained model has no data to pin the demand
curve while a bounded prior stays sane. The published probe already refuted that against
B; against a correctly-implemented Q-DT it does not merely fail, it reverses.

---

## 3. What DOES survive — and it is the better claim

### 3.1 The channel contrast is real, large, and now causally demonstrated

`pricing_dt/diagnostics/diag_trust_region.py` interpolates continuously between the goal channel and the
action channel using the *same* fitted model, by admitting the top-`m` actions of
the DT's own ranking and choosing among them by the model's planning value:

| m | 1 | 2 | 3 | 4 | 6 | 9 | 11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **fitted structured model** | **+0.670** | +0.585 | −0.905 | −2.976 | −4.148 | −4.135 | **−4.543** |
| **exact Q\* (control)** | +0.670 | +0.897 | +0.960 | +0.980 | +1.000 | +1.000 | **+1.000** |
| in-model value (fitted) | 456 | 505 | 683 | 593 | 724 | 1461 | 1800 |
| logged support of chosen action | 0.453 | 0.401 | 0.116 | 0.039 | 0.020 | 0.030 | 0.030 |

`m=1` reproduces the structured DT (0.670) and `m=11` reproduces
estimate-then-optimize (−4.543, in-model 1800, curse gap +1631) — both published
endpoints, hit exactly, so the axis is validated end-to-end.

The control is what makes this an experiment: with an **accurate** model, widening
the same trust region *improves* value monotonically to the optimum (+1.000, curse
gap 0.000 throughout). So the damage is model error being cashed in, not the loss of
imitation per se. The mechanism claim moves from inference to intervention.

Note also how narrow the safe region is: admitting just the top **3** of 11 actions
is already enough to go negative.

### 3.2 The data channel is not a third safe channel — it is a truncation dial

`pricing_dt/diagnostics/diag_data_channel.py`, same model, Dyna/MBPO-style synthetic transitions:

| synthetic-action source | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| greedy w.r.t. the model (Dyna-faithful) | +0.383 | +0.129 | −1.713 | **−3.826** |
| from a BC policy (stays near support) | +0.411 | +0.399 | +0.401 | +0.411 |

With model-greedy rollouts the data channel degrades smoothly toward the action
channel's −4.543 as the rollout lengthens; its safety comes **entirely** from
short-rollout truncation, which is exactly what MBPO's truncation is for. Keep the
synthetic actions inside the logged support and it is flat and safe — but then it is
also no better than plain BC (0.407).

So the three channels are not three separate regimes; they are one axis, and what
positions you on it is **how far the model is allowed to move actions off the logged
support**. That is a cleaner and more general statement than "three channels have
different fates".

### 3.3 A two-line baseline matches the proposed method

Model-*filtered* BC — score logged trajectories by the model, keep the top half,
behaviour-clone them:

| arm | value | vs A_structured |
|---|---:|---|
| filtBC keep 0.5 | **0.706** | +0.036, p = 0.23 (tie) |
| A_structured | 0.670 | — |
| filtBC keep 0.25 | 0.628 | −0.042, p = 0.38 (tie) |
| real_DT (vanilla) | 0.441 | −0.228, p = 0.004 |
| real_BC | 0.407 | −0.263, p = 0.002 |

The cheapest possible use of the domain model — touching neither actions, nor
targets, nor the transition distribution, only which real data survives —
**statistically ties** the proposed method. It does not beat it, but the proposed
method's machinery is not earning its complexity against it.

---

## 4. New mechanistic finding: the goal channel does not want an accurate target

The oracle rung was missing from the ladder. Added (`relabel.oracle_rtg`, exactly
`Qstar[t, b_t, a_t]`), it is the **worst** relabelling arm:

| controlled swap (identical target form, only the model changes) | Δ | p |
|---|---:|---:|
| A_structured (0.670) − oracle_Qstar (0.359) | **+0.311** | 0.002 |
| QDT_qsa (0.604) − oracle_Qstar (0.359) | +0.245 | 0.004 |

Both a structured-model estimate *and* a bootstrapped estimate of `Q*` beat `Q*`
itself. This survives the conditioning confound (0.684 vs 0.568, p = 0.027).

**Why**, from `pricing_dt/diagnostics/diag_target_stats.py`: `Q*(s_t,a_t)` already contains the optimal
continuation, so a *bad* logged action still scores high and the target barely
separates good from bad trajectories at the same state — its within-state
discrimination ratio is 0.080 against the structured target's 0.334, i.e. it is
nearly as degenerate as the *broken* Q-DT (0.000). The correct value function is a
poor conditioning signal precisely because it is correct.

Two things this rules out:

- **Not magnitude.** Target inflation ranges 0.99× to 16.2× across arms with no
  relation to value (Spearman +0.14, p = 0.76) — the DT standardises its return
  token, so level is washed out. This also explains why the published
  `vanilla_matchA` control found nothing: globally rescaling a training column is a
  near-no-op under standardisation. The live knob is the *ask*, not the column.
- **Not fidelity.** Across arms, higher cross-state fidelity to true achievable
  value goes with *lower* policy value (Pearson −0.87, p = 0.012; Spearman −0.63,
  p = 0.13 — directional, underpowered at n = 6 arms). The load-bearing evidence is
  the controlled swap above, not this correlation.

### 4.1 Which part of the target does the work

`pricing_dt/diagnostics/diag_target_decomp.py` (n = 9 seeds; the 10th was lost to a process death and
recovered from the log at 2 dp) crosses the two terms of
`R_hat_t = r_hat(s_t,a_t) + sum_{k>t} max_p r_hat(s_k,p)`:

| action \ potential | model | oracle | mean | shuffle |
|---|---:|---:|---:|---:|
| model | **0.666** | 0.389 | −0.839 | 0.250 |
| oracle | 0.650 | 0.378 | −2.077 | 0.369 |
| mean | 0.673 | 0.410 | 0.357 | 0.364 |
| shuffle | 0.653 | 0.403 | 0.309 | 0.321 |

| manipulation | Δ | p |
|---|---:|---:|
| delete the ACTION term | +0.007 | 0.91 |
| shuffle the ACTION term | −0.013 | 1.00 |
| make the ACTION term exact | −0.016 | 0.09 |
| delete the POTENTIAL term | **−1.505** | 0.004 |
| shuffle the POTENTIAL term | −0.416 | 0.004 |
| make the POTENTIAL term exact | **−0.277** | 0.004 |
| delete BOTH | −0.309 | 0.004 |

The outcome is fixed by the potential term alone; the action term is inert under
deletion, shuffling, *and* correction.

This looks paradoxical against the project's own action-dependence fix, and the
resolution is worth stating in the paper: in a reference-price MDP the logged action
enters the potential term too, through the transition (the roll-forward starts from
the post-action reference). The action-dependence that matters is the **downstream**
one — how the action moves the state — not the immediate-revenue one. An action's
value here is mostly its effect on the future reference price.

Also note deleting *both* terms (0.357, ≈ BC's 0.407) beats deleting only the
potential term (−0.839): a **misleading** target is worse than no target at all.

---

## 4.2 The honest protocol — held-out conditioning-target selection

`pricing_dt/diagnostics/diag_heldout_protocol.py`. Start bins split disjointly per seed (30% selection /
70% test); every arm sweeps the same grid (q0.50–q1.0 plus max ×{1.25,1.5,2.0} — it
must cover every arm's optimum, or the protocol invents a new unfairness), picks its
target on selection, and the test set is read once. `filtBC` needs no target and is
therefore protocol-immune. The two target-decomposition cells that carry the
mechanism claim are included, since they were exposed to exactly the same confound.

| arm | **held-out (test)** | sd | at the default q0.95 | gain from honest selection |
|---|---:|---:|---:|---:|
| action_deleted | **0.705** | 0.128 | 0.678 | +0.027 |
| filtBC keep 0.5 (ask-free) | **0.704** | 0.063 | 0.704 | — |
| A_structured | 0.681 | 0.087 | 0.668 | **+0.013** |
| QDT_td | 0.665 | 0.034 | 0.584 | +0.081 |
| QDT_qsa | 0.622 | 0.064 | 0.608 | +0.014 |
| vanilla | 0.573 | 0.120 | 0.448 | +0.125 |
| oracle_Qstar | 0.556 | 0.101 | 0.374 | **+0.182** |
| potential_exact | 0.554 | 0.096 | 0.375 | +0.179 |
| B_unconstrained | 0.518 | 0.097 | 0.327 | +0.191 |

The last column is the §2 confound quantified: the default rule cost every
comparator 0.08–0.19 and cost the proposed method 0.013.

Paired vs `A_structured`, Holm over 8 comparisons — **nothing survives**:
B_unconstrained +0.163 (p=.0098, holm=.078), potential_exact +0.127 (holm=.150),
oracle_Qstar +0.125 (holm=.176), vanilla +0.108 (holm=.420), QDT_qsa +0.059,
QDT_td +0.016 (p=.77), filtBC −0.023, action_deleted −0.024.

**Read this as underpowered, not as a demonstrated null.** With 8 comparisons at
n=10 seeds, Holm requires raw p ≤ 0.00625 for the leading comparison, while the
observed spread is 0.187 with a stable ordering. "No arm separates" is a statement
about this design's resolution.

What the honest protocol does settle:

- **The main-line claim is finished.** A ties the fixed Q-DT (+0.016, p=0.77), ties
  the ask-free filtered-BC baseline (−0.023), and ties the variant with its own
  action term deleted (−0.024). No qualification left.
- **The oracle-Q\* finding shrinks and loses significance**: +0.311 (default rule)
  → +0.125 (honest, raw p=0.029, holm=0.176). It cannot carry a section on this
  evidence. Directionally it holds, and `potential_exact` (+0.127) moves with it, as
  the same "accuracy hurts" family should.
- **Per-arm-best was a good approximation after all.** Transfer losses are 0.015–0.042
  and near-uniform (A 0.028 vs oracle 0.018), so the earlier per-arm-best estimate
  (+0.115, p=0.027) and the honest one (+0.125, p=0.029) agree. The concern that
  test-set-max flattered the most peaked arm was correct in direction and ~0.01 in size.
- **The q0.95 peak did not transfer**: A's modal held-out choice is q0.975 and
  selection matched the test-set optimum in only 50% of seeds. The apparent
  coincidence of the default rule with A's optimum was partly test-set overfitting.
- **The channel/trust-region result is untouched**, and by construction: at large `m`
  the DT's probabilities are irrelevant, so the −4.543 endpoint does not depend on any
  conditioning choice.

### Pre-registered decision, and the one deviation worth flagging

The registered rule for this outcome was: collapse to the axis paper, drop the
potential-shift probe, demote `Q*` to a single sentence. That rule is followed here.

The one honest caveat: the data are equally consistent with real effects this design
cannot resolve. If the `Q*` result is wanted as a section rather than a sentence, the
legitimate move is a **confirmatory replication declared in advance** — 30 seeds,
three arms (`A_structured`, `oracle_Qstar`, `potential_exact`), **one** pre-specified
comparison so multiplicity is not paying for eight. Same cost as this run
(3 arms × 30 seeds ≈ 9 arms × 10 seeds). Continuing to add seeds *because* p=0.029
was seen, without declaring it confirmatory first, is optional stopping and would
repeat the methodological failure this whole exercise exists to correct — so this is
a decision to take deliberately, not a default.

### 4.3 Confirmatory replication of the `Q*` finding

Design fixed in advance in `PREREGISTRATION_QSTAR.md` and executed unchanged
(`pricing_dt/diagnostics/diag_qstar_confirm.py`). **Fresh seeds 10–69** (n = 60); seeds 0–9 generated the
hypothesis and were excluded. Two co-primary manipulations of the same claim — replace
the whole target with the truth, or replace only the aspiration field with the truth —
paired Wilcoxon, Holm over exactly those two comparisons. Sized by bootstrap power
against a **halved** effect (n = 60 → 0.77; n = 30 → 0.42).

| arm | mean | sd |
|---|---:|---:|
| `A_structured` | +0.6218 | 0.1643 |
| `oracle_Qstar` | +0.5481 | 0.1324 |
| `potential_exact` | +0.5271 | 0.1389 |

| hypothesis | effect | 95% CI | seeds positive | p | Holm |
|---|---:|---|---:|---:|---:|
| H1  A − `oracle_Qstar` | **+0.0738** | [+0.0168, +0.1259] | 47/60 | .00073 | **.00073** |
| H2  A − `potential_exact` | **+0.0947** | [+0.0361, +0.1492] | 49/60 | .00020 | **.00040** |

**Both confirmed.** The finding carries a section. Two things must travel with it:

- The effect is **+0.074**, not the exploratory +0.125 — a 41% shrinkage, the winner's
  curse the design anticipated. Any figure or abstract number must use the replication.
- It remains above the +0.05 smallest-effect-of-interest declared before the run, so it
  is meaningful by a standard fixed in advance rather than chosen afterwards.

The mechanism measurement (§4, within-state discrimination ratio 0.080 for `Q*` against
0.334 for the structured target) is exploratory and was not part of the confirmatory
design; it is an explanation offered for a confirmed effect, not itself confirmed.

## 4.4 Second environment — what replicates and what does not

`pricing_dt/envs/inventory.py` (lost-sales inventory, overdispersed censored demand, exact DP
anchors), `pricing_dt/diagnostics/diag_env2_probe.py`, `pricing_dt/diagnostics/diag_env2_channels.py`. The structural prior is
**Poisson**, whose variance is identically its mean and which therefore cannot
represent overdispersion — the analogue of bounded elasticity, and a textbook
inventory model rather than a strawman.

**The feasibility probe earned its keep by failing twice first.** Two design beliefs
were wrong and were caught before any suite was built on them:

1. *Censoring alone does not create exploitable optimism.* A naively fitted model
   under-estimates demand, so its in-model value falls WITH the truth — ordinary
   estimation error, not the optimizer's curse. An explicit **stockout penalty** is
   load-bearing: only then does under-stating demand variance make the planner
   fail to foresee a cost it will actually pay.
2. *Too conservative a logger destroys the normalisation anchor.* At order-up-to 6 the
   logging policy is itself catastrophic (value −8.0 against an 80.9 optimum), so
   "beats the logger" becomes a trivial bar and hides the collapse. At order-up-to 14
   the logger is competent (69.1) and the collapse is visible.

**Condition (b) holds** at order-up-to 14: planning on the Poisson prior gives
nv **−3.246** with in-model value 94.8 (curse gap **+60.3**); the prior under-states
demand variance by 79% (4.6 vs 22.0).

### Goal channel, 10 seeds

| arm | mean | sd |
|---|---:|---:|
| QDT (fixed) | **0.382** | 0.177 |
| vanilla | 0.311 | 0.202 |
| BC | 0.307 | 0.174 |
| oracle `Q*` | 0.220 | 0.300 |
| **structured** | **0.169** | 0.321 |
| empirical | −0.044 | 0.400 |

### Trust-region sweep (the signature figure, replicated)

| m | 1 | 2 | 3 | 5 | 8 | 11 |
|---|---:|---:|---:|---:|---:|---:|
| **structured** | 0.169 | −0.053 | −0.388 | −1.676 | −3.467 | **−3.646** |
| **oracle (control)** | 0.169 | 0.429 | 0.584 | 0.796 | 0.931 | **+0.972** |

The `m = 11` endpoint (−3.646) lands on the independently measured planner (−3.246),
and the accurate-model control rises to the optimum, exactly as in pricing.

### What this second environment establishes

**Replicates:**
- **The channel/support axis.** The same Poisson model is catastrophic when it chooses
  actions (−3.246) and merely unhelpful when it sets targets (+0.169) — a ~3.4-unit
  swing in normalised value from one model, with the scissors and its accurate-model
  control intact. This is the paper's central claim and it now holds in two
  environments with structurally different priors.
- **The main-line null.** The structured relabel again fails to beat a correct Q-DT
  (0.169 vs 0.382).

**Does NOT replicate:**
- **"Structured relabelling is beneficial."** In pricing it was 0.670 against vanilla's
  0.441; here it is 0.169 against vanilla's 0.311 and BC's 0.307 — it actively *hurts*.
  The goal channel is *safe*, not *useful*.
- **"The exact `Q*` is a worse conditioning target." REVERSES**: structured − oracle =
  **−0.051, p = 0.0059**, against the +0.074 confirmed in pricing.

### The reversal supports the mechanism while limiting the claim

Within-state discrimination ratios (5 seeds):

| arm | pricing | inventory |
|---|---:|---:|
| structured | 0.334 | 0.259 |
| **oracle `Q*`** | **0.080** | **0.208** |

`Q*` is nearly non-discriminative in pricing and ordinarily discriminative in
inventory, because ordering too much *and* too little are both penalised, so
`Q*(x,a)` genuinely varies across actions at a state — whereas in pricing the optimal
continuation compensates for a bad current price and flattens the target. The
discrimination account therefore **predicts** the reversal.

Consequence for §4.3: the pre-registered replication established the `Q*` effect on
fresh seeds *within the pricing environment*; it does **not** generalise. The
defensible claim is the mechanism — *return-conditioned relabelling needs a
within-state discriminative target* — with "`Q*` is a poor target" as a corollary that
holds only where `Q*` happens to be non-discriminative. That is a weaker headline than
§4.3 alone suggests and must be stated wherever the finding appears.

---

## 5. What the paper can now claim

Defensible, with the evidence above:

1. **Where a domain model is allowed to act determines whether its error is
   harmful.** Same model, same information: 0.670 in the goal channel, −4.543 in the
   action channel, and a continuous, validated path between them.
2. **The protective mechanism is confinement to the logged action support, and it is
   causally demonstrated** — with an accurate-model control showing the damage is
   model error, not the loss of imitation. The safe region is narrow (top-3 of 11).
3. **The data channel is the same axis, not a separate one**: its safety is
   truncation, and untruncated it converges on the action channel.
4. **Return-conditioned relabelling needs within-state discriminative targets, and
   `Q*` is a poor choice precisely because it is correct.** **Confirmed** by a
   pre-registered replication on fresh seeds (§4.3): +0.074, 95% CI [+0.017, +0.126],
   p = 0.0007, n = 60 — but **within the pricing environment only**: it reverses in the
   inventory environment (−0.051, p = 0.006), where `Q*` is not degenerate (§4.4). The
   generalisable claim is the discrimination mechanism, not "`Q*` is a bad target".

No longer defensible without qualification:

5. "Structured demand-model relabelling beats bootstrapped-value relabelling."
   Under held-out target selection (§4.2) it ties a correct Q-DT (+0.016, p = 0.77),
   ties ask-free model-filtered BC (-0.023), and ties the variant with its own action
   term deleted (-0.024). The published +0.391 rested on a comparator with zero
   within-state target signal.
6. "Economic structure is what earns the gain." The structured *action* term is
   inert (§4.1); what matters is a non-degenerate aspiration field, which a
   bootstrapped value supplies just as well (§2.4). What remains is the much narrower
   "a bounded parameterisation beats an unconstrained MLP, and its correctness is
   irrelevant".

7. "The advantage holds across the calibrated elasticity band." It is negative at
   three of four points and −0.110 at the data's own median elasticity (§2.5).

Gate 2 full corrected-pricing rerun addendum (`results_gate2_pricing_full_20260818/`, a superseded batch not included in this package):
across 720 policy evaluations, Structured DT with a support mask has the best overall
mean normalised value (0.745), followed by corrected Q-DT `td` denoised (0.711),
unmasked Structured DT (0.688), Q-DT `q_sa` (0.626), IQL (0.569), support-masked
Vanilla DT (0.552), and Vanilla DT (0.490). The support mask improves Structured DT by
+0.0567 (raw p = 3.7e-11) and Vanilla DT by +0.0620 (raw p = 3.5e-09), while also driving
unseen-action rate to zero. However, corrected Q-DT `td` is the strongest method at
`N=1600` (0.771). This strengthens the support/channel mechanism and weakens any
attempt to present the work as a universal Structured-DT win.

Recommended framing: the contribution is the **channel/support axis and its
mechanism**, with the structured relabeller as one instance rather than the claim.
Section 4 is a genuine positive finding about return-conditioned methods in general.

---

## 6. Artifacts

New: `pricing_dt/diagnostics/diag_channel_ladder.py`, `pricing_dt/diagnostics/diag_trust_region.py`, `pricing_dt/diagnostics/diag_target_decomp.py`,
`pricing_dt/diagnostics/diag_conditioning.py`, `pricing_dt/diagnostics/diag_data_channel.py`, `pricing_dt/diagnostics/diag_target_stats.py`,
`pricing_dt/diagnostics/diag_c2_fixed.py`; `relabel.oracle_rtg`; `qdt.value_relabel(mode=...)`.

CSVs in `results/`: `channel_ladder{,_alpha}.csv`, `trust_region_{scan,summary}.csv`,
`target_decomp{,_matrix}.csv`, `conditioning_{sweep,summary}.csv`,
`data_channel{,_summary}.csv`, `target_stats{,_summary}.csv`, `c2_fixed_*.csv`.

Figures: `trust_region_scissors.png` (signature), `channel_ladder.png`,
`conditioning_sweep.png`, `target_decomp.png`, `data_channel.png`.

`qdt.value_relabel` now defaults to the fixed `mode="td"`; `mode="state_value"`
reproduces the pre-fix baseline. Any published number computed against Q-DT needs
recomputation — the original `results/` tree is untouched for comparison.
