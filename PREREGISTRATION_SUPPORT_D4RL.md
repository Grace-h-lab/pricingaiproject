# Pre-registration: does the support-mask contraction hold on a public benchmark?

> **Pointer note, added 2026-09-02 and changing nothing else.** Chapter 4 was renumbered after
> this was written, and the companion paper draft was folded into the dissertation. The body
> below is left exactly as it was written; read its cross-references as: §4.7.4 is now §4.6.2
> (*How much of a masked policy is the mask?*); §4.8 is now §4.7 (*What bounds these results*)
> together with Appendix F.3.1, where the failed replication is in fact reported; the paper's
> §7 and §9 are §4.6.2 and §5.5; and the E2 support crossing ships as `results_env2_support_n30_20260824/`.

**Written before the confirmatory run was executed.** Timestamp: 2026-08-25, after the E1
and E2 support crossings (`results_gate2_masked_20260821/`, `results_env2_support_20260824/`)
and before any MiniGrid policy was trained.

## Why this run is needed

Chapter 5 §5.5 and the paper's §9 both name the same gap in the same words: the support
results are masked-versus-bare comparisons, they need no exact optimum, they *could* be run
on a standard discrete-action offline benchmark, and they were not. `D4RL_FEASIBILITY.md`
explains why the *channel* and `Q*` results cannot be ported — they are stated in units of a
true optimality gap — but that argument does not cover the contraction claim, which is
scale-free. This run closes the one gap the argument leaves open.

## What is being tested

The claim, as stated in §4.7.4 and the paper's §7:

> Confining a policy to logged action support is a **contraction toward whatever the logged
> support itself achieves**, not an improvement operator. Whether an arm gains depends on
> where it sat relative to that band.

In E1 the mask improved all five families (+0.052 to +0.326); in E2 it significantly harmed
the strongest arm. The account attributes the difference to where each environment's logging
band sits. That explanation was constructed after seeing both results. **This run tests it
prospectively**, on data neither environment produced.

## The benchmark, and why it is the right one

`D4RL/minigrid/fourrooms-v0` and `D4RL/minigrid/fourrooms-random-v0` (Farama Minari
distribution of the D4RL MiniGrid datasets). Both are logs of the *same* environment,
`MiniGrid-FourRooms-v0`, with `Discrete(7)` actions, and they differ almost only in the
logging policy:

| | episodes | steps | mean return | success | logged action histogram |
|---|---:|---:|---:|---:|---|
| `fourrooms-v0` (expert) | 590 | 10,010 | 0.847 | 100% | `[849, 1182, 7979, 0, 0, 0, 0]` |
| `fourrooms-random-v0` | 10,174 | 1,000,070 | 0.018 | 3.1% | `[143454, 142333, 142965, 142997, 142679, 142578, 143064]` |

This is the sharpest available test of the account, because the two logs bracket the
mechanism it proposes. Under the expert logger the band is **high** and support is **narrow**
— four of seven actions are never logged at all. Under the random logger the band is **low**
and support is **maximally wide** — the logged action distribution is uniform, so a support
constraint has almost nothing to remove. Everything else is held fixed: same environment,
same observation space, same algorithms, same seeds.

## Design, fixed in advance

**Operator.** Identical to E1/E2: restrict the policy's argmax to the top-*k* actions by
logged probability, `k = 3`. Because MiniGrid states are not tabular, logged probability is
supplied by a behaviour-cloning estimate `π_β(a|s)` rather than by per-state counts. A
second, published form is reported as a robustness check: the discrete-BCQ threshold
`A_τ(s) = {a : π_β(a|s) / max_a' π_β(a'|s) ≥ τ}` at `τ = 0.3`. The mask is applied at
inference to an already-trained policy, exactly as in E1/E2; no arm is retrained under it.

**Arms (5 learners + 1 no-learner floor).** Chosen to mirror the five E1 families:
behaviour cloning; discrete CQL; discrete IQL (expectile 0.7, AWR β = 3); a
return-conditioned Decision Transformer on logged return-to-go; and a Q-relabelled Decision
Transformer using the CQL critic (the analogue of Q-DT `td`). The floor is a uniform random
policy carrying the identical mask.

**Seeds and evaluation.** 3 training seeds per (dataset, arm); each policy evaluated on 100
episodes at fixed evaluation seeds shared across all arms, so comparisons are paired.

**Statistics.** All scale-free; none requires an optimum.

1. `compression` = sd of mean return across the five learner arms, bare ÷ masked.
2. `Spearman(bare off-support rate, mask gain)` across the five arms.
3. per-arm mask gain, paired across seeds.
4. the no-learner floor: uniform random under the same mask.

## Predictions, fixed in advance

| | Prediction | Refuted by |
|---|---|---|
| **P1** | `compression > 1` under the expert logger — masked arms span less than bare arms | `compression ≤ 1` |
| **P2** | Mask gain is positively rank-correlated with how far the bare arm strays off support (Spearman > 0), reproducing +1.000 in E1 and E2 | Spearman ≤ 0 |
| **P3** | **Compression under the random logger is materially smaller than under the expert logger.** The mask can only contract toward a band it actually binds to, and against a uniform logger it removes almost nothing | comparable or larger compression under the random logger |
| **P4** | The no-learner floor is far higher under the expert logger than under the random logger, because the floor is the log's own value | floors within noise of each other |

**P3 is the load-bearing one.** P1 and P2 would merely reproduce E1. P3 is a prediction the
contraction account makes and an "inference-time masking is a general improvement" account
does not: the latter has no reason for the gain to depend on the logger's entropy.

## Decision rule, fixed in advance

- **P1, P2 and P3 all hold** → the contraction account is externally validated on a public
  benchmark, and §4.7.4 may say so.
- **P1 and P2 hold, P3 fails** → the phenomenon replicates but the explanation does not.
  Reported as such, in those words, and the account is downgraded to a description.
- **P1 or P2 fails** → the contraction result does not generalise beyond the two
  environments built for this study. Reported as a failed replication in §4.8, and the claim
  in §4.7.4 is restricted to E1 and E2 by name.

No arm, dataset, seed count or statistic will be added after seeing results. If the run is
incomplete at submission, this document is reported as a declared but unexecuted experiment
and nothing is claimed from partial output.

**Result lineage.** Outputs are written to `results_minigrid_support_20260825/` and are
never merged with the pricing or inventory CSVs.

## Deviations

**No deviation from the design.** Arms, datasets, seed count, operators, statistics and the
decision rule were executed as written. Two reporting notes:

1. **The compression statistic is reported as a median across seeds, not a mean**, and as
   undefined when the masked standard deviation falls below 1e-4. One seed produced masked
   returns that were near-identical across arms, so the mean of the per-seed ratio was
   dominated by a near-zero denominator (it evaluated to 3.7e6). This is a presentation
   choice made after seeing the data; it changes no verdict — P1 and P3 resolve identically
   under either summary.
2. **A Spearman correlation is undefined when every arm's gain is exactly zero.** An
   undefined statistic cannot support a prediction, so it is counted as a failure of that
   prediction rather than skipped.

## Outcome

**The replication failed.** P2 is refuted under both mask forms, so by the decision rule
fixed above: *the contraction result does not generalise beyond the two environments built
for this study, and the claim in §4.7.4 must be restricted to E1 and E2 by name.*

| | top-*k*, k = 3 (primary) | discrete-BCQ τ = 0.3 |
|---|---|---|
| **P1** compression > 1, expert | **REFUTED** — 1.00× | **HOLDS** — 1.17× (per-seed 1.17 / 0.72 / 1.94) |
| **P2** Spearman(off-support, gain) > 0 | **REFUTED** — undefined | **REFUTED** — 0.00 (per-seed 0.0 / +0.7 / −0.7) |
| **P3** compression smaller under the random logger | **REFUTED** — 2.09× vs 1.00× | **HOLDS** — 1.00× vs 1.17× |
| **P4** floor higher under the expert logger | **HOLDS** — 0.0466 vs 0.0061 | **HOLDS** — 0.4153 vs 0.0209 |

**Why the primary operator failed.** Top-3 is *vacuous* on the expert log: that logger only
ever plays 3 of its 7 actions, so the admissible set is those 3 actions at every state
(`|A₃| = 3.00`, off-support 0.000 for all five learners) and every gain is exactly 0.0000.
On the random log it does bind, and there every gain is **negative** (IQL −0.0065, DT
−0.0210, Q-DT −0.0033, floor −0.0166) with Spearman −0.701 — the opposite of the predicted
ordering. The E1/E2 operator's *k* is not portable: it has to be chosen relative to the
logging policy's entropy rather than as an absolute.

**Why the secondary operator did not rescue it.** Under τ the mask binds on the expert log
(`|A_τ| = 1.47`) and compresses weakly, but the gain ordering is uninformative: the two arms
that stray furthest are *harmed* (DT −0.0261 at off-support 0.197; Q-DT −0.0047 at 0.196)
while the two that stray least gain slightly (CQL +0.0013, IQL +0.0092). On the random log
τ is vacuous in the other direction — a uniform `π_β` puts all seven actions within the
threshold (`|A_τ| = 7.00`), so every gain is exactly 0.0000.

**What did replicate.** P4, in both forms and decisively: the no-learner floor tracks the
logging policy's quality (0.4153 under the expert log against 0.0209 under the random one).
And on the expert log under τ, the masked floor (0.4153) is **above the best masked learner**
(0.4141) — the E1 finding that a masked score is evidence about a method only insofar as it
clears its own floor, reproduced on public data.

**Two weaknesses of this test, stated so the failure is not over-read.** Neither licenses
retro-fitting the account; both bound what the failure shows.

- On the random log no arm learns much (returns 0.000–0.027 against a dataset mean of
  0.018), so its spread and rank statistics are close to noise.
- On the expert log all five learners (0.304–0.440) sit well below their own logging policy
  (0.847), unlike E1 and E2. The deliberately simple observation encoder is the likely
  cause. A configuration in which the arms bracketed their logger would give P2 more room
  than it had here.

So the honest summary is: **the prediction failed, and the benchmark also gave it less room
to succeed than E1 and E2 did.** Both halves belong in the write-up.
