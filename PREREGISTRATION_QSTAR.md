# Pre-registration: is an exact `Q*` a worse conditioning target than an approximate one?

**Written before the confirmatory run was executed.** Timestamp: 2026-07-27, after the
exploratory results in `CHANNEL_RESULTS.md` §4/§4.2 and before any seed in the range
below was evaluated.

## Why a confirmatory study is needed

The exploratory finding is that the exact `Q*(s_t,a_t)` is the *worst* relabelling
target — worse than a structured-model estimate of the same quantity — with a measured
explanation (its within-state discrimination ratio is 0.080 against the structured
target's 0.334, because `Q*` folds in the optimal continuation so a bad logged action
still scores highly).

Under the honest held-out protocol the gap is **+0.125, raw p = 0.029, Holm n.s.**
across 8 comparisons at n = 10 seeds. That is not enough to carry a paper section. The
legitimate way to promote it is a replication **declared in advance**, not additional
seeds appended because p = 0.029 was observed — the latter is optional stopping and
would repeat the methodological failure this whole exercise exists to correct.

## Design, fixed in advance

**Fresh seeds.** Seeds **10–69** (n = 60). Seeds 0–9 are deliberately excluded: they
generated the hypothesis, so reusing them would not be a replication.

**Cell and protocol.** Identical to `pricing_dt/diagnostics/diag_heldout_protocol.py`: `N = 100`,
`noise = 0.5`, `delta = 3`; start bins split 30% selection / 70% test per seed; each
arm sweeps the same target grid (q0.50–q1.0 plus max ×{1.25, 1.5, 2.0}), picks its
conditioning target on the selection bins, and the test bins are read once.

**Arms (3).**
| arm | target |
|---|---|
| `A_structured` | fitted structured demand model (the reference) |
| `oracle_Qstar` | exact `Qstar[t, b_t, a_t]` |
| `potential_exact` | structured action term + **exact** potential term |

Two arms rather than one because the claim — *target accuracy is not what the goal
channel wants* — is tested by two independent manipulations: replacing the whole
target with the truth, and replacing only the aspiration field with the truth.

**Hypotheses (both directional in the sense that a negative result refutes the claim,
but tested two-sided).**
- **H1 (primary):** `A_structured − oracle_Qstar > 0`
- **H2 (co-primary):** `A_structured − potential_exact > 0`

**Test.** Paired Wilcoxon signed-rank across seeds, two-sided, Holm correction over the
**two** pre-specified comparisons (so the leading comparison needs raw p ≤ 0.025). No
other comparison computed in this run may be promoted to confirmatory status; anything
else is descriptive.

**Sample size.** n = 60 chosen from a bootstrap power analysis over the exploratory
paired differences, at α = 0.025:

| n | effect = 100% of observed | 75% | 50% | 33% |
|---:|---:|---:|---:|---:|
| 30 | 0.99 | 0.91 | 0.42 | 0.16 |
| **60** | **1.00** | **1.00** | **0.77** | 0.35 |
| 80 | 1.00 | 1.00 | 0.89 | 0.46 |

The exploratory estimate (+0.125) is subject to winner's curse, so the design is sized
against a **halved** effect. Smallest effect of interest declared as **+0.05** in
normalised value, on the grounds that the entire surviving structured-vs-fixed-Q-DT gap
is +0.016 and the structured-vs-unconstrained residue is +0.163.

## Decision rule, fixed in advance

- **Both H1 and H2 significant after Holm** → the finding is confirmed and may carry a
  section of the paper, reported at the replication's own effect size (not the
  exploratory one).
- **Exactly one significant** → reported as a partially-replicated result, one
  paragraph, with the failed manipulation stated explicitly.
- **Neither significant** → the finding is **permanently demoted** to a single
  descriptive sentence with a pointer to the data. No further seeds will be added, and
  no alternative analysis will be substituted to rescue it.

Effect sizes are reported with a bootstrap 95% CI on the paired mean difference
regardless of outcome. The exploratory n = 10 result is reported alongside, so the
shrinkage between exploratory and confirmatory estimates is visible.

## Deviations

None. The design above was executed as written (`pricing_dt/diagnostics/diag_qstar_confirm.py`, seeds 10-69,
n = 60), and no analysis choice was changed after seeing the data.

## Outcome (recorded after execution)

| arm | mean | sd |
|---|---:|---:|
| `A_structured` | +0.6218 | 0.1643 |
| `oracle_Qstar` | +0.5481 | 0.1324 |
| `potential_exact` | +0.5271 | 0.1389 |

| hypothesis | effect | 95% CI | seeds positive | p | Holm |
|---|---:|---|---:|---:|---:|
| H1  A − `oracle_Qstar` | **+0.0738** | [+0.0168, +0.1259] | 47/60 | .00073 | **.00073** |
| H2  A − `potential_exact` | **+0.0947** | [+0.0361, +0.1492] | 49/60 | .00020 | **.00040** |

**Both confirmed** → per the decision rule the finding may carry a section, reported at
the replication effect size.

Winner's curse materialised as anticipated: the effect shrank 41% from the exploratory
+0.125 to +0.0738. This is why the study was sized against a halved effect — at n = 30
the power against an effect this size sits between 0.42 and 0.91, so the original
proposal of 30 seeds carried a real risk of a false negative that the decision rule
would have made permanent.

The replicated effect (+0.074) remains above the SESOI of +0.05 declared in advance.

**All reporting of this finding must use +0.074 [+0.017, +0.126], not the exploratory
+0.125.**
