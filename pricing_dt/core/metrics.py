"""Metrics and statistics.

Implements: normalised value, Equation (3.3), in `normalised_value`, and the paired
Wilcoxon with Holm correction of §3.5.3 in `paired_test` and `holm`.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: normalised_value(), paired_test(), holm(), stitching_score().
#
# The measurement layer: normalised value against exact anchors, paired significance
# testing across seeds, and step-down multiplicity correction.
#
# Implements or follows:
#   - Fu, J., Kumar, A., Nachum, O., Tucker, G. and Levine, S. (2020) 'D4RL: Datasets for
#     Deep Data-Driven Reinforcement Learning'. arXiv:2004.07219.
#   - Wilcoxon, F. (1945) 'Individual comparisons by ranking methods', Biometrics
#     Bulletin, 1(6).
#   - Holm, S. (1979) 'A simple sequentially rejective multiple test procedure',
#     Scandinavian Journal of Statistics, 6(2).
#   - Brandfonbrener, D. et al. (2022) 'When does return-conditioned supervised learning
#     work for offline reinforcement learning?', NeurIPS 35.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
from scipy.stats import wilcoxon


def expected_trajectory_return(tr, mdp):
    """De-noised expected return of a trajectory's action sequence: the sum of
    EXPECTED (not noise-realised) rewards of the actions actually taken along its
    true reference path. This is the skill content of the trajectory, stripped of
    favourable/unfavourable demand noise (cf. claim C1)."""
    return sum(float(mdp.R[int(tr.actions[t]), int(tr.ref_bins[t])])
               for t in range(len(tr.actions)))


def best_logged_by_start(trajs, mdp, n_bins):
    """Per-start ceiling: for each initial reference bin, the best de-noised
    trajectory return among logged trajectories that START in that bin. Returns a
    dict {bin: best_return}. This is the comparable, well-posed ceiling: a learned
    policy's value FROM a given start is compared against the best logged
    trajectory FROM THE SAME start."""
    best = {}
    for tr in trajs:
        b0 = int(tr.ref_bins[0])
        r = expected_trajectory_return(tr, mdp)
        if b0 not in best or r > best[b0]:
            best[b0] = r
    return best


def stitching_score(policy_value_by_bin, trajs, mdp):
    """Well-posed C1 quantity. For each logged start bin, compare the learned
    policy's value-from-that-start against the best logged trajectory from the
    same start; report (i) the average margin and (ii) the fraction of starts the
    policy beats. `policy_value_by_bin` is a dict {bin: V_learned_from_bin}.
    Positive average margin / >0.5 fraction => genuine stitching."""
    ceil = best_logged_by_start(trajs, mdp, mdp.cfg.n_ref_bins)
    margins, wins = [], []
    for b0, c in ceil.items():
        if b0 in policy_value_by_bin:
            m = policy_value_by_bin[b0] - c
            margins.append(m); wins.append(m > 1e-6)
    if not margins:
        return dict(avg_margin=float("nan"), beat_fraction=float("nan"))
    return dict(avg_margin=float(np.mean(margins)),
                beat_fraction=float(np.mean(wins)))


def normalised_value(v, v_anchor, v_optimal):
    """Equation (3.3): policy value on the normalised scale, 0 at the lower reference
    and 1 at the optimal sequential policy.

    Section 3.5.1 sets the lower reference per environment, and this function serves E1,
    where it is the ORACLE MYOPIC policy. E2's lower reference is its logging policy, a
    weaker zero, and E2 forms the ratio inline in its own diagnostics rather than here.
    E1 and E2 normalised values are therefore not the same unit and must not be pooled
    or differenced; `diagnostics.diag_env2_channels` states the same at its own anchor.

    For E1, `v_anchor` is NOT the logging policy. Callers pass the value of
    argmax_a R[a, b] under the TRUE reward -- the perfect-information
    contextual-bandit solution (see experiments._setup). The parameter was
    once called `v_behaviour`, and the released CSVs still label the columns
    `v_behaviour` / `v_behaviour_expected`; those names are retained only so
    that previously published result files stay readable.

    Both anchors must be EXPECTED (noiseless) policy values so that
    v_optimal >= v_anchor by construction. Passing realised/noisy returns can
    make the denominator non-positive; we return NaN in that case rather than
    dividing by a flooring constant, which would manufacture garbage.
    """
    denom = v_optimal - v_anchor
    if denom <= 1e-8:
        return float("nan")
    return (v - v_anchor) / denom


def regret(v_learned, v_optimal):
    """Shortfall from the optimum, in the environment's own reward units.

    Unnormalised, so unlike `normalised_value` it is not comparable across environments
    and is reported only within one.
    """
    return v_optimal - v_learned


def paired_test(a, b):
    """Paired two-sided Wilcoxon signed-rank across seeds (Section 3.5.3).

    `a` is the proposed arm, `b` the comparator; returns (median_diff, p_value). The
    inferential unit is the seed, so with the ten seeds reported here the smallest
    attainable two-sided p is 2/2**10 = 0.00195, before any Holm correction.
    Returns a NaN p for fewer than three seeds or an all-zero difference.
    """
    a, b = np.asarray(a), np.asarray(b)
    diff = a - b
    if len(diff) < 3 or np.allclose(diff, 0):
        return float(np.median(diff)), float("nan")
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        p = float("nan")
    return float(np.median(diff)), float(p)


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (Section 3.5.3).

    The family is whatever the caller passes in, and Section 3.5.3 fixes what that is:
    the comparisons reported in one analysis, not every comparison in the study. Passing
    a wider set than the analysis reports would over-correct; a narrower one would
    under-correct.
    """
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        prev = max(prev, val)
        adj[idx] = min(prev, 1.0)
    return adj
