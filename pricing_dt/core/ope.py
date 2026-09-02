"""Off-policy evaluation for RQ3 / C3.

Demonstrates the non-stationarity failure: when a single ("pooled") behaviour
policy is fit to a log produced by a drifting (still-learning) logger, the
importance weights are systematically wrong and the doubly robust estimate is
biased. A change-point-"segmented" estimator that fits a behaviour policy per
segment recovers the truth.

No new estimator is proposed here: segmented DR builds on
the OPE and non-stationary-OPE literature (Jiang and Li, 2016; Liu et al.,
2023; Uehara, Shi and Kallus, 2026; Shimizu et al., 2025). The contribution
here is the application to pricing-DT uplift validation plus the simulator-
grounded bias demonstration.

IMPORTANT (estimator-visibility note). The non-stationarity bias is only
*observable* when the doubly robust estimator actually leans on its importance
weights. If the direct-method outcome model q_hat were perfectly specified, DR
would be consistent regardless of (pooled, wrong) propensities and the bias would
be invisible — a subtlety that can make a correct mechanism look like a null
result. Two consequences for this code:
  (1) The headline E3 here sidesteps propensity estimation entirely by using the
      simulator's TRUE per-segment loggers, so the pooled-vs-segmented gap is a
      pure consequence of mixing different segment policies — the cleanest
      demonstration.
  (2) If E3 is later extended to ESTIMATE pi_b from data (the realistic setting),
      q_hat must be deliberately imperfect (e.g. a low-capacity or state-coarse
      direct method) for the bias to surface; a high-capacity q_hat that fits the
      returns well will mask it. This is a property of DR, not a bug.

Implements: the off-policy estimators of the secondary strand, reported in Appendix
F.3.4. The effective sample fraction of Equation (3.11) is not here: it is computed
alongside the real-data weights, in `diagnostics.diag_gate3_real_ope._ess`, and is
reported in Section 5.5.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: dr_value(), pooled_dr(), segmented_dr(), marginalised_is_value(), estimate_behaviour_policy().
#
# Off-policy evaluation for the non-stationary logging strand: a fitted outcome model,
# doubly-robust value estimation, per-regime segmentation, behaviour-policy estimation
# with partial pooling, and a marginalised occupancy-ratio estimator.
#
# Implements or follows:
#   - Precup, D., Sutton, R.S. and Singh, S. (2000) 'Eligibility Traces for Off-Policy
#     Policy Evaluation', ICML.
#   - Dudík, M., Langford, J. and Li, L. (2011) 'Doubly Robust Policy Evaluation and
#     Learning', ICML. arXiv:1103.4601.
#   - Jiang, N. and Li, L. (2016) 'Doubly Robust Off-policy Value Evaluation for
#     Reinforcement Learning', ICML.
#   - Thomas, P.S. and Brunskill, E. (2016) 'Data-Efficient Off-Policy Policy Evaluation
#     for Reinforcement Learning', ICML.
#   - Uehara, M., Shi, C. and Kallus, N. (2026) 'A Review of Off-Policy Evaluation in
#     Reinforcement Learning', Statistical Science. arXiv:2212.06355.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


# ----------------------- behaviour policy estimation -----------------------
def estimate_behaviour_policy(trajs, n_actions, clip=1e-3):
    """Fit pi_b(a|s) by multinomial logistic regression on the logged pairs.

    Probabilities are floored at `clip` because they become importance-weight
    denominators: an unclipped near-zero produces a weight large enough to carry the
    estimate on its own.
    """
    S = np.concatenate([t.obs for t in trajs])
    A = np.concatenate([t.actions for t in trajs])
    classes = np.unique(A)
    if len(classes) < 2:
        # degenerate segment: fall back to the empirical action frequency
        freq = np.full(n_actions, clip)
        for a in A:
            freq[a] += 1.0
        freq = freq / freq.sum()
        return lambda obs: freq
    clf = LogisticRegression(max_iter=500)
    clf.fit(S, A)

    def probs(obs):
        p = np.full(n_actions, clip)
        pr = clf.predict_proba(obs.reshape(1, -1))[0]
        for cls, val in zip(clf.classes_, pr):
            p[cls] = max(val, clip)
        return p / p.sum()

    return probs


def estimate_behaviour_policy_shrunk(seg_trajs, global_pb, n_actions, kappa=50.0, clip=1e-3):
    """Partially-pooled (hierarchical / empirical-Bayes) per-segment propensity.

    The naive segmented estimator fits an independent logistic policy on each
    detected segment; when change-point detection splits the log, each segment's
    fit is starved of data and its per-step propensities are noisy, and that added
    variance can outweigh the bias reduction segmentation is meant to buy (the C3
    data-driven negative result). This estimator instead *shrinks* each segment's
    local estimate toward the global pooled estimate with a data-adaptive weight

        lam_s = n_s / (n_s + kappa),

    so small segments borrow statistical strength from the pool (lam->0) while
    large, well-populated segments trust their local fit (lam->1). kappa=0 recovers
    the naive per-segment estimator; kappa->inf recovers the pooled estimator. This
    is the standard James–Stein / hierarchical-Bayes remedy for the small-sample
    variance that breaks naive segmentation."""
    n_s = sum(len(t.actions) for t in seg_trajs)
    local = estimate_behaviour_policy(seg_trajs, n_actions, clip)
    lam = n_s / (n_s + kappa)

    def probs(obs):
        p = lam * local(obs) + (1.0 - lam) * global_pb(obs)
        p = np.maximum(p, clip)
        return p / p.sum()

    return probs


# ----------------------- direct method Q-hat -----------------------
def fit_qhat(trajs, n_actions, state_dependent=True):
    """Ridge regression of return-to-go onto the outcome features.

    state_dependent=True : features = [state, one-hot action] (a capable direct
        method). DR is then near-consistent even with wrong propensities, so the
        non-stationarity bias is MASKED (the estimator-visibility note above).
    state_dependent=False: features = [one-hot action] only — a deliberately weak,
        STATE-INDEPENDENT q_hat (q_hat(a) = mean RTG of action a). DR must then lean
        on its importance weights, so a mis-specified (pooled) propensity surfaces as
        bias. This is the principled choice for the headline E3 demonstration; the
        state-dependent variant is reported alongside it to show the masking is real,
        not a null mechanism.
    """
    X, y = [], []
    for tr in trajs:
        for t in range(len(tr.actions)):
            oh = np.zeros(n_actions); oh[tr.actions[t]] = 1.0
            feat = np.concatenate([tr.obs[t], oh]) if state_dependent else oh
            X.append(feat); y.append(tr.rtg[t])
    reg = Ridge(alpha=1.0).fit(np.array(X), np.array(y))

    def qhat(obs, a):
        oh = np.zeros(n_actions); oh[a] = 1.0
        feat = np.concatenate([obs, oh]) if state_dependent else oh
        return float(reg.predict(feat.reshape(1, -1))[0])

    return qhat


# ----------------------- marginalised importance sampling -----------------------
def marginalised_is_value(trajs, mdp, pe_argmax, init_bins, n_actions):
    """Marginalised IS via state-action OCCUPANCY ratios — the estimator proposed in
    Appendix F.3.4 as the secondary off-policy-evaluation strand.

    Standard per-decision IS/DR multiply per-step propensity ratios over the horizon,
    so they (a) suffer product-of-weights variance and (b) require a correctly-specified
    per-step behaviour policy pi_b(a|s) — which, under non-stationary logging, must be
    estimated per segment and is exactly what the data-driven C3 pipeline fails to do.

    A marginalised estimator avoids both. It reweights logged rewards by the *marginal*
    state-action density ratio w(s,a) = d^pi_e(s,a) / d^pi_b(s,a):
        V_MIS = E_{(s,a,r)~log}[ w(s,a) r ] = sum_{s,a} d^pi_e(s,a) * rhat(s,a).
    Here the discrete state is (ref-bin, t), so the closed form is used:
      - d^pi_b(s,a) is estimated EMPIRICALLY from the pooled log (its visitation already
        reflects whatever drifting mixture of loggers produced it — no per-segment
        propensity model, no change-point detection);
      - rhat(s,a) is the pooled empirical mean logged reward at (ref-bin, action) — the
        reward function is stationary, so pooling across segments and time is unbiased and
        maximises sample size;
      - d^pi_e(s,a) is computed EXACTLY by forward-propagating the deterministic target
        policy from the initial distribution under the KNOWN reference transition
        (ref' = (1-eta)ref + eta*price), which in pricing is a modelled, controllable
        dynamic, not something that must be learnt.
    Unsupported target cells (visited by pi_e but never by the log) fall back to the
    action-marginal then global empirical reward; the supported occupancy mass is
    returned as `coverage` so the support limitation is visible rather than hidden.

    pe_argmax(obs) -> int action (deterministic target policy).
    Returns (V_MIS, coverage).
    """
    A, B, H = n_actions, mdp.cfg.n_ref_bins, mdp.H
    rsum = np.zeros((A, B)); rcnt = np.zeros((A, B))
    for tr in trajs:
        for t in range(len(tr.actions)):
            b, _ = mdp.decode_obs(tr.obs[t])
            a = int(tr.actions[t])
            rsum[a, b] += tr.rewards[t]; rcnt[a, b] += 1.0
    rcnt_a = rcnt.sum(axis=1); rsum_a = rsum.sum(axis=1)
    rbar_a = np.where(rcnt_a > 0, rsum_a / np.maximum(rcnt_a, 1.0), np.nan)
    rglob = rsum.sum() / max(rcnt.sum(), 1.0)

    def reward_at(a, b):
        if rcnt[a, b] > 0:
            return rsum[a, b] / rcnt[a, b]
        if rcnt_a[a] > 0:
            return rbar_a[a]
        return rglob

    # forward-propagate the deterministic target occupancy under the known transition
    d = np.zeros(B)
    for b0 in init_bins:
        d[b0] += 1.0
    d /= d.sum()
    v_mis, supported, total = 0.0, 0.0, 0.0
    for t in range(H):
        dnext = np.zeros(B)
        for b in range(B):
            if d[b] <= 0:
                continue
            a = int(pe_argmax(mdp.obs(mdp.ref_grid[b], t)))
            v_mis += d[b] * reward_at(a, b)
            total += d[b]
            if rcnt[a, b] > 0:
                supported += d[b]
            dnext[mdp.N[a, b]] += d[b]
        d = dnext
    return float(v_mis), float(supported / max(total, 1e-9))


# ----------------------- per-decision doubly robust -----------------------
def dr_value(trajs, pi_e_probs, pi_b_probs, qhat, n_actions, gamma=1.0, w_clip=10.0):
    """Per-decision doubly robust estimator, following Jiang & Li (2016), with two
    departures from the estimator as published:

    - the cumulative weight is clipped at `w_clip` at every step, which bounds the
      variance the product of ratios would otherwise carry and introduces a bias the
      published estimator does not have;
    - the residual carries a single `gamma`, not `gamma ** t`. Every call in this project
      passes gamma = 1.0 (ExpConfig sets undiscounted finite-horizon returns), so no
      reported number depends on it, but the parameter is not the standard discount.
    """
    ests = []
    for tr in trajs:
        H = len(tr.actions)
        v0 = sum(pi_e_probs(tr.obs[0])[a] * qhat(tr.obs[0], a) for a in range(n_actions))
        acc = v0
        w = 1.0
        for t in range(H):
            pe = pi_e_probs(tr.obs[t]); pb = pi_b_probs(tr.obs[t])
            rho = pe[tr.actions[t]] / max(pb[tr.actions[t]], 1e-6)
            w = min(w * rho, w_clip)
            q_sa = qhat(tr.obs[t], tr.actions[t])
            acc += w * (tr.rewards[t] - q_sa)
            if t + 1 < H:
                v_next = sum(pi_e_probs(tr.obs[t + 1])[a] * qhat(tr.obs[t + 1], a)
                             for a in range(n_actions))
                acc += w * gamma * v_next
        ests.append(acc)
    return float(np.mean(ests))


# ----------------------- change-point detection -----------------------
def detect_segments(trajs, max_segments, min_gain_frac=0.05):
    """PENALISED greedy binary segmentation on per-trajectory mean action (a cheap
    proxy for behaviour-policy shift). Returns a segment id per trajectory.

    A split is only accepted if its variance-reduction exceeds `min_gain_frac` of the
    total variance, so the detector does NOT over-segment: with no real drift it
    returns a single segment (segmented == pooled, benefit 0 by construction), and it
    adds boundaries only where the logging policy genuinely shifts. Without this
    stopping rule the detector splits to `max_segments` regardless of drift, starving
    each segment's behaviour-policy estimate of data and inflating its variance."""
    means = np.array([tr.actions.mean() for tr in trajs])
    n = len(means)
    total = ((means - means.mean()) ** 2).sum()
    boundaries = [0, n]
    if total <= 1e-9:
        return np.zeros(n, dtype=int)

    def cost(a, b):
        seg = means[a:b]
        return ((seg - seg.mean()) ** 2).sum() if b > a else 0.0

    while len(boundaries) - 1 < max_segments:
        best_gain, best_b, best_pos = 0.0, None, None
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            base = cost(a, b)
            for split in range(a + 1, b):
                gain = base - cost(a, split) - cost(split, b)
                if gain > best_gain:
                    best_gain, best_b, best_pos = gain, split, i
        # stop unless the best split explains a non-trivial share of total variance
        if best_b is None or best_gain < min_gain_frac * total:
            break
        boundaries.insert(best_pos + 1, best_b)
    seg_id = np.zeros(n, dtype=int)
    for s in range(len(boundaries) - 1):
        seg_id[boundaries[s]:boundaries[s + 1]] = s
    return seg_id


# ----------------------- pooled vs segmented DR -----------------------
def pooled_dr(trajs, pi_e_probs, n_actions, gamma=1.0):
    """Doubly-robust value with ONE behaviour policy fitted to the whole log.

    Under a drifting logger no single policy generated the data, so the importance
    weights are wrong in a direction that does not average out. This is the biased
    estimate the RQ3 demonstration is built around, not a recommended estimator.
    """
    pb = estimate_behaviour_policy(trajs, n_actions)
    qh = fit_qhat(trajs, n_actions)
    return dr_value(trajs, pi_e_probs, pb, qh, n_actions, gamma)


def segmented_dr(trajs, pi_e_probs, n_actions, gamma=1.0, oracle=True, max_segments=6):
    """Doubly-robust value with a behaviour policy fitted per regime.

    Recovers what `pooled_dr` loses. `oracle=True` uses the simulator's true segment
    boundaries and so measures the ceiling of the correction; `oracle=False` detects the
    change points from the log, which is what a practitioner could actually do.
    """
    if oracle:
        seg_ids = np.array([tr.seg for tr in trajs])
    else:
        seg_ids = detect_segments(trajs, max_segments)
    qh = fit_qhat(trajs, n_actions)              # DM can be global
    vals, weights = [], []
    for s in np.unique(seg_ids):
        sub = [tr for tr, sid in zip(trajs, seg_ids) if sid == s]
        if len(sub) < 3:
            continue
        pb = estimate_behaviour_policy(sub, n_actions)
        vals.append(dr_value(sub, pi_e_probs, pb, qh, n_actions, gamma))
        weights.append(len(sub))
    weights = np.array(weights, float); weights /= weights.sum()
    return float(np.dot(weights, vals))
