"""Structure-enabled return-to-go relabelling (the proposed mechanism).

At each logged state the structured demand model estimates the achievable
expected revenue under a counterfactually better price; the return-to-go is
recomputed against this structurally grounded target. This supplies the
dynamic-programming-like recombination Q-DT gets from a bootstrapped value,
but anchored to economic structure -> lower variance in low-data regions and
no value-overestimation.

We exploit the KNOWN reference-price transition (it is part of the testbed) to
roll the demand model forward greedily over the remaining horizon. In a setting
where the transition is unknown, this would be replaced by a learned or assumed
transition; that substitution is a clean extension point.

References:
  - Yamagata, Khalil and Santos-Rodriguez (2023), bootstrapped value relabelling
    for Q-DT.
  - Hu et al. (2024), Gao et al. (2024), and Kim et al. (2024), modern
    Q-/advantage-aided conditional sequence modelling.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: relabel_dataset(), achievable_rtg(), oracle_rtg(), logged_rtg().
#
# Structured return relabelling: rewrite each trajectory's conditioning target using the
# domain model, decomposed into an action term and a potential (roll-forward) term.
#
# Implements or follows:
#   - Andrychowicz, M. et al. (2017) 'Hindsight Experience Replay', NeurIPS 30.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
import torch
from pricing_dt.core.torch_utils import resolve_device


def achievable_rtg(tr, demand_model, mdp, lam=1.0, device=None):
    """Return an ACTION-DEPENDENT relabelled RTG vector [H] for one trajectory.

    The structured relabelling target of §3.4, in its action-dependent form:

        R_hat_t = r_hat(s_t, a_t)  +  sum_{k>t} max_p r_hat(s_k, p)

    The FIRST term is the demand-model (de-noised) revenue of the action the log
    ACTUALLY took at t — not the optimal price. Only the continuation (k > t) is
    the greedy demand-model optimum, rolled forward under the known reference-
    price transition. Why this matters: if every step used the optimal price, the
    relabelled return-to-go would be identical regardless of the logged action,
    and the Decision Transformer would learn to associate a high target with the
    sub-optimal actions actually present in the data. Making the current step
    action-dependent means a sub-optimal logged action receives a lower R_hat, so
    the DT correctly associates the high achievable target with the good action.
    (This is the action-dependent analogue of Q-DT's value relabelling, anchored
    to economic structure rather than a bootstrapped value.)

    The de-noised current-step revenue also separates skill from luck: the DT is
    conditioned on the achievable expected return of the action taken, not on a
    noise-inflated realised return — which targets the stochasticity failure.

    Blended with the logged g_t by anchor weight lam (lam=1 => fully structured).
    """
    rev, cont = _model_tables(demand_model, mdp, device)
    return _apply_tables(tr, rev, cont, mdp, lam)


def _model_tables(demand_model, mdp, device=None):
    """Precompute the two tables the relabeller actually needs, in ONE forward.

    Both the current-step term and the greedy continuation only ever query the
    demand model at grid points, because the reference price starts on the grid
    and the known transition `mdp.N` maps grid points to grid points. So the
    whole relabeller is determined by

        rev[k, b, a] = price_a * E[demand](price_a, obs(ref_grid[b], k))

    over the H x n_ref_bins x n_prices grid — a single batched evaluation. The
    greedy continuation then follows by a pure-numpy backward pass:

        cont[H, b] = 0
        cont[k, b] = rev[k, b, a*] + cont[k+1, N[a*, b]],   a* = argmax_a rev[k, b, a]

    Note a* is the MYOPIC argmax of immediate revenue, faithfully reproducing the
    original greedy rollout (it is deliberately not a dynamic-programming optimum).
    """
    device = resolve_device(device, demand_model)
    H, B, A = mdp.H, mdp.cfg.n_ref_bins, len(mdp.prices)
    obs = np.stack([mdp.obs_batch(mdp.ref_grid, k) for k in range(H)])      # [H, B, obs]
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    s = obs_t.unsqueeze(2).expand(H, B, A, obs_t.shape[-1]).reshape(-1, obs_t.shape[-1])
    prices = torch.as_tensor(mdp.prices, dtype=torch.float32, device=device)
    p = prices.view(1, 1, A).expand(H, B, A).reshape(-1)

    demand_model.eval()
    with torch.no_grad():
        dem = demand_model(p, s).reshape(H, B, A)
        rev = (prices.view(1, 1, A) * dem).cpu().numpy().astype(np.float32)

    cont = np.zeros((H + 1, B), np.float32)
    bins = np.arange(B)
    for k in reversed(range(H)):
        a_best = rev[k].argmax(axis=1)                                     # [B]
        cont[k] = rev[k, bins, a_best] + cont[k + 1, mdp.N[a_best, bins]]
    return rev, cont


def _apply_tables(tr, rev, cont, mdp, lam):
    """Table lookup form of the per-trajectory relabelled RTG."""
    H = mdp.H
    steps = np.arange(H)
    b = np.asarray(tr.ref_bins[:H], dtype=int)
    a = np.asarray(tr.actions[:H], dtype=int)
    # current step uses the LOGGED action; the continuation starts from the bin
    # that action leads to, one timestep later.
    relabelled = (rev[steps, b, a] + cont[steps + 1, mdp.N[a, b]]).astype(np.float32)
    return (1 - lam) * tr.rtg + lam * relabelled


def relabel_dataset(trajs, demand_model, mdp, lam=1.0, device=None):
    """Relabel a whole dataset. The demand-model tables depend only on the model
    and the MDP grid, never on the trajectory, so they are built once here rather
    than rebuilt for every trajectory."""
    rev, cont = _model_tables(demand_model, mdp, device)
    return [_apply_tables(tr, rev, cont, mdp, lam) for tr in trajs]


def oracle_rtg(trajs, mdp, lam=1.0):
    """ORACLE relabel — the ceiling of the goal channel.

    Exactly the same action-dependent form as `achievable_rtg`, but computed with
    the TRUE demand model instead of a fitted one. Because the continuation term
    is then the true optimal continuation, the target collapses to the exact
    action-value of the logged action:

        R_hat_t = R[a_t, b_t] + Vstar[t+1, N[a_t, b_t]]  ==  Qstar[t, b_t, a_t]

    Why this arm matters. All three relabellers in the study are estimators of the
    SAME quantity Q*(s_t, a_t): the structured relabeller estimates it with a
    prior-constrained demand model, Q-DT estimates it by bootstrapping, and this
    arm IS it. So it upper-bounds both, and it separates two mechanisms that the
    structured prior confounds — a target that is *accurate* from a target that is
    merely *stable across seeds* (Q* is both; the fitted structured prior is only
    the latter, being prior-dominated and data-independent). If the structured
    relabel beats the oracle, target accuracy is not what the goal channel wants.
    """
    if not hasattr(mdp, "Qstar"):
        mdp.solve_optimal()
    out = []
    for tr in trajs:
        H = len(tr.actions)
        g = np.array([mdp.Qstar[t, int(tr.ref_bins[t]), int(tr.actions[t])]
                      for t in range(H)], np.float32)
        out.append((1 - lam) * tr.rtg + lam * g)
    return out


def logged_rtg(trajs):
    return [tr.rtg for tr in trajs]
