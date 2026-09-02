"""Offline contextual-bandit baselines — the paradigm actually deployed in pricing.

WHY THIS RUN EXISTS. Every learner in this project is sequential: DT, Q-DT, CQL,
IQL and the estimate-then-optimize planner all reason about the fact that today's
price moves tomorrow's reference price. That is the right comparison *within*
offline RL, but it is not the comparison a pricing practitioner faces. Commercially
deployed pricing is dominated by rule-based / elasticity-driven systems and, in the
fastest-growing patent cluster, by CONTEXTUAL BANDITS — which treat each
(state, timestep) as an independent decision and ignore the intertemporal coupling
entirely.

The project already contains the bandit as a *reference point* rather than a
method: `normalised_value` is anchored so that

    nv = 0  <->  the ORACLE myopic policy  (argmax_a of the TRUE reward table R)
    nv = 1  <->  the optimal sequential policy

and the oracle myopic policy is exactly the contextual bandit solution under
perfect information. So every nv already reads as "how much of the
bandit -> sequential gap did this method close". What is missing is the other half:
a bandit that has to LEARN the reward from the same logs everything else sees. That
policy is the realistic alternative, and it can score below zero, because it pays
estimation error on top of being myopic.

ARMS. Three standard offline bandit policy learners, all myopic by construction
(they optimise immediate logged reward only and never model the transition):

    bandit_DM    direct method: fit rhat(s,a) by regression on logged (s,a,r),
                 then act argmax_a rhat(s,a). This is the one-step form of the
                 elasticity-driven "estimate demand, then optimise" pipeline, and
                 the natural bandit analogue of `diag_estimate_optimize`.
    bandit_IPS   counterfactual risk minimisation on a softmax policy with
                 self-normalised, clipped importance weights
                 w = pi_theta(a|s) / pihat_b(a|s)   (Swaminathan and Joachims, 2015).
    bandit_DR    doubly robust policy optimisation: the DM model as the baseline
                 plus the IPS correction on its residual (Dudik, Langford and Li,
                 2011), i.e. the policy-learning twin of `ope.dr_value`.

Each arm is evaluated twice: bare, and with the SAME top-3 logged-support mask the
DT / Q-DT / IQL arms carry in `diag_gate2_pricing`. Masking is inference-only, so
the masked twin reuses the identical trained model and the pair is exact. Given the
measured dose-response (mask gain rises monotonically with off-support rate), a
learner that argmaxes a fitted reward surface over the full price grid is expected
to leave support often and therefore to gain a lot from the mask.

`bandit_oracle` is a correctness check, not a result: it is the myopic argmax of the
TRUE reward table, so it must evaluate to nv = 0.000 exactly. If it does not, the
anchoring assumption above is wrong and nothing else in this file can be read.

PROTOCOL. Identical cells, seeds, anchors, start bins and row schema as
`diag_gate2_pricing`, so the outputs merge straight into that comparison table.
Nuisances are estimated with the project's existing OPE machinery
(`ope.estimate_behaviour_policy`) rather than a fresh implementation, so the
propensity model here is the same one the C3 results are computed against.

Implements: the offline contextual-bandit reference point of Appendix F.1.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: main().
#
# The deployed-practice reference point: three offline contextual-bandit learners --
# direct method, self-normalised counterfactual risk minimisation and a doubly-robust
# hybrid -- myopic by construction, on the same logs and protocol as the sequential
# arms.
#
# Implements or follows:
#   - Dudík, M., Langford, J. and Li, L. (2011) 'Doubly Robust Policy Evaluation and
#     Learning', ICML. arXiv:1103.4601.
#   - Swaminathan, A. and Joachims, T. (2015) 'Batch Learning from Logged Bandit Feedback
#     through Counterfactual Risk Minimization', JMLR 16.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core.baselines import (MLP, make_support_masked_qnet_policy,
                                       policy_from_qnet)
from pricing_dt.core.ope import estimate_behaviour_policy
from pricing_dt.core.torch_utils import default_device
from pricing_dt.diagnostics.diag_gate2_pricing import (_cell_id, _evaluate_method,
                                                       _parse_cells, _parse_seeds,
                                                       _summarise, _support_counts)
from pricing_dt.experiments.experiments import _seed, _setup, _traj_start_bins


METHODS = [
    "Bandit DM",
    "Bandit DM support top3",
    "Bandit IPS",
    "Bandit IPS support top3",
    "Bandit DR",
    "Bandit DR support top3",
    "Bandit oracle (myopic, true R)",
]


def _bandit_data(trajs, n_actions):
    """Flatten trajectories to contextual-bandit tuples (s, a, r).

    The transition is deliberately discarded: that discarding IS the bandit
    assumption, and the size of the resulting loss is what this diagnostic
    measures.
    """
    S, A, R = [], [], []
    for tr in trajs:
        for t in range(len(tr.actions)):
            S.append(tr.obs[t])
            A.append(int(tr.actions[t]))
            R.append(float(tr.rewards[t]))
    return (np.asarray(S, dtype=np.float32),
            np.asarray(A, dtype=np.int64),
            np.asarray(R, dtype=np.float32))


def _logged_propensities(trajs, S, A, n_actions):
    """pihat_b(a_i | s_i) for every logged tuple, via the project's OPE estimator."""
    pb = estimate_behaviour_policy(trajs, n_actions)
    return np.asarray([pb(s)[a] for s, a in zip(S, A)], dtype=np.float32)


def _fit_reward_model(S, A, R, obs_dim, n_actions, hidden, updates, lr, batch,
                      device, seed):
    """rhat(s, a): MSE regression of the logged reward onto (state, action).

    Outputs one head per action so the model is queried exactly like a Q-net; this
    is what lets the support mask and `policy_from_qnet` be reused unchanged.
    """
    torch.manual_seed(seed)
    net = MLP(obs_dim, n_actions, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    s = torch.as_tensor(S, device=device)
    a = torch.as_tensor(A, device=device)
    r = torch.as_tensor(R, device=device)
    n = len(S)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(updates):
        idx = torch.randint(0, n, (min(batch, n),), generator=g).to(device)
        pred = net(s[idx]).gather(1, a[idx].unsqueeze(1)).squeeze(1)
        loss = ((pred - r[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def _fit_policy(S, A, R, PB, obs_dim, n_actions, *, hidden, updates, lr, batch,
                w_clip, entropy_coef, device, seed, rhat=None):
    """Softmax policy trained by self-normalised IPS, or by DR when `rhat` is given.

    IPS objective (maximised):   sum_i w_i r_i / sum_i w_i,   w_i = pi(a_i|s_i)/pihat_b(a_i|s_i)
    DR objective  (maximised):   E_a~pi[rhat(s_i,a)] + w_i (r_i - rhat(s_i,a_i))

    Self-normalisation and weight clipping are the standard variance controls; the
    small entropy bonus stops the softmax collapsing onto a single price in the
    first few hundred updates, which is a practical failure mode of plain CRM on
    logs this small. POEM's variance regulariser is NOT implemented, which is recorded
    machine-readably in the `not_implemented` field this script writes.
    """
    torch.manual_seed(seed + 1)
    net = MLP(obs_dim, n_actions, hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    s = torch.as_tensor(S, device=device)
    a = torch.as_tensor(A, device=device)
    r = torch.as_tensor(R, device=device)
    pb = torch.as_tensor(PB, device=device).clamp_min(1e-6)
    if rhat is not None:
        with torch.no_grad():
            rh_all = rhat(s)                                    # [n, A]
            rh_taken = rh_all.gather(1, a.unsqueeze(1)).squeeze(1)
    n = len(S)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    for _ in range(updates):
        idx = torch.randint(0, n, (min(batch, n),), generator=g).to(device)
        logp = torch.log_softmax(net(s[idx]), dim=-1)
        p_taken = logp.gather(1, a[idx].unsqueeze(1)).squeeze(1).exp()
        w = (p_taken / pb[idx]).clamp(max=w_clip)
        if rhat is None:
            obj = (w * r[idx]).sum() / w.sum().clamp_min(1e-8)
        else:
            direct = (logp.exp() * rh_all[idx]).sum(dim=1)
            obj = (direct + w * (r[idx] - rh_taken[idx])).mean()
        entropy = -(logp.exp() * logp).sum(dim=1).mean()
        loss = -(obj + entropy_coef * entropy)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


def _oracle_myopic_policy(mdp):
    """argmax_a of the TRUE reward table — the perfect-information bandit.

    This is the policy `normalised_value` is anchored on, so it must score exactly
    nv = 0.000. It is included as an assertion, not as a comparator.
    """
    def fn(obs):
        b = mdp.obs_to_bins(np.asarray(obs, dtype=np.float32).reshape(1, -1))[0]
        return int(mdp.R[:, int(b)].argmax())

    def batched(obs, t=None):
        bins = mdp.obs_to_bins(np.asarray(obs, dtype=np.float32))
        return mdp.R[:, bins].argmax(axis=0)

    fn.batched = batched
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results_bandit_baseline")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--w-clip", type=float, default=10.0)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--support-topk", type=int, default=3)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, n_actions = 2, cfg.sim.n_prices
    seeds = _parse_seeds(args.seeds) if args.seeds else cfg.exp.seeds
    cells = _parse_cells(args.cells) if args.cells else (
        [(min(cfg.exp.data_sizes), max(cfg.exp.noise_levels))] if args.smoke
        else [(n, z) for n in cfg.exp.data_sizes for z in cfg.exp.noise_levels])
    updates = 300 if args.smoke else 2000
    lr, batch = 3e-4, 256
    device = default_device()

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "bandit_protocol.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "stage": "Offline contextual-bandit baselines (the deployed paradigm)",
            "preset": "smoke" if args.smoke else "full",
            "cells": cells, "seeds": seeds, "methods": METHODS,
            "framing": ("nv = 0 is the ORACLE myopic (perfect-information bandit) "
                        "policy and nv = 1 is the optimal sequential policy, so "
                        "these arms measure what a LEARNED bandit gives up "
                        "relative to that anchor. They are a realistic alternative "
                        "to compare against, not a comparator to beat."),
            "learners": {
                "DM": "MLP rhat(s,a) by MSE on logged (s,a,r); act argmax_a rhat",
                "IPS": "softmax policy, self-normalised clipped IPS (CRM)",
                "DR": "softmax policy, DM baseline + IPS residual correction",
            },
            "propensity": "ope.estimate_behaviour_policy (logistic on obs), the same estimator C3 uses",
            "hyperparams": {"hidden": args.hidden, "updates": updates, "lr": lr,
                            "batch": batch, "w_clip": args.w_clip,
                            "entropy_coef": args.entropy_coef},
            "not_implemented": "POEM variance regularisation (Swaminathan and Joachims 2015)",
            "support_mask": f"top{args.support_topk}, inference-only, identical to diag_gate2_pricing",
        }, f, indent=2)

    raw_rows = []
    for n, noise in cells:
        cell = _cell_id(n, noise)
        for seed in seeds:
            print(f"\n=== Bandit baselines cell={cell} seed={seed} ===", flush=True)
            _seed(seed)
            mdp, _, _, _ = _setup(cfg, {"demand_noise": float(noise)}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, int(n), float(noise), seed,
                                               cfg.data.expert_q)
            init_bins = _traj_start_bins(trajs)
            v_opt = float(mdp.Vstar[0, init_bins].mean())
            myopic = _oracle_myopic_policy(mdp)
            v_beh, _ = mdp.evaluate_policy_fn(myopic, init_bins)
            counts, totals = _support_counts(trajs, mdp)

            S, A, R = _bandit_data(trajs, n_actions)
            PB = _logged_propensities(trajs, S, A, n_actions)

            rhat = _fit_reward_model(S, A, R, obs_dim, n_actions, args.hidden,
                                     updates, lr, batch, device, seed)
            pi_ips = _fit_policy(S, A, R, PB, obs_dim, n_actions,
                                 hidden=args.hidden, updates=updates, lr=lr,
                                 batch=batch, w_clip=args.w_clip,
                                 entropy_coef=args.entropy_coef, device=device,
                                 seed=seed, rhat=None)
            pi_dr = _fit_policy(S, A, R, PB, obs_dim, n_actions,
                                hidden=args.hidden, updates=updates, lr=lr,
                                batch=batch, w_clip=args.w_clip,
                                entropy_coef=args.entropy_coef, device=device,
                                seed=seed, rhat=rhat)

            nets = [("Bandit DM", rhat), ("Bandit IPS", pi_ips), ("Bandit DR", pi_dr)]
            for name, net in nets:
                for sfx, mask_name in [("", "none"),
                                       (" support top3", f"top{args.support_topk}")]:
                    policy = (policy_from_qnet(net) if mask_name == "none"
                              else make_support_masked_qnet_policy(
                                  net, mdp, counts, topk=args.support_topk))
                    _evaluate_method(
                        raw_rows, gate="gate2D_bandit", cell_id=cell, n=n,
                        noise=noise, seed=seed, method=name + sfx, family="Bandit",
                        policy_fn=policy, mdp=mdp, init_bins=init_bins,
                        v_beh=v_beh, v_opt=v_opt, counts=counts, totals=totals,
                        extra={"support_mask": mask_name,
                               "bandit_updates": int(updates),
                               "w_clip": float(args.w_clip)})
                    print(f"  {name + sfx}: done", flush=True)

            _evaluate_method(
                raw_rows, gate="gate2D_bandit", cell_id=cell, n=n, noise=noise,
                seed=seed, method="Bandit oracle (myopic, true R)", family="Bandit",
                policy_fn=myopic, mdp=mdp, init_bins=init_bins, v_beh=v_beh,
                v_opt=v_opt, counts=counts, totals=totals,
                extra={"support_mask": "none"})
            print("  Bandit oracle (myopic, true R): done", flush=True)

            raw = pd.DataFrame(raw_rows)
            raw.to_csv(os.path.join(args.outdir, "bandit_raw.csv"), index=False)
            _summarise(raw).to_csv(os.path.join(args.outdir, "bandit_summary.csv"),
                                   index=False)

    raw = pd.DataFrame(raw_rows)
    means = (raw.groupby("method")
             .agg(n_runs=("nv", "size"), mean_nv=("nv", "mean"),
                  median_nv=("nv", "median"),
                  mean_unseen_rate=("selected_unseen_rate", "mean"),
                  mean_behavior_prob=("mean_behavior_prob", "mean"))
             .reset_index().sort_values("mean_nv", ascending=False))
    means.to_csv(os.path.join(args.outdir, "bandit_method_means.csv"), index=False)

    anchor = raw[raw.method == "Bandit oracle (myopic, true R)"].nv.abs().max()
    print(f"\nANCHOR CHECK: max |nv| of the oracle myopic policy = {anchor:.3e} "
          f"(must be ~0; nv=0 is defined as this policy)")
    print(means.to_string(index=False))


if __name__ == "__main__":
    main()
