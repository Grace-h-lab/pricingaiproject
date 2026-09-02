"""How much of the support mask's value is the logging policy's own competence?

Every support result in this study is measured on a log whose two region specialists
jointly contain the optimal action for each state, because `expert_q = 1.0` makes each
specialist play the exact dynamic-programming optimum inside its own region. Chapter 5
concedes that the absolute level of the masked scores depends on this; nothing has ever
measured how much.

This sweeps `expert_q` and records the whole chain at each setting:

    logger competence  ->  what the mask admits  ->  what any policy could get inside it
                                                ->  what a policy with no learner gets
                                                ->  what learners actually get

The ceiling is the quantity that was missing. A masked score of 0.97 means something
very different when the best achievable inside the mask is 0.99 than when it is 0.6.

Two exact quantities are computed by backward induction rather than sampled: the value
of a uniform policy confined to the mask (the no-learner floor) and the value of the
best policy confined to the mask (the ceiling).

The planners here use a CONSTRAINED backup: when the policy may only act inside the
mask, the value function backs up over the mask too. `diag_family_table` backs up over
all actions and restricts only the acting step, which lets a planner value continuations
it is not allowed to take; that inconsistency is fixed here.

Implements: the ceiling of Equation (3.10) and the floor of Equation (3.6) across
logging-policy competence, drawn as Figure 4.6.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: main().
#
# Sweeps the logging policy's competence and re-measures the ceiling reachable inside
# the mask, the no-learner floor and the arms, showing what a constrained score actually
# buys.
#
# Implements or follows:
#   - Fujimoto, S., Conti, E., Ghavamzadeh, M. and Pineau, J. (2019) 'Benchmarking Batch
#     Deep Reinforcement Learning Algorithms', eq. 17. arXiv:1910.01708.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import argparse
import csv
import json
import os

import numpy as np
import torch

from pricing_dt.core import config as C
from pricing_dt.core import provenance
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.baselines import (make_support_masked_qnet_policy,
                                       policy_from_qnet, train_bc)
from pricing_dt.core.demand_model import (StructuredDemandModel,
                                          UnconstrainedDemandModel,
                                          fit_demand_model)
from pricing_dt.core.dt import (_supported_actions, make_dt_policy,
                                make_support_masked_dt_policy, train_dt)
from pricing_dt.core.relabel import logged_rtg, relabel_dataset
from pricing_dt.diagnostics.diag_bandit_baseline import _oracle_myopic_policy
from pricing_dt.diagnostics.diag_gate2_pricing import _support_counts
from pricing_dt.diagnostics.diag_logger_value import _specialist_value_exact
from pricing_dt.experiments.experiments import _seed, _setup, _traj_start_bins


def _support_grid(counts, mdp, topk):
    """Boolean [H, B, A] mask of admissible actions."""
    S = np.zeros((mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=bool)
    for t in range(mdp.H):
        for b in range(mdp.cfg.n_ref_bins):
            S[t, b] = _supported_actions(counts, t, b, mdp.cfg.n_prices, topk=topk)
    return S


def _exact_backward(mdp, R, S=None, reduce="max"):
    """Exact value of the best (or uniform) policy, optionally confined to S.

    `R` is [H, B, A] expected reward; with `reduce="mean"` this is the value of
    choosing uniformly among the admissible actions, which is the no-learner floor.
    """
    B = mdp.cfg.n_ref_bins
    V = np.zeros((mdp.H + 1, B))
    for t in reversed(range(mdp.H)):
        for b in range(B):
            q = R[t, b, :] + V[t + 1, mdp.N[:, b]]
            ok = np.ones(len(q), bool) if S is None else S[t, b]
            if not ok.any():
                ok = np.ones(len(q), bool)
            V[t, b] = q[ok].max() if reduce == "max" else q[ok].mean()
    return V


def _fitted_rewards(dm, mdp, device="cpu"):
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    Rhat = np.zeros((H, B, A))
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(B):
                s = torch.tensor(mdp.obs(mdp.ref_grid[b], t), dtype=torch.float32,
                                 device=device).unsqueeze(0).repeat(A, 1)
                Rhat[t, b, :] = mdp.prices * dm(prices, s).cpu().numpy()
    return Rhat


def _constrained_planner(Rhat, mdp, S=None):
    """Greedy policy whose value function backs up over the SAME action set it may use."""
    B = mdp.cfg.n_ref_bins
    V = np.zeros((mdp.H + 1, B))
    pi = np.zeros((mdp.H, B), dtype=int)
    for t in reversed(range(mdp.H)):
        for b in range(B):
            q = Rhat[t, b, :] + V[t + 1, mdp.N[:, b]]
            ok = np.ones(len(q), bool) if S is None else S[t, b]
            if not ok.any():
                ok = np.ones(len(q), bool)
            masked = np.where(ok, q, -np.inf)
            pi[t, b] = int(masked.argmax())
            V[t, b] = masked.max()

    def fn(obs):
        b, t = mdp.decode_obs(obs)
        return int(pi[t, b])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_expertq_sweep")
    ap.add_argument("--expert-q", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--sizes", default="100,400,1600")
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--with-dt", action="store_true",
                    help="also train the GOAL-CHANNEL arms, so the channel contrast "
                         "itself can be traced against logger competence")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = C.smoke() if a.smoke else C.full()
    qs = [float(x) for x in a.expert_q.split(",")]
    sizes = [int(x) for x in a.sizes.split(",")]
    seeds = range(1 if a.smoke else a.seeds)
    if a.smoke:
        qs, sizes = qs[:2], sizes[:1]
    obs_dim, n_actions = 2, cfg.sim.n_prices
    os.makedirs(a.outdir, exist_ok=True)
    # Record the commit, device and library versions this run actually used; the
    # parameters alone do not pin a result. See REPRODUCE.md.
    provenance.stamp(a.outdir, replace=True)

    rows = []
    for q in qs:
        for n in sizes:
            for seed in seeds:
                _seed(seed)
                cfg.data.n_train_traj = int(n)
                cfg.data.expert_q = float(q)
                mdp, _, _, _ = _setup(cfg, {"demand_noise": a.noise}, seed=seed)
                trajs = D.make_stitching_necessary(mdp, int(n), a.noise, seed, q)
                init = _traj_start_bins(trajs)
                counts, _ = _support_counts(trajs, mdp)
                S = _support_grid(counts, mdp, a.topk)
                v_opt = float(mdp.Vstar[0, init].mean())
                v_anchor, _ = mdp.evaluate_policy_fn(_oracle_myopic_policy(mdp), init)
                nv = lambda v: float(M.normalised_value(float(v), v_anchor, v_opt))

                Rtrue = np.stack([mdp.R.T for _ in range(mdp.H)])      # [H, B, A]
                ceiling = nv(_exact_backward(mdp, Rtrue, S, "max")[0, init].mean())
                floor = nv(_exact_backward(mdp, Rtrue, S, "mean")[0, init].mean())
                floor_bare = nv(_exact_backward(mdp, Rtrue, None, "mean")[0, init].mean())
                astar_in = float(np.mean([S[t, b, int(mdp.pistar[t, b])]
                                          for t in range(mdp.H)
                                          for b in range(mdp.cfg.n_ref_bins)]))
                v_log = 0.5 * (_specialist_value_exact(mdp, True, init, q)
                               + _specialist_value_exact(mdp, False, init, q))

                dt_out = {}
                dm_goal = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo,
                                                cfg.model.elasticity_hi)
                fit_demand_model(dm_goal, trajs, mdp, cfg.model.demand_epochs)

                bc = train_bc(trajs, obs_dim, n_actions, cfg.model, seed=seed)
                v_bc, _ = mdp.evaluate_policy_fn(policy_from_qnet(bc), init)
                v_bcm, _ = mdp.evaluate_policy_fn(
                    make_support_masked_qnet_policy(bc, mdp, counts, topk=a.topk), init)

                if a.with_dt:
                    # The goal channel. Without this the sweep measures how logger
                    # competence bounds the MASK, but not how it bounds the CHANNEL
                    # contrast -- and the channel swing is the headline claim.
                    for tag, rtg in (("vanilla", logged_rtg(trajs)),
                                     ("structured", relabel_dataset(trajs, dm_goal, mdp))):
                        m = train_dt(D.pack_dt(trajs, rtg), obs_dim, n_actions,
                                     cfg.model, seed=seed)
                        target = float(np.quantile(np.concatenate(rtg), 0.95))
                        vb, _ = mdp.evaluate_policy_fn(make_dt_policy(m, mdp, target), init)
                        vm, _ = mdp.evaluate_policy_fn(
                            make_support_masked_dt_policy(m, mdp, target, counts,
                                                          topk=a.topk), init)
                        dt_out[f"nv_dt_{tag}"] = nv(vb)
                        dt_out[f"nv_dt_{tag}_masked"] = nv(vm)

                out = dict(expert_q=q, N=n, seed=seed, astar_in_mask=astar_in,
                           nv_logger=nv(v_log), nv_ceiling=ceiling,
                           nv_floor_masked=floor, nv_floor_bare=floor_bare,
                           nv_bc=nv(v_bc), nv_bc_masked=nv(v_bcm), **dt_out)
                for tag, dm in (("struct", StructuredDemandModel(
                                    obs_dim, cfg.model.elasticity_lo,
                                    cfg.model.elasticity_hi)),
                                ("uncon", UnconstrainedDemandModel(obs_dim))):
                    fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
                    Rhat = _fitted_rewards(dm, mdp)
                    vb, _ = mdp.evaluate_policy_fn(_constrained_planner(Rhat, mdp), init)
                    vm, _ = mdp.evaluate_policy_fn(_constrained_planner(Rhat, mdp, S), init)
                    out[f"nv_eto_{tag}"] = nv(vb)
                    out[f"nv_eto_{tag}_masked"] = nv(vm)
                rows.append(out)
                # write after every config: a long sweep must not be all-or-nothing
                with open(os.path.join(a.outdir, "expertq_sweep.csv"), "w",
                          newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)
                print(f"  q={q:.2f} N={n:5d} s={seed}  logger {out['nv_logger']:+.3f}  "
                      f"ceiling {ceiling:+.3f}  floor {floor:+.3f}  "
                      f"a*in {astar_in:.1%}  EtOu_masked {out['nv_eto_uncon_masked']:+.3f}",
                      flush=True)

    path = os.path.join(a.outdir, "expertq_sweep.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(vars(a), open(os.path.join(a.outdir, "protocol.json"), "w"), indent=1)
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
