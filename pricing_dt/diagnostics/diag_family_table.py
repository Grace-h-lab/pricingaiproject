"""The two method families missing from the single comparable multi-family run.

`results_bandit_20260821/combined_raw.csv` puts offline contextual bandits, IQL,
both corrected Q-DT read-outs and both Decision-Transformer arms on one protocol --
same cells, same seeds, same start bins, same anchors, each with and without the
logged-support mask. It does not contain behaviour cloning, nor the *sequential*
estimate-then-optimise planner, so a four-family comparison can otherwise only be
assembled across runs that use different start-state conventions (Chapter 3,
"Value normalisation").

This script produces exactly those missing arms on the identical protocol, so the
rows can be concatenated with the existing file rather than compared across it.
Nothing here re-runs or overwrites a published arm.

The anchors are recomputed rather than read, and then CHECKED against the published
file: if they disagree the concatenation is inadmissible and the script says so.

Implements: the cross-family comparison under one mask and one protocol, reported as
Table 4.11.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: main().
#
# The cross-family comparison: every arm restricted to the same top-3 logged action set
# and evaluated on one protocol, so the ranking reflects the objective rather than the
# constraint.
#
# Implements or follows:
#   - Fujimoto, S., Conti, E., Ghavamzadeh, M. and Pineau, J. (2019) 'Benchmarking Batch
#     Deep Reinforcement Learning Algorithms', eq. 17. arXiv:1910.01708.
#   - Kostrikov, I., Nair, A. and Levine, S. (2022) 'Offline Reinforcement Learning with
#     Implicit Q-Learning'. arXiv:2110.06169.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from pricing_dt.core import config as C
from pricing_dt.core import provenance
from pricing_dt.core import data as D
from pricing_dt.core.baselines import (make_support_masked_qnet_policy,
                                       policy_from_qnet, train_bc)
from pricing_dt.core.demand_model import (StructuredDemandModel,
                                          UnconstrainedDemandModel,
                                          fit_demand_model)
from pricing_dt.core.dt import _supported_actions
from pricing_dt.diagnostics.diag_bandit_baseline import _oracle_myopic_policy
from pricing_dt.diagnostics.diag_gate2_pricing import (_cell_id, _evaluate_method,
                                                       _parse_cells, _parse_seeds,
                                                       _summarise, _support_counts)
from pricing_dt.experiments.experiments import _seed, _setup, _traj_start_bins


def fit_transition_regression(trajs, mdp):
    """Estimate the reference-price update from the log, by least squares.

    The planner is otherwise given the TRUE transition, which is an information
    advantage over the learners it is ranked against. This estimates it instead
    from the same trajectories, on the same footing as the demand model: a linear
    fit of next reference price on (reference price, chosen price).
    """
    X, y = [], []
    for tr in trajs:
        refs = mdp.ref_grid[tr.ref_bins]
        for t in range(len(tr.actions) - 1):
            X.append([refs[t], mdp.prices[int(tr.actions[t])], 1.0])
            y.append(refs[t + 1])
    X, y = np.asarray(X), np.asarray(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    A, B = mdp.cfg.n_prices, mdp.cfg.n_ref_bins
    N = np.zeros((A, B), dtype=int)
    for a in range(A):
        for b in range(B):
            pred = coef[0] * mdp.ref_grid[b] + coef[1] * mdp.prices[a] + coef[2]
            N[a, b] = mdp.ref_to_bin(float(np.clip(pred, mdp.cfg.p_min, mdp.cfg.p_max)))
    return N, coef


def empirical_transition(trajs, mdp):
    """Transition table built ONLY from observed (action, reference bin) pairs.

    Where the log never played action a in bin b there is no evidence about where
    that action leads, so the planner is left with a self-loop -- the least
    informative assumption available without inventing dynamics. This makes the
    dynamics as support-limited as the reward, which is the faithful analogue of the
    situation the learners face.
    """
    A, B = mdp.cfg.n_prices, mdp.cfg.n_ref_bins
    N = np.tile(np.arange(B)[None, :], (A, 1))          # unobserved -> self-loop
    seen = np.zeros((A, B), dtype=bool)
    for tr in trajs:
        for t in range(len(tr.actions) - 1):
            a, b = int(tr.actions[t]), int(tr.ref_bins[t])
            N[a, b] = int(tr.ref_bins[t + 1])
            seen[a, b] = True
    return N, float(seen.mean())


def plan_q(dm, mdp, device="cpu", N=None):
    """Backward induction under the fitted demand and a transition model.

    Same recursion as `diag_estimate_optimize.plan_with_dm`, but the full Q table is
    returned so that a support-masked planner can re-argmax over the admissible set
    instead of being handed an already-collapsed policy, and the transition may be
    supplied (`N`) rather than taken from the simulator.
    """
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    Nmat = mdp.N if N is None else N
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    Rhat = np.zeros((H, B, A))
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(B):
                s = torch.tensor(mdp.obs(mdp.ref_grid[b], t), dtype=torch.float32,
                                 device=device).unsqueeze(0).repeat(A, 1)
                Rhat[t, b, :] = mdp.prices * dm(prices, s).cpu().numpy()
    V = np.zeros((H + 1, B))
    Q = np.zeros((H, B, A))
    for t in reversed(range(H)):
        for b in range(B):
            Q[t, b, :] = Rhat[t, b, :] + V[t + 1, Nmat[:, b]]
            V[t, b] = Q[t, b, :].max()
    return Q, V


def planner_policy(Q, mdp, counts=None, topk=None):
    """Greedy under the planner's own Q, optionally restricted to logged support."""
    def fn(obs):
        b, t = mdp.decode_obs(obs)
        q = Q[t, b, :]
        if counts is None:
            return int(q.argmax())
        sup = _supported_actions(counts, t, b, mdp.cfg.n_prices, topk=topk)
        return int(np.where(sup, q, -np.inf).argmax())
    return fn


def _check_anchors(rows, published):
    """The concatenation is only admissible if the anchors agree exactly."""
    if published is None or not os.path.exists(published):
        return "not checked (published file absent)"
    pub = pd.read_csv(published)
    key = ["cell_id", "seed"]
    mine = pd.DataFrame(rows).drop_duplicates(key)[key + ["v_behaviour_expected",
                                                          "v_optimal_same_start"]]
    ref = pub.drop_duplicates(key)[key + ["v_behaviour_expected",
                                          "v_optimal_same_start"]]
    m = mine.merge(ref, on=key, suffixes=("_new", "_pub"))
    if m.empty:
        return "no overlapping (cell, seed) to check"
    da = float((m.v_behaviour_expected_new - m.v_behaviour_expected_pub).abs().max())
    do = float((m.v_optimal_same_start_new - m.v_optimal_same_start_pub).abs().max())
    ok = da < 1e-9 and do < 1e-9
    return (f"{'ADMISSIBLE' if ok else 'INADMISSIBLE'} — {len(m)} matched cells, "
            f"max abs anchor diff = {da:.3e}, max abs optimum diff = {do:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results_family_table")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--support-topk", type=int, default=3)
    ap.add_argument("--published",
                    default="results_bandit_20260821/bandit_raw.csv",
                    help="file whose anchors this run must reproduce")
    ap.add_argument("--merge-with",
                    default="results_bandit_20260821/combined_raw.csv",
                    help=("rows for the arms this script does not compute. Concatenated with "
                          "family_raw.csv to write four_family_raw.csv, which is the file the "
                          "cross-family claims are recomputed from. Pass '' to skip."))
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, n_actions = 2, cfg.sim.n_prices
    cells = (_parse_cells(args.cells) if args.cells else
             [(n, x) for n in cfg.exp.data_sizes for x in cfg.exp.noise_levels])
    seeds = _parse_seeds(args.seeds) if args.seeds else list(cfg.exp.seeds)
    if args.smoke:
        cells, seeds = cells[:1], seeds[:1]
    os.makedirs(args.outdir, exist_ok=True)
    # Record the commit, device and library versions this run actually used; the
    # parameters alone do not pin a result. See REPRODUCE.md.
    provenance.stamp(args.outdir, replace=True)

    raw_rows = []
    for n, noise in cells:
        cell = _cell_id(n, noise)
        for seed in seeds:
            print(f"\n=== family table cell={cell} seed={seed} ===", flush=True)
            _seed(seed)
            cfg.data.n_train_traj = int(n)
            mdp, _, _, _ = _setup(cfg, {"demand_noise": float(noise)}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, int(n), float(noise), seed,
                                               cfg.data.expert_q)
            init_bins = _traj_start_bins(trajs)
            v_opt = float(mdp.Vstar[0, init_bins].mean())
            v_beh, _ = mdp.evaluate_policy_fn(_oracle_myopic_policy(mdp), init_bins)
            counts, totals = _support_counts(trajs, mdp)

            bc = train_bc(trajs, obs_dim, n_actions, cfg.model, seed=seed)
            arms = [("Behaviour cloning", "BC", policy_from_qnet(bc), "none"),
                    (f"Behaviour cloning support top{args.support_topk}", "BC",
                     make_support_masked_qnet_policy(bc, mdp, counts,
                                                     topk=args.support_topk),
                     f"top{args.support_topk}")]

            N_learned, _coef = fit_transition_regression(trajs, mdp)
            N_emp, seen_frac = empirical_transition(trajs, mdp)
            transitions = [("", None), (" learned-transition", N_learned),
                           (" empirical-transition", N_emp)]

            for tag, dm in (("structured",
                             StructuredDemandModel(obs_dim, cfg.model.elasticity_lo,
                                                   cfg.model.elasticity_hi)),
                            ("unconstrained", UnconstrainedDemandModel(obs_dim))):
                fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
                for tsfx, Nmat in transitions:
                    Q, _ = plan_q(dm, mdp, N=Nmat)
                    base = f"Estimate-then-optimise {tag}{tsfx}"
                    arms.append((base, "EtO", planner_policy(Q, mdp), "none"))
                    arms.append((f"{base} support top{args.support_topk}", "EtO",
                                 planner_policy(Q, mdp, counts, args.support_topk),
                                 f"top{args.support_topk}"))
            print(f"  [(a,b) pairs observed in the log: {seen_frac:.1%}]", flush=True)

            for method, family, policy, mask in arms:
                _evaluate_method(raw_rows, gate="gate2E_family", cell_id=cell, n=n,
                                 noise=noise, seed=seed, method=method,
                                 family=family, policy_fn=policy, mdp=mdp,
                                 init_bins=init_bins, v_beh=v_beh, v_opt=v_opt,
                                 counts=counts, totals=totals,
                                 extra={"support_mask": mask})
                print(f"  {method}: nv={raw_rows[-1]['nv']:+.4f}", flush=True)

            pd.DataFrame(raw_rows).to_csv(
                os.path.join(args.outdir, "family_raw.csv"), index=False)

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(os.path.join(args.outdir, "family_raw.csv"), index=False)
    _summarise(raw).to_csv(os.path.join(args.outdir, "family_summary.csv"),
                           index=False)
    # four_family_raw.csv is the file `verify_claims.py` recomputes eleven of
    # the headline figures from, and until this pass nothing in the repository produced it:
    # the archived copy had been assembled by hand from these rows plus the arms computed by
    # the bandit/IQL/Q-DT runs. Writing it here makes the documented command sufficient.
    if args.merge_with and os.path.exists(args.merge_with):
        other = pd.read_csv(args.merge_with)
        overlap = set(zip(raw["cell_id"], raw["seed"], raw["method"])) &                   set(zip(other["cell_id"], other["seed"], other["method"]))
        if overlap:
            raise SystemExit(f"{len(overlap)} rows are in both this run and "
                             f"{args.merge_with}; the two are meant to be disjoint arms")
        pd.concat([raw, other], ignore_index=True).to_csv(
            os.path.join(args.outdir, "four_family_raw.csv"), index=False)
        print(f"wrote four_family_raw.csv  ({len(raw)} computed here + {len(other)} from "
              f"{args.merge_with})")
    elif args.merge_with:
        print(f"NOTE: {args.merge_with} absent; four_family_raw.csv not written")

    verdict = _check_anchors(raw_rows, args.published)
    json.dump({"cells": cells, "seeds": seeds, "support_topk": args.support_topk,
               "anchor_check": verdict,
               "purpose": "arms missing from results_bandit_20260821/combined_raw.csv",
               "protocol": "identical to diag_bandit_baseline / diag_gate2_pricing"},
              open(os.path.join(args.outdir, "protocol.json"), "w"), indent=2)

    print(f"\nANCHOR CHECK vs {args.published}: {verdict}")
    print(raw.groupby("method")["nv"].agg(["size", "mean", "median"]).round(4).to_string())


if __name__ == "__main__":
    main()
