"""Profit-objective sensitivity for the controlled pricing testbed.

The main pricing MDP uses revenue = price * demand. This diagnostic checks the
commercially natural profit objective

    profit = (price - unit_cost) * demand

without changing the main experiment pipeline. It keeps exact dynamic programming
and compares two placements of the same fitted demand model:

  - Action channel / estimate-then-optimize:
      demand model -> expected profit -> DP argmax price policy.
  - Goal channel:
      demand model -> profit return-to-go relabel -> Decision Transformer.

Demand fitting is profit-aware. Because profit logs satisfy
profit = (price - unit_cost) * demand, the diagnostic converts logged profit back
to demand before fitting the demand model; it does not divide profit by price.

Output: profit_sensitivity.csv
"""
import argparse
import copy
import os

import numpy as np
import pandas as pd
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.demand_model import (
    StructuredDemandModel,
    UnconstrainedDemandModel,
    fit_demand_model,
)
from pricing_dt.core.dt import make_dt_policy, train_dt
from pricing_dt.core.relabel import logged_rtg
from pricing_dt.core.simulator import PricingMDP
from pricing_dt.experiments.experiments import _seed


class ProfitPricingMDP(PricingMDP):
    """Pricing MDP with profit, not revenue, as the reward."""

    def __init__(self, cfg, unit_cost):
        self.unit_cost = float(unit_cost)
        super().__init__(cfg)

    def expected_reward(self, price, ref):
        return (price - self.unit_cost) * self.expected_demand(price, ref)

    def sample_reward(self, price, ref, noise):
        q = self.expected_demand(price, ref)
        if noise > 0:
            q = q * np.exp(self.rng.normal(0.0, noise))
        return (price - self.unit_cost) * q


def _setup_profit(cfg, unit_cost, noise, seed):
    sc = copy.deepcopy(cfg.sim)
    sc.demand_noise = noise
    sc.seed = seed
    if unit_cost >= sc.p_min:
        raise ValueError(
            f"unit_cost={unit_cost} must be below p_min={sc.p_min}; otherwise "
            "the lowest price has non-positive margin and demand recovery is unstable."
        )
    mdp = ProfitPricingMDP(sc, unit_cost)
    mdp.solve_optimal()
    init = mdp.initial_bins(cfg.data.n_eval_episodes)
    v_opt = float(mdp.Vstar[0, init].mean())
    myopic = lambda o: int(mdp.R[:, mdp.decode_obs(o)[0]].argmax())
    v_beh, _ = mdp.evaluate_policy_fn(myopic, init)
    return mdp, init, v_opt, v_beh


def _revenue_view_trajs(trajs, mdp):
    """Convert profit-reward trajectories to revenue-reward trajectories.

    fit_demand_model expects demand = reward / price. Under a profit objective the
    logged reward is margin * demand, so we first recover demand from margin and
    then feed price * demand into the existing demand-fitting code.
    """
    out = []
    for tr in trajs:
        prices = mdp.prices[tr.actions]
        margins = np.maximum(prices - mdp.unit_cost, 1e-6)
        demand = tr.rewards / margins
        revenue = demand * prices
        out.append(D.Trajectory(tr.obs, tr.ref_bins, tr.actions, revenue.astype(np.float32), tr.seg))
    return out


def _fit_demand_models_profit(trajs, mdp, cfg, obs_dim):
    rev_trajs = _revenue_view_trajs(trajs, mdp)
    dm_a = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
    fit_demand_model(dm_a, rev_trajs, mdp, cfg.model.demand_epochs)
    dm_b = UnconstrainedDemandModel(obs_dim)
    fit_demand_model(dm_b, rev_trajs, mdp, cfg.model.demand_epochs)
    return dm_a, dm_b


def _profit_relabel_one(tr, demand_model, mdp, lam=1.0, device="cpu"):
    H = mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    margins = torch.tensor(mdp.prices - mdp.unit_cost, dtype=torch.float32, device=device)
    logged = tr.rtg
    relabelled = np.zeros(H, np.float32)

    demand_model.eval()
    with torch.no_grad():
        for t in range(H):
            ref = mdp.ref_grid[tr.ref_bins[t]]

            a_t = int(tr.actions[t])
            s_t = torch.tensor(mdp.obs(ref, t), dtype=torch.float32, device=device).unsqueeze(0)
            dem_t = demand_model(prices[a_t].unsqueeze(0), s_t)
            total = float(margins[a_t] * dem_t.squeeze())
            ref = mdp.ref_grid[mdp.N[a_t, mdp.ref_to_bin(ref)]]

            for k in range(t + 1, H):
                s = torch.tensor(mdp.obs(ref, k), dtype=torch.float32, device=device)
                s_rep = s.unsqueeze(0).repeat(len(prices), 1)
                dem = demand_model(prices, s_rep)
                profit = margins * dem
                a_best = int(torch.argmax(profit).item())
                total += float(profit[a_best].item())
                ref = mdp.ref_grid[mdp.N[a_best, mdp.ref_to_bin(ref)]]

            relabelled[t] = total
    return (1 - lam) * logged + lam * relabelled


def _profit_relabel_dataset(trajs, demand_model, mdp, lam=1.0):
    return [_profit_relabel_one(tr, demand_model, mdp, lam=lam) for tr in trajs]


def _plan_with_profit_dm(dm, mdp, device="cpu"):
    """Backward induction under fitted expected profit."""
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    margins = mdp.prices - mdp.unit_cost
    Rhat = np.zeros((H, B, A))
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(B):
                s = torch.tensor(mdp.obs(mdp.ref_grid[b], t), dtype=torch.float32, device=device)
                s_rep = s.unsqueeze(0).repeat(A, 1)
                dem = dm(prices, s_rep).cpu().numpy()
                Rhat[t, b, :] = margins * dem

    V = np.zeros((H + 1, B))
    pi = np.zeros((H, B), dtype=int)
    for t in reversed(range(H)):
        for b in range(B):
            q = Rhat[t, b, :] + V[t + 1, mdp.N[:, b]]
            pi[t, b] = int(q.argmax())
            V[t, b] = q.max()
    return pi, V


def _tabular_policy_fn(pi, mdp):
    def fn(obs):
        b, t = mdp.decode_obs(obs)
        return int(pi[t, b])
    return fn


def _eval_dt(model, mdp, init, train_rtg):
    target = float(np.quantile(np.concatenate(train_rtg), 0.95))
    pol = make_dt_policy(model, mdp, target)
    v, _ = mdp.evaluate_policy_fn(pol, init)
    return v, target


def _parse_costs(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--costs", default="0.0,0.2,0.4",
                    help="Comma-separated unit costs. Must be below SimConfig.p_min.")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, n_actions = 2, cfg.sim.n_prices
    n_traj = min(cfg.exp.data_sizes)
    noise = max(cfg.exp.noise_levels)
    costs = _parse_costs(args.costs)
    rows = []

    print(f"profit sensitivity: N={n_traj} noise={noise} costs={costs} seeds={cfg.exp.seeds}")
    for unit_cost in costs:
        for seed in cfg.exp.seeds:
            _seed(seed)
            mdp, init, v_opt, v_beh = _setup_profit(cfg, unit_cost, noise, seed)
            trajs = D.make_stitching_necessary(mdp, n_traj, noise, seed, cfg.data.expert_q)

            dm_a, dm_b = _fit_demand_models_profit(trajs, mdp, cfg, obs_dim)
            rtg_vanilla = logged_rtg(trajs)
            rtg_struct = _profit_relabel_dataset(trajs, dm_a, mdp, cfg.model.relabel_lambda)

            dt_v = train_dt(D.pack_dt(trajs, rtg_vanilla), obs_dim, n_actions, cfg.model, seed=seed)
            v_dt, _ = _eval_dt(dt_v, mdp, init, rtg_vanilla)
            dt_s = train_dt(D.pack_dt(trajs, rtg_struct), obs_dim, n_actions, cfg.model, seed=seed)
            v_sdt, _ = _eval_dt(dt_s, mdp, init, rtg_struct)

            row = dict(
                unit_cost=unit_cost,
                seed=seed,
                v_opt_profit=round(v_opt, 3),
                v_beh_profit=round(v_beh, 3),
                vanillaDT=round(M.normalised_value(v_dt, v_beh, v_opt), 3),
                structuredDT_profit=round(M.normalised_value(v_sdt, v_beh, v_opt), 3),
            )
            for tag, dm in (("structured", dm_a), ("unconstrained", dm_b)):
                pi, Vhat = _plan_with_profit_dm(dm, mdp)
                v_true, _ = mdp.evaluate_policy_fn(_tabular_policy_fn(pi, mdp), init)
                row[f"EtO_{tag}"] = round(M.normalised_value(v_true, v_beh, v_opt), 3)
                row[f"inmodel_{tag}"] = round(float(Vhat[0, init].mean()), 3)
                row[f"truegap_{tag}"] = round(float(Vhat[0, init].mean()) - v_true, 3)
            rows.append(row)
            print(
                f"cost {unit_cost:.2f} seed {seed}: "
                f"DT={row['vanillaDT']:.3f} structuredDT={row['structuredDT_profit']:.3f} "
                f"EtO_struct={row['EtO_structured']:.3f} EtO_uncon={row['EtO_unconstrained']:.3f}"
            )

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "profit_sensitivity.csv")
    df.to_csv(out, index=False)

    print(f"\nWrote {out}")
    summary = df.groupby("unit_cost").mean(numeric_only=True).reset_index()
    print("\n=== profit sensitivity summary ===")
    for _, r in summary.iterrows():
        print(
            f"cost {r.unit_cost:.2f}: vanillaDT={r.vanillaDT:.3f} "
            f"structuredDT={r.structuredDT_profit:.3f} "
            f"EtO_struct={r.EtO_structured:.3f} EtO_uncon={r.EtO_unconstrained:.3f} "
            f"truegap_struct={r.truegap_structured:+.1f}"
        )


if __name__ == "__main__":
    main()
