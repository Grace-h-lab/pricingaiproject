"""Commercial pricing constraints diagnostic.

The main channel study already contains two forms of constraints: a discrete
price grid and logged-support confinement through imitation/trust regions. This
optional diagnostic makes the commercial "constraints" layer explicit for the
estimate-then-optimise pipeline:

  - margin floor: do not offer prices below cost plus minimum margin;
  - price-change limit: do not move too far from the current reference price;
  - logged support: restrict to the top-k logged actions for the state.

The diagnostic keeps exact dynamic programming. It asks whether business-style
constraints make a direct demand-model optimiser safer, and whether they change
the interpretation of the channel warning.

Output:
  - pricing_constraints.csv
  - pricing_constraints_summary.csv
  - pricing_constraints_tests.csv
"""
import argparse
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
from pricing_dt.experiments.experiments import _seed, _setup


def _support_counts(trajs, mdp):
    counts = np.zeros((mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=float)
    for tr in trajs:
        for t, a in enumerate(tr.actions):
            counts[t, int(tr.ref_bins[t]), int(a)] += 1.0
    return counts


def _topk_support_mask(counts, t, b, k, n_actions):
    if k is None or k <= 0:
        return np.ones(n_actions, dtype=bool)
    mask = np.zeros(n_actions, dtype=bool)
    top = np.argsort(counts[t, b])[-int(k):]
    mask[top] = counts[t, b, top] > 0
    if not mask.any():
        mask[int(np.argmax(counts[t, b]))] = True
    return mask


def _constraint_mask(mdp, counts, t, b, mode, unit_cost, min_margin,
                     max_ref_move, support_topk):
    prices = mdp.prices
    ref = mdp.ref_grid[b]
    masks = []
    if mode in {"margin", "all"}:
        masks.append(prices >= unit_cost + min_margin)
    if mode in {"price_change", "all"}:
        masks.append(np.abs(prices - ref) <= max_ref_move)
    if mode in {"support", "all"}:
        masks.append(_topk_support_mask(counts, t, b, support_topk, mdp.cfg.n_prices))
    if not masks:
        mask = np.ones(mdp.cfg.n_prices, dtype=bool)
    else:
        mask = np.logical_and.reduce(masks)

    if mask.any():
        return mask

    # Conservative fallback: preserve margin if possible, then choose the closest
    # feasible price to the reference. This keeps the DP well-defined and makes
    # empty support cells visible through support metrics rather than crashes.
    fallback = prices >= unit_cost + min_margin
    if not fallback.any():
        fallback = np.ones_like(prices, dtype=bool)
    idx = np.where(fallback)[0]
    chosen = int(idx[np.argmin(np.abs(prices[idx] - ref))])
    mask = np.zeros(mdp.cfg.n_prices, dtype=bool)
    mask[chosen] = True
    return mask


def _reward_surface(dm, mdp, device="cpu"):
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    Rhat = np.zeros((H, B, A), dtype=float)
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(B):
                obs = torch.tensor(
                    mdp.obs(mdp.ref_grid[b], t),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0).repeat(A, 1)
                dem = dm(prices, obs).detach().cpu().numpy()
                Rhat[t, b, :] = mdp.prices * dem
    return Rhat


def _true_reward_surface(mdp):
    return np.repeat(mdp.R.T[None, :, :], mdp.H, axis=0)


def _plan_constrained(Rhat, mdp, counts, mode, unit_cost, min_margin,
                      max_ref_move, support_topk):
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    V = np.zeros((H + 1, B), dtype=float)
    pi = np.zeros((H, B), dtype=int)
    feasible_counts = []
    for t in reversed(range(H)):
        for b in range(B):
            mask = _constraint_mask(
                mdp,
                counts,
                t,
                b,
                mode,
                unit_cost,
                min_margin,
                max_ref_move,
                support_topk,
            )
            feasible_counts.append(int(mask.sum()))
            q = np.full(A, -np.inf, dtype=float)
            q[mask] = Rhat[t, b, mask] + V[t + 1, mdp.N[mask, b]]
            pi[t, b] = int(np.argmax(q))
            V[t, b] = float(np.max(q))
    return pi, V, float(np.mean(feasible_counts))


def _tabular_policy_fn(pi, mdp):
    def fn(obs):
        b, t = mdp.decode_obs(obs)
        return int(pi[t, b])
    return fn


def _policy_support_metrics(pi, mdp, init_bins, counts):
    unseen, logged_counts, price_moves, margin_ok = [], [], [], []
    totals = counts.sum(axis=2)
    for b0 in init_bins:
        ref = mdp.ref_grid[int(b0)]
        for t in range(mdp.H):
            b = mdp.ref_to_bin(ref)
            a = int(pi[t, b])
            c = float(counts[t, b, a])
            unseen.append(c <= 0.0)
            logged_counts.append(c)
            price_moves.append(abs(float(mdp.prices[a] - ref)))
            margin_ok.append(float(mdp.prices[a]))
            ref = mdp.ref_grid[mdp.N[a, b]]
    return {
        "selected_unseen_rate": float(np.mean(unseen)),
        "mean_logged_count": float(np.mean(logged_counts)),
        "mean_abs_price_ref_move": float(np.mean(price_moves)),
        "visited_state_support": float(np.mean(np.asarray(logged_counts) > 0.0)),
        "mean_state_total_count": float(
            np.mean([totals[t, mdp.ref_to_bin(mdp.ref_grid[int(b0)])]
                     for b0 in init_bins for t in range(mdp.H)])
        ),
    }


def _parse_modes(text):
    allowed = {"none", "margin", "price_change", "support", "all"}
    modes = []
    for raw in text.split(","):
        mode = raw.strip().lower().replace("-", "_")
        if not mode:
            continue
        if mode not in allowed:
            raise ValueError(f"Unknown constraint mode '{raw}'. Expected {sorted(allowed)}.")
        modes.append(mode)
    return modes or ["none", "margin", "price_change", "support", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--modes", default="none,margin,price_change,support,all")
    ap.add_argument("--unit-cost", type=float, default=0.35)
    ap.add_argument("--min-margin", type=float, default=0.30)
    ap.add_argument("--max-ref-move", type=float, default=0.45)
    ap.add_argument("--support-topk", type=int, default=3)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim = 2
    n_traj = min(cfg.exp.data_sizes)
    noise = max(cfg.exp.noise_levels)
    modes = _parse_modes(args.modes)
    rows = []
    print(
        "pricing constraints diagnostic: "
        f"N={n_traj} noise={noise} modes={modes} seeds={cfg.exp.seeds}"
    )

    for seed in cfg.exp.seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, n_traj, noise, seed, cfg.data.expert_q)
        counts = _support_counts(trajs, mdp)

        dm_struct = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm_struct, trajs, mdp, cfg.model.demand_epochs)
        dm_uncon = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dm_uncon, trajs, mdp, cfg.model.demand_epochs)

        surfaces = {
            "oracle": _true_reward_surface(mdp),
            "structured": _reward_surface(dm_struct, mdp),
            "unconstrained": _reward_surface(dm_uncon, mdp),
        }
        for model_name, Rhat in surfaces.items():
            for mode in modes:
                pi, Vhat, mean_feasible = _plan_constrained(
                    Rhat,
                    mdp,
                    counts,
                    mode,
                    args.unit_cost,
                    args.min_margin,
                    args.max_ref_move,
                    args.support_topk,
                )
                v_true, _ = mdp.evaluate_policy_fn(_tabular_policy_fn(pi, mdp), init)
                v_inmodel = float(Vhat[0, init].mean())
                support = _policy_support_metrics(pi, mdp, init, counts)
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "constraint": mode,
                        "v_opt": round(v_opt, 3),
                        "v_behaviour": round(v_beh, 3),
                        "nv": round(M.normalised_value(v_true, v_beh, v_opt), 3),
                        "v_true": round(v_true, 3),
                        "v_inmodel": round(v_inmodel, 3),
                        "truegap": round(v_inmodel - v_true, 3),
                        "mean_feasible_actions": round(mean_feasible, 3),
                        **{k: round(v, 4) for k, v in support.items()},
                    }
                )
        current = [r for r in rows if r["seed"] == seed and r["model"] == "structured"]
        msg = " ".join(f"{r['constraint']}={r['nv']:+.2f}" for r in current)
        print(f"seed {seed} structured EtO nv: {msg}")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    raw_path = os.path.join(args.outdir, "pricing_constraints.csv")
    df.to_csv(raw_path, index=False)

    summary = (
        df.groupby(["model", "constraint"], dropna=False)
        .agg(
            nv=("nv", "mean"),
            nv_sd=("nv", "std"),
            v_true=("v_true", "mean"),
            v_inmodel=("v_inmodel", "mean"),
            truegap=("truegap", "mean"),
            mean_feasible_actions=("mean_feasible_actions", "mean"),
            selected_unseen_rate=("selected_unseen_rate", "mean"),
            mean_abs_price_ref_move=("mean_abs_price_ref_move", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary_path = os.path.join(args.outdir, "pricing_constraints_summary.csv")
    summary.to_csv(summary_path, index=False)

    tests = []
    for model in ["structured", "unconstrained", "oracle"]:
        base = df[(df.model == model) & (df.constraint == "none")].sort_values("seed")
        for mode in modes:
            if mode == "none":
                continue
            comp = df[(df.model == model) & (df.constraint == mode)].sort_values("seed")
            if len(base) == len(comp) and len(base) > 0:
                med, p = M.paired_test(comp["nv"], base["nv"])
                tests.append(
                    {
                        "model": model,
                        "comparison": f"{mode} - none",
                        "median_diff": round(med, 3),
                        "p_value": round(p, 6) if np.isfinite(p) else np.nan,
                    }
                )
    tests_df = pd.DataFrame(tests)
    tests_path = os.path.join(args.outdir, "pricing_constraints_tests.csv")
    tests_df.to_csv(tests_path, index=False)

    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tests_path}")
    print("\n=== pricing constraints summary ===")
    for _, r in summary.sort_values(["model", "constraint"]).iterrows():
        print(
            f"{r.model:13s} {r.constraint:12s} "
            f"nv={r.nv:+.3f} gap={r.truegap:+.1f} "
            f"feasible={r.mean_feasible_actions:.1f} unseen={r.selected_unseen_rate:.3f}"
        )


if __name__ == "__main__":
    main()
