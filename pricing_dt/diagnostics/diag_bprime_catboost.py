"""B'-CatBoost probe: strong tabular demand predictor, no economic structure.

This mirrors diag_bprime_xgb.py but swaps the boosted-tree backend for CatBoost.
The diagnostic asks whether a strong tabular forecaster can replace the structured
demand prior as a return-relabel signal in the hardest pricing cell.

Output: bprime_catboost.csv

Reference:
  - Prokhorenkova et al. (2018), CatBoost: unbiased boosting with categorical
    features.
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
from pricing_dt.core.relabel import relabel_dataset
from pricing_dt.diagnostics.diag_denoise_verdict import (
    _shrink_to_var,
    _smooth_by_start,
    _target_noise,
    _train_eval,
)
from pricing_dt.experiments.experiments import _seed, _setup


try:
    from catboost import CatBoostRegressor
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise RuntimeError(
        "CatBoost is required for diag_bprime_catboost.py. Install catboost or run "
        "diag_bprime_xgb.py for the XGBoost/sklearn fallback diagnostic."
    ) from exc


class CatBoostDemandModel:
    """CatBoost model with the torch demand-model call interface."""

    def __init__(self, reg, log_clamp_max=10.0):
        self.reg = reg
        self.log_clamp_max = log_clamp_max

    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, price, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        price_col = price.reshape(-1, 1)
        x = torch.cat([s, price_col], dim=-1).detach().cpu().numpy()
        log_pred = np.clip(self.reg.predict(x), a_min=None, a_max=self.log_clamp_max)
        dem = np.exp(log_pred)
        return torch.as_tensor(dem, dtype=price.dtype, device=price.device)


def _extract(trajs, mdp):
    states, prices, demand = [], [], []
    for tr in trajs:
        states.append(tr.obs)
        prices.append(mdp.prices[tr.actions])
        demand.append(tr.rewards / np.maximum(mdp.prices[tr.actions], 1e-6))
    return np.concatenate(states), np.concatenate(prices), np.concatenate(demand)


def fit_catboost(trajs, mdp, seed, iterations=400, depth=5, learning_rate=0.05):
    states, prices, demand = _extract(trajs, mdp)
    x = np.concatenate([states, prices[:, None]], axis=1)
    y = np.log(np.clip(demand, 1e-3, None))
    reg = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(iterations),
        depth=int(depth),
        learning_rate=float(learning_rate),
        l2_leaf_reg=3.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    reg.fit(x, y)
    return CatBoostDemandModel(reg)


def _heldout_log_mse(dm, ho_trajs, mdp):
    states, prices, demand = _extract(ho_trajs, mdp)
    log_demand = np.log(np.clip(demand, 1e-3, None))
    price_t = torch.tensor(prices, dtype=torch.float32)
    state_t = torch.tensor(states, dtype=torch.float32)
    dm.eval()
    with torch.no_grad():
        pred = dm(price_t, state_t).detach().cpu().numpy()
    log_pred = np.log(np.clip(pred, 1e-3, None))
    return float(np.mean((log_pred - log_demand) ** 2))


def _true_demand_rmse(dm, mdp):
    prices = torch.tensor(mdp.prices, dtype=torch.float32)
    se, n = 0.0, 0
    dm.eval()
    with torch.no_grad():
        for t in range(mdp.H):
            for b, ref in enumerate(mdp.ref_grid):
                states = torch.tensor(mdp.obs(ref, t), dtype=torch.float32).unsqueeze(0)
                states = states.repeat(len(mdp.prices), 1)
                pred = dm(prices, states).detach().cpu().numpy()
                true = mdp.expected_demand(mdp.prices, ref)
                se += float(np.sum((pred - true) ** 2))
                n += len(mdp.prices)
    return float(np.sqrt(se / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, n_actions = 2, cfg.sim.n_prices
    n_traj = min(cfg.exp.data_sizes)
    noise = max(cfg.exp.noise_levels)
    print(
        f"backend=catboost cell: N={n_traj} noise={noise} seeds={cfg.exp.seeds} "
        f"holdout={args.holdout} iterations={args.iterations}"
    )

    rows = []
    for seed in cfg.exp.seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, n_traj, noise, seed, cfg.data.expert_q)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(trajs))
        n_ho = max(2, int(round(args.holdout * len(trajs))))
        ho_trajs = [trajs[i] for i in idx[:n_ho]]
        tr_trajs = [trajs[i] for i in idx[n_ho:]]

        dm_a = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm_a, tr_trajs, mdp, cfg.model.demand_epochs)
        dm_b = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dm_b, tr_trajs, mdp, cfg.model.demand_epochs)
        dm_cat = fit_catboost(
            tr_trajs,
            mdp,
            seed,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
        )

        rtg_a = relabel_dataset(tr_trajs, dm_a, mdp, cfg.model.relabel_lambda)
        rtg_b = relabel_dataset(tr_trajs, dm_b, mdp, cfg.model.relabel_lambda)
        rtg_cat = relabel_dataset(tr_trajs, dm_cat, mdp, cfg.model.relabel_lambda)

        std_a = float(np.concatenate(rtg_a).std())
        std_cat = float(np.concatenate(rtg_cat).std())
        resid_a, disp_a = _target_noise(dm_a, mdp)
        resid_cat, disp_cat = _target_noise(dm_cat, mdp)

        nv_a = _train_eval(rtg_a, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, n_actions, seed)
        nv_b = _train_eval(rtg_b, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, n_actions, seed)
        nv_cat = _train_eval(rtg_cat, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, n_actions, seed)
        nv_cat_sm = _train_eval(
            _smooth_by_start(rtg_cat, tr_trajs),
            tr_trajs,
            mdp,
            init,
            v_beh,
            v_opt,
            cfg,
            obs_dim,
            n_actions,
            seed,
        )
        nv_cat_lv = _train_eval(
            _shrink_to_var(rtg_cat, std_a),
            tr_trajs,
            mdp,
            init,
            v_beh,
            v_opt,
            cfg,
            obs_dim,
            n_actions,
            seed,
        )

        rows.append(
            dict(
                seed=seed,
                N=n_traj,
                noise=noise,
                backend="catboost",
                ho_logmse_A=round(_heldout_log_mse(dm_a, ho_trajs, mdp), 4),
                ho_logmse_B=round(_heldout_log_mse(dm_b, ho_trajs, mdp), 4),
                ho_logmse_CatBoost=round(_heldout_log_mse(dm_cat, ho_trajs, mdp), 4),
                true_rmse_A=round(_true_demand_rmse(dm_a, mdp), 3),
                true_rmse_B=round(_true_demand_rmse(dm_b, mdp), 3),
                true_rmse_CatBoost=round(_true_demand_rmse(dm_cat, mdp), 3),
                A_structured=round(nv_a, 3),
                B_mlp=round(nv_b, 3),
                CatBoost=round(nv_cat, 3),
                CatBoost_smoothByStart=round(nv_cat_sm, 3),
                CatBoost_lowvar=round(nv_cat_lv, 3),
                tgt_std_A=round(std_a, 1),
                tgt_std_CatBoost=round(std_cat, 1),
                qsurf_resid_A=round(resid_a, 2),
                qsurf_resid_CatBoost=round(resid_cat, 2),
                qsurf_disp_A=round(disp_a, 2),
                qsurf_disp_CatBoost=round(disp_cat, 2),
            )
        )
        print(
            f"seed {seed}: value A={nv_a:.3f} B={nv_b:.3f} Cat={nv_cat:.3f} "
            f"Cat_sm={nv_cat_sm:.3f} Cat_lv={nv_cat_lv:.3f}"
        )

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "bprime_catboost.csv")
    df.to_csv(out, index=False)

    mean = df.mean(numeric_only=True)
    med, p = M.paired_test(df.A_structured.values, df.CatBoost.values)
    gap = mean.A_structured - mean.CatBoost
    closed = (
        (mean.CatBoost_smoothByStart - mean.CatBoost) / gap
        if abs(gap) > 1e-9
        else float("nan")
    )
    print(f"\nWrote {out}")
    print("\n=== B'-CatBoost summary ===")
    print(f"held-out log-MSE: A={mean.ho_logmse_A:.3f} B={mean.ho_logmse_B:.3f} CatBoost={mean.ho_logmse_CatBoost:.3f}")
    print(f"true-demand RMSE: A={mean.true_rmse_A:.3f} B={mean.true_rmse_B:.3f} CatBoost={mean.true_rmse_CatBoost:.3f}")
    print(f"policy value:     A={mean.A_structured:.3f} B={mean.B_mlp:.3f} CatBoost={mean.CatBoost:.3f}")
    print(f"CatBoost controls: smoothByStart={mean.CatBoost_smoothByStart:.3f} lowvar={mean.CatBoost_lowvar:.3f}")
    print(f"A - CatBoost policy gap = {gap:+.3f}, paired p={p:.4f}, smoothing closes {closed * 100:.0f}%")


if __name__ == "__main__":
    main()
