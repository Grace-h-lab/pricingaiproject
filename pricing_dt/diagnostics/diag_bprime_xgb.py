"""B'-XGBoost probe — "strong tabular predictor, no economic structure".

WHY THIS RUN EXISTS.
The E2-AB ablation already has B = UnconstrainedDemandModel (an MLP with no prior)
and shows B loses policy value to the structured model A, while the mechanism chain
(diag_optimism_verdict -> diag_action_shape -> diag_denoise_verdict) localised A's
edge to the SHAPE / low-variance smoothness of its optimistic target, NOT to raw
predictive correctness ("A tracks true value WORSE than B" yet wins).

A reasonable objection to using B as the "no-structure" control is that B is only
a weak MLP: maybe a genuinely STRONG tabular learner would both forecast logged
demand better AND, through that accuracy, relabel better. B' answers exactly that by
swapping the MLP for a gradient-boosted-tree regressor (XGBoost, or a scikit-learn
HistGradientBoosting fallback if XGBoost is not installed) on the SAME [state, price]
-> log-demand target, with NO monotonicity / elasticity constraints.

THE CONFOUND WE MUST CONTROL. Boosted trees produce a piecewise-constant, typically
HIGHER-variance, non-smooth achievable-Q surface. The project has already shown the
mechanism is (at least partly) target smoothness. So a naive "B' loses" result would
be un-interpretable: is B' worse because it lacks economic structure, or merely
because trees give a noisier conditioning signal? We therefore carry the denoise-
verdict controls: B'_smoothByStart and B'_lowvar apply the SAME variance/smoothness
corrections used on B in diag_denoise_verdict. The informative outcomes are:

  * B' forecasts logged demand BEST (lowest held-out log-MSE) yet its policy value is
    still below A  -> predictive accuracy is NOT sufficient for good return relabelling.
  * AND smoothing / variance-matching B' does NOT close the gap to A -> A's advantage
    is not merely the smoothness confound either; the structured prior contributes an
    action-ranking (shape) benefit beyond both accuracy and smoothness.
  * If instead smoothing B' recovers A, the read is that B''s deficit was
    target-noise, and accuracy-vs-structure was not cleanly separated by this probe.

Design (kept deliberately minimal — a DIAGNOSTIC, not a main experiment):
  * Hardest cell only: N=min(data_sizes), noise=max(noise_levels) — identical to
    e2ab_ablation and diag_denoise_verdict.
  * One trajectory-level 80/20 split per seed: A, B, B' are all fit on the SAME 80%
    train logs; the held-out 20% gives an honest logged-demand forecast metric. Policy
    value is trained/evaluated on that same 80% (offline: you relabel the logs you fit
    on). Absolute policy values therefore sit slightly below the full-N e2ab numbers;
    the A-vs-B-vs-B' ORDERING under identical conditions is the invariant we test.

Metrics per seed:
  demand fit :  ho_logdemand_mse (held-out logged log-demand MSE; the accuracy axis)
                true_demand_rmse (RMSE of E[demand] vs the simulator's true DGP on the
                                  full (ref_bin x t x price) grid — coverage-free)
  policy val :  normalised value of A / B / B' and of denoised B'
  target var :  tgt_std and Q-surface residual (shows trees give a noisier target)

Output: results/bprime_xgb.csv plus a printed summary table and verdict.

Reference:
  - Chen and Guestrin (2016), XGBoost: A Scalable Tree Boosting System.
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset
# reuse the denoise-verdict machinery so B' is judged on the SAME variance controls
from pricing_dt.diagnostics.diag_denoise_verdict import _train_eval, _smooth_by_start, _shrink_to_var, _target_noise

# --- strong tabular backend: XGBoost if available, else sklearn HistGBR (always present) ---
try:
    from xgboost import XGBRegressor
    _BACKEND = "xgboost"

    def _make_regressor(seed):
        return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                            random_state=seed, n_jobs=0, verbosity=0)
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _BACKEND = "sklearn_hist_gbr"

    def _make_regressor(seed):
        return HistGradientBoostingRegressor(max_iter=400, max_depth=4,
                                             learning_rate=0.05, l2_regularization=1.0,
                                             random_state=seed)


class XGBDemandModel:
    """Gradient-boosted-tree demand model (B'): a STRONG tabular predictor with NO
    economic structure. Exposes the SAME forward(price, s) -> E[demand] interface as
    the torch demand models, so it drops into relabel.achievable_rtg and
    diag_action_shape.demand_q unchanged. It is non-differentiable / non-torch and is
    fitted separately by fit_xgb (it does not go through fit_demand_model)."""

    def __init__(self, reg, log_clamp_max=10.0):
        self.reg = reg
        self.log_clamp_max = log_clamp_max

    # no-op mode switches so callers' dm.eval() / dm.train() work like an nn.Module
    def eval(self):
        return self

    def train(self):
        return self

    def __call__(self, price, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        price_col = price.reshape(-1, 1)
        X = torch.cat([s, price_col], dim=-1).detach().cpu().numpy()
        log_pred = np.clip(self.reg.predict(X), a_min=None, a_max=self.log_clamp_max)
        dem = np.exp(log_pred)
        return torch.as_tensor(dem, dtype=price.dtype, device=price.device)


def _extract(trajs, mdp):
    """Logged (state, price, demand) at the transition level. demand = reward / price,
    exactly as fit_demand_model does (so B' sees the identical target as A and B)."""
    S, P, Dm = [], [], []
    for tr in trajs:
        S.append(tr.obs)
        P.append(mdp.prices[tr.actions])
        Dm.append(tr.rewards / np.maximum(mdp.prices[tr.actions], 1e-6))
    return np.concatenate(S), np.concatenate(P), np.concatenate(Dm)


def fit_xgb(trajs, mdp, seed):
    """Fit the boosted-tree demand model on logged [state, price] -> log-demand."""
    S, P, Dm = _extract(trajs, mdp)
    logD = np.log(np.clip(Dm, 1e-3, None))
    X = np.concatenate([S, P[:, None]], axis=1)
    reg = _make_regressor(seed)
    reg.fit(X, logD)
    return XGBDemandModel(reg)


def _heldout_log_mse(dm, ho_trajs, mdp, device="cpu"):
    """Held-out logged-demand log-MSE — the predictive-accuracy axis. Works for any
    model exposing forward(price, s): the torch A/B and the XGB B' alike."""
    S, P, Dm = _extract(ho_trajs, mdp)
    logD = np.log(np.clip(Dm, 1e-3, None))
    price = torch.tensor(P, dtype=torch.float32, device=device)
    s = torch.tensor(S, dtype=torch.float32, device=device)
    dm.eval()
    with torch.no_grad():
        pred = dm(price, s).detach().cpu().numpy()
    log_pred = np.log(np.clip(pred, 1e-3, None))
    return float(np.mean((log_pred - logD) ** 2))


def _true_demand_rmse(dm, mdp, device="cpu"):
    """RMSE of predicted E[demand] against the simulator's TRUE expected demand over
    the full (ref_bin x t x price) grid — a coverage-free 'did you recover the DGP'
    fit metric, in the same spirit as the project's 'tracks true value' framing."""
    Bn, H, A = mdp.cfg.n_ref_bins, mdp.H, len(mdp.prices)
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    se, n = 0.0, 0
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(Bn):
                ref = mdp.ref_grid[b]
                s = torch.tensor(mdp.obs(ref, t), dtype=torch.float32,
                                 device=device).unsqueeze(0).repeat(A, 1)
                pred = dm(prices, s).detach().cpu().numpy()
                true = mdp.expected_demand(mdp.prices, ref)   # vectorised over prices
                se += float(np.sum((pred - true) ** 2)); n += A
    return float(np.sqrt(se / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--holdout", type=float, default=0.2)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    print(f"backend={_BACKEND}  delta={cfg.sim.delta}  cell: N={N} noise={noise}  "
          f"seeds={seeds}  holdout={args.holdout}\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

        # one trajectory-level 80/20 split: fit on train, forecast-eval on held-out
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(trajs))
        n_ho = max(2, int(round(args.holdout * len(trajs))))
        ho_trajs = [trajs[i] for i in idx[:n_ho]]
        tr_trajs = [trajs[i] for i in idx[n_ho:]]

        # --- fit the three demand models on the SAME train logs ---
        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, tr_trajs, mdp, cfg.model.demand_epochs)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, tr_trajs, mdp, cfg.model.demand_epochs)
        dmBp = fit_xgb(tr_trajs, mdp, seed)

        # --- demand FIT metrics (accuracy axis + true-DGP recovery) ---
        ho_A = _heldout_log_mse(dmA, ho_trajs, mdp)
        ho_B = _heldout_log_mse(dmB, ho_trajs, mdp)
        ho_Bp = _heldout_log_mse(dmBp, ho_trajs, mdp)
        tr_A = _true_demand_rmse(dmA, mdp)
        tr_B = _true_demand_rmse(dmB, mdp)
        tr_Bp = _true_demand_rmse(dmBp, mdp)

        # --- relabel the train logs and read off target variance ---
        rtgA = relabel_dataset(tr_trajs, dmA, mdp, cfg.model.relabel_lambda)
        rtgB = relabel_dataset(tr_trajs, dmB, mdp, cfg.model.relabel_lambda)
        rtgBp = relabel_dataset(tr_trajs, dmBp, mdp, cfg.model.relabel_lambda)
        stdA = float(np.concatenate(rtgA).std())
        stdBp = float(np.concatenate(rtgBp).std())
        residA, _ = _target_noise(dmA, mdp)
        residBp, _ = _target_noise(dmBp, mdp)

        # --- policy value: same relabel -> train_dt -> exact eval pipeline ---
        nvA = _train_eval(rtgA, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        nvB = _train_eval(rtgB, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        nvBp = _train_eval(rtgBp, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        # --- CONFOUND CONTROL: give B' A's smoothness / variance and re-test ---
        rtgBp_sm = _smooth_by_start(rtgBp, tr_trajs)
        nvBp_sm = _train_eval(rtgBp_sm, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        rtgBp_lv = _shrink_to_var(rtgBp, stdA)          # match A's target variance
        nvBp_lv = _train_eval(rtgBp_lv, tr_trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        rows.append(dict(
            seed=seed, N=N, noise=noise, backend=_BACKEND,
            # demand fit
            ho_logmse_A=round(ho_A, 4), ho_logmse_B=round(ho_B, 4), ho_logmse_Bp=round(ho_Bp, 4),
            true_rmse_A=round(tr_A, 3), true_rmse_B=round(tr_B, 3), true_rmse_Bp=round(tr_Bp, 3),
            # policy value
            A_structured=round(nvA, 3), B_mlp=round(nvB, 3), Bp_xgb=round(nvBp, 3),
            Bp_smoothByStart=round(nvBp_sm, 3), Bp_lowvar=round(nvBp_lv, 3),
            # target variance (the confound, made visible)
            tgt_std_A=round(stdA, 1), tgt_std_Bp=round(stdBp, 1),
            qsurf_resid_A=round(residA, 2), qsurf_resid_Bp=round(residBp, 2)))
        print(f"seed {seed}: fit(ho log-mse) A={ho_A:.3f} B={ho_B:.3f} B'={ho_Bp:.3f} | "
              f"value A={nvA:.3f} B={nvB:.3f} B'={nvBp:.3f} "
              f"(B'sm={nvBp_sm:.3f} B'lv={nvBp_lv:.3f}) | tgt_std A={stdA:.0f} B'={stdBp:.0f}")

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "bprime_xgb.csv"), index=False)

    m = df.mean(numeric_only=True)
    print(f"\n=== B' ({_BACKEND}) summary (mean over {len(df)} seeds) ===")
    print("                      A_struct   B_mlp    B'_xgb")
    print(f"  held-out log-MSE  {m.ho_logmse_A:8.3f} {m.ho_logmse_B:8.3f} {m.ho_logmse_Bp:8.3f}   (lower = better logged forecast)")
    print(f"  true-demand RMSE  {m.true_rmse_A:8.3f} {m.true_rmse_B:8.3f} {m.true_rmse_Bp:8.3f}   (lower = closer to true DGP)")
    print(f"  policy value      {m.A_structured:8.3f} {m.B_mlp:8.3f} {m.Bp_xgb:8.3f}   (higher = better)")
    print(f"  target std        {m.tgt_std_A:8.1f} {'      -':>8s} {m.tgt_std_Bp:8.1f}   (B' noisier target => the confound)")
    print(f"  Q-surf residual   {m.qsurf_resid_A:8.2f} {'      -':>8s} {m.qsurf_resid_Bp:8.2f}")
    print(f"\n  denoised B':  smoothByStart={m.Bp_smoothByStart:.3f}   lowvar(matchA)={m.Bp_lowvar:.3f}")

    gap = m.A_structured - m.Bp_xgb
    med, p = M.paired_test(df.A_structured.values, df.Bp_xgb.values)
    closed_sm = (m.Bp_smoothByStart - m.Bp_xgb) / gap if gap > 1e-9 else float("nan")
    best_forecast = "B'" if (m.ho_logmse_Bp <= m.ho_logmse_A and m.ho_logmse_Bp <= m.ho_logmse_B) else \
                    ("B" if m.ho_logmse_B <= m.ho_logmse_A else "A")
    print(f"\n  best logged forecaster: {best_forecast}   |   A - B' policy value = {gap:+.3f} (p={p:.4f})")
    print(f"  smoothing/variance-matching closes {closed_sm*100:4.0f}% of the A-B' gap")

    if gap > 0.02 and best_forecast in ("B", "B'") and closed_sm < 0.25:
        verdict = ("accuracy is NOT sufficient AND not a smoothness artefact: a strong "
                   "tabular forecaster relabels worse than A even after variance-matching "
                   "-> structured optimistic SHAPE is the operative signal.")
    elif gap > 0.02 and closed_sm >= 0.6:
        verdict = ("B''s deficit is largely TARGET-NOISE (denoising recovers most of it) "
                   "-> accuracy-vs-structure not cleanly separated; report as smoothness, "
                   "consistent with diag_denoise_verdict.")
    elif gap > 0.02:
        verdict = ("A still ahead of B'; denoising partially explains it -> shape + "
                   "smoothness both contribute.")
    else:
        verdict = ("no clear A-over-B' policy-value gap at this scale -> B' probe "
                   "inconclusive (do NOT over-claim).")
    print(f"\n  >>> VERDICT: {verdict}")


if __name__ == "__main__":
    main()
