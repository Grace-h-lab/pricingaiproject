"""Denoising-verdict probe — the last link in the elimination chain.

Established so far (hardest cell N=100, noise=0.5, delta=3.0, 10 seeds):
  - A_structured beats magnitude-matched B -> NOT magnitude.
  - A tracks true value WORSE than B at every granularity (fit, start-rank,
    action-rank) -> NOT correctness.
The only un-tested hypothesis for A's net-value win is DENOISING / variance: the
bounded-monotone prior produces SMOOTHER, lower-variance relabel targets (it
cannot chase the multiplicative demand noise the way unconstrained B does), giving
the DT a cleaner return-conditioning signal.

Two tests:

(1) Variance of the relabelled targets. Measure the dispersion of each model's
    per-(state,action) achievable-Q surface that is NOT explained by the true
    Qstar — i.e. the residual "noise" the demand model injects into the target.
    Predict A << B (A is smoother / less noise-chasing).

(2) DECISIVE: build "B_denoised" by SMOOTHING B's relabelled RTG toward its
    structure-free trend and retrain+eval a DT. If denoised-B closes the gap to A,
    the mechanism IS the smoothness of the optimistic conditioning signal (not the
    economic structure per se). If it does NOT, even denoising fails to explain A
    and the structured prior contributes something beyond smoothness.

    We denoise two ways for robustness:
      B_smoothByStart : replace each step's RTG by the mean RTG over trajectories
                        sharing its (start bin) — removes per-trajectory noise.
      B_lowvar(rho)   : shrink each RTG toward the global mean by factor rho,
                        matching A's target-variance (a pure variance match).
"""
import argparse
import numpy as np

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed, _eval_dt
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset
from pricing_dt.core.dt import train_dt
from pricing_dt.diagnostics.diag_action_shape import demand_q


def _train_eval(rtg_list, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed):
    m = train_dt(D.pack_dt(trajs, rtg_list), obs_dim, A, cfg.model, seed=seed)
    v, _ = _eval_dt(m, mdp, init, rtg_list)
    return M.normalised_value(v, v_beh, v_opt)


def _target_noise(dm, mdp):
    """Mean over states of the within-state residual std of the demand-Q surface
    after regressing out Qstar (linear) — the part of the target NOT tracking truth.
    Also returns the raw across-action std averaged over states (dispersion)."""
    H, B = mdp.H, mdp.cfg.n_ref_bins
    resid, disp = [], []
    for t in range(H):
        for b in range(B):
            qhat = demand_q(dm, mdp, b, t)
            qstar = mdp.Qstar[t, b, :]
            if qhat.std() < 1e-9:
                continue
            disp.append(float(qhat.std()))
            if qstar.std() > 1e-9:
                # residual after best linear fit qhat ~ a*qstar+b
                a, c = np.polyfit(qstar, qhat, 1)
                resid.append(float((qhat - (a * qstar + c)).std()))
    return (float(np.mean(resid)) if resid else float("nan"),
            float(np.mean(disp)) if disp else float("nan"))


def _smooth_by_start(rtg_list, trajs):
    """Replace each trajectory's RTG with the per-step mean RTG over trajectories
    sharing its start bin (removes per-trajectory demand-noise from the target)."""
    groups = {}
    for tr, g in zip(trajs, rtg_list):
        groups.setdefault(int(tr.ref_bins[0]), []).append(g)
    means = {b: np.mean(np.stack(gs), axis=0) for b, gs in groups.items()}
    return [means[int(tr.ref_bins[0])].astype(np.float32) for tr in trajs]


def _shrink_to_var(rtg_list, target_std):
    """Shrink RTG toward its global mean so its overall std == target_std (variance match)."""
    flat = np.concatenate(rtg_list)
    mu, sd = float(flat.mean()), float(flat.std())
    if sd < 1e-9:
        return [g.copy() for g in rtg_list]
    rho = target_std / sd
    return [(mu + rho * (g - mu)).astype(np.float32) for g in rtg_list]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)

        rtgA = relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda)
        rtgB = relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda)

        # (1) target-noise / dispersion of each model's Q surface
        noiseA, dispA = _target_noise(dmA, mdp)
        noiseB, dispB = _target_noise(dmB, mdp)
        # per-trajectory target std (the conditioning-signal variance the DT sees)
        stdA = float(np.concatenate(rtgA).std())
        stdB = float(np.concatenate(rtgB).std())

        nvA = _train_eval(rtgA, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        nvB = _train_eval(rtgB, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        # (2) denoised B
        rtgB_sm = _smooth_by_start(rtgB, trajs)
        nvB_sm = _train_eval(rtgB_sm, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
        rtgB_lv = _shrink_to_var(rtgB, stdA)        # match A's target variance
        nvB_lv = _train_eval(rtgB_lv, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        rows.append(dict(seed=seed,
                         A_structured=round(nvA, 3), B_calibrated=round(nvB, 3),
                         B_smoothByStart=round(nvB_sm, 3), B_lowvar=round(nvB_lv, 3),
                         tgt_std_A=round(stdA, 1), tgt_std_B=round(stdB, 1),
                         qsurf_resid_A=round(noiseA, 2), qsurf_resid_B=round(noiseB, 2),
                         qsurf_disp_A=round(dispA, 2), qsurf_disp_B=round(dispB, 2)))
        print(f"seed {seed}: A={nvA:.3f} B={nvB:.3f} B_smooth={nvB_sm:.3f} B_lowvar={nvB_lv:.3f} | "
              f"tgt_std A={stdA:.0f} B={stdB:.0f} | qresid A={noiseA:.1f} B={noiseB:.1f}")

    import pandas as pd, os
    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "denoise_verdict.csv"), index=False)

    print("\n=== denoising summary (mean over seeds) ===")
    for col in ["A_structured", "B_calibrated", "B_smoothByStart", "B_lowvar"]:
        print(f"  {col:16s}: {df[col].mean():.3f}")
    print(f"\n  target std            A={df.tgt_std_A.mean():.0f}  B={df.tgt_std_B.mean():.0f}")
    print(f"  Q-surface residual    A={df.qsurf_resid_A.mean():.2f}  B={df.qsurf_resid_B.mean():.2f}  (lower=smoother)")
    print(f"  Q-surface dispersion  A={df.qsurf_disp_A.mean():.2f}  B={df.qsurf_disp_B.mean():.2f}")
    gapA = df.A_structured.mean() - df.B_calibrated.mean()
    closed_sm = (df.B_smoothByStart.mean() - df.B_calibrated.mean()) / gapA if gapA > 1e-9 else float("nan")
    closed_lv = (df.B_lowvar.mean() - df.B_calibrated.mean()) / gapA if gapA > 1e-9 else float("nan")
    medS, pS = M.paired_test(df.A_structured.values, df.B_smoothByStart.values)
    print(f"\n  A-B gap = {gapA:+.3f}")
    print(f"  smoothing closes {closed_sm*100:4.0f}% of the A-B gap   (A vs B_smooth: mean "
          f"{df.A_structured.mean()-df.B_smoothByStart.mean():+.3f}, p={pS:.4f})")
    print(f"  variance-match closes {closed_lv*100:4.0f}% of the A-B gap")
    if closed_sm > 0.6:
        verdict = "DENOISING explains it (smoothed B recovers A) -> mechanism = smooth optimistic signal"
    elif closed_sm > 0.25:
        verdict = "DENOISING is PARTIAL (smoothing helps but does not fully recover A)"
    else:
        verdict = "NOT denoising (smoothed B still far below A) -> structured prior adds beyond smoothness"
    print(f"\n  >>> VERDICT: {verdict}")


if __name__ == "__main__":
    main()
