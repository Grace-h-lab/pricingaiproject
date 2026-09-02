"""⑥ C3 absolute: recovering the segmentation benefit under ESTIMATED propensities.

The headline C3 result (experiments.e3_ope) shows that segmenting the doubly-robust
estimator reduces non-stationarity bias when the per-segment behaviour policies are the
simulator's TRUE loggers. The deployable test — estimate pi_b from the logged
(state, action) pairs and detect change-points from data — *destroys* that benefit
(Table 4.7): splitting the log starves each segment's logistic propensity fit, and the
added variance outweighs the bias reduction, so segmenting slightly INCREASES bias.

This script tests the fix flagged as future work #1: a partially-pooled (hierarchical /
empirical-Bayes) per-segment propensity that shrinks each segment's local estimate toward
the global pooled estimate with weight lam_s = n_s/(n_s+kappa). It borrows statistical
strength across segments, trading the variance that breaks naive segmentation for a small,
controlled pooling bias.

It reproduces the e3_ope data-driven pipeline exactly (same drift sweep, same memoryless
structured-DT target whose TRUE value is computed, same weak state-independent q_hat, same
penalised change-point detector) and adds a third estimator arm:

  bias_pooled_estpi            : one pooled propensity (mis-specified under drift)
  bias_segmented_estpi         : naive per-segment propensity (the failure)
  bias_segmented_shrunk_estpi  : SHRINKAGE per-segment propensity (the proposed fix)

Outputs:
  results/c3_shrinkage.csv          per (drift, seed) biases, all three arms
  results/c3_shrinkage_summary.csv  mean |bias| and benefits by drift
  results/c3_shrinkage_kappa.csv    kappa sweep at the highest drift (bias-variance curve)
"""
import argparse
import os
import numpy as np
import pandas as pd

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core import ope
from pricing_dt.experiments.experiments import _setup, _seed, _mean_pairwise_tv
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset
from pricing_dt.core.dt import train_dt, dt_action_probs


def _build_cell(cfg, drift, seed, noise, obs_dim, A):
    """Replicates the e3_ope per-cell construction: non-stationary log, a memoryless
    structured-DT target policy, its exact true value, and the data-driven OPE pieces."""
    _seed(seed)
    mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
    trajs, loggers = D.make_nonstationary(mdp, cfg.data.n_segments,
                                          cfg.data.traj_per_segment, noise,
                                          temp=0.5, seed=seed, spread=drift)
    tv = _mean_pairwise_tv(loggers, mdp)
    dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
    fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
    rtg = relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)
    dt = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
    target = float(np.quantile(np.concatenate(rtg), 0.95))
    _pcache = {}

    def pe_dist(o):
        k = (round(float(o[0]), 4), round(float(o[1]), 4))
        if k not in _pcache:
            _pcache[k] = dt_action_probs(dt, mdp, target, o)
        return _pcache[k]

    pol = lambda o: int(np.argmax(pe_dist(o)))
    v_true, _ = mdp.evaluate_policy_fn(pol, init)

    def pe(o):
        p = np.zeros(A); p[pol(o)] = 1.0; return p

    return mdp, trajs, pe, v_true, v_beh, tv, init


def _data_driven_biases(trajs, pe, v_true, A, n_segments, kappa):
    """Pooled, naive-segmented, and shrinkage-segmented DR bias under estimated pi_b
    and detected change-points (weak state-independent q_hat)."""
    qh = ope.fit_qhat(trajs, A, state_dependent=False)
    pb_pool = ope.estimate_behaviour_policy(trajs, A)
    v_pool = ope.dr_value(trajs, pe, pb_pool, qh, A)

    seg_ids = ope.detect_segments(trajs, max_segments=n_segments + 2)
    naive_vals, shrunk_vals, ws = [], [], []
    for s in np.unique(seg_ids):
        sub = [tr for tr, sid in zip(trajs, seg_ids) if sid == s]
        if len(sub) < 3:
            continue
        pb_naive = ope.estimate_behaviour_policy(sub, A)
        pb_shrunk = ope.estimate_behaviour_policy_shrunk(sub, pb_pool, A, kappa=kappa)
        naive_vals.append(ope.dr_value(sub, pe, pb_naive, qh, A))
        shrunk_vals.append(ope.dr_value(sub, pe, pb_shrunk, qh, A))
        ws.append(len(sub))
    ws = np.array(ws, float); ws /= ws.sum()
    v_naive = float(np.dot(ws, naive_vals))
    v_shrunk = float(np.dot(ws, shrunk_vals))
    return (v_pool - v_true, v_naive - v_true, v_shrunk - v_true,
            int(len(np.unique(seg_ids))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--kappa", type=float, default=50.0)
    args = ap.parse_args()
    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    noise = cfg.exp.noise_levels[len(cfg.exp.noise_levels) // 2]
    drifts = cfg.exp.drift_levels
    seeds = cfg.exp.seeds

    rows = []
    cells = {}  # cache built cells for reuse in the kappa sweep
    for drift in drifts:
        for seed in seeds:
            mdp, trajs, pe, v_true, v_beh, tv, _init = _build_cell(cfg, drift, seed, noise, obs_dim, A)
            cells[(drift, seed)] = (trajs, pe, v_true)
            bp, bn, bs, ndet = _data_driven_biases(trajs, pe, v_true, A,
                                                   cfg.data.n_segments, args.kappa)
            rows.append(dict(drift=drift, seed=seed, logger_tv=round(tv, 3),
                             v_true=round(v_true, 2),
                             bias_pooled_estpi=round(bp, 2),
                             bias_segmented_estpi=round(bn, 2),
                             bias_segmented_shrunk_estpi=round(bs, 2),
                             n_detected_segments=ndet))
            print(f"drift={drift} seed={seed}: |bias| pooled={abs(bp):.1f} "
                  f"naive-seg={abs(bn):.1f} shrunk-seg={abs(bs):.1f}  (segs={ndet})")

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "c3_shrinkage.csv"), index=False)

    summ = df.groupby("drift").agg(
        logger_tv=("logger_tv", lambda x: round(x.mean(), 3)),
        mean_abs_bias_pooled=("bias_pooled_estpi", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_naive_seg=("bias_segmented_estpi", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_shrunk_seg=("bias_segmented_shrunk_estpi", lambda x: round(np.abs(x).mean(), 2)),
        n_detected_segments=("n_detected_segments", lambda x: round(x.mean(), 1)),
    ).reset_index()
    summ["benefit_naive"] = (summ["mean_abs_bias_pooled"] - summ["mean_abs_bias_naive_seg"]).round(2)
    summ["benefit_shrunk"] = (summ["mean_abs_bias_pooled"] - summ["mean_abs_bias_shrunk_seg"]).round(2)
    summ.to_csv(os.path.join(args.outdir, "c3_shrinkage_summary.csv"), index=False)

    # ---- kappa sweep at the highest drift: the bias-variance tradeoff ----
    hi = max(drifts)
    ksweep = [0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 1e9]  # 0=naive, inf=pooled
    krows = []
    for kappa in ksweep:
        biases = []
        for seed in seeds:
            trajs, pe, v_true = cells[(hi, seed)]
            _, _, bs, _ = _data_driven_biases(trajs, pe, v_true, A, cfg.data.n_segments, kappa)
            biases.append(abs(bs))
        krows.append(dict(kappa=kappa, drift=hi,
                          mean_abs_bias_shrunk_seg=round(float(np.mean(biases)), 2),
                          se=round(float(np.std(biases) / np.sqrt(len(biases))), 2)))
    kdf = pd.DataFrame(krows)
    kdf.to_csv(os.path.join(args.outdir, "c3_shrinkage_kappa.csv"), index=False)

    print("\n=== C3 data-driven: segmentation benefit by drift (kappa={}) ===".format(args.kappa))
    print(summ[["drift", "logger_tv", "mean_abs_bias_pooled", "mean_abs_bias_naive_seg",
                "mean_abs_bias_shrunk_seg", "benefit_naive", "benefit_shrunk"]].to_string(index=False))
    print("\n=== kappa sweep at drift={} (bias-variance) ===".format(hi))
    print(kdf.to_string(index=False))


if __name__ == "__main__":
    main()
