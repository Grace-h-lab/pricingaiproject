"""⑥ C3 absolute, second attempt: marginalised IS under estimated propensities.

The shrinkage probe (diag_c3_shrinkage.py) ruled out the *variance* diagnosis of the
data-driven C3 failure: sharing strength across segments does not recover the benefit,
because the bottleneck is per-step propensity-model mis-specification, not data-splitting
variance. This script tests the other candidate from Appendix F.3.4 — a *marginalised IS*
estimator that sidesteps per-step (and hence per-segment) propensities entirely by
reweighting pooled empirical rewards with state-action OCCUPANCY ratios
(ope.marginalised_is_value). It uses the known reference transition (a modelled,
controllable dynamic in pricing) to compute the target occupancy exactly, and estimates
the behaviour occupancy + reward model from the pooled log.

For each drift level it compares, all evaluated against the known true value:
  bias_pooled_estpi : pooled per-step DR with estimated pi_b (the deployable baseline)
  bias_mis          : marginalised IS (proposed deployable estimator)
and reports the target-occupancy coverage (how much target mass the log supports).
The oracle per-segment DR benefit (from e3_ope) is the upper bound this aims to match.

Outputs: results/c3_mis.csv, results/c3_mis_summary.csv
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: main().
#
# Marginalised occupancy-ratio estimation under non-stationary logging, as the second
# attempt at recovering the segmentation benefit with estimated propensities.
#
# Implements or follows:
#   - Uehara, M., Shi, C. and Kallus, N. (2026) 'A Review of Off-Policy Evaluation in
#     Reinforcement Learning', Statistical Science. arXiv:2212.06355.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import argparse
import os
import numpy as np
import pandas as pd

from pricing_dt.core import config as C
from pricing_dt.core import ope
from pricing_dt.diagnostics.diag_c3_shrinkage import _build_cell, _data_driven_biases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    noise = cfg.exp.noise_levels[len(cfg.exp.noise_levels) // 2]
    drifts = cfg.exp.drift_levels
    seeds = cfg.exp.seeds

    rows = []
    for drift in drifts:
        for seed in seeds:
            mdp, trajs, pe, v_true, v_beh, tv, init = _build_cell(
                cfg, drift, seed, noise, obs_dim, A)
            pol = lambda o: int(np.argmax(pe(o)))         # deterministic target
            # deployable per-step DR baseline (pooled, estimated pi_b)
            bp, _, _, _ = _data_driven_biases(trajs, pe, v_true, A, cfg.data.n_segments, 50.0)
            # marginalised IS (occupancy-ratio, estimated from data + known transition)
            v_mis, coverage = ope.marginalised_is_value(trajs, mdp, pol, init, A)
            bias_mis = v_mis - v_true
            rows.append(dict(drift=drift, seed=seed, logger_tv=round(tv, 3),
                             v_true=round(v_true, 2),
                             bias_pooled_estpi=round(bp, 2),
                             bias_mis=round(bias_mis, 2),
                             mis_coverage=round(coverage, 3)))
            print(f"drift={drift} seed={seed}: |bias| pooled={abs(bp):6.1f} "
                  f"MIS={abs(bias_mis):6.1f}  coverage={coverage:.2f}")

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "c3_mis.csv"), index=False)

    summ = df.groupby("drift").agg(
        logger_tv=("logger_tv", lambda x: round(x.mean(), 3)),
        mean_abs_bias_pooled=("bias_pooled_estpi", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_mis=("bias_mis", lambda x: round(np.abs(x).mean(), 2)),
        mean_coverage=("mis_coverage", lambda x: round(x.mean(), 3)),
    ).reset_index()
    summ["benefit_mis"] = (summ["mean_abs_bias_pooled"] - summ["mean_abs_bias_mis"]).round(2)
    summ.to_csv(os.path.join(args.outdir, "c3_mis_summary.csv"), index=False)

    print("\n=== C3 data-driven: marginalised IS vs pooled DR, by drift ===")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
