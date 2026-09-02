"""C2 re-measured against a FAIR Q-DT baseline, across the data-size axis.

The published C2 headline is `adv_structured_minus_QDT = +0.391` over the E2 grid,
with the direction note that the advantage GROWS with N because "Q-DT/CQL degrades
with more data" — the opposite of the pre-registered guess. The channel-ladder
diagnostic shows why: the shipped Q-DT relabelled with an action-INDEPENDENT state
value, whose within-state target spread is exactly 0.000, so it carried no signal
distinguishing trajectories from the same state.

This script re-measures the advantage along the N axis at the hardest noise level,
which is the specific claim at risk: if the "grows with N" trend was an artefact of
the action-independent baseline, it should flatten or reverse once the comparator
depends on the logged action.

It replaces a full `run.py --exp e2` re-run, which is infeasible here: `e2_core`
evaluates each arm from every unique start bin separately AND over the full
256-bin initial distribution (~277 rollouts per arm per cell) across 9 cells. This
script uses the memoised evaluator (the rollout is deterministic per start bin, so
~21 rollouts suffice for an identical number) and holds noise at its hardest value,
trading the noise axis — which the published grid already shows to be the less
informative one — for a tractable, fair N axis.

Arms per (N, seed): vanilla DT, Q-DT legacy (action-independent), Q-DT td and q_sa
(action-dependent),
structured DT. All share one fitted demand model and one fitted Q-net per cell.
"""
import argparse
import numpy as np
import pandas as pd
import os

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset, logged_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_trust_region import eval_cached

ARMS = ["vanillaDT", "QDT_legacy", "QDT_td", "QDT_qsa", "structuredDT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--noise", type=float, default=None)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    noise = args.noise if args.noise is not None else max(cfg.exp.noise_levels)
    Ns = cfg.exp.data_sizes
    seeds = cfg.exp.seeds
    print(f"delta={cfg.sim.delta}  noise={noise}  N={Ns}  seeds={seeds}\n")

    rows = []
    for N in Ns:
        for seed in seeds:
            _seed(seed)
            mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
            dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)

            cols = {"vanillaDT": logged_rtg(trajs),
                    "structuredDT": relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)}
            for tag, mode in (("QDT_legacy", "state_value"), ("QDT_td", "td"),
                              ("QDT_qsa", "q_sa")):
                cols[tag], _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                             seed=seed, mode=mode, q=q)
            out = {"N": N, "seed": seed}
            for tag in ARMS:
                rtg = cols[tag]
                m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
                tgt = float(np.quantile(np.concatenate(rtg), 0.95))
                v, _, _, _, _ = eval_cached(make_dt_policy(m, mdp, tgt), mdp, init)
                out[f"nv_{tag}"] = round(M.normalised_value(v, v_beh, v_opt), 3)
            rows.append(out)
            print(f"N={N:5d} seed {seed}: " +
                  " ".join(f"{t}={out['nv_'+t]:+.3f}" for t in ARMS))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "c2_fixed_raw.csv"), index=False)

    print("\n=== mean normalised value by N ===")
    summ = []
    for N in Ns:
        cell = df[df.N == N]
        r = {"N": N}
        for t in ARMS:
            r[t] = round(cell[f"nv_{t}"].mean(), 3)
        med_l, p_l = M.paired_test(cell["nv_structuredDT"].values, cell["nv_QDT_legacy"].values)
        best_fixed = max(["QDT_td", "QDT_qsa"], key=lambda t: cell[f"nv_{t}"].mean())
        med_f, p_f = M.paired_test(cell["nv_structuredDT"].values, cell[f"nv_{best_fixed}"].values)
        r["adv_vs_legacy"] = round(cell["nv_structuredDT"].mean() - cell["nv_QDT_legacy"].mean(), 3)
        r["p_legacy"] = round(p_l, 4)
        r["adv_vs_fixed"] = round(cell["nv_structuredDT"].mean() - cell[f"nv_{best_fixed}"].mean(), 3)
        r["p_fixed"] = round(p_f, 4)
        r["best_fixed_arm"] = best_fixed
        summ.append(r)
    sdf = pd.DataFrame(summ)
    ps = np.nan_to_num(sdf["p_fixed"].values, nan=1.0)
    sdf["p_fixed_holm"] = M.holm(ps).round(4)
    sdf.to_csv(os.path.join(args.outdir, "c2_fixed_summary.csv"), index=False)
    print(sdf.to_string(index=False))

    print("\n=== the claim at risk: does the advantage still GROW with N? ===")
    print(f"  vs the BROKEN baseline : " +
          "  ".join(f"N={r['N']}: {r['adv_vs_legacy']:+.3f}" for r in summ))
    print(f"  vs the FIXED  baseline : " +
          "  ".join(f"N={r['N']}: {r['adv_vs_fixed']:+.3f}" for r in summ))
    trend_l = summ[-1]["adv_vs_legacy"] - summ[0]["adv_vs_legacy"]
    trend_f = summ[-1]["adv_vs_fixed"] - summ[0]["adv_vs_fixed"]
    print(f"  trend (largest N minus smallest N): action-independent {trend_l:+.3f}, "
          f"action-dependent {trend_f:+.3f}")
    if trend_l > 0.05 and trend_f <= 0.05:
        print("  >>> the 'advantage grows with N' direction note was an ARTEFACT of the "
              "action-independent baseline; it does not survive an action-dependent one.")
    elif trend_f > 0.05:
        print("  >>> the 'advantage grows with N' trend SURVIVES the fix.")
    else:
        print("  >>> neither baseline shows a clear N trend at this noise level.")


if __name__ == "__main__":
    main()
