"""Elasticity-band robustness, re-run fairly.

The published Table 4.5 is entirely a table of "C2 advantage over Q-DT" across the
Online Retail II elasticity band (median 1.37, IQR 0.95-1.83): advantage +0.00 / +0.23
/ +0.30 / +0.51 at beta = 0.95 / 1.37 / 1.83 / 2.00, read as "holds at and above the
median, attenuates at the inelastic corner". Every entry is measured against the
action-independent Q-DT and at each arm's own 0.95 quantile, so the whole table
inherits both confounds.

Re-run with the fixed Q-DT and held-out conditioning-target selection. The question is
no longer "how large is the advantage across the band" but "does any advantage exist
anywhere in the band once the comparator depends on the logged action".

Run:  python -m pricing_dt.diagnostics.diag_elasticity_fixed --outdir results
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
from pricing_dt.core.relabel import relabel_dataset
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_heldout_protocol import target_grid, per_bin_values, nv_on

BETAS = [0.95, 1.37, 1.83, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sel-frac", type=float, default=0.3)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    betas = BETAS if not args.smoke else [0.95, 2.0]
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  betas={betas}\n")

    rows = []
    for beta in betas:
        for seed in cfg.exp.seeds:
            _seed(seed)
            rng = np.random.default_rng(1000 + seed)
            mdp, init, _vo, _vb = _setup(cfg, {"demand_noise": noise, "beta": beta}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

            allb = np.arange(cfg.sim.n_ref_bins)
            perm = rng.permutation(allb)
            n_sel = max(2, int(round(args.sel_frac * len(allb))))
            sel_bins, test_bins = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
            myopic = lambda o: int(mdp.R[:, mdp.ref_to_bin(
                mdp.cfg.p_min + o[0] * (mdp.cfg.p_max - mdp.cfg.p_min))].argmax())
            v_beh_b = per_bin_values(myopic, mdp, allb)
            v_opt_b = {int(b): float(mdp.Vstar[0, b]) for b in allb}

            dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
            arms = {
                "structured": relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda),
                "QDT_fixed": value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                           seed=seed, mode="td", q=q)[0],
                "QDT_broken": value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                            seed=seed, mode="state_value", q=q)[0],
            }
            out = {"beta": beta, "seed": seed}
            for name, rtg in arms.items():
                m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
                sel, test = {}, {}
                for label, tgt in target_grid(rtg):
                    vals = per_bin_values(make_dt_policy(m, mdp, tgt), mdp, allb)
                    sel[label] = nv_on(vals, sel_bins, v_beh_b, v_opt_b)
                    test[label] = nv_on(vals, test_bins, v_beh_b, v_opt_b)
                out[name] = round(test[max(sel, key=sel.get)], 3)
            rows.append(out)
            print(f"beta={beta} seed {seed}: " +
                  "  ".join(f"{k}={out[k]:+.2f}" for k in arms))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "elasticity_fixed.csv"), index=False)

    print("\n=== elasticity band, held-out protocol "
          "(published advantage: +0.00 / +0.23 / +0.30 / +0.51) ===")
    summ, ps = [], []
    for beta in betas:
        cell = df[df.beta == beta]
        med_f, p_f = M.paired_test(cell["structured"].values, cell["QDT_fixed"].values)
        med_b, p_b = M.paired_test(cell["structured"].values, cell["QDT_broken"].values)
        ps.append(1.0 if p_f != p_f else p_f)
        summ.append(dict(beta=beta,
                         structured=round(cell["structured"].mean(), 3),
                         QDT_fixed=round(cell["QDT_fixed"].mean(), 3),
                         QDT_broken=round(cell["QDT_broken"].mean(), 3),
                         adv_vs_fixed=round(cell["structured"].mean() - cell["QDT_fixed"].mean(), 3),
                         p_fixed=round(p_f, 4),
                         adv_vs_broken=round(cell["structured"].mean() - cell["QDT_broken"].mean(), 3)))
    sdf = pd.DataFrame(summ)
    sdf["p_fixed_holm"] = M.holm(np.array(ps)).round(4)
    sdf.to_csv(os.path.join(args.outdir, "elasticity_fixed_summary.csv"), index=False)
    print(sdf.to_string(index=False))
    surv = sdf[sdf.p_fixed_holm < 0.05]
    print("\n=== verdict ===")
    if len(surv):
        print(f"  >>> advantage survives at beta = {list(surv.beta)}")
    else:
        print("  >>> no point in the elasticity band retains a significant advantage "
              "over an action-dependent Q-DT after Holm.")


if __name__ == "__main__":
    main()
