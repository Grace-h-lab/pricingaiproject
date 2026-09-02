"""E2-AB prior-isolation ablation, re-run fairly.

The published ablation is A_full_prior ~ C_misspecified (~0.67) >> D_bootstrapped
(~0.44) > B_no_constraints (~0.33), read as "a structured relabel beats the
bootstrapped value, and an unconstrained one does not". Arm **D is the action-independent Q-DT
read-out** (within-state target spread exactly 0.000), and all four arms were
evaluated at their own 0.95 quantile — the two confounds established in
`audit/CHANNEL_RESULTS.md` §1 and §4.2.

Re-run with D split into its action-independent and action-dependent forms, under
held-out conditioning-target
selection (bins split 30/70 per seed; ask chosen on selection, test read once).

Arms, matching `experiments.e2ab_ablation`:
  A_full_prior      structured prior, correct elasticity bounds
  B_no_constraints  unconstrained MLP demand model
  C_misspecified    structured prior with bounds shifted to [e_hi, 2*e_hi]
  D_bootstrap_fixed r_t + V(s_{t+1})            <- action-dependent
  D_bootstrap_broken V(s_t)                      <- action-independent; what the
                                                   published table used
"""
import argparse
import numpy as np
import pandas as pd
import os

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_heldout_protocol import target_grid, per_bin_values, nv_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sel-frac", type=float, default=0.3)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={cfg.exp.seeds}\n")

    rows = []
    for seed in cfg.exp.seeds:
        _seed(seed)
        rng = np.random.default_rng(1000 + seed)
        mdp, init, _vo, _vb = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

        allb = np.arange(cfg.sim.n_ref_bins)
        perm = rng.permutation(allb)
        n_sel = max(2, int(round(args.sel_frac * len(allb))))
        sel_bins, test_bins = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])
        myopic = lambda o: int(mdp.R[:, mdp.ref_to_bin(
            mdp.cfg.p_min + o[0] * (mdp.cfg.p_max - mdp.cfg.p_min))].argmax())
        v_beh_b = per_bin_values(myopic, mdp, allb)
        v_opt_b = {int(b): float(mdp.Vstar[0, b]) for b in allb}

        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)
        dmC = StructuredDemandModel(obs_dim, cfg.model.elasticity_hi, cfg.model.elasticity_hi * 2)
        fit_demand_model(dmC, trajs, mdp, cfg.model.demand_epochs)
        q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)

        arms = {
            "A_full_prior": relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda),
            "B_no_constraints": relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda),
            "C_misspecified": relabel_dataset(trajs, dmC, mdp, cfg.model.relabel_lambda),
            "D_bootstrap_fixed": value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                               seed=seed, mode="td", q=q)[0],
            "D_bootstrap_broken": value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                                seed=seed, mode="state_value", q=q)[0],
        }
        for name, rtg in arms.items():
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            sel, test = {}, {}
            for label, tgt in target_grid(rtg):
                vals = per_bin_values(make_dt_policy(m, mdp, tgt), mdp, allb)
                sel[label] = nv_on(vals, sel_bins, v_beh_b, v_opt_b)
                test[label] = nv_on(vals, test_bins, v_beh_b, v_opt_b)
            ch = max(sel, key=sel.get)
            rows.append(dict(seed=seed, arm=name, nv_heldout=round(test[ch], 3),
                             nv_default=round(test.get("q0.95", np.nan), 3), chosen=ch))
        cur = {r["arm"]: r["nv_heldout"] for r in rows if r["seed"] == seed}
        print(f"seed {seed}: " + "  ".join(f"{k}={v:+.2f}" for k, v in cur.items()))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.outdir, "e2ab_fixed.csv"), index=False)
    summ = df.groupby("arm").agg(heldout=("nv_heldout", "mean"), sd=("nv_heldout", "std"),
                                 default_rule=("nv_default", "mean")).round(3)
    summ = summ.sort_values("heldout", ascending=False)
    summ.to_csv(os.path.join(args.outdir, "e2ab_fixed_summary.csv"))
    print("\n=== E2-AB under held-out conditioning (published: A~C 0.67 >> D 0.44 > B 0.33) ===")
    print(summ.to_string())

    piv = df.pivot_table(index="seed", columns="arm", values="nv_heldout")
    print("\n=== the two claims the ablation was built to make ===")
    for lhs, rhs, claim in (("A_full_prior", "D_bootstrap_fixed",
                             "structured relabel beats the bootstrapped value"),
                            ("A_full_prior", "B_no_constraints",
                             "the economic prior beats an unconstrained demand model"),
                            ("A_full_prior", "C_misspecified",
                             "prior CORRECTNESS matters (published: it does not)")):
        med, p = M.paired_test(piv[lhs].values, piv[rhs].values)
        d = piv[lhs].mean() - piv[rhs].mean()
        print(f"  {lhs} - {rhs:20s}: {d:+.3f}  p={p:.4f}   [{claim}]")
    med, p = M.paired_test(piv["D_bootstrap_fixed"].values, piv["D_bootstrap_broken"].values)
    print(f"\n  cost of the baseline defect: D_fixed - D_broken = "
          f"{piv['D_bootstrap_fixed'].mean() - piv['D_bootstrap_broken'].mean():+.3f} (p={p:.4f})")


if __name__ == "__main__":
    main()
