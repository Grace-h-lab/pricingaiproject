"""Sparse price coverage, re-checked under the symmetric conditioning protocol.

The support account of the goal channel predicts that its protection comes from
confinement to the logged action support, and so must weaken when that support is
thin. This scan tests that boundary by snapping logged actions to K distinct prices
and measuring how each arm's value moves as price coverage is thinned.

Re-run with held-out target selection (bins split 30/70 per seed), 10 seeds, and with
the FIXED Q-DT added — the published probe compared only A against B, so it could not
say whether thin support hurts the structured relabeller specifically or every
return-conditioned method equally. For the axis claim, "equally" is the prediction.

Anchor note: normalised value uses the same unsnapped ORACLE MYOPIC anchor as
every other experiment here -- argmax over the true reward, not the logging
policy -- so numbers stay comparable across the project. Under snapping the
actual logger is not that policy, which is why values can fall well below 0.

Implements: the thinned-coverage scan of Appendix F.3.2.

Run:  python -m pricing_dt.diagnostics.diag_sparse_fixed --outdir results
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
from pricing_dt.core.relabel import relabel_dataset, logged_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_sparse_coverage import reroll_sparse
from pricing_dt.diagnostics.diag_heldout_protocol import target_grid, per_bin_values, nv_on


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sel-frac", type=float, default=0.3)
    ap.add_argument("--N", type=int, default=100)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    noise = max(cfg.exp.noise_levels)
    Ks = [A, 5, 3, 2] if not args.smoke else [A, 3]
    seeds = cfg.exp.seeds
    print(f"N={args.N} noise={noise} delta=3.0  K={Ks}  seeds={seeds}\n")

    rows = []
    for K in Ks:
        allowed = np.unique(np.round(np.linspace(0, A - 1, K)).astype(int))
        for seed in seeds:
            _seed(seed)
            rng = np.random.default_rng(1000 + seed)
            mdp, init, _vo, _vb = _setup(cfg, {"demand_noise": noise, "delta": 3.0}, seed=seed)
            base = D.make_stitching_necessary(mdp, args.N, noise, seed, cfg.data.expert_q)
            trajs = [reroll_sparse(tr, mdp, allowed, noise, rng) for tr in base]

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
            q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
            arms = {
                "A_structured": relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda),
                "B_unconstrained": relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda),
                "QDT_fixed": value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                           seed=seed, mode="td", q=q)[0],
                "vanilla": logged_rtg(trajs),
            }
            out = {"K": K, "seed": seed}
            for name, rtg in arms.items():
                m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
                sel, test = {}, {}
                for label, tgt in target_grid(rtg):
                    vals = per_bin_values(make_dt_policy(m, mdp, tgt), mdp, allb)
                    sel[label] = nv_on(vals, sel_bins, v_beh_b, v_opt_b)
                    test[label] = nv_on(vals, test_bins, v_beh_b, v_opt_b)
                ch = max(sel, key=sel.get)
                out[name] = round(test[ch], 3)
                out[f"{name}_default"] = round(test.get("q0.95", np.nan), 3)
            rows.append(out)
            print(f"K={K:2d} seed {seed}: " +
                  "  ".join(f"{k}={out[k]:+.2f}" for k in arms))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "sparse_fixed.csv"), index=False)

    names = ["A_structured", "B_unconstrained", "QDT_fixed", "vanilla"]
    print("\n=== normalised value vs price coverage, held-out protocol ===")
    print("  published: A-B = +0.245 at K=11 (p>=0.06), ~0 at K<=3; both ~ -1.96 at K=2\n")
    summ = []
    for K in Ks:
        cell = df[df.K == K]
        r = {"K": K}
        for n in names:
            r[n] = round(cell[n].mean(), 3)
        med, p = M.paired_test(cell["A_structured"].values, cell["B_unconstrained"].values)
        r["A_minus_B"] = round(cell["A_structured"].mean() - cell["B_unconstrained"].mean(), 3)
        r["p_AB"] = round(p, 4)
        med2, p2 = M.paired_test(cell["A_structured"].values, cell["QDT_fixed"].values)
        r["A_minus_QDT"] = round(cell["A_structured"].mean() - cell["QDT_fixed"].mean(), 3)
        r["p_AQ"] = round(p2, 4)
        summ.append(r)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(args.outdir, "sparse_fixed_summary.csv"), index=False)
    print(sdf.to_string(index=False))

    print("\n=== the two readings this probe has to support ===")
    full, thin = sdf.iloc[0], sdf.iloc[-1]
    print(f"  (1) A's edge over B is confined to full coverage: "
          f"K={int(full.K)} A-B={full.A_minus_B:+.3f} (p={full.p_AB}) -> "
          f"K={int(thin.K)} A-B={thin.A_minus_B:+.3f} (p={thin.p_AB})")
    drop = {n: round(sdf.iloc[0][n] - sdf.iloc[-1][n], 3) for n in names}
    print(f"  (2) thin support degrades EVERY return-conditioned arm, not just the "
          f"structured one:")
    for n in names:
        print(f"        {n:16s}: K={int(full.K)} {sdf.iloc[0][n]:+.3f} -> "
              f"K={int(thin.K)} {sdf.iloc[-1][n]:+.3f}   (drop {drop[n]:+.3f})")
    spread = max(drop.values()) - min(drop.values())
    print(f"\n  >>> {'COMMON collapse' if spread < 0.5 else 'UNEVEN collapse'}: "
          f"largest-minus-smallest drop across arms = {spread:.3f}. "
          + ("Thin action support hurts return-conditioned methods as a class, which is "
             "what the support/trust-region account predicts."
             if spread < 0.5 else
             "The arms degrade at materially different rates; the support account needs "
             "to say why."))


if __name__ == "__main__":
    main()
