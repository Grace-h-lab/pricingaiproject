"""Conditioning-target sensitivity — closing a confound in every DT comparison here.

All arms are evaluated by conditioning the DT on the 0.95 quantile of ITS OWN
training return-to-go column (`experiments._eval_dt`). That is standard DT
practice, but across relabelling arms it is not a fixed protocol: each relabeller
produces a different RTG distribution, so each arm is asked for a different thing
at test time. A relabeller can then look better purely because the 0.95-quantile
rule happens to pick a luckier "ask" for its particular target distribution —
which would be a property of the evaluation rule, not of the relabelling.

This is not hypothetical here. The structured target is inflated to ~1.48x the true
optimum, so its 0.95 quantile is an aggressive ask; the oracle Q* target is
calibrated, so the same rule asks it for much less relative to what is achievable.
Any comparison between them at "each arm's own 0.95 quantile" therefore confounds
relabelling quality with conditioning aggressiveness.

The fix is to give every arm the same treatment: sweep the conditioning target over
a wide grid and report (a) the value at the default rule, (b) the BEST value the arm
can reach at any target, and (c) where that best sits. Conclusions that survive at
the per-arm best are conclusions about the relabelling; conclusions that only exist
at the default rule are conclusions about the rule.

Targets swept, per arm: quantiles {0.5, 0.75, 0.9, 0.95, 0.99, 1.0} of its own RTG
column, plus multiples {1.25, 1.5, 2.0} of that column's maximum (asking for more
than anything in the data — the regime return-conditioning is supposed to exploit).
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
from pricing_dt.core.relabel import relabel_dataset, logged_rtg, oracle_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_trust_region import eval_cached

QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
MAX_MULTS = [1.25, 1.5, 2.0]


def target_grid(rtg_list):
    flat = np.concatenate(rtg_list)
    out = [(f"q{q}", float(np.quantile(flat, q))) for q in QUANTILES]
    mx = float(flat.max())
    out += [(f"max x{m}", mx * m) for m in MAX_MULTS]
    return out


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

        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        rtg_qdt, _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model, seed=seed, mode="td")

        arms = {"A_structured": relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda),
                "QDT_td": rtg_qdt,
                "oracle_Qstar": oracle_rtg(trajs, mdp),
                "vanilla": logged_rtg(trajs)}

        for name, rtg in arms.items():
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            for label, tgt in target_grid(rtg):
                pol = make_dt_policy(m, mdp, tgt)
                v, _, _, _, _ = eval_cached(pol, mdp, init)
                rows.append(dict(seed=seed, arm=name, target_rule=label,
                                 target=round(tgt, 1),
                                 nv=round(M.normalised_value(v, v_beh, v_opt), 3),
                                 is_default=(label == "q0.95")))
        best = {a: max(r["nv"] for r in rows if r["seed"] == seed and r["arm"] == a)
                for a in arms}
        dflt = {a: [r["nv"] for r in rows if r["seed"] == seed and r["arm"] == a
                    and r["is_default"]][0] for a in arms}
        print(f"seed {seed}: " + "  ".join(
            f"{a}={dflt[a]:+.2f}->{best[a]:+.2f}" for a in arms))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "conditioning_sweep.csv"), index=False)

    print("\n=== value by conditioning target (mean over seeds) ===")
    piv = df.groupby(["arm", "target_rule"]).nv.mean().unstack().round(3)
    order = [f"q{q}" for q in QUANTILES] + [f"max x{m}" for m in MAX_MULTS]
    piv = piv.reindex(columns=order)
    print(piv.to_string())

    print("\n=== default rule vs per-arm best ===")
    summ = []
    for arm in df.arm.unique():
        sub = df[df.arm == arm]
        d = sub[sub.is_default].groupby("seed").nv.max()
        b = sub.groupby("seed").nv.max()
        summ.append(dict(arm=arm, default=round(d.mean(), 3), best=round(b.mean(), 3),
                         gain=round(b.mean() - d.mean(), 3),
                         best_rule=piv.loc[arm].idxmax()))
    sdf = pd.DataFrame(summ).sort_values("best", ascending=False)
    sdf.to_csv(os.path.join(args.outdir, "conditioning_summary.csv"), index=False)
    print(sdf.to_string(index=False))

    piv_seed = df.groupby(["arm", "seed"]).nv.max().unstack()
    print("\n=== paired tests at each arm's BEST conditioning target ===")
    base = piv_seed.loc["A_structured"].values
    for arm in [a for a in piv_seed.index if a != "A_structured"]:
        med, p = M.paired_test(base, piv_seed.loc[arm].values)
        print(f"  A_structured - {arm:14s}: "
              f"{base.mean() - piv_seed.loc[arm].values.mean():+.3f} "
              f"(paired median {med:+.3f}, p={p:.4f})")

    win_default = sdf.sort_values("default", ascending=False).arm.iloc[0]
    win_best = sdf.arm.iloc[0]
    print(f"\n  best arm under the DEFAULT 0.95-quantile rule : {win_default}")
    print(f"  best arm at each arm's OWN BEST target        : {win_best}")
    print("  >>> " + ("ranking is ROBUST to the conditioning rule."
                      if win_default == win_best else
                      "ranking DEPENDS on the conditioning rule -- the default-rule "
                      "comparison is confounded and must be reported at the per-arm best."))


if __name__ == "__main__":
    main()
