"""C1 (stitching) re-measured under a fair comparator AND the even-handed protocol.

C1 is re-measured here because two properties of the original protocol affect it in
the same way they affect C2:

  - the published margins compare against `Q-DT -25.314`, produced by the
    action-INDEPENDENT relabeller whose within-state target spread is exactly 0.000;
  - every arm is evaluated at the 0.95 quantile of its OWN return column, which §2/§4.2
    shows costs the comparators more normalised value than it costs the proposed
    method.

Published: mean per-start stitching margin
structured -3.950 vs vanilla -15.514 vs Q-DT -25.314; strict per-start success
6.7% / 2.2% / 0.0%; "best of the three in 95.6% of cells, p < 1e-16".

METRIC (unchanged, `metrics.stitching_score`). For each logged start bin, the learned
policy's exact value from that bin minus the best DE-NOISED logged return from the
same bin. Positive margin => genuine stitching. Reported as the mean margin and the
fraction of starts beaten.

PROTOCOL. Start bins are split 30/70 per seed. Under `heldout`, each arm picks its
conditioning target on the selection bins and the margin is computed on the test bins
only; `default` reports the published q0.95 rule on the same test bins, so the two
columns are like-for-like and isolate the protocol's contribution.

Run:  python -m pricing_dt.diagnostics.diag_c1_fixed --outdir results
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
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_heldout_protocol import target_grid, per_bin_values

ARMS = ["vanillaDT", "QDT_legacy", "QDT_td", "QDT_qsa", "structuredDT", "oracle_Qstar"]


def stitch_on(vals, bins, ceiling):
    """Mean margin and beat-fraction restricted to `bins` that have a logged ceiling."""
    margins = [vals[int(b)] - ceiling[int(b)] for b in bins if int(b) in ceiling]
    if not margins:
        return float("nan"), float("nan")
    return float(np.mean(margins)), float(np.mean([m > 1e-6 for m in margins]))


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

        # de-noised per-start ceiling from the logged trajectories
        ceiling = M.best_logged_by_start(trajs, mdp, cfg.sim.n_ref_bins)

        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
        cols = {
            "vanillaDT": logged_rtg(trajs),
            "structuredDT": relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda),
            "oracle_Qstar": oracle_rtg(trajs, mdp),
        }
        for tag, mode in (("QDT_legacy", "state_value"), ("QDT_td", "td"), ("QDT_qsa", "q_sa")):
            cols[tag], _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                         seed=seed, mode=mode, q=q)

        for tag in ARMS:
            rtg = cols[tag]
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            grid = target_grid(rtg)
            best_lab, best_sel = None, -1e18
            per_target = {}
            for label, tgt in grid:
                vals = per_bin_values(make_dt_policy(m, mdp, tgt), mdp, allb)
                per_target[label] = vals
                s_marg, _ = stitch_on(vals, sel_bins, ceiling)
                if s_marg == s_marg and s_marg > best_sel:
                    best_sel, best_lab = s_marg, label
            ho_marg, ho_beat = stitch_on(per_target[best_lab], test_bins, ceiling)
            df_marg, df_beat = stitch_on(per_target.get("q0.95", per_target[best_lab]),
                                         test_bins, ceiling)
            rows.append(dict(seed=seed, arm=tag,
                             margin_heldout=round(ho_marg, 2), beat_heldout=round(ho_beat, 3),
                             margin_default=round(df_marg, 2), beat_default=round(df_beat, 3),
                             chosen=best_lab))
        cur = {r["arm"]: r["margin_heldout"] for r in rows if r["seed"] == seed}
        print(f"seed {seed}: " + "  ".join(f"{k}={v:+.1f}" for k, v in cur.items()))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "c1_fixed.csv"), index=False)
    summ = df.groupby("arm").agg(
        margin_heldout=("margin_heldout", "mean"), beat_heldout=("beat_heldout", "mean"),
        margin_default=("margin_default", "mean"), beat_default=("beat_default", "mean"),
    ).round(3).sort_values("margin_heldout", ascending=False)
    summ.to_csv(os.path.join(args.outdir, "c1_fixed_summary.csv"))
    print("\n=== C1 stitching, test bins only (mean over seeds) ===")
    print("  published: structured -3.950 | vanilla -15.514 | Q-DT(legacy) -25.314")
    print("  strict success published: 6.7% / 2.2% / 0.0%\n")
    print(summ.to_string())

    piv = df.pivot_table(index="seed", columns="arm", values="margin_heldout")
    print("\n=== paired vs structuredDT, even-handed protocol ===")
    base = piv["structuredDT"].values
    labels, diffs, ps = [], [], []
    for a in [c for c in piv.columns if c != "structuredDT"]:
        med, p = M.paired_test(base, piv[a].values)
        labels.append(a); diffs.append(base.mean() - piv[a].values.mean())
        ps.append(1.0 if p != p else p)
    holm = M.holm(np.array(ps))
    for a, d, p, h in sorted(zip(labels, diffs, ps, holm), key=lambda z: -z[1]):
        print(f"  structuredDT - {a:14s}: {d:+8.2f}  p={p:.4f}  holm={h:.4f}")

    sig = [a for a, h in zip(labels, holm) if h < 0.05]
    print("\n=== verdict ===")
    if "QDT_td" in sig or "QDT_qsa" in sig:
        print("  >>> C1 SURVIVES: structured still stitches significantly better than an "
              "action-dependent Q-DT under the even-handed protocol.")
    else:
        print("  >>> C1 DOES NOT SURVIVE as a comparative claim against an "
              "action-dependent Q-DT "
              f"(significant only vs {sig if sig else 'nothing'}).")
    print("  Note the published Q-DT margin of -25.314 came from the zero-signal target; "
          "compare it with the QDT_legacy row above under the same even-handed protocol.")


if __name__ == "__main__":
    main()
