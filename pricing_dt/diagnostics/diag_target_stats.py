"""Which STATISTICAL PROPERTY of a conditioning target predicts policy value?

The channel ladder rules out the obvious answers. Across its relabelling arms:

  - target SCALE is irrelevant: inflation varies over an order of magnitude with no
    relation to value — the DT standardises its return token, so level is washed
    out.
  - target FIDELITY is negatively related to value, and in the controlled swap —
    identical target form, only the model changed — BOTH the structured-model
    estimate and the bootstrapped estimate beat the EXACT Q*.

So an accurate conditioning target is actively worse than an inaccurate one. This
script measures the target-distribution properties that could explain it, WITHOUT
training anything (pure relabeller statistics, so it is seconds not hours):

  spread_total      sd of all targets                (overall dispersion)
  spread_between    sd of per-start mean targets     (cross-state variation)
  spread_within     mean sd of targets within a start(discrimination between
                                                      trajectories from the same
                                                      state — the quantity the DT
                                                      needs to prefer one logged
                                                      continuation over another)
  disc_ratio        spread_within / spread_between   (how much of the variation is
                                                      usable discrimination rather
                                                      than state indexing)
  action_gap        mean over states of (target of the best logged action minus
                                         target of the worst logged action),
                                        normalised by spread_total
  rank_corr_action  mean per-state Spearman of the target against Qstar over the
                                         actions actually logged at that state

The hypothesis the ladder implies: Q* is a POOR conditioning signal because
Q*(s,a) already includes the optimal continuation, so a bad logged action still
scores high and the target barely separates good from bad trajectories at the same
state. Estimation error, by contrast, spreads targets apart. If that is right,
`spread_within` / `disc_ratio` should track value while fidelity anti-tracks it.

Implements: the within-state discrimination ratio of Equation (3.7), reported in §4.4.
"""
import argparse
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset, logged_rtg, oracle_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql


def stats_for(rtg_list, trajs, mdp, bins=None):
    """Target-distribution statistics, grouped by the state a trajectory starts in.

    `bins` names the discrete state: E1 groups by reference-price bin, E2 by inventory
    level. The ratio itself is Equation (3.7) and does not depend on which.
    """
    if bins is None:
        bins = lambda tr: tr.ref_bins
    flat = np.concatenate(rtg_list)
    by_start, by_state_act = {}, {}
    for tr, g in zip(trajs, rtg_list):
        by_start.setdefault(int(bins(tr)[0]), []).append(float(g[0]))
        for t in range(len(tr.actions)):
            key = (t, int(bins(tr)[t]))
            by_state_act.setdefault(key, []).append((int(tr.actions[t]), float(g[t])))

    means = np.array([np.mean(v) for v in by_start.values()])
    withins = [np.std(v) for v in by_start.values() if len(v) > 1]
    sw = float(np.mean(withins)) if withins else 0.0
    sb = float(means.std())

    gaps, rcs = [], []
    for (t, b), pairs in by_state_act.items():
        if len(pairs) < 3:
            continue
        acts = np.array([p[0] for p in pairs]); tg = np.array([p[1] for p in pairs])
        if tg.std() < 1e-9:
            gaps.append(0.0); continue
        qs = np.array([mdp.Qstar[t, b, a] for a in acts])
        gaps.append(float(tg.max() - tg.min()))
        if qs.std() > 1e-9 and len(set(acts)) > 2:
            rc, _ = spearmanr(tg, qs)
            if rc == rc:
                rcs.append(float(rc))
    st = float(flat.std())
    return dict(spread_total=st, spread_between=sb, spread_within=sw,
                disc_ratio=sw / sb if sb > 1e-9 else np.nan,
                action_gap=float(np.mean(gaps)) / st if st > 1e-9 and gaps else np.nan,
                rank_corr_action=float(np.mean(rcs)) if rcs else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--env2", action="store_true",
                    help="measure the ratio in E2 (inventory) instead of E1")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--level", type=int, default=14)
    args = ap.parse_args()
    if args.env2:
        return main_env2(args)
    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)

    rows = []
    for seed in cfg.exp.seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
        arms = {}
        arms["A_structured"] = relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)
        for tag, mode in (("QDT_legacy", "state_value"), ("QDT_td", "td"), ("QDT_qsa", "q_sa")):
            arms[tag], _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                         seed=seed, mode=mode, q=q)
        arms["oracle_Qstar"] = oracle_rtg(trajs, mdp)
        arms["vanilla"] = logged_rtg(trajs)
        for tag, rtg in arms.items():
            rows.append(dict(seed=seed, arm=tag, **stats_for(rtg, trajs, mdp)))
        print(f"seed {seed} done")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "target_stats.csv"), index=False)
    agg = df.groupby("arm").mean(numeric_only=True).drop(columns=["seed"]).round(3)

    # attach the measured policy values from the ladder / conditioning sweep
    nv = {}
    lad = os.path.join(args.outdir, "channel_ladder.csv")
    if os.path.exists(lad):
        L = pd.read_csv(lad)
        for a in ["A_structured", "QDT_legacy", "QDT_td", "QDT_qsa", "oracle_Qstar"]:
            if a in L.columns:
                nv[a] = L[a].mean()
    cond = os.path.join(args.outdir, "conditioning_sweep.csv")
    if os.path.exists(cond) and "vanilla" not in nv:
        Cd = pd.read_csv(cond)
        v = Cd[(Cd.arm == "vanilla") & (Cd.is_default)]
        if len(v):
            nv["vanilla"] = v.nv.mean()
    agg["nv"] = pd.Series(nv)
    agg.to_csv(os.path.join(args.outdir, "target_stats_summary.csv"))
    print("\n=== target-distribution properties (mean over seeds) ===")
    print(agg.to_string())

    sub = agg.dropna(subset=["nv"])
    print(f"\n=== association with policy value (n={len(sub)} arms) ===")
    for col in ["spread_total", "spread_between", "spread_within", "disc_ratio",
                "action_gap", "rank_corr_action"]:
        s = sub.dropna(subset=[col])
        if len(s) < 4:
            continue
        rs, ps = spearmanr(s[col], s["nv"])
        print(f"  {col:18s}: spearman {rs:+.3f} (p={ps:.3f})")
    print("\n  (spread_* are in revenue units and differ hugely in scale across arms;"
          "\n   disc_ratio / action_gap / rank_corr_action are the scale-free ones.)")


def main_env2(args):
    """The same ratio in E2, on the arms Table 4.10 reports.

    E2's state is the inventory level, so trajectories are grouped by `states` rather
    than by reference-price bin. The structured target is the Poisson-fitted Q and the
    oracle target the environment's exact Q*, both built exactly as
    `diag_env2_channels` builds them, so the two files measure the same objects.
    """
    from pricing_dt.envs.inventory import InventoryMDP, InvConfig, OrderUpTo
    from pricing_dt.diagnostics.diag_env2_channels import (
        roll_trajectories, q_from_pmf, relabel_from_Q, qdt_relabel,
        generate_logs, fit_poisson)
    from pricing_dt.core.torch_utils import device_report

    cfg = C.smoke() if args.smoke else C.full()
    print(device_report())
    rows = []
    for seed in cfg.exp.seeds:
        _seed(seed)
        rng = np.random.default_rng(seed)
        mdp = InventoryMDP(InvConfig(seed=seed))
        mdp.solve_optimal()
        logger = OrderUpTo(mdp, args.level, rng)
        trajs = roll_trajectories(mdp, logger, args.episodes, rng)
        log = generate_logs(mdp, logger, args.episodes, rng)
        Qp, _ = q_from_pmf(mdp, fit_poisson(log, mdp.cfg.max_demand))
        qnet = train_cql(trajs, mdp, 2, mdp.n_actions, cfg.model, seed=seed)
        arms = {
            "A_structured": relabel_from_Q(trajs, Qp),
            "oracle_Qstar": relabel_from_Q(trajs, mdp.Qstar),
            "QDT_td": qdt_relabel(trajs, qnet),
            "vanilla": [tr.rtg for tr in trajs],
        }
        for tag, rtg in arms.items():
            rows.append(dict(seed=seed, arm=tag,
                             **stats_for(rtg, trajs, mdp, bins=lambda tr: tr.states)))
        print(f"seed {seed} done")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "target_stats_env2.csv"), index=False)
    agg = df.groupby("arm").mean(numeric_only=True).drop(columns=["seed"]).round(3)
    agg.to_csv(os.path.join(args.outdir, "target_stats_env2_summary.csv"))
    print(chr(10) + "=== E2 target-distribution properties (mean over %d seeds) ==="
          % df.seed.nunique())
    print(agg.to_string())
    return 0


if __name__ == "__main__":
    main()
