"""Channel ladder — an action-dependent comparator and the missing oracle rung.

Two questions, one protocol (the same hardest cell / seeds / anchors as
`diag_optimism_verdict.py` and `diag_estimate_optimize.py`, so every number drops
straight into the main comparison table).

(1) IS THE Q-DT BASELINE FAIR?  The shipped Q-DT relabelled with the STATE value
    V(s_t) = max_a Q(s_t,a), which does not depend on the logged action — the
    exact defect Appendix E.1 records and corrects for the structured relabeller. Arms
    `QDT_legacy` vs `QDT_td` / `QDT_qsa` measure what that cost the comparator.
    Every published advantage-over-Q-DT number rests on the answer.

(2) WHAT DOES THE GOAL CHANNEL ACTUALLY WANT?  All three relabellers estimate the
    SAME quantity Q*(s_t,a_t) — structured-model estimate, bootstrapped estimate,
    and (new) the exact article. `oracle_Qstar` is therefore the ceiling of the
    goal channel and the ceiling of the Q-DT family at once, and it disentangles
    two confounded properties of the structured target:

        accurate?   stable across seeds?
        --------    --------------------
        structured  NO (rank corr vs Qstar is negative)   YES (prior-dominated)
        bootstrapped  ~                                 NO  (refit per seed)
        oracle      YES                                 YES

    If oracle >> structured, the channel wants accuracy and the structured prior
    is a cheap surrogate for it. If oracle ~ structured, what matters is a stable
    state-indexed aspiration field, not its accuracy. If oracle < structured, the
    channel wants OPTIMISM specifically, and the paper's mechanism claim has to be
    written that way. `oracle_matchA` (oracle rescaled to the structured target's
    magnitude) separates accuracy from magnitude within the oracle itself.

Arms per seed, each a full DT train + exact evaluation:
    A_structured   structured demand-model relabel (the proposed method)
    QDT_legacy     V(s_t)                       -- action-independent read-out
    QDT_td         r_t + V(s_{t+1})             -- action-dependent, matches structured form
    QDT_td_dn      R[a_t,b_t] + V(s_{t+1})      -- action-dependent + de-noised current step
    QDT_qsa        Q(s_t, a_t)                  -- action-dependent, both terms from the Q-net
    oracle_Qstar   Qstar[t, b_t, a_t]           -- exact Q* of the logged action
    oracle_matchA  oracle rescaled to A's 0.95-quantile target

The four Q-DT arms share ONE fitted Q-net per seed: they differ only in how the
same value function is read out, so refitting per arm would inject variance
rather than rigour.

--alpha-sweep additionally sweeps the CQL conservatism coefficient and reports the
BEST alpha per seed for the fixed Q-DT — deliberately generous to the baseline
(it is oracle-tuned on the test objective), so the reported advantage over it is a
lower bound.
"""
import argparse
import numpy as np
import pandas as pd
import os
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed, _eval_dt
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset, oracle_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.dt import train_dt


def _quantile_target(rtg_list, q=0.95):
    return float(np.quantile(np.concatenate(rtg_list), q))


def _scale_to_target(rtg_list, target_match, q=0.95):
    cur = _quantile_target(rtg_list, q)
    c = target_match / cur if abs(cur) > 1e-8 else 1.0
    return [r * c for r in rtg_list], c


def _train_eval(rtg_list, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed):
    m = train_dt(D.pack_dt(trajs, rtg_list), obs_dim, A, cfg.model, seed=seed)
    v, tgt = _eval_dt(m, mdp, init, rtg_list)
    return M.normalised_value(v, v_beh, v_opt), tgt


def _inflation(rtg_list, trajs, mdp):
    """mean step-0 target / mean TRUE achievable optimum from the same starts.
    1.0 == calibrated; >1 == optimistic."""
    tgt = np.mean([float(g[0]) for g in rtg_list])
    truth = np.mean([float(mdp.Vstar[0, int(tr.ref_bins[0])]) for tr in trajs])
    return tgt / truth if abs(truth) > 1e-8 else float("nan")


def _shape_corr(rtg_list, trajs, mdp):
    """Cross-state correlation: does the target rank which STARTS have higher
    achievable value correctly? (The ordering the DT can actually consume.)"""
    by = {}
    for tr, g in zip(trajs, rtg_list):
        by.setdefault(int(tr.ref_bins[0]), []).append(float(g[0]))
    bins = sorted(by)
    x = np.array([np.mean(by[b]) for b in bins])
    y = np.array([float(mdp.Vstar[0, b]) for b in bins])
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--alpha-sweep", action="store_true",
                    help="sweep CQL alpha and report the best per seed (generous to the baseline)")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    alphas = [0.1, 1.0, 5.0] if args.alpha_sweep else []
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}")
    print(f"arms: A_structured, QDT_legacy, QDT_td, QDT_td_dn, QDT_qsa, "
          f"oracle_Qstar, oracle_matchA" + (f"  + alpha sweep {alphas}" if alphas else "") + "\n")

    rows, alpha_rows = [], []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        out = {"seed": seed, "v_opt": round(v_opt, 1)}

        def arm(tag, rtg):
            nv, _ = _train_eval(rtg, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
            out[tag] = round(nv, 3)
            out[f"infl_{tag}"] = round(_inflation(rtg, trajs, mdp), 3)
            out[f"corr_{tag}"] = round(_shape_corr(rtg, trajs, mdp), 3)
            return nv

        # --- the proposed method (reference point: 0.670, audit/CHANNEL_RESULTS.md) ---
        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        rtgA = relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda)
        tgtA = _quantile_target(rtgA)
        arm("A_structured", rtgA)

        # --- Q-DT family: ONE Q-net, four read-outs ---
        q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
        for tag, mode, dn in (("QDT_legacy", "state_value", False),
                              ("QDT_td", "td", False),
                              ("QDT_td_dn", "td", True),
                              ("QDT_qsa", "q_sa", False)):
            rtg, _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                   seed=seed, mode=mode, denoise=dn, q=q)
            arm(tag, rtg)

        # --- the missing rung: exact Q* ---
        rtgO = oracle_rtg(trajs, mdp)
        arm("oracle_Qstar", rtgO)
        rtgOm, cO = _scale_to_target(rtgO, tgtA)
        arm("oracle_matchA", rtgOm)
        out["scale_oracle_to_A"] = round(cO, 2)

        rows.append(out)
        print(f"seed {seed}: A={out['A_structured']:.3f} | "
              f"QDT legacy={out['QDT_legacy']:.3f} td={out['QDT_td']:.3f} "
              f"td_dn={out['QDT_td_dn']:.3f} qsa={out['QDT_qsa']:.3f} | "
              f"oracle={out['oracle_Qstar']:.3f} oracle_matchA={out['oracle_matchA']:.3f}")

        # --- optional: tune the baseline's conservatism, generously ---
        for al in alphas:
            import copy
            mc = copy.deepcopy(cfg.model); mc.cql_alpha = al
            qa = train_cql(trajs, mdp, obs_dim, A, mc, seed=seed)
            rtg, _ = value_relabel(trajs, mdp, obs_dim, A, mc, seed=seed, mode="td", q=qa)
            nv, _ = _train_eval(rtg, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
            alpha_rows.append(dict(seed=seed, cql_alpha=al, nv_QDT_td=round(nv, 3)))
            print(f"         alpha={al}: QDT_td={nv:.3f}")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "channel_ladder.csv"), index=False)
    if alpha_rows:
        adf = pd.DataFrame(alpha_rows)
        adf.to_csv(os.path.join(args.outdir, "channel_ladder_alpha.csv"), index=False)

    arms = ["A_structured", "QDT_legacy", "QDT_td", "QDT_td_dn", "QDT_qsa",
            "oracle_Qstar", "oracle_matchA"]
    print("\n=== mean normalised value over seeds ===")
    for a in sorted(arms, key=lambda a: -df[a].mean()):
        print(f"  {a:16s}: {df[a].mean():+.3f}  (sd {df[a].std():.3f})  "
              f"inflation {df['infl_'+a].mean():.2f}x  cross-state corr {df['corr_'+a].mean():.3f}")

    print("\n=== (1) what the baseline fix costs the headline ===")
    for fixed in ["QDT_td", "QDT_td_dn", "QDT_qsa"]:
        med, p = M.paired_test(df[fixed].values, df["QDT_legacy"].values)
        print(f"  {fixed} - QDT_legacy : {df[fixed].mean() - df['QDT_legacy'].mean():+.3f} "
              f"(paired median {med:+.3f}, p={p:.4f})")
    best_fixed = max(["QDT_td", "QDT_td_dn", "QDT_qsa"], key=lambda a: df[a].mean())
    med, p = M.paired_test(df["A_structured"].values, df[best_fixed].values)
    print(f"  ADVANTAGE vs BEST fixed Q-DT ({best_fixed}): "
          f"{df['A_structured'].mean() - df[best_fixed].mean():+.3f} "
          f"(paired median {med:+.3f}, p={p:.4f})")
    print(f"  [pre-fix advantage vs QDT_legacy was "
          f"{df['A_structured'].mean() - df['QDT_legacy'].mean():+.3f}]")

    print("\n=== (2) what the goal channel wants ===")
    med, p = M.paired_test(df["A_structured"].values, df["oracle_Qstar"].values)
    d = df["A_structured"].mean() - df["oracle_Qstar"].mean()
    print(f"  A_structured - oracle_Qstar : {d:+.3f} (paired median {med:+.3f}, p={p:.4f})")
    if d > 0.05:
        verdict = ("OPTIMISM -- the structured target BEATS the exact Q*. The goal channel "
                   "does not want accuracy; rewrite the mechanism claim around optimistic shaping.")
    elif d < -0.05:
        verdict = ("ACCURACY -- the exact Q* wins. The structured prior is a cheap surrogate "
                   "for an accurate target; the current framing survives.")
    else:
        verdict = ("STABILITY -- structured ties the exact Q* despite being inaccurate. What the "
                   "channel consumes is a stable state-indexed aspiration field, not its accuracy.")
    print(f"  >>> VERDICT: {verdict}")

    if alpha_rows:
        print("\n=== CQL alpha sweep (fixed Q-DT, best-per-seed = generous baseline) ===")
        piv = adf.pivot(index="seed", columns="cql_alpha", values="nv_QDT_td")
        for al in alphas:
            print(f"  alpha={al}: mean {piv[al].mean():+.3f}")
        best = piv.max(axis=1)
        print(f"  best-per-seed mean : {best.mean():+.3f}")
        med, p = M.paired_test(df["A_structured"].values, best.values)
        print(f"  ADVANTAGE vs oracle-tuned Q-DT: "
              f"{df['A_structured'].mean() - best.mean():+.3f} "
              f"(paired median {med:+.3f}, p={p:.4f})")


if __name__ == "__main__":
    main()
