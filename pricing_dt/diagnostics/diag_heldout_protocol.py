"""Held-out conditioning-target selection — the hygiene standard applied to ourselves.

WHY THIS RUN EXISTS. `diag_conditioning.py` showed the arm ranking depends on the
conditioning target, and reported each arm at ITS OWN BEST target. That is equal
treatment, but it is oracle-tuned on the test objective, and the bias is NOT equal
across arms: it is largest for the arm whose value-vs-target curve is most peaked.
The structured arm is by far the most peaked (0.670 at q0.95, collapsing to -0.74 at
q0.50 and -0.30 at max x2), while vanilla and oracle_Qstar are comparatively flat.
So "each arm at its own best" systematically FLATTERS the structured arm relative to
its comparators, and every surviving comparison — the tie with a fixed Q-DT, the
+0.115 over oracle Q*, the tie with model-filtered BC — is measured on a reading
that favours it.

This script removes that. The conditioning target is chosen on a HELD-OUT set of
start states and the test set is read once.

PROTOCOL (pre-registered).
  - Start states are split disjointly per seed: ~30% SELECTION, ~70% TEST. The split
    is over reference bins, so selection and test are genuinely different states —
    the realistic difficulty is precisely that the best ask may not transfer.
  - Every arm gets identical treatment: sweep the target grid, pick the argmax on
    SELECTION, report that single target's value on TEST.
  - The grid must cover every arm's known optimum or the protocol manufactures a new
    unfairness. A q0.50-q0.99 grid would exclude the max x1.25 optima of vanilla and
    oracle_Qstar, so the grid spans q0.50-q1.0 AND max x{1.25, 1.5, 2.0}.
  - Evaluation is deterministic per start bin, so all bins are rolled out once per
    (arm, target) and the two splits are read off the same per-bin values. Anchors
    (behaviour, optimum) are recomputed per split.
  - Paired Wilcoxon across seeds + Holm.

ARMS. The five relabelling arms, plus two that the previous analysis left exposed to
the same confound and which carry the mechanism claim:
    potential_exact  = target decomposition cell (model action x ORACLE potential):
                       the "making the aspiration field exact HURTS" result (-0.277)
    action_deleted   = decomposition cell (MEAN action x model potential):
                       the "the action term is inert" result (+0.007)
Both were measured at their own q0.95 exactly like everything else, and both belong
to the same "accuracy hurts" family as the oracle-Q* finding, so they get the same
hygiene rather than being exempted.
    filtBC           = model-filtered behaviour cloning. Needs no ask at all, so it is
                       protocol-IMMUNE and is the natural reference: if the ask-free
                       baseline wins under the even-handed protocol, the conditioning
                       machinery is not paying for itself.

ALSO RECORDED, because they decide what comes next:
  - the quantile SELECTED per seed, and whether it matches the arm's test-set optimum
    (i.e. was the q0.95 peak a stable property or test-set overfitting?)
  - each arm's within-state discrimination ratio, feeding the discrimination-vs-value
    regression without needing a separate run.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: target_grid(), nv_on(), main().
#
# The even-handed evaluation protocol: every arm's conditioning target is selected the
# same way on held-out data, which removed the proposed method's apparent advantage.
#
# Implements or follows:
#   - Wilcoxon, F. (1945) 'Individual comparisons by ranking methods', Biometrics
#     Bulletin, 1(6).
#   - Holm, S. (1979) 'A simple sequentially rejective multiple test procedure',
#     Scandinavian Journal of Statistics, 6(2).
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import argparse
import numpy as np
import pandas as pd
import os

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset, logged_rtg, oracle_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_cql, train_bc, policy_from_qnet
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_trust_region import model_q_surface
from pricing_dt.diagnostics.diag_target_decomp import compute_terms, build
from pricing_dt.diagnostics.diag_target_stats import stats_for

QUANTILES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
             0.925, 0.95, 0.975, 0.99, 1.0]
MAX_MULTS = [1.25, 1.5, 2.0]


def target_grid(rtg_list):
    flat = np.concatenate(rtg_list)
    g = [(f"q{q}", float(np.quantile(flat, q))) for q in QUANTILES]
    mx = float(flat.max())
    return g + [(f"max x{m}", mx * m) for m in MAX_MULTS]


def per_bin_values(policy_fn, mdp, bins):
    """Exact expected return from each bin (deterministic rollout)."""
    out = {}
    for b0 in bins:
        ref = mdp.ref_grid[b0]
        tot = 0.0
        for t in range(mdp.H):
            b = mdp.ref_to_bin(ref)
            a = int(policy_fn(mdp.obs(ref, t)))
            tot += mdp.R[a, b]
            ref = mdp.ref_grid[mdp.N[a, b]]
        out[int(b0)] = float(tot)
    return out


def nv_on(vals, bins, v_beh_b, v_opt_b):
    v = np.mean([vals[int(b)] for b in bins])
    beh = np.mean([v_beh_b[int(b)] for b in bins])
    opt = np.mean([v_opt_b[int(b)] for b in bins])
    return M.normalised_value(v, beh, opt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sel-frac", type=float, default=0.3)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}")
    print(f"held-out selection: {args.sel_frac:.0%} of start bins select the ask, "
          f"the rest is read once\n")

    rows, sel_rows, stat_rows = [], [], []
    for seed in seeds:
        _seed(seed)
        rng = np.random.default_rng(1000 + seed)
        mdp, init, _vo, _vb = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

        # disjoint bin split
        allb = np.arange(cfg.sim.n_ref_bins)
        perm = rng.permutation(allb)
        n_sel = max(2, int(round(args.sel_frac * len(allb))))
        sel_bins, test_bins = np.sort(perm[:n_sel]), np.sort(perm[n_sel:])

        # per-bin anchors
        myopic = lambda o: int(mdp.R[:, mdp.ref_to_bin(
            mdp.cfg.p_min + o[0] * (mdp.cfg.p_max - mdp.cfg.p_min))].argmax())
        v_beh_b = per_bin_values(myopic, mdp, allb)
        v_opt_b = {int(b): float(mdp.Vstar[0, b]) for b in allb}

        # ---- models shared by the arms ----
        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)
        q = train_cql(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
        terms = compute_terms(trajs, dmA, mdp)

        arms = {
            "A_structured":    relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda),
            "B_unconstrained": relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda),
            "QDT_td":          value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                             seed=seed, mode="td", q=q)[0],
            "QDT_qsa":         value_relabel(trajs, mdp, obs_dim, A, cfg.model,
                                             seed=seed, mode="q_sa", q=q)[0],
            "oracle_Qstar":    oracle_rtg(trajs, mdp),
            "vanilla":         logged_rtg(trajs),
            # decomposition cells carrying the mechanism claim
            "potential_exact": build(terms, "model", "oracle", rng),
            "action_deleted":  build(terms, "mean", "model", rng),
        }

        for name, rtg in arms.items():
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            grid = target_grid(rtg)
            sel_nv, test_nv = {}, {}
            for label, tgt in grid:
                vals = per_bin_values(make_dt_policy(m, mdp, tgt), mdp, allb)
                sel_nv[label] = nv_on(vals, sel_bins, v_beh_b, v_opt_b)
                test_nv[label] = nv_on(vals, test_bins, v_beh_b, v_opt_b)
            chosen = max(sel_nv, key=sel_nv.get)
            oracle_best = max(test_nv, key=test_nv.get)
            rows.append(dict(seed=seed, arm=name,
                             nv_heldout=round(test_nv[chosen], 3),
                             nv_default=round(test_nv.get("q0.95", np.nan), 3),
                             nv_testbest=round(test_nv[oracle_best], 3),
                             chosen=chosen, testbest=oracle_best,
                             transfer_loss=round(test_nv[oracle_best] - test_nv[chosen], 3)))
            sel_rows.append(dict(seed=seed, arm=name, chosen=chosen, testbest=oracle_best,
                                 matched=(chosen == oracle_best)))
            st = stats_for(rtg, trajs, mdp)
            stat_rows.append(dict(seed=seed, arm=name, disc_ratio=st["disc_ratio"],
                                  nv_heldout=test_nv[chosen]))

        # ---- ask-free reference: model-filtered BC ----
        _R, _V, Qs = model_q_surface(dmA, mdp)
        scores = np.array([Qs[0, int(tr.ref_bins[0]), int(tr.actions[0])] for tr in trajs])
        idx = np.argsort(-scores)[:max(2, len(trajs) // 2)]
        b2 = train_bc([trajs[i] for i in idx], obs_dim, A, cfg.model, seed=seed)
        vals = per_bin_values(policy_from_qnet(b2), mdp, allb)
        tnv = nv_on(vals, test_bins, v_beh_b, v_opt_b)
        rows.append(dict(seed=seed, arm="filtBC_keep0.5", nv_heldout=round(tnv, 3),
                         nv_default=round(tnv, 3), nv_testbest=round(tnv, 3),
                         chosen="(no ask)", testbest="(no ask)", transfer_loss=0.0))

        cur = {r["arm"]: r["nv_heldout"] for r in rows if r["seed"] == seed}
        print(f"seed {seed}: " + "  ".join(f"{k}={v:+.2f}" for k, v in cur.items()))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.outdir, "heldout_protocol.csv"), index=False)
    pd.DataFrame(sel_rows).to_csv(os.path.join(args.outdir, "heldout_selection.csv"), index=False)
    pd.DataFrame(stat_rows).to_csv(os.path.join(args.outdir, "heldout_disc.csv"), index=False)

    summ = df.groupby("arm").agg(
        heldout=("nv_heldout", "mean"), sd=("nv_heldout", "std"),
        default_rule=("nv_default", "mean"), test_best=("nv_testbest", "mean"),
        transfer_loss=("transfer_loss", "mean")).round(3).sort_values("heldout", ascending=False)
    summ.to_csv(os.path.join(args.outdir, "heldout_summary.csv"))
    print("\n=== TEST-set value under held-out target selection (mean over seeds) ===")
    print(summ.to_string())

    piv = df.pivot_table(index="seed", columns="arm", values="nv_heldout")
    print("\n=== paired vs A_structured, under the even-handed protocol ===")
    base = piv["A_structured"].values
    others = [a for a in piv.columns if a != "A_structured"]
    ps, labels, diffs = [], [], []
    for a in others:
        med, p = M.paired_test(base, piv[a].values)
        ps.append(1.0 if p != p else p); labels.append(a)
        diffs.append(base.mean() - piv[a].values.mean())
    holm = M.holm(np.array(ps))
    for a, d, p, h in sorted(zip(labels, diffs, ps, holm), key=lambda z: -z[1]):
        print(f"  A_structured - {a:16s}: {d:+.3f}  p={p:.4f}  holm={h:.4f}")

    print("\n=== did the q0.95 peak transfer? (selection choice vs test-set optimum) ===")
    sel = pd.DataFrame(sel_rows)
    for a in sel.arm.unique():
        s = sel[sel.arm == a]
        top = s.chosen.value_counts().idxmax()
        print(f"  {a:16s}: matched test-optimum in {s.matched.mean():.0%} of seeds; "
              f"most-chosen ask = {top}")

    print("\n=== outcome ===")
    top_arm = summ.index[0]
    a_val = summ.loc["A_structured", "heldout"]
    spread = summ["heldout"].max() - summ[summ.index != "filtBC_keep0.5"]["heldout"].min()
    sig = [l for l, h in zip(labels, holm) if h < 0.05]
    if top_arm == "filtBC_keep0.5":
        print("  >>> D: the ask-free filtered-BC baseline WINS under the "
              "even-handed protocol.")
    elif not sig:
        print("  >>> A: no arm separates. The conditioning target is not a real lever at "
              "m=1; the only lever that matters is the trust-region width m.")
    else:
        print(f"  >>> B/C: A_structured retains significant separation from {sig}.")
    print(f"      (A_structured held-out {a_val:+.3f}; best arm {top_arm} "
          f"{summ['heldout'].max():+.3f}; spread across conditioned arms {spread:.3f})")


if __name__ == "__main__":
    main()
