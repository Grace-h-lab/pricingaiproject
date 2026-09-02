"""CONFIRMATORY replication: is an exact `Q*` a worse conditioning target?

Executes exactly the design fixed in `PREREGISTRATION_QSTAR.md`, which was written
before any seed in 10-69 was evaluated. Read that file first; nothing here may be
changed after the fact without recording a deviation there.

  arms        A_structured, oracle_Qstar, potential_exact
  seeds       10-69 (FRESH; 0-9 generated the hypothesis and are excluded)
  protocol    as diag_heldout_protocol.py -- 30/70 start-bin split, target chosen on
              selection bins, test bins read once
  hypotheses  H1 A_structured - oracle_Qstar    > 0   (primary)
              H2 A_structured - potential_exact > 0   (co-primary)
  test        paired Wilcoxon, two-sided, Holm over exactly these TWO comparisons
  decision    both significant -> section; one -> paragraph; neither -> one sentence,
              permanently, with no further seeds added

Implements: the pre-registered confirmatory replication of §3.5.5, reported in §4.5.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: boot_ci(), main().
#
# The pre-registered confirmatory replication: fresh seeds, sample size fixed in advance
# against a halved effect, two co-primary hypotheses, Holm over exactly those two.
#
# Implements or follows:
#   - Nosek, B.A., Ebersole, C.R., DeHaven, A.C. and Mellor, D.T. (2018) 'The
#     preregistration revolution', PNAS, 115(11).
#   - Efron, B. and Tibshirani, R.J. (1993) An Introduction to the Bootstrap. Chapman &
#     Hall.
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
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset, oracle_rtg
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_heldout_protocol import target_grid, per_bin_values, nv_on
from pricing_dt.diagnostics.diag_target_decomp import compute_terms, build

PRIMARY = [("A_structured", "oracle_Qstar"), ("A_structured", "potential_exact")]


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--sel-frac", type=float, default=0.3)
    ap.add_argument("--seed-lo", type=int, default=10)
    ap.add_argument("--seed-hi", type=int, default=69)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = list(range(args.seed_lo, args.seed_hi + 1))
    if args.smoke:
        seeds = seeds[:2]
    print(f"CONFIRMATORY (see PREREGISTRATION_QSTAR.md)")
    print(f"cell N={N} noise={noise} delta={cfg.sim.delta}  seeds {seeds[0]}-{seeds[-1]} "
          f"(n={len(seeds)})\n")

    rows = []
    for seed in seeds:
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

        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        terms = compute_terms(trajs, dm, mdp)
        arms = {
            "A_structured": relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda),
            "oracle_Qstar": oracle_rtg(trajs, mdp),
            "potential_exact": build(terms, "model", "oracle", rng),
        }
        out = {"seed": seed}
        for name, rtg in arms.items():
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            sel, test = {}, {}
            for label, tgt in target_grid(rtg):
                vals = per_bin_values(make_dt_policy(m, mdp, tgt), mdp, allb)
                sel[label] = nv_on(vals, sel_bins, v_beh_b, v_opt_b)
                test[label] = nv_on(vals, test_bins, v_beh_b, v_opt_b)
            out[name] = round(test[max(sel, key=sel.get)], 4)
        rows.append(out)
        print(f"seed {seed:3d}: " + "  ".join(f"{k}={out[k]:+.3f}" for k in arms))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "qstar_confirm.csv"), index=False)

    print("\n=== arm means (confirmatory sample) ===")
    for a in ["A_structured", "oracle_Qstar", "potential_exact"]:
        print(f"  {a:16s}: {df[a].mean():+.4f}  (sd {df[a].std():.4f}, n={len(df)})")

    print("\n=== pre-specified comparisons (Holm over 2) ===")
    ps, res = [], []
    for lhs, rhs in PRIMARY:
        d = (df[lhs] - df[rhs]).values
        med, p = M.paired_test(df[lhs].values, df[rhs].values)
        lo, hi = boot_ci(d)
        ps.append(1.0 if p != p else p)
        res.append((lhs, rhs, d.mean(), lo, hi, p, int((d > 0).sum()), len(d)))
    holm = M.holm(np.array(ps))
    exploratory = {"oracle_Qstar": 0.125, "potential_exact": 0.127}
    for (lhs, rhs, m_, lo, hi, p, npos, n), h in zip(res, holm):
        star = "SIGNIFICANT" if h < 0.05 else "not significant"
        print(f"  {lhs} - {rhs}")
        print(f"    effect {m_:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"positive in {npos}/{n} seeds")
        print(f"    p={p:.5f}  holm={h:.5f}  -> {star}")
        print(f"    exploratory (n=10) estimate was {exploratory[rhs]:+.3f}; "
              f"shrinkage {m_ - exploratory[rhs]:+.3f}")

    nsig = int((holm < 0.05).sum())
    print("\n=== pre-registered decision ===")
    if nsig == 2:
        print("  >>> BOTH confirmed. The finding may carry a section, reported at the "
              "REPLICATION effect size above, not the exploratory one.")
    elif nsig == 1:
        ok = [f"{l}-{r}" for (l, r, *_), h in zip(res, holm) if h < 0.05]
        bad = [f"{l}-{r}" for (l, r, *_), h in zip(res, holm) if h >= 0.05]
        print(f"  >>> PARTIAL. Confirmed: {ok}. Failed: {bad}. One paragraph, with the "
              "failed manipulation stated explicitly.")
    else:
        print("  >>> NOT CONFIRMED. Per the pre-registration the finding is permanently "
              "demoted to one descriptive sentence. No further seeds; no substitute "
              "analysis.")


if __name__ == "__main__":
    main()
