"""Optimism-verdict diagnostic — the decisive test for a POSITIVE claim.

The structure-verdict diagnostic established that A (structured prior) beats B
(unconstrained, better-calibrated demand fit) in net value, and attributed it to
"target inflation". But the DT standardises its return token, so a naive question
arises: is A's advantage just a MAGNITUDE effect (any sufficiently optimistic
target would do) or a SHAPE effect (the structured prior puts relatively higher
targets on the states where higher value is genuinely achievable)?

This script answers it by adding the missing control: take B's calibrated
relabelled RTG and SCALE it per-seed so its conditioning target (0.95-quantile)
MATCHES A's inflated target. Then train+eval a DT on it.

  - If  A ≈ B_matchA : the advantage is pure optimism MAGNITUDE. Unstructured
    optimism of matched size reproduces it -> structure does not earn its keep ->
    stay with the diagnostic (negative) framing.
  - If  A >  B_matchA : the advantage is the SHAPE of the structured target (where
    the optimism is placed), which a flat rescaling cannot reproduce -> structure
    genuinely helps -> a positive "structured DT is better (via principled
    optimistic target shaping)" claim is honestly available.

Arms (hardest cell N=min, noise=max), per seed, each a full DT train+eval:
  A_structured   : StructuredDemandModel relabel (proposed)
  B_calibrated   : UnconstrainedDemandModel relabel
  B_matchA       : B's RTG globally scaled so its 0.95-target == A's 0.95-target
  vanilla_matchA : logged RTG globally scaled to A's 0.95-target (naive optimism)
  D_QDT          : bootstrapped value relabel (Q-DT)

Plus an anchor-weight (lam) FRONTIER for the structured prior: lam in {0..1}
blends logged (lam=0, == vanilla) -> fully structured (lam=1), tracing net value
against structure-shaping strength (the "tunable optimism knob").

Also reports, per seed, the cross-sectional CORRELATION between each model's
per-start relabelled target and the TRUE per-start achievable optimum (mdp.Vstar):
if A/C correlate with truth better than B despite worse absolute fit, that is the
mechanism — the prior ranks achievable value correctly across states.
"""
import argparse
import numpy as np

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed, _eval_dt
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset, logged_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.dt import train_dt


def _quantile_target(rtg_list, q=0.95):
    return float(np.quantile(np.concatenate(rtg_list), q))


def _scale_to_target(rtg_list, target_match, q=0.95):
    """Globally scale a per-trajectory RTG list so its q-quantile == target_match."""
    cur = _quantile_target(rtg_list, q)
    c = target_match / cur if abs(cur) > 1e-8 else 1.0
    return [r * c for r in rtg_list], c


def _train_eval(rtg_list, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed):
    m = train_dt(D.pack_dt(trajs, rtg_list), obs_dim, A, cfg.model, seed=seed)
    v, tgt = _eval_dt(m, mdp, init, rtg_list)
    return M.normalised_value(v, v_beh, v_opt), tgt


def _per_start_target(rtg_list, trajs):
    """Mean relabelled step-0 RTG grouped by logged start bin -> {bin: mean R_hat_0}."""
    by = {}
    for tr, g in zip(trajs, rtg_list):
        b0 = int(tr.ref_bins[0])
        by.setdefault(b0, []).append(float(g[0]))
    return {b: float(np.mean(v)) for b, v in by.items()}


def _shape_corr(rtg_list, trajs, mdp):
    """Spearman-free Pearson corr between per-start relabel target and the TRUE
    per-start achievable optimum Vstar[0, bin]. High => the relabel ranks which
    starts have higher achievable value correctly (the structural-knowledge claim)."""
    ps = _per_start_target(rtg_list, trajs)
    bins = sorted(ps)
    x = np.array([ps[b] for b in bins])
    y = np.array([float(mdp.Vstar[0, b]) for b in bins])
    if x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny config to validate the pipeline")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    lams = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}\n")

    verdict_rows, frontier_rows = [], []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

        # --- A: structured prior ---
        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        rtgA = relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda)
        tgtA = _quantile_target(rtgA)
        nvA, _ = _train_eval(rtgA, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        # --- B: unconstrained (calibrated) ---
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)
        rtgB = relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda)
        tgtB = _quantile_target(rtgB)
        nvB, _ = _train_eval(rtgB, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        # --- B_matchA: B scaled so its target == A's (decisive control) ---
        rtgBm, cB = _scale_to_target(rtgB, tgtA)
        nvBm, _ = _train_eval(rtgBm, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        # --- vanilla_matchA: logged RTG scaled to A's target (naive optimism) ---
        rtgV = logged_rtg(trajs)
        rtgVm, cV = _scale_to_target(rtgV, tgtA)
        nvVm, _ = _train_eval(rtgVm, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        # --- D: bootstrapped value (Q-DT) ---
        rtgD, _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
        nvD, _ = _train_eval(rtgD, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)

        corrA = _shape_corr(rtgA, trajs, mdp)
        corrB = _shape_corr(rtgB, trajs, mdp)
        verdict_rows.append(dict(seed=seed, v_opt=round(v_opt, 1),
                                 A_structured=round(nvA, 3), B_calibrated=round(nvB, 3),
                                 B_matchA=round(nvBm, 3), vanilla_matchA=round(nvVm, 3),
                                 D_QDT=round(nvD, 3),
                                 tgtA=round(tgtA, 1), tgtB=round(tgtB, 1),
                                 scaleB=round(cB, 2), scaleV=round(cV, 2),
                                 shapecorr_A=round(corrA, 3), shapecorr_B=round(corrB, 3)))
        print(f"seed {seed}: A={nvA:.3f} B={nvB:.3f} B_matchA={nvBm:.3f} "
              f"van_matchA={nvVm:.3f} D={nvD:.3f} | shapecorr A={corrA:.2f} B={corrB:.2f}")

        # --- lam frontier (structured prior) ---
        for lam in lams:
            rtg = relabel_dataset(trajs, dmA, mdp, lam)
            nv, _ = _train_eval(rtg, trajs, mdp, init, v_beh, v_opt, cfg, obs_dim, A, seed)
            frontier_rows.append(dict(seed=seed, lam=lam, nv=round(nv, 3)))

    import pandas as pd, os
    vdf = pd.DataFrame(verdict_rows)
    fdf = pd.DataFrame(frontier_rows)
    os.makedirs(args.outdir, exist_ok=True)
    vdf.to_csv(os.path.join(args.outdir, "optimism_verdict.csv"), index=False)
    fdf.to_csv(os.path.join(args.outdir, "optimism_frontier.csv"), index=False)

    print("\n=== verdict summary (mean over seeds) ===")
    for col in ["A_structured", "B_calibrated", "B_matchA", "vanilla_matchA", "D_QDT"]:
        print(f"  {col:16s}: {vdf[col].mean():.3f}")
    # decisive paired test: structure vs matched-magnitude unstructured optimism
    med, p = M.paired_test(vdf["A_structured"].values, vdf["B_matchA"].values)
    print(f"\n  DECISIVE  A_structured - B_matchA: mean "
          f"{vdf['A_structured'].mean() - vdf['B_matchA'].mean():+.3f}, "
          f"paired median {med:+.3f}, Wilcoxon p={p:.4f}")
    print(f"  shape-corr-with-truth   A={vdf['shapecorr_A'].mean():.3f}  "
          f"B={vdf['shapecorr_B'].mean():.3f}")
    verdict = ("SHAPE (structure earns its keep -> positive claim available)"
               if (vdf['A_structured'].mean() - vdf['B_matchA'].mean()) > 0.05 and (p < 0.1 or p != p)
               else "MAGNITUDE (matched optimism reproduces A -> stay diagnostic)")
    print(f"\n  >>> VERDICT: {verdict}")
    print("\n=== lam frontier (mean nv over seeds) ===")
    for lam in lams:
        sub = fdf[fdf.lam == lam]
        print(f"  lam={lam:.2f}: nv={sub['nv'].mean():.3f}")


if __name__ == "__main__":
    main()
