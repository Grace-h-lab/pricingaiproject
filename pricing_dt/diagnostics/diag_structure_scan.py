"""Structure-verdict scan across the reference-price strength delta.

For each delta in {1,2,3,4} and several seeds, at the E2-AB hardest cell, fit the
structured (A) and unconstrained (B) demand models and measure -- with NO DT
training:

  fit_logmse_A/B : in-sample log-space demand-fit error (lower = fits logged
                   demand better).
  target_A/B     : the 0.95-quantile relabelled return-to-go (the DT conditioning
                   target produced by that demand model).
  v_opt          : true achievable optimum (exact rollout) from logged starts.
  inflation_A/B  : target / v_opt  (1.0 = calibrated; >1 = optimistic inflation).

Outputs:
  results/structure_verdict_scan.csv
  results/figures/structure_inflation_vs_delta.png   (inflation grows with delta)
  results/figures/structure_fit_vs_target.png        (B fits better yet A's target
                                                       is inflated, at delta=3)
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset


def _fit_logmse(model, trajs, mdp):
    S, P, Dm = [], [], []
    for tr in trajs:
        S.append(tr.obs)
        P.append(mdp.prices[tr.actions])
        Dm.append(tr.rewards / np.maximum(mdp.prices[tr.actions], 1e-6))
    S = torch.tensor(np.concatenate(S), dtype=torch.float32)
    P = torch.tensor(np.concatenate(P), dtype=torch.float32)
    Dm = torch.tensor(np.concatenate(Dm), dtype=torch.float32)
    logD = torch.log(torch.clamp(Dm, min=1e-3))
    model.eval()
    with torch.no_grad():
        logp = torch.log(torch.clamp(model(P, S), min=1e-3))
        return float(((logp - logD) ** 2).mean())


def run_scan(deltas=(1.0, 2.0, 3.0, 4.0), n_seeds=5):
    cfg = C.full()
    obs_dim = 2
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    rows = []
    for delta in deltas:
        for seed in cfg.exp.seeds[:n_seeds]:
            _seed(seed)
            mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise, "delta": delta}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

            dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
            fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
            dmB = UnconstrainedDemandModel(obs_dim)
            fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)

            mseA, mseB = _fit_logmse(dmA, trajs, mdp), _fit_logmse(dmB, trajs, mdp)
            tgtA = float(np.quantile(np.concatenate(relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda)), 0.95))
            tgtB = float(np.quantile(np.concatenate(relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda)), 0.95))
            rows.append(dict(delta=delta, seed=seed, v_opt=v_opt,
                             fit_logmse_A=mseA, fit_logmse_B=mseB,
                             target_A=tgtA, target_B=tgtB,
                             inflation_A=tgtA / v_opt, inflation_B=tgtB / v_opt))
            print(f"delta={delta} seed={seed}: v_opt={v_opt:6.1f} "
                  f"mseA={mseA:.3f} mseB={mseB:.3f} inflA={tgtA/v_opt:.2f}x inflB={tgtB/v_opt:.2f}x")
    return rows


def agg(rows, key):
    """mean per delta of rows[*][key]."""
    out = {}
    for r in rows:
        out.setdefault(r["delta"], []).append(r[key])
    return {d: float(np.mean(v)) for d, v in sorted(out.items())}


def main(replot_only=False, outdir="results", smoke=False):
    import pandas as pd
    figdir = os.path.join(outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    csv = os.path.join(outdir, "structure_verdict_scan.csv")
    if replot_only and os.path.exists(csv):
        df = pd.read_csv(csv)
        rows = df.to_dict("records")
        print("replotting from", csv)
    else:
        rows = run_scan(deltas=(1.0, 3.0), n_seeds=1) if smoke else run_scan()
        df = pd.DataFrame(rows)
        df.to_csv(csv, index=False)
        print("\nwrote", csv)

    # --- item 3: inflation factor vs delta ---
    iA, iB = agg(rows, "inflation_A"), agg(rows, "inflation_B")
    ds = sorted(iA)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(ds, [iA[d] for d in ds], "o-", color="#c0392b", label="A: structured prior")
    ax.plot(ds, [iB[d] for d in ds], "s-", color="#2980b9", label="B: unconstrained")
    ax.axhline(1.0, ls="--", color="gray", lw=1, label="true optimum (calibrated)")
    for d in ds:
        ax.annotate(f"{iA[d]:.2f}", (d, iA[d]), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8, color="#c0392b")
        ax.annotate(f"{iB[d]:.2f}", (d, iB[d]), textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=8, color="#2980b9")
    ax.set_xlabel("reference-price strength  $\\delta$")
    ax.set_ylabel("relabel target / true optimum")
    ax.set_title("In the sequential regime ($\\delta\\geq3$) the structured prior\ninflates its target while the unconstrained model self-calibrates")
    ax.set_xticks(ds); ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    f1 = os.path.join(figdir, "structure_inflation_vs_delta.png")
    fig.savefig(f1, dpi=150); plt.close(fig)

    # --- item 2: fit error vs target inflation at delta=3 ---
    d3 = df[df["delta"] == 3.0]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.scatter(d3["fit_logmse_A"], d3["inflation_A"], color="#c0392b", s=45, label="A: structured prior")
    ax.scatter(d3["fit_logmse_B"], d3["inflation_B"], color="#2980b9", s=45, marker="s", label="B: unconstrained")
    ax.scatter([d3["fit_logmse_A"].mean()], [d3["inflation_A"].mean()], color="#c0392b",
               s=220, marker="*", edgecolor="k", zorder=5)
    ax.scatter([d3["fit_logmse_B"].mean()], [d3["inflation_B"].mean()], color="#2980b9",
               s=220, marker="*", edgecolor="k", zorder=5)
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("demand-fit error  (log-space MSE, lower = better fit)")
    ax.set_ylabel("relabel target / true optimum")
    ax.set_title("$\\delta=3$: B fits demand better, yet A inflates the target")
    ax.legend(fontsize=8); fig.tight_layout()
    f2 = os.path.join(figdir, "structure_fit_vs_target.png")
    fig.savefig(f2, dpi=150); plt.close(fig)

    print("wrote", f1)
    print("wrote", f2)
    print("\n=== summary (mean over seeds) ===")
    for d in ds:
        print(f"delta={d}: inflation A={iA[d]:.2f}x  B={iB[d]:.2f}x")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--replot-only", action="store_true")
    args = ap.parse_args()
    main(replot_only=args.replot_only, outdir=args.outdir, smoke=args.smoke)
