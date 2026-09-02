"""Sparse-coverage probe: is there a regime where structure GENUINELY helps?

The triangle (diag_structure_scan.py) shows that at full price coverage A>B in
value is a target-inflation artefact (B fits demand better, A inflates the target).
The original motivation for structure, though, was sparse coverage: where the
logged data exercises only K distinct prices, the unconstrained model B has no
data to pin the demand curve in the gaps and may extrapolate wildly, while the
bounded-elasticity A is forced to stay monotone/sane there.

This probe drives price coverage from full (K=11) down to K=2 and measures, against
the simulator's GROUND-TRUTH demand, where each model is right:

  incov_logmse_A/B  : log-demand error vs truth at COVERED prices (seen).      B≤A expected.
  oocov_logmse_A/B  : log-demand error vs truth at UNCOVERED prices (extrap).  HYPOTHESIS: A<B as K shrinks.
  inflation_A/B     : relabel target (0.95-q) / true optimum.                  genuine benefit => A->1, B-> away.

Decision rule (genuine, not artefact): structure genuinely helps in a cell iff the
out-of-coverage ordering REVERSES (A closer to truth than B at unseen prices) AND
A's target is better calibrated than B's. Mere A>B in normalised value is NOT used
here (it already holds at full coverage and is the artefact).

Run:  python -m pricing_dt.diagnostics.diag_sparse_coverage --outdir results
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core.data import Trajectory
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model
from pricing_dt.core.relabel import relabel_dataset


def reroll_sparse(tr, mdp, allowed_idx, noise, rng):
    """Snap a trajectory's actions to the K allowed price indices and re-simulate
    rewards/refs/obs consistently, yielding a valid episode with sparse price coverage."""
    H = mdp.H
    ref = mdp.ref_grid[int(tr.ref_bins[0])]
    obs, rbins, acts, rews = [], [], [], []
    for t in range(H):
        a0 = int(tr.actions[t])
        a = int(allowed_idx[np.argmin(np.abs(allowed_idx - a0))])   # snap to nearest allowed
        b = mdp.ref_to_bin(ref)
        obs.append(mdp.obs(ref, t)); rbins.append(b); acts.append(a)
        rews.append(mdp.sample_reward(mdp.prices[a], ref, noise))
        ref = mdp.ref_grid[mdp.N[a, b]]
    return Trajectory(np.array(obs, np.float32), np.array(rbins, int),
                      np.array(acts, int), np.array(rews, np.float32))


def truth_fit_error(model, trajs, mdp, allowed_idx):
    """Log-demand MSE vs the simulator's true expected demand, at every grid price
    over the states visited in `trajs`, split into covered / uncovered prices."""
    # unique visited (ref_bin, t) states
    seen = set()
    for tr in trajs:
        for t in range(mdp.H):
            seen.add((int(tr.ref_bins[t]), t))
    allowed = set(int(a) for a in allowed_idx)
    nP = mdp.cfg.n_prices
    cov_e, ooc_e = [], []
    model.eval()
    with torch.no_grad():
        for (b, t) in seen:
            ref = mdp.ref_grid[b]
            s = torch.tensor(mdp.obs(ref, t), dtype=torch.float32).unsqueeze(0).repeat(nP, 1)
            p = torch.tensor(mdp.prices, dtype=torch.float32)
            pred = model(p, s).cpu().numpy()
            true = mdp.expected_demand(mdp.prices, ref)
            err = (np.log(np.clip(pred, 1e-3, None)) - np.log(np.clip(true, 1e-3, None))) ** 2
            for a in range(nP):
                (cov_e if a in allowed else ooc_e).append(err[a])
    return (float(np.mean(cov_e)) if cov_e else float("nan"),
            float(np.mean(ooc_e)) if ooc_e else float("nan"))


def run(Ks=(11, 5, 3, 2), Ns=(100, 50), n_seeds=5):
    cfg = C.full()
    obs_dim = 2
    noise = max(cfg.exp.noise_levels)
    nP = cfg.sim.n_prices
    rows = []
    for N in Ns:
        for K in Ks:
            allowed_idx = np.unique(np.round(np.linspace(0, nP - 1, K)).astype(int))
            for seed in cfg.exp.seeds[:n_seeds]:
                _seed(seed)
                mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise, "delta": 3.0}, seed=seed)
                rng = np.random.default_rng(1000 + seed)
                base = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
                trajs = [reroll_sparse(tr, mdp, allowed_idx, noise, rng) for tr in base]

                dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
                fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
                dmB = UnconstrainedDemandModel(obs_dim)
                fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)

                covA, oocA = truth_fit_error(dmA, trajs, mdp, allowed_idx)
                covB, oocB = truth_fit_error(dmB, trajs, mdp, allowed_idx)
                tgtA = float(np.quantile(np.concatenate(relabel_dataset(trajs, dmA, mdp, cfg.model.relabel_lambda)), 0.95))
                tgtB = float(np.quantile(np.concatenate(relabel_dataset(trajs, dmB, mdp, cfg.model.relabel_lambda)), 0.95))
                rows.append(dict(N=N, K=K, seed=seed, v_opt=v_opt,
                                 incov_logmse_A=covA, incov_logmse_B=covB,
                                 oocov_logmse_A=oocA, oocov_logmse_B=oocB,
                                 inflation_A=tgtA / v_opt, inflation_B=tgtB / v_opt))
                print(f"N={N} K={K:2d} seed={seed}: ooc A={oocA:.3f} B={oocB:.3f} "
                      f"({'A<B genuine' if oocA < oocB else 'B<A artefact'}) | "
                      f"infl A={tgtA/v_opt:.2f} B={tgtB/v_opt:.2f}")
    return rows


def main():
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir, figdir = args.outdir, os.path.join(args.outdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    rows = run(Ks=(11, 3), Ns=(50,), n_seeds=1) if args.smoke else run()
    df = pd.DataFrame(rows)
    csv = os.path.join(outdir, "sparse_coverage_scan.csv")
    df.to_csv(csv, index=False)
    print("\nwrote", csv)

    print("\n=== summary by (N,K): mean over seeds ===")
    print(f"{'N':>4} {'K':>3} | {'ooc_A':>6} {'ooc_B':>6} {'A<B?':>5} | {'infl_A':>6} {'infl_B':>6}")
    g = df.groupby(["N", "K"]).mean(numeric_only=True).reset_index()
    for _, r in g.iterrows():
        flag = "A<B" if r.oocov_logmse_A < r.oocov_logmse_B else "B<A"
        print(f"{int(r.N):>4} {int(r.K):>3} | {r.oocov_logmse_A:6.3f} {r.oocov_logmse_B:6.3f} "
              f"{flag:>5} | {r.inflation_A:6.2f} {r.inflation_B:6.2f}")

    # figure: out-of-coverage demand error vs K (N=100), A and B
    for N in sorted(df.N.unique()):
        sub = g[g.N == N].sort_values("K")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
        ax1.plot(sub.K, sub.oocov_logmse_A, "o-", color="#c0392b", label="A: structured")
        ax1.plot(sub.K, sub.oocov_logmse_B, "s-", color="#2980b9", label="B: unconstrained")
        ax1.set_xlabel("price coverage K (distinct prices)"); ax1.set_ylabel("out-of-coverage demand error vs truth")
        ax1.set_title(f"N={N}: extrapolation error at unseen prices"); ax1.invert_xaxis()
        ax1.legend(fontsize=8)
        ax2.plot(sub.K, sub.inflation_A, "o-", color="#c0392b", label="A: structured")
        ax2.plot(sub.K, sub.inflation_B, "s-", color="#2980b9", label="B: unconstrained")
        ax2.axhline(1.0, ls="--", color="gray", lw=1)
        ax2.set_xlabel("price coverage K (distinct prices)"); ax2.set_ylabel("relabel target / true optimum")
        ax2.set_title(f"N={N}: target calibration"); ax2.invert_xaxis(); ax2.legend(fontsize=8)
        fig.tight_layout()
        f = os.path.join(figdir, f"sparse_coverage_N{N}.png")
        fig.savefig(f, dpi=150); plt.close(fig)
        print("wrote", f)


if __name__ == "__main__":
    main()
