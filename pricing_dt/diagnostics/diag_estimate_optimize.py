"""Estimate-then-optimize pricing baseline — the canonical OR/industry method
for pricing, and the comparator the rest of the study lacks.

Instead of relabelling return-to-go and training a sequence model, this is the
classic two-stage pricing pipeline:
  (1) ESTIMATE a demand model from the logged (state, price, demand) data.
  (2) OPTIMIZE: treat the fitted demand as if it were the truth and solve the
      finite-horizon pricing MDP by backward induction under the KNOWN
      reference-price transition (the same transition the structured relabeller
      exploits), giving a tabular policy pi_hat[t, b].
We then evaluate pi_hat's TRUE value on the exact simulator.

Two demand models -> two pricing baselines, run on the SAME hardest cell, seeds,
init distribution and (v_beh, v_opt) anchors as diag_optimism_verdict.py, so the
numbers drop straight into the same comparison table as
A_structured / B_calibrated / Q-DT:
  EtO_structured     : plan with StructuredDemandModel
  EtO_unconstrained  : plan with UnconstrainedDemandModel

Also reports, per seed, the planner's IN-MODEL value (what it THINKS pi_hat is
worth under the fitted demand) vs the realised TRUE value: the gap is the
estimation error propagating through the argmax (the optimizer's-curse / model-
mismatch cost), and ties directly to the project's fidelity-vs-value theme.
"""
import argparse
import numpy as np
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model


def plan_with_dm(dm, mdp, device="cpu"):
    """Backward induction using the FITTED demand's expected reward and the KNOWN
    transition. Returns tabular policy pi[t,b] and the planner's in-model V[t,b]."""
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    Rhat = np.zeros((H, B, A))
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(B):
                s = torch.tensor(mdp.obs(mdp.ref_grid[b], t), dtype=torch.float32,
                                 device=device).unsqueeze(0).repeat(A, 1)
                dem = dm(prices, s).cpu().numpy()
                Rhat[t, b, :] = mdp.prices * dem
    V = np.zeros((H + 1, B))
    pi = np.zeros((H, B), dtype=int)
    for t in reversed(range(H)):
        for b in range(B):
            q = Rhat[t, b, :] + V[t + 1, mdp.N[:, b]]
            pi[t, b] = int(q.argmax())
            V[t, b] = q.max()
    return pi, V


def tabular_policy_fn(pi, mdp):
    def fn(obs):
        b, t = mdp.decode_obs(obs)
        return int(pi[t, b])
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim = 2
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)

        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)

        out = {"seed": seed, "v_opt": round(v_opt, 1)}
        for tag, dm in (("structured", dmA), ("unconstrained", dmB)):
            pi, Vhat = plan_with_dm(dm, mdp)
            v_true, _ = mdp.evaluate_policy_fn(tabular_policy_fn(pi, mdp), init)
            v_inmodel = float(Vhat[0, init].mean())
            out[f"EtO_{tag}"] = round(M.normalised_value(v_true, v_beh, v_opt), 3)
            out[f"inmodel_{tag}"] = round(v_inmodel, 1)
            out[f"truegap_{tag}"] = round(v_inmodel - v_true, 1)  # optimizer's-curse gap
        rows.append(out)
        print(f"seed {seed}: EtO_struct nv={out['EtO_structured']:.3f} "
              f"(in-model {out['inmodel_structured']:.0f}, true-gap {out['truegap_structured']:+.0f}) | "
              f"EtO_uncon nv={out['EtO_unconstrained']:.3f} "
              f"(in-model {out['inmodel_unconstrained']:.0f}, true-gap {out['truegap_unconstrained']:+.0f})")

    import pandas as pd, os
    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "estimate_optimize.csv"), index=False)

    print("\n=== estimate-then-optimize summary (mean over seeds) ===")
    print(f"  EtO_structured      nv : {df.EtO_structured.mean():.3f}")
    print(f"  EtO_unconstrained   nv : {df.EtO_unconstrained.mean():.3f}")
    print(f"  optimizer's-curse gap (in-model - true): "
          f"struct {df.truegap_structured.mean():+.0f}  uncon {df.truegap_unconstrained.mean():+.0f}")

    # pull the DT-family numbers from the optimism verdict for a side-by-side table
    ref = os.path.join(args.outdir, "optimism_verdict.csv")
    if os.path.exists(ref):
        ov = pd.read_csv(ref)
        print("\n=== same cell, same seeds: pricing methods vs offline-RL methods ===")
        table = [("EtO_structured (plan)", df.EtO_structured.mean()),
                 ("EtO_unconstrained (plan)", df.EtO_unconstrained.mean()),
                 ("A_structured DT", ov.A_structured.mean()),
                 ("Q-DT", ov.D_QDT.mean()),
                 ("vanilla DT (lam=0)", ov.vanilla_matchA.mean()),
                 ("B_calibrated DT", ov.B_calibrated.mean())]
        for name, val in sorted(table, key=lambda x: -x[1]):
            print(f"  {name:28s}: {val:.3f}")


if __name__ == "__main__":
    main()
