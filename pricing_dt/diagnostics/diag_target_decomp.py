"""Target decomposition — WHICH PART of the structured target does the work?

The structured relabel target splits exactly into two terms with different jobs:

    R_hat_t  =  r_hat(s_t, a_t)               <- ACTION term: discriminates the
                                                 logged action against alternatives
                                                 at the same state
              + sum_{k>t} max_p r_hat(s_k,p)  <- POTENTIAL term: a state-indexed
                                                 aspiration field ("how good is it
                                                 to be here"), nearly independent
                                                 of the logged action

This matters because the existing diagnostics have eliminated every obvious
explanation of the structured prior's advantage and left "shape" as a residual:

  - not magnitude   (B rescaled to A's target recovers only ~0.08 of ~0.34)
  - not accuracy    (A's per-state action-rank corr vs Qstar is NEGATIVE, -0.185,
                     while the better-ranking B loses)
  - not denoising   (A's targets are HIGHER variance than B's)
  - not cross-state ranking either: A's per-start target correlation with Vstar is
                     0.92 while B's is 0.99 — B is better on that too, and loses

"Shape" is not a mechanism until it is localised. This script localises it by
building each term from an independent source and crossing them:

    source in {model, oracle, mean, shuffle}
      model   : the fitted structured demand model (as in the method)
      oracle  : the exact truth  (R[a_t,b_t] for the action term,
                                  Vstar[t+1, N[a_t,b_t]] for the potential term)
      mean    : the term's own average, i.e. the term is REMOVED while leaving the
                target's overall level untouched
      shuffle : the term's values permuted across the dataset — same marginal
                distribution, destroyed correspondence. This is the control that
                separates "this term carries information" from "this term carries
                variance".

Key arms. `model x model` reproduces the method (~0.670). `mean x model` deletes
action discrimination; `model x mean` deletes the aspiration field. Whichever
deletion costs more is where the effect lives. The oracle rows say whether that
term wants to be ACCURATE or merely structured: if `oracle x model` <
`model x model`, the action term is actively better off wrong.

Implements: the target decomposition of §4.4, reported as Table 4.7 and Figure 4.4.
"""
import argparse
import itertools
import numpy as np
import pandas as pd
import os
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed, _eval_dt
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.dt import train_dt

SOURCES = ["model", "oracle", "mean", "shuffle"]


def compute_terms(trajs, dm, mdp, device="cpu"):
    """Both terms of the structured target, per trajectory, for every source.

    Returns dict src -> (action_terms, potential_terms), each a list of [H] arrays.
    The 'model' terms reproduce relabel.achievable_rtg exactly when summed.
    """
    H = mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    if not hasattr(mdp, "Qstar"):
        mdp.solve_optimal()

    act_m, pot_m, act_o, pot_o = [], [], [], []
    dm.eval()
    with torch.no_grad():
        for tr in trajs:
            am = np.zeros(H, np.float32); pm = np.zeros(H, np.float32)
            ao = np.zeros(H, np.float32); po = np.zeros(H, np.float32)
            for t in range(H):
                b_t = int(tr.ref_bins[t])
                ref = mdp.ref_grid[b_t]
                a_t = int(tr.actions[t])
                # --- action term ---
                s_t = torch.tensor(mdp.obs(ref, t), dtype=torch.float32,
                                   device=device).unsqueeze(0)
                am[t] = float(prices[a_t] * dm(prices[a_t].unsqueeze(0), s_t).squeeze())
                ao[t] = float(mdp.R[a_t, b_t])
                # --- potential term: greedy model roll-forward after the logged action ---
                r2 = mdp.ref_grid[mdp.N[a_t, b_t]]
                tot = 0.0
                for k in range(t + 1, H):
                    s = torch.tensor(mdp.obs(r2, k), dtype=torch.float32, device=device)
                    rev = prices * dm(prices, s.unsqueeze(0).repeat(len(prices), 1))
                    a_b = int(torch.argmax(rev).item())
                    tot += float(rev[a_b].item())
                    r2 = mdp.ref_grid[mdp.N[a_b, mdp.ref_to_bin(r2)]]
                pm[t] = tot
                po[t] = float(mdp.Vstar[t + 1, mdp.N[a_t, b_t]]) if t + 1 <= H else 0.0
            act_m.append(am); pot_m.append(pm); act_o.append(ao); pot_o.append(po)
    return dict(model=(act_m, pot_m), oracle=(act_o, pot_o))


def _mean_of(terms):
    """Replace every entry by the global mean: removes the term's variation while
    preserving the target's level."""
    mu = float(np.mean(np.concatenate(terms)))
    return [np.full_like(x, mu) for x in terms]


def _shuffle_of(terms, rng):
    """Permute all entries across the whole dataset: identical marginal, destroyed
    correspondence with (state, action)."""
    flat = np.concatenate(terms)
    perm = rng.permutation(flat.size)
    shuffled = flat[perm]
    out, i = [], 0
    for x in terms:
        out.append(shuffled[i:i + x.size].reshape(x.shape).astype(np.float32)); i += x.size
    return out


def build(terms, src_a, src_p, rng):
    (am, pm), (ao, po) = terms["model"], terms["oracle"]
    a = {"model": am, "oracle": ao, "mean": _mean_of(am), "shuffle": _shuffle_of(am, rng)}[src_a]
    p = {"model": pm, "oracle": po, "mean": _mean_of(pm), "shuffle": _shuffle_of(pm, rng)}[src_p]
    return [x + y for x, y in zip(a, p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    cells = [(a, p) for a, p in itertools.product(SOURCES, SOURCES)]
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}")
    print(f"{len(cells)} action x potential cells\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        rng = np.random.default_rng(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        terms = compute_terms(trajs, dm, mdp)

        line = {}
        for src_a, src_p in cells:
            rtg = build(terms, src_a, src_p, rng)
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            v, tgt = _eval_dt(m, mdp, init, rtg)
            nv = M.normalised_value(v, v_beh, v_opt)
            # Provenance: confirm the arms really do differ in their target column.
            # (Identical nv across action sources is a substantive finding only if the
            # inputs differ; if these stats coincide the ablation is not biting.)
            flat = np.concatenate(rtg)
            rows.append(dict(seed=seed, action=src_a, potential=src_p, nv=round(nv, 3),
                             rtg_mean=round(float(flat.mean()), 1),
                             rtg_std=round(float(flat.std()), 1),
                             eval_target=round(float(tgt), 1)))
            line[f"{src_a[:3]}x{src_p[:3]}"] = round(nv, 3)
        print(f"seed {seed}: " + " ".join(f"{k}={v:+.2f}" for k, v in line.items()))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "target_decomp.csv"), index=False)
    piv = df.groupby(["action", "potential"]).nv.mean().unstack().round(3)
    piv = piv.reindex(index=SOURCES, columns=SOURCES)
    piv.to_csv(os.path.join(args.outdir, "target_decomp_matrix.csv"))

    print("\n=== mean normalised value:  rows = ACTION term, cols = POTENTIAL term ===")
    print(piv.to_string())

    ref = piv.loc["model", "model"]
    print(f"\n  reference (model x model, the method) : {ref:+.3f}")
    print(f"  delete the ACTION term    (mean x model) : {piv.loc['mean','model']:+.3f} "
          f"({piv.loc['mean','model'] - ref:+.3f})")
    print(f"  delete the POTENTIAL term (model x mean) : {piv.loc['model','mean']:+.3f} "
          f"({piv.loc['model','mean'] - ref:+.3f})")
    print(f"  delete BOTH               (mean x mean)  : {piv.loc['mean','mean']:+.3f} "
          f"({piv.loc['mean','mean'] - ref:+.3f})")
    print(f"  make the ACTION term exact   (oracle x model) : {piv.loc['oracle','model']:+.3f} "
          f"({piv.loc['oracle','model'] - ref:+.3f})")
    print(f"  make the POTENTIAL term exact(model x oracle) : {piv.loc['model','oracle']:+.3f} "
          f"({piv.loc['model','oracle'] - ref:+.3f})")
    print(f"  make BOTH exact              (oracle x oracle): {piv.loc['oracle','oracle']:+.3f} "
          f"({piv.loc['oracle','oracle'] - ref:+.3f})")

    d_act = ref - piv.loc["mean", "model"]
    d_pot = ref - piv.loc["model", "mean"]
    sub = df[(df.action == "model") & (df.potential == "model")].nv.values
    for tag, (sa, sp) in (("action", ("mean", "model")), ("potential", ("model", "mean"))):
        o = df[(df.action == sa) & (df.potential == sp)].nv.values
        med, p = M.paired_test(sub, o)
        print(f"  paired test, deleting the {tag} term: median {med:+.3f}, p={p:.4f}")
    print(f"\n  >>> the effect sits mainly in the "
          f"{'POTENTIAL (aspiration field)' if d_pot > d_act else 'ACTION (discrimination)'} term "
          f"(cost of deleting: action {d_act:+.3f} vs potential {d_pot:+.3f})")


if __name__ == "__main__":
    main()
