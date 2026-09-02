"""Trust-region width — the interventional test of "imitation clips model optimism".

The mechanism claim is that the goal channel is safe because the DT's imitation
objective confines actions to the logged support, so the demand model's optimism
can only ever be an *aspiration* and never a *choice*. Until now that was inferred
from a two-point comparison (structured relabel 0.670 vs estimate-then-optimize
-4.543). This script turns it into a dose-response curve by dismantling the trust
region continuously, with everything else held fixed.

THE KNOB. At each state, rank actions by the DT's own conditional probability and
keep only the top m; choose among those by the demand model's planning value:

    pi_m(s) = argmax_{a in TopM_m(pi_DT(.|s))}  Qhat_model(s, a)

    m = 1  -> the DT's greedy action                      == structured DT (channel 3)
    m = A  -> unrestricted argmax of the model's Q surface == EtO planner   (channel 1)

So a single axis interpolates the two channels using the SAME fitted model, and
both endpoints are already known independently (0.670 and -4.543) — the curve has
to land on them, which makes it self-checking. (Ranking by DT probability rather
than thresholding it avoids having to calibrate a threshold against an arbitrarily
peaked softmax; m is also directly interpretable as trust-region width in actions.)

WHY A TEMPERATURE SWEEP WOULD NOT DO. Raising the DT's sampling temperature moves
the policy toward UNIFORM, not toward the model's optimistic region. Any decline
it produces is confounded with "random actions are worse". The top-m relaxation is
*directed*: every action it admits is one the model actively prefers, so a decline
is attributable to model error being cashed in, which is the claim under test.

THE CONTROL (this is what makes it an experiment rather than a demonstration).
The same sweep is run with the demand model replaced by the EXACT Q*. If the
collapse were about trust-region width per se, both would fall. The prediction is
the opposite: with an accurate model, widening the trust region IMPROVES value
monotonically to the optimum (m=A is then the optimal policy, nv=1.0). Damage from
widening is therefore attributable to model error, not to leaving imitation.

READ-OUTS per (model, m): true value, the planner's IN-MODEL value (what the model
thinks the policy is worth), the optimizer's-curse gap between them, and the mean
empirical logged frequency of the chosen actions (how far outside the data support
the policy actually went).

Implements: the trust-region sweep of Equation (3.1), reported in §4.3 and drawn as
Figures 4.1 and 4.2.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: make_topm_policy(), oracle_q_surface(), main().
#
# The interventional control at the centre of the dissertation: one knob admits
# progressively more actions, and the identical sweep is repeated with the exact Q*
# instead of the fitted model.
#
# Implements or follows:
#   - Smith, J.E. and Winkler, R.L. (2006) 'The Optimizer's Curse', Management Science,
#     52(3).
#   - Elmachtoub, A.N. and Grigas, P. (2022) 'Smart Predict, then Optimize', Management
#     Science, 68(1).
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
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
from pricing_dt.core.relabel import relabel_dataset
from pricing_dt.core.dt import train_dt, _step_logits


# ----------------------------------------------------------------------------
def model_q_surface(dm, mdp, device="cpu"):
    """Backward induction on the FITTED demand model.
    Returns Rhat[t,b,a], V[t+1,b], Q[t,b,a] = Rhat + V(next). argmax_a Q[t,b] is
    exactly the estimate-then-optimize policy (diag_estimate_optimize.plan_with_dm)."""
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    Rhat = np.zeros((H, B, A))
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b in range(B):
                s = torch.tensor(mdp.obs(mdp.ref_grid[b], t), dtype=torch.float32,
                                 device=device).unsqueeze(0).repeat(A, 1)
                Rhat[t, b, :] = mdp.prices * dm(prices, s).cpu().numpy()
    V = np.zeros((H + 1, B))
    Q = np.zeros((H, B, A))
    for t in reversed(range(H)):
        for b in range(B):
            Q[t, b] = Rhat[t, b, :] + V[t + 1, mdp.N[:, b]]
            V[t, b] = Q[t, b].max()
    return Rhat, V, Q


def oracle_q_surface(mdp):
    """The same object computed exactly: Rhat -> R, Q -> Qstar."""
    if not hasattr(mdp, "Qstar"):
        mdp.solve_optimal()
    H, B, A = mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices
    Rhat = np.zeros((H, B, A))
    for t in range(H):
        Rhat[t] = mdp.R.T
    return Rhat, mdp.Vstar, mdp.Qstar


# ----------------------------------------------------------------------------
def make_topm_policy(model, mdp, target_return, Q, m, device="cpu"):
    """DT-restricted model-greedy policy: the trust region Top_m of Equation (3.1).

    Ranks by the DT's own probability, not by logged counts -- the logged-count mask is
    Supp_k, in `core.dt._supported_actions`. Keeps the DT's history bookkeeping identical
    to dt.make_dt_policy so m=1 reproduces the structured DT exactly.
    """
    st = {"rtg": [], "s": [], "a": [], "t": []}

    def policy_fn(obs):
        t = int(round(obs[1] * (mdp.H - 1)))
        if t == 0:
            st.update(rtg=[float(target_return)], s=[], a=[], t=[])
        else:
            st["rtg"].append(st["rtg"][-1])
        st["s"].append(obs.astype(np.float32))
        st["t"].append(t)
        logits = _step_logits(model, st["rtg"], st["s"], st["a"] + [0], st["t"], device)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        ref = mdp.cfg.p_min + obs[0] * (mdp.cfg.p_max - mdp.cfg.p_min)
        b = mdp.ref_to_bin(ref)
        if m <= 1:
            a = int(np.argmax(p))                      # DT greedy == structured DT
        else:
            allowed = np.argsort(-p)[:m]               # top-m by DT probability
            a = int(allowed[np.argmax(Q[t, b, allowed])])
        st["a"].append(a)
        st["rtg"][-1] = st["rtg"][-1] - float(mdp.R[a, b])
        return a

    return policy_fn


def eval_cached(policy_fn, mdp, init_bins):
    """Exact expected evaluation, memoised by start bin.

    The rollout is deterministic given a start bin (expected-reward transitions,
    greedy policy), and `init_bins` samples only ~n_ref_bins distinct values with
    multiplicity, so evaluating each distinct start once and re-weighting is exact
    and ~12x cheaper. Also returns the realised (t, b, a) visits for the support
    and in-model diagnostics."""
    uniq, counts = np.unique(np.asarray(init_bins), return_counts=True)
    vals, visits = {}, {}
    for b0 in uniq:
        ref = mdp.ref_grid[b0]
        total, path = 0.0, []
        for t in range(mdp.H):
            b = mdp.ref_to_bin(ref)
            a = int(policy_fn(mdp.obs(ref, t)))
            path.append((t, b, a))
            total += mdp.R[a, b]
            ref = mdp.ref_grid[mdp.N[a, b]]
        vals[int(b0)] = float(total)
        visits[int(b0)] = path
    v = float(np.average([vals[int(b)] for b in uniq], weights=counts))
    return v, vals, visits, uniq, counts


def inmodel_value(visits, uniq, counts, Rhat):
    """What the MODEL thinks the executed policy is worth (sum of its own expected
    rewards along the realised path). At m=A this is the planner's V, i.e. the
    value that estimate-then-optimize believes its own policy achieves."""
    per = [sum(Rhat[t, b, a] for (t, b, a) in visits[int(b0)]) for b0 in uniq]
    return float(np.average(per, weights=counts))


def empirical_support(trajs, mdp):
    """emp[b, a]: logged empirical action frequency at each reference bin."""
    B, A = mdp.cfg.n_ref_bins, mdp.cfg.n_prices
    cnt = np.zeros((B, A))
    for tr in trajs:
        for t in range(len(tr.actions)):
            cnt[int(tr.ref_bins[t]), int(tr.actions[t])] += 1
    tot = cnt.sum(1, keepdims=True)
    return np.divide(cnt, np.maximum(tot, 1e-9))


def support_mass(visits, uniq, counts, emp):
    per = [np.mean([emp[b, a] for (_, b, a) in visits[int(b0)]]) for b0 in uniq]
    return float(np.average(per, weights=counts))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    ms = list(range(1, A + 1))
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}")
    print(f"trust-region widths m={ms}  (m=1 -> structured DT, m={A} -> EtO planner)\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        emp = empirical_support(trajs, mdp)

        # the structured DT of the main result: model, relabel, DT (trained ONCE)
        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        rtg = relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)
        dtm = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
        target = float(np.quantile(np.concatenate(rtg), 0.95))

        surfaces = {"structured": model_q_surface(dm, mdp), "oracle": oracle_q_surface(mdp)}

        for mdl, (Rhat, _V, Q) in surfaces.items():
            for m in ms:
                pol = make_topm_policy(dtm, mdp, target, Q, m)
                v, _vals, visits, uniq, counts = eval_cached(pol, mdp, init)
                nv = M.normalised_value(v, v_beh, v_opt)
                vin = inmodel_value(visits, uniq, counts, Rhat)
                rows.append(dict(seed=seed, model=mdl, m=m,
                                 nv=round(nv, 3), v_true=round(v, 1),
                                 v_inmodel=round(vin, 1),
                                 curse_gap=round(vin - v, 1),
                                 support=round(support_mass(visits, uniq, counts, emp), 4)))
            sub = [r for r in rows if r["seed"] == seed and r["model"] == mdl]
            print(f"seed {seed} [{mdl:10s}] nv by m: " +
                  " ".join(f"{r['nv']:+.2f}" for r in sub))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "trust_region_scan.csv"), index=False)
    summ = df.groupby(["model", "m"]).agg(
        nv=("nv", "mean"), nv_sd=("nv", "std"),
        v_inmodel=("v_inmodel", "mean"), curse_gap=("curse_gap", "mean"),
        support=("support", "mean")).round(3).reset_index()
    summ.to_csv(os.path.join(args.outdir, "trust_region_summary.csv"), index=False)

    print("\n=== trust-region sweep (mean over seeds) ===")
    for mdl in ["structured", "oracle"]:
        s = summ[summ.model == mdl]
        print(f"\n  [{mdl}]")
        print(f"  {'m':>3} {'nv':>8} {'in-model':>10} {'curse gap':>10} {'support':>9}")
        for _, r in s.iterrows():
            print(f"  {int(r.m):>3} {r.nv:>+8.3f} {r.v_inmodel:>10.1f} "
                  f"{r.curse_gap:>+10.1f} {r.support:>9.4f}")

    st = summ[summ.model == "structured"]
    orc = summ[summ.model == "oracle"]
    d_struct = float(st[st.m == A].nv.iloc[0] - st[st.m == 1].nv.iloc[0])
    d_oracle = float(orc[orc.m == A].nv.iloc[0] - orc[orc.m == 1].nv.iloc[0])
    print(f"\n  widening the trust region 1 -> {A}:")
    print(f"    with the FITTED structured model : {d_struct:+.3f}")
    print(f"    with the EXACT Q* (control)      : {d_oracle:+.3f}")
    if d_struct < -0.1 and d_oracle > 0.1:
        v = ("CONFIRMED -- widening is harmful ONLY with a wrong model. The imitation "
             "constraint is what contains model optimism; the damage is model error "
             "being cashed in, not the loss of imitation per se.")
    elif d_struct < -0.1 and d_oracle <= 0.1:
        v = ("PARTIAL -- widening hurts with both models, so the trust region is doing "
             "work beyond containing model error; report the confound honestly.")
    else:
        v = ("NOT CONFIRMED -- widening does not reproduce the planning collapse; the "
             "two-point 0.670 vs -4.543 contrast has another explanation.")
    print(f"  >>> VERDICT: {v}")


if __name__ == "__main__":
    main()
