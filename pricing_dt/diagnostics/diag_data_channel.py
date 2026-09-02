"""The DATA channel — the missing row of the channel matrix.

The study compares putting a domain model in the ACTION channel (estimate-then-
optimize: plan against the model; true value -4.543) with putting it in the GOAL
channel (relabel the return-to-go; 0.670). The obvious third option is absent, and
it is the one a reviewer asks about first: what if the SAME demand model is used
the way Dyna and MBPO use a model — to GENERATE synthetic transitions that augment
the training set? There the model's error enters as contaminated data rather than
as a chosen action or a conditioning target.

This script fills that row, plus a fourth (data SELECTION), using the same fitted
StructuredDemandModel, the same cell, seeds and anchors as every other diagnostic.

ARMS

  real_BC / real_DT           no model at all (floor references)

  dyna_*   synthetic transitions branched from logged states, rewards from the
           fitted demand model, transitions from the KNOWN reference-price update
           (the same knowledge the relabeller uses, so the channels are compared
           at equal information). Two sub-designs, because the choice of behaviour
           policy in the synthetic rollout IS the design decision that decides how
           much model error gets in:
             action_src="bc"     actions from a BC policy fitted to the log
                                 -> synthetic data stays near the logged support
             action_src="greedy" actions greedy w.r.t. the model's own Q surface
                                 -> synthetic data chases the model's optimum, the
                                    faithful Dyna/MBPO analogue of what the planner
                                    does, and the arm where model exploitation
                                    should reappear if the channel cannot contain it
           swept over branch length k (MBPO's short-rollout truncation, the
           standard control for model error) and over the synthetic:real ratio.

  filtBC_* data SELECTION rather than data generation: keep only the logged
           trajectories the model scores highest, then behaviour-clone them. This
           is the cheapest possible use of a domain model and a genuinely different
           channel — the model touches neither actions, nor targets, nor the
           transition distribution, only which real data survives.


"""
import argparse
import numpy as np
import pandas as pd
import os
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.relabel import logged_rtg
from pricing_dt.core.baselines import train_bc, policy_from_qnet
from pricing_dt.core.dt import train_dt, make_dt_policy
from pricing_dt.diagnostics.diag_trust_region import model_q_surface, eval_cached


def model_rollout(mdp, dm, Q, trajs, n_traj, k, action_src, bc, rng, device="cpu"):
    """Branch synthetic trajectories of length k from logged states.

    Rewards come from the fitted demand model; the reference-price transition is
    the known one. Returns Trajectory objects carrying MODEL rewards, so a DT
    trained on them conditions on model-generated returns."""
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    out = []
    dm.eval()
    with torch.no_grad():
        for _ in range(n_traj):
            tr0 = trajs[rng.integers(len(trajs))]
            t0 = int(rng.integers(0, mdp.H))
            b = int(tr0.ref_bins[t0])
            ref = mdp.ref_grid[b]
            obs, rb, acts, rews = [], [], [], []
            for j in range(min(k, mdp.H - t0)):
                t = t0 + j
                b = mdp.ref_to_bin(ref)
                o = mdp.obs(ref, t)
                if action_src == "greedy":
                    a = int(np.argmax(Q[t, b]))                     # chase the model optimum
                else:
                    s = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                    p = torch.softmax(bc(s)[0], dim=-1).cpu().numpy()
                    a = int(rng.choice(len(p), p=p / p.sum()))       # stay near logged support
                s = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
                r = float(prices[a] * dm(prices[a].unsqueeze(0), s).squeeze())
                obs.append(o); rb.append(b); acts.append(a); rews.append(r)
                ref = mdp.ref_grid[mdp.N[a, b]]
            if not acts:
                continue
            out.append(D.Trajectory(np.array(obs, np.float32), np.array(rb),
                                    np.array(acts), np.array(rews, np.float32)))
    return out


def pad_to_horizon(trajs, H):
    """DT packing needs equal-length trajectories; repeat the final step of short
    synthetic branches so they can be batched with the real ones."""
    out = []
    for tr in trajs:
        h = len(tr.actions)
        if h == H:
            out.append(tr); continue
        rep = H - h
        out.append(D.Trajectory(
            np.concatenate([tr.obs, np.repeat(tr.obs[-1:], rep, 0)]),
            np.concatenate([tr.ref_bins, np.repeat(tr.ref_bins[-1:], rep)]),
            np.concatenate([tr.actions, np.repeat(tr.actions[-1:], rep)]),
            np.concatenate([tr.rewards, np.zeros(rep, np.float32)])))
    return out


def model_traj_value(tr, Q):
    return float(Q[0, int(tr.ref_bins[0]), int(tr.actions[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    H = cfg.sim.horizon
    ratios = [0.25, 0.5, 0.75]
    ks = [1, 2, 4, H]
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        rng = np.random.default_rng(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
        _Rhat, _V, Q = model_q_surface(dm, mdp)
        bc = train_bc(trajs, obs_dim, A, cfg.model, seed=seed)

        def rec(arm, nv, **kw):
            rows.append(dict(seed=seed, arm=arm, nv=round(nv, 3), **kw))

        def eval_dt_on(all_trajs, rtg):
            m = train_dt(D.pack_dt(all_trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            tgt = float(np.quantile(np.concatenate(rtg), 0.95))
            v, _, _, _, _ = eval_cached(make_dt_policy(m, mdp, tgt), mdp, init)
            return M.normalised_value(v, v_beh, v_opt)

        # ---- no-model references ----
        v, _, _, _, _ = eval_cached(policy_from_qnet(bc), mdp, init)
        rec("real_BC", M.normalised_value(v, v_beh, v_opt))
        rec("real_DT", eval_dt_on(trajs, logged_rtg(trajs)))

        # ---- DATA channel: Dyna/MBPO-style augmentation ----
        # n_syn such that synthetic/(synthetic+real) == the stated ratio, so the
        # labels mean what they say in both sweeps.
        def n_for(ratio):
            return int(len(trajs) * ratio / max(1e-9, 1 - ratio))

        for src in ["bc", "greedy"]:
            for k in ks:                                   # ratio fixed at 0.5
                syn = pad_to_horizon(model_rollout(mdp, dm, Q, trajs, n_for(0.5), k, src, bc, rng), H)
                allt = trajs + syn
                rec(f"dyna_{src}_k{k}", eval_dt_on(allt, logged_rtg(allt)), k=k, ratio=0.5, src=src)
            for ratio in ratios:                           # k fixed at 2
                syn = pad_to_horizon(model_rollout(mdp, dm, Q, trajs, n_for(ratio), 2, src, bc, rng), H)
                allt = trajs + syn
                rec(f"dyna_{src}_r{ratio}", eval_dt_on(allt, logged_rtg(allt)), k=2, ratio=ratio, src=src)

        # ---- DATA-SELECTION channel: model-filtered BC ----
        scores = np.array([model_traj_value(tr, Q) for tr in trajs])
        for keep in [0.25, 0.5]:
            idx = np.argsort(-scores)[:max(2, int(len(trajs) * keep))]
            sub = [trajs[i] for i in idx]
            b2 = train_bc(sub, obs_dim, A, cfg.model, seed=seed)
            v, _, _, _, _ = eval_cached(policy_from_qnet(b2), mdp, init)
            rec(f"filtBC_keep{keep}", M.normalised_value(v, v_beh, v_opt), keep=keep)

        cur = [r for r in rows if r["seed"] == seed]
        print(f"seed {seed}: " + "  ".join(f"{r['arm']}={r['nv']:+.2f}"
                                           for r in cur if "_k2" not in r["arm"]))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "data_channel.csv"), index=False)
    summ = df.groupby("arm").nv.agg(["mean", "std"]).round(3).sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(args.outdir, "data_channel_summary.csv"))
    print("\n=== data / selection channel (mean normalised value over seeds) ===")
    print(summ.to_string())

    best_data = summ.drop(index=[i for i in summ.index if not i.startswith("dyna")]).index[0]
    print(f"\n  best DATA-channel arm : {best_data}  ({summ.loc[best_data,'mean']:+.3f})")
    print(f"  reference points: goal channel (structured relabel) +0.670, "
          f"action channel (estimate-then-optimize) -4.543,")
    print(f"                    real_DT {summ.loc['real_DT','mean']:+.3f}, "
          f"real_BC {summ.loc['real_BC','mean']:+.3f}")
    print("\n  greedy-vs-bc synthetic actions (does the data channel cash in model optimism?)")
    for k in ks:
        g = summ.loc[f"dyna_greedy_k{k}", "mean"]; b = summ.loc[f"dyna_bc_k{k}", "mean"]
        print(f"    k={k}: greedy {g:+.3f}  bc {b:+.3f}  (greedy - bc {g-b:+.3f})")


if __name__ == "__main__":
    main()
