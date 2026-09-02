"""E2 replication: does the logged-support constraint generalise beyond pricing?

WHY THIS RUN EXISTS. Chapter 4's strongest evidence is now the support constraint,
not the structured relabeller: masking to logged support improves all five target
families in E1 (+0.052 to +0.326, Holm-corrected) in an order rank-identical to how
far each strays, and a no-learner floor bounds what any masked score demonstrates.
All of that rests on ONE environment. The channel claim (C1) and the trust-region
mechanism (C2) were both replicated in the inventory testbed; the mask crossing
never was.

This closes that gap, and it is the right way to spend the remaining external-validity
budget. A continuous-control benchmark cannot test this claim at all: the top-k
logged-support mask has no direct definition over a continuous action space, so
porting it would require inventing a continuous analogue -- itself a research
contribution rather than a replication. The inventory environment, by contrast, is
discrete (11 order quantities against 11 prices), exactly solvable, and already
built, so the identical constraint applies unchanged.

PROTOCOL. Deliberately the same as `diag_env2_channels`: same seeds, same logging
policy, same episode counts, same conditioning-target selection on a held-out split.
Each arm's target is chosen once on the bare arm and REUSED by its masked twin, so
the two differ only by the mask and the pair is exact -- the same device used in
`diag_gate2_pricing`.

ARMS. The five relabelling targets plus behaviour cloning, each bare and masked, and
two controls that make the numbers interpretable:

    random           uniform over the order grid; the chance floor
    random + mask    what the support set alone buys with NO learner, the E1
                     analogue of the 0.448 floor of §4.6.2. Without it a
                     masked score cannot be attributed to the method.

CAVEAT, inherited from the environment. E2's transition is stochastic, so learned
policy values are Monte-Carlo rather than exact; the anchors stay exact by dynamic
programming. E2 comparisons are therefore weaker than E1's, and are reported as
such.

Outputs: env2_support_raw.csv, env2_support_summary.csv

Implements: the support crossing repeated in E2 over thirty seeds, reported in §4.6.1.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core.baselines import train_bc, train_cql
from pricing_dt.core.dt import train_dt
from pricing_dt.diagnostics.diag_env2_channels import (obs_batch, q_from_pmf,
                                                       qdt_relabel, relabel_from_Q,
                                                       roll_trajectories, step_batch,
                                                       target_grid)
from pricing_dt.diagnostics.diag_env2_probe import (fit_empirical, fit_poisson,
                                                    generate_logs)
from pricing_dt.envs.inventory import InvConfig, InventoryMDP, OrderUpTo


def support_counts(trajs, mdp):
    """counts[t, inventory_level, order_quantity] over the logged trajectories.

    The pricing analogue is indexed by (timestep, reference-price bin); here the
    state is the inventory level. Same object, different state variable.
    """
    counts = np.zeros((mdp.H, mdp.n_states, mdp.n_actions), dtype=float)
    for tr in trajs:
        for t in range(len(tr.actions)):
            counts[t, int(tr.states[t]), int(tr.actions[t])] += 1.0
    return counts


def supported_batch(counts, t, x, n_actions, topk):
    """Boolean [B, n_actions] mask: the top-k most-logged actions in each state.

    Mirrors `dt._supported_actions` exactly, including its fallback: if a state was
    never visited, keep the single most-logged action so the policy always has a
    legal move rather than an empty choice set.
    """
    out = np.zeros((len(x), n_actions), dtype=bool)
    for i, xi in enumerate(x):
        c = counts[t, int(xi)]
        top = np.argsort(c)[-int(topk):]
        out[i, top] = c[top] > 0
        if not out[i].any():
            out[i, int(np.argmax(c))] = True
    return out


@torch.no_grad()
def mc_eval_masked(mdp, model, target, n_ep, seed, ref_counts, enforce=False, topk=3):
    """Batched Monte-Carlo value of a DT controller, optionally support-masked.

    Returns (value, off_support_rate), the second being Equation (3.5). `ref_counts` is always supplied because the
    off-support rate is measured against the same top-k set whether or not the mask
    was ENFORCED -- that is what puts the bare and masked arms on one axis. `enforce`
    controls only whether unsupported actions are actually forbidden; when true the
    logits are set to -inf exactly as in `dt.make_support_masked_dt_policy`, so the
    masked arm plays the model's own highest-ranked SUPPORTED action. The rate is
    measured along the realised rollout, matching how E1 reports it.
    """
    rng = np.random.default_rng(seed)
    B, H, A = n_ep, mdp.H, mdp.n_actions
    x = rng.integers(0, mdp.cfg.max_inventory // 2 + 1, size=B)
    rtg = np.zeros((B, H), np.float32)
    S = np.zeros((B, H, 2), np.float32)
    Aa = np.zeros((B, H), np.int64)
    T = np.tile(np.arange(H), (B, 1))
    cur = np.full(B, float(target), np.float32)
    total = np.zeros(B)
    off = []
    for t in range(H):
        rtg[:, t] = cur
        S[:, t] = obs_batch(mdp, x, t)
        logits = model(torch.from_numpy(rtg[:, :t + 1]),
                       torch.from_numpy(S[:, :t + 1]),
                       torch.from_numpy(Aa[:, :t + 1]),
                       torch.from_numpy(T[:, :t + 1]))[:, -1]
        ref = supported_batch(ref_counts, t, x, A, topk)
        if enforce:
            m_ = torch.from_numpy(ref).to(logits.device)
            logits = logits.masked_fill(~m_, float("-inf"))
        a = logits.argmax(1).cpu().numpy()
        off.append(~ref[np.arange(B), a])
        Aa[:, t] = a
        rew, x = step_batch(mdp, x, a, rng)
        total += rew
        cur = cur - rew.astype(np.float32)
    return float(total.mean()), float(np.mean(np.concatenate(off)))


def mc_eval_random(mdp, n_ep, seed, ref_counts, enforce=False, topk=3):
    """Uniform policy, optionally masked -- the no-learner control."""
    rng = np.random.default_rng(seed)
    pick = np.random.default_rng(seed + 7)
    B, A = n_ep, mdp.n_actions
    x = rng.integers(0, mdp.cfg.max_inventory // 2 + 1, size=B)
    total = np.zeros(B)
    off = []
    for t in range(mdp.H):
        ref = supported_batch(ref_counts, t, x, A, topk)
        if enforce:
            a = np.array([pick.choice(np.flatnonzero(r)) for r in ref])
        else:
            a = pick.integers(0, A, size=B)
        off.append(~ref[np.arange(B), a])
        rew, x = step_batch(mdp, x, a, rng)
        total += rew
    return float(total.mean()), float(np.mean(np.concatenate(off)))


@torch.no_grad()
def mc_eval_bc_masked(mdp, bc, n_ep, seed, ref_counts, enforce=False, topk=3):
    rng = np.random.default_rng(seed)
    B, A = n_ep, mdp.n_actions
    x = rng.integers(0, mdp.cfg.max_inventory // 2 + 1, size=B)
    total = np.zeros(B)
    off = []
    for t in range(mdp.H):
        logits = bc(torch.from_numpy(obs_batch(mdp, x, t)))
        ref = supported_batch(ref_counts, t, x, A, topk)
        if enforce:
            m_ = torch.from_numpy(ref).to(logits.device)
            logits = logits.masked_fill(~m_, float("-inf"))
        a = logits.argmax(1).cpu().numpy()
        off.append(~ref[np.arange(B), a])
        rew, x = step_batch(mdp, x, a, rng)
        total += rew
    return float(total.mean()), float(np.mean(np.concatenate(off)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_env2_support")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-ep", type=int, default=400)
    ap.add_argument("--sel-ep", type=int, default=200)
    ap.add_argument("--level", type=int, default=14)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    mcfg = C.full().model
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        mdp = InventoryMDP(InvConfig(seed=seed))
        mdp.solve_optimal()
        init = np.zeros(mdp.n_states)
        init[: mdp.cfg.max_inventory // 2 + 1] = 1.0
        init /= init.sum()
        logger = OrderUpTo(mdp, args.level, rng)
        v_beh = mdp.evaluate_policy_fn(lambda o: logger.action(*mdp.decode_obs(o)), init)
        v_opt = float(init @ mdp.Vstar[0])
        denom = v_opt - v_beh

        trajs = roll_trajectories(mdp, logger, args.episodes, rng)
        counts = support_counts(trajs, mdp)
        log = generate_logs(mdp, logger, args.episodes, rng)
        Qp, _ = q_from_pmf(mdp, fit_poisson(log, mdp.cfg.max_demand))
        Qe, _ = q_from_pmf(mdp, fit_empirical(log, mdp.cfg.max_demand))
        obs_dim, A = 2, mdp.n_actions
        qnet = train_cql(trajs, mdp, obs_dim, A, mcfg, seed=seed)

        arms = {"vanilla": [tr.rtg for tr in trajs],
                "structured": relabel_from_Q(trajs, Qp),
                "empirical": relabel_from_Q(trajs, Qe),
                "oracle": relabel_from_Q(trajs, mdp.Qstar),
                "QDT": qdt_relabel(trajs, qnet)}

        def rec(arm, mask, v, off):
            rows.append(dict(seed=seed, arm=arm, mask=mask,
                             nv=round((v - v_beh) / denom, 4),
                             off_support=round(off, 4),
                             v_beh=round(v_beh, 2), v_opt=round(v_opt, 2)))

        for name, rtg in arms.items():
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, mcfg, seed=seed)
            # target selected once, on the BARE arm, then reused by the masked twin
            best, best_v = None, -1e18
            for _, tgt in target_grid(rtg):
                v, _ = mc_eval_masked(mdp, m, tgt, args.sel_ep, seed=10_000 + seed,
                                      ref_counts=counts, enforce=False, topk=args.topk)
                if v > best_v:
                    best_v, best = v, tgt
            rec(name, "none",
                *mc_eval_masked(mdp, m, best, args.eval_ep, seed=20_000 + seed,
                                ref_counts=counts, enforce=False, topk=args.topk))
            rec(name, f"top{args.topk}",
                *mc_eval_masked(mdp, m, best, args.eval_ep, seed=20_000 + seed,
                                ref_counts=counts, enforce=True, topk=args.topk))

        bc = train_bc(trajs, obs_dim, A, mcfg, seed=seed)
        rec("BC", "none", *mc_eval_bc_masked(mdp, bc, args.eval_ep, 20_000 + seed,
                                             ref_counts=counts, enforce=False,
                                             topk=args.topk))
        rec("BC", f"top{args.topk}",
            *mc_eval_bc_masked(mdp, bc, args.eval_ep, 20_000 + seed,
                               ref_counts=counts, enforce=True, topk=args.topk))
        rec("random", "none",
            *mc_eval_random(mdp, args.eval_ep, 20_000 + seed, ref_counts=counts,
                            enforce=False, topk=args.topk))
        rec("random", f"top{args.topk}",
            *mc_eval_random(mdp, args.eval_ep, 20_000 + seed, ref_counts=counts,
                            enforce=True, topk=args.topk))
        print(f"seed {seed}: done", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "env2_support_raw.csv"), index=False)
    summ = (df.groupby(["arm", "mask"])
            .agg(nv=("nv", "mean"), sd=("nv", "std"), off_support=("off_support", "mean"))
            .reset_index().sort_values("nv", ascending=False).round(4))
    summ.to_csv(os.path.join(args.outdir, "env2_support_summary.csv"), index=False)
    print("\n=== E2 support crossing (normalised value) ===")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
