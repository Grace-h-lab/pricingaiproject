"""Second environment, GOAL-CHANNEL half: does the channel contrast replicate?

The probe (`diag_env2_probe.py`) established condition (b) on the inventory
environment: planning against the Poisson structural prior collapses to nv -3.246
while the model believes it is worth 94.8 (optimizer's-curse gap +60.3). That is the
ACTION-channel half. This script supplies the GOAL-channel half — the same fitted
models used to relabel return-to-go instead of to choose actions — so the pair can be
compared as it was in the pricing testbed:

    pricing    goal channel +0.670   vs   action channel -4.543
    inventory  goal channel   ???    vs   action channel -3.246

TWO NECESSARY DEVIATIONS FROM THE PRICING PROTOCOL, both forced by this environment
rather than chosen:

1. **Monte-Carlo evaluation.** The pricing MDP had a deterministic transition, so a
   history-dependent DT policy could be evaluated exactly by rolling one path per start
   bin. Here x' = max(0, x + q - D) is stochastic, and a DT's action depends on its
   running return-to-go, which is path-dependent — exact evaluation would require
   carrying the RTG in the state and is combinatorial. We therefore estimate values by
   Monte Carlo with a fixed episode budget, paired across arms via a common seed per
   (seed, arm) so the comparison is within-sample. Anchors (logging value, optimum) are
   still computed EXACTLY by DP, so only the learned-policy values carry sampling noise.

2. **Relabelling via DP rather than a greedy roll-forward.** In pricing,
   `R_hat_t = r_hat(s_t,a_t) + sum_{k>t} max_p r_hat(s_k,p)` rolled the model forward
   along one deterministic path. Under a stochastic kernel the same quantity IS the
   fitted model's action-value, so the relabeller is `R_hat_t = Qhat[t, x_t, a_t]`,
   obtained by backward induction on the fitted demand. NOTE this is not literally the
   same estimator as E1's: E1 rolls the fitted model forward MYOPICALLY (a* maximises
   immediate revenue), whereas backward induction follows the fitted model's DP-optimal
   continuation. Measured on E1, that difference does not move the within-state
   discrimination the goal channel consumes (0.328 vs 0.331, against 0.083 for exact Q*),
   so the two are treated as the same idea under different kernels -- but they are not the
   same object, and the oracle arm is exactly `Qstar[t, x_t, a_t]` in both.

ARMS
  vanilla        logged return-to-go
  structured     Qhat from the POISSON prior (the model that is toxic when planned on)
  empirical      Qhat from the free histogram (unconstrained comparator)
  oracle         exact Qstar  (the goal-channel ceiling / the Q* question again)
  QDT            bootstrapped value relabel, action-dependent (r_t + V(s_{t+1}))
  BC             behaviour cloning, no return conditioning

Conditioning targets are selected on a held-out episode set and read once on a disjoint
test set, matching `diag_heldout_protocol.py`.
"""
import argparse
import numpy as np
import pandas as pd
import os
import torch

from pricing_dt.envs.inventory import InvConfig, InventoryMDP, OrderUpTo
from pricing_dt.diagnostics.diag_env2_probe import generate_logs, fit_poisson, fit_empirical
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.dt import train_dt, _step_logits
from pricing_dt.core.baselines import train_bc, train_cql
from pricing_dt.core import config as C

QUANTILES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0]
MAX_MULTS = [1.25, 1.5]


class InvTraj:
    """Minimal trajectory object compatible with data.pack_dt (needs .obs/.actions)."""
    __slots__ = ("obs", "states", "actions", "rewards")

    def __init__(self, obs, states, actions, rewards):
        self.obs, self.states = obs, states
        self.actions, self.rewards = actions, rewards

    @property
    def rtg(self):
        return np.cumsum(self.rewards[::-1])[::-1].copy()


def roll_trajectories(mdp, logger, n_ep, rng):
    trajs = []
    for _ in range(n_ep):
        x = int(rng.integers(0, mdp.cfg.max_inventory // 2 + 1))
        obs, st, ac, rw = [], [], [], []
        for t in range(mdp.H):
            a = logger.action(x, t)
            r, nxt, _s, _c = mdp.step(x, a, rng)
            obs.append(mdp.obs(x, t)); st.append(x); ac.append(a); rw.append(r)
            x = nxt
        trajs.append(InvTraj(np.array(obs, np.float32), np.array(st),
                             np.array(ac), np.array(rw, np.float32)))
    return trajs


def q_from_pmf(mdp, pmf):
    """Backward induction on a fitted demand pmf -> Qhat[t,x,a], Vhat."""
    Rh, Ph = mdp._step_arrays(pmf)
    Q, V, _pi = mdp.solve_optimal(R=Rh, P=Ph)
    Q, V = Q.copy(), V.copy()
    mdp.solve_optimal()                       # restore the true Q*/V*/pi*
    return Q, V


def relabel_from_Q(trajs, Q):
    return [np.array([Q[t, int(tr.states[t]), int(tr.actions[t])]
                      for t in range(len(tr.actions))], np.float32) for tr in trajs]


def qdt_relabel(trajs, qnet, device="cpu"):
    """Action-dependent bootstrapped target r_t + V(s_{t+1}) (the fixed form)."""
    out = []
    with torch.no_grad():
        for tr in trajs:
            s = torch.tensor(tr.obs, dtype=torch.float32, device=device)
            v = qnet(s).max(1).values.cpu().numpy().astype(np.float32)
            H = len(tr.actions)
            g = np.zeros(H, np.float32)
            g[:H - 1] = tr.rewards[:H - 1] + v[1:]
            g[H - 1] = tr.rewards[H - 1]
            out.append(g)
    return out


def obs_batch(mdp, x, t):
    """Vectorised mdp.obs over a batch of inventory levels."""
    return np.stack([np.asarray(x, np.float32) / max(1, mdp.cfg.max_inventory),
                     np.full(len(x), t / max(1, mdp.H - 1), np.float32)], axis=1)


def step_batch(mdp, x, a, rng):
    """Vectorised mdp.step over a batch."""
    c = mdp.cfg
    d = rng.choice(len(mdp.demand_pmf), size=len(x), p=mdp.demand_pmf)
    avail = np.minimum(x + a, c.max_inventory)
    sales = np.minimum(d, avail)
    end = avail - sales
    unmet = np.maximum(d - avail, 0)
    rew = (c.price * sales - c.order_cost * a
           - c.hold_cost * end - c.stockout_penalty * unmet)
    return rew.astype(np.float64), end.astype(int)


@torch.no_grad()
def mc_eval(mdp, model, target, n_ep, seed, topm=None, Q=None, device="cpu"):
    """Batched Monte-Carlo value of a DT controller.

    All episodes are advanced in lockstep, so each timestep costs ONE forward pass
    over a batch rather than one pass per episode -- the per-episode form was ~B times
    slower and made the full grid infeasible. The DT keeps its true running history
    (RTG decremented by the realised reward), so this is the deployed controller, not a
    memoryless approximation.

    `topm`/`Q` optionally restrict the choice to the DT's top-m actions and pick among
    them by the Q surface: the trust-region knob, m=1 being the DT itself.
    """
    rng = np.random.default_rng(seed)
    B, H = n_ep, mdp.H
    x = rng.integers(0, mdp.cfg.max_inventory // 2 + 1, size=B)
    rtg = np.zeros((B, H), np.float32)
    S = np.zeros((B, H, 2), np.float32)
    Aa = np.zeros((B, H), np.int64)
    T = np.tile(np.arange(H), (B, 1))
    cur = np.full(B, float(target), np.float32)
    total = np.zeros(B)
    for t in range(H):
        rtg[:, t] = cur
        S[:, t] = obs_batch(mdp, x, t)
        logits = model(torch.from_numpy(rtg[:, :t + 1]),
                       torch.from_numpy(S[:, :t + 1]),
                       torch.from_numpy(Aa[:, :t + 1]),
                       torch.from_numpy(T[:, :t + 1]))[:, -1]
        if topm is None or topm <= 1:
            a = logits.argmax(1).cpu().numpy()
        else:
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            order = np.argsort(-p, axis=1)[:, :topm]
            qv = Q[t][x[:, None], order]                 # [B, topm]
            a = order[np.arange(B), qv.argmax(1)]
        Aa[:, t] = a
        rew, x = step_batch(mdp, x, a, rng)
        total += rew
        cur = cur - rew.astype(np.float32)
    return float(total.mean())


@torch.no_grad()
def mc_eval_bc(mdp, bc, n_ep, seed):
    rng = np.random.default_rng(seed)
    B = n_ep
    x = rng.integers(0, mdp.cfg.max_inventory // 2 + 1, size=B)
    total = np.zeros(B)
    for t in range(mdp.H):
        a = bc(torch.from_numpy(obs_batch(mdp, x, t))).argmax(1).cpu().numpy()
        rew, x = step_batch(mdp, x, a, rng)
        total += rew
    return float(total.mean())


def target_grid(rtg_list):
    flat = np.concatenate(rtg_list)
    g = [(f"q{q}", float(np.quantile(flat, q))) for q in QUANTILES]
    mx = float(flat.max())
    return g + [(f"max x{m}", mx * m) for m in MAX_MULTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--eval-ep", type=int, default=400)
    ap.add_argument("--sel-ep", type=int, default=200)
    ap.add_argument("--level", type=int, default=14)
    ap.add_argument("--trust", action="store_true", help="also run the trust-region sweep")
    args = ap.parse_args()

    mcfg = C.full().model
    rows, trust_rows = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        mdp = InventoryMDP(InvConfig(seed=seed))
        mdp.solve_optimal()
        init = np.zeros(mdp.n_states); init[: mdp.cfg.max_inventory // 2 + 1] = 1.0
        init /= init.sum()
        logger = OrderUpTo(mdp, args.level, rng)
        # NOTE ON THE ANCHOR. This is the LOGGING POLICY, unlike E1, where the
        # identically-named `v_beh` holds the oracle myopic value (see
        # experiments.py::_setup and metrics.py::normalised_value). E2's zero is
        # therefore its logger and E1's is a strictly stronger reference, so E1 and
        # E2 normalised values are NOT the same unit and must not be pooled,
        # differenced, or read against one another as if they were. Every
        # comparison inside this file is internally consistent because all arms and
        # both anchors are computed here the same way. Dissertation section 3.5.1
        # states the asymmetry.
        v_beh = mdp.evaluate_policy_fn(lambda o: logger.action(*mdp.decode_obs(o)), init)
        v_opt = float(init @ mdp.Vstar[0])
        denom = v_opt - v_beh

        trajs = roll_trajectories(mdp, logger, args.episodes, rng)
        log = generate_logs(mdp, logger, args.episodes, rng)
        pois = fit_poisson(log, mdp.cfg.max_demand)
        emp = fit_empirical(log, mdp.cfg.max_demand)
        Qp, _ = q_from_pmf(mdp, pois)
        Qe, _ = q_from_pmf(mdp, emp)
        obs_dim, A = 2, mdp.n_actions
        qnet = train_cql(trajs, mdp, obs_dim, A, mcfg, seed=seed)

        arms = {
            "vanilla": [tr.rtg for tr in trajs],
            "structured": relabel_from_Q(trajs, Qp),
            "empirical": relabel_from_Q(trajs, Qe),
            "oracle": relabel_from_Q(trajs, mdp.Qstar),
            "QDT": qdt_relabel(trajs, qnet),
        }
        out = {"seed": seed, "v_beh": round(v_beh, 1), "v_opt": round(v_opt, 1)}
        for name, rtg in arms.items():
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, mcfg, seed=seed)
            best, best_v = None, -1e18
            for label, tgt in target_grid(rtg):
                v = mc_eval(mdp, m, tgt, args.sel_ep, seed=10_000 + seed)
                if v > best_v:
                    best_v, best = v, tgt
            v_test = mc_eval(mdp, m, best, args.eval_ep, seed=20_000 + seed)
            out[name] = round((v_test - v_beh) / denom, 3)
            if args.trust and name == "structured":
                for mm in [1, 2, 3, 5, 8, A]:
                    for tag, Qs in (("structured", Qp), ("oracle", mdp.Qstar)):
                        vv = mc_eval(mdp, m, best, args.eval_ep,
                                     seed=20_000 + seed, topm=mm, Q=Qs)
                        trust_rows.append(dict(seed=seed, model=tag, m=mm,
                                               nv=round((vv - v_beh) / denom, 3)))
        bc = train_bc(trajs, obs_dim, A, mcfg, seed=seed)
        out["BC"] = round((mc_eval_bc(mdp, bc, args.eval_ep, seed=20_000 + seed)
                           - v_beh) / denom, 3)
        rows.append(out)
        print(f"seed {seed}: " + "  ".join(
            f"{k}={out[k]:+.2f}" for k in list(arms) + ["BC"]))

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(args.outdir, "env2_channels.csv"), index=False)
    names = ["structured", "oracle", "empirical", "QDT", "vanilla", "BC"]
    summ = df[names].agg(["mean", "std"]).T.round(3).sort_values("mean", ascending=False)
    summ.to_csv(os.path.join(args.outdir, "env2_channels_summary.csv"))
    print("\n=== GOAL channel, inventory environment (normalised value) ===")
    print(summ.to_string())
    print(f"\n  ACTION channel on the same models (from diag_env2_probe):")
    print(f"    plan on the Poisson prior     : -3.246   (in-model 94.8, curse gap +60.3)")
    print(f"    plan on the empirical model   : -1.192")

    print("\n=== the channel contrast, both environments ===")
    st = float(summ.loc["structured", "mean"])
    print(f"  pricing    goal +0.670   action -4.543")
    print(f"  inventory  goal {st:+.3f}   action -3.246")
    # "beneficial" must mean beating the no-model baselines, not merely beating the
    # logging policy -- BC clears that bar too, so `st > 0` would report replication for
    # a structured arm sitting below both no-model baselines.
    base = max(float(summ.loc["vanilla", "mean"]), float(summ.loc["BC", "mean"]))
    action_channel = -3.246
    if st > base:
        print("  >>> FULLY REPLICATED: the structural prior is beneficial as a conditioning "
              "target and destructive as a planner.")
    elif st > action_channel + 1.0:
        print(f"  >>> SAFETY HALF ONLY: the same model is catastrophic in the action channel "
              f"({action_channel:+.3f}) and merely unhelpful in the goal channel "
              f"({st:+.3f}, vs {base:+.3f} for the best no-model baseline). The CHANNEL "
              f"claim replicates; 'structured relabelling helps' does NOT.")
    else:
        print("  >>> NOT REPLICATED: the goal channel collapses here too.")
    med, p = M.paired_test(df["structured"].values, df["oracle"].values)
    print(f"\n  structured - oracle: {df['structured'].mean() - df['oracle'].mean():+.3f} "
          f"(p={p:.4f})  [the Q* question, re-asked in env 2]")

    if trust_rows:
        tdf = pd.DataFrame(trust_rows)
        tdf.to_csv(os.path.join(args.outdir, "env2_trust.csv"), index=False)
        piv = tdf.groupby(["model", "m"]).nv.mean().unstack().round(3)
        print("\n=== trust-region sweep (rows = Q surface used to pick within the top-m) ===")
        print(piv.to_string())


if __name__ == "__main__":
    main()
