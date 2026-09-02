"""Commercial-context robustness diagnostic for the pricing channel study.

The main pricing experiment intentionally isolates a reference-price MDP. This
optional diagnostic adds two deployment-facing context dimensions while keeping
exact dynamic programming:

  - seasonal context: an observed discrete season that shifts demand;
  - product heterogeneity: product types with different baseline demand,
    elasticity and reference-price sensitivity.

It then repeats the core channel comparison:

  - action channel: fitted demand model -> DP argmax price policy;
  - goal channel: fitted demand model -> RTG relabel -> Decision Transformer.

The point is not to replace the controlled main result. It asks whether the
channel warning survives when the state looks more like a commercial pricing
pipeline.

Output:
  - commercial_context.csv
  - commercial_context_tests.csv
"""
import argparse
import copy
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import torch

from pricing_dt.core import config as C
from pricing_dt.core import provenance
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.demand_model import (
    StructuredDemandModel,
    UnconstrainedDemandModel,
    fit_demand_model,
)
from pricing_dt.core.dt import _step_logits, train_dt
from pricing_dt.core.relabel import logged_rtg
from pricing_dt.core.torch_utils import resolve_device
from pricing_dt.experiments.experiments import _seed


@dataclass
class ContextTrajectory:
    obs: np.ndarray
    ref_bins: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    product_ids: np.ndarray
    season_ids: np.ndarray
    seg: int = 0

    @property
    def rtg(self):
        return np.cumsum(self.rewards[::-1])[::-1].copy()

    @property
    def total_return(self):
        return float(self.rewards.sum())


class CommercialContextPricingMDP:
    """Reference-price pricing MDP with observed season and product context."""

    def __init__(
        self,
        cfg,
        n_products=3,
        n_seasons=4,
        use_product=True,
        use_season=True,
        season_amp=0.35,
    ):
        self.cfg = cfg
        self.prices = np.linspace(cfg.p_min, cfg.p_max, cfg.n_prices)
        self.ref_grid = np.linspace(cfg.p_min, cfg.p_max, cfg.n_ref_bins)
        self.H = cfg.horizon
        self.rng = np.random.default_rng(cfg.seed)
        self.use_product = bool(use_product)
        self.use_season = bool(use_season)
        self.n_products = int(n_products) if self.use_product else 1
        self.n_seasons = int(n_seasons) if self.use_season else 1
        self.season_amp = float(season_amp) if self.use_season else 0.0
        self._build_context_parameters()
        self._build_dynamics()

    @property
    def obs_dim(self):
        return 4 + self.n_products

    def _build_context_parameters(self):
        if self.n_products == 1:
            centered = np.array([0.0])
        else:
            centered = np.linspace(-1.0, 1.0, self.n_products)

        self.product_alpha = 0.35 * centered
        self.product_beta_mult = np.clip(1.0 + 0.30 * centered, 0.35, None)
        self.product_delta_mult = np.clip(1.0 + 0.20 * centered, 0.35, None)
        self.product_market_mult = np.clip(1.0 - 0.15 * centered, 0.50, None)
        self.product_season_sens = np.clip(1.0 + 0.25 * centered, 0.25, None)

        season_idx = np.arange(self.n_seasons)
        self.season_encoding = np.column_stack(
            [
                np.sin(2.0 * np.pi * season_idx / self.n_seasons),
                np.cos(2.0 * np.pi * season_idx / self.n_seasons),
            ]
        )
        self.season_effect = self.season_amp * self.season_encoding[:, 0]

    def expected_demand(self, price, ref, product_id=0, season_id=0):
        p = int(product_id)
        s = int(season_id)
        c = self.cfg
        u = (
            c.alpha
            + self.product_alpha[p]
            + self.product_season_sens[p] * self.season_effect[s]
            - c.beta * self.product_beta_mult[p] * price
            + c.delta * self.product_delta_mult[p] * (ref - price)
        )
        return c.market_size * self.product_market_mult[p] * _sigmoid(u)

    def expected_reward(self, price, ref, product_id=0, season_id=0):
        return price * self.expected_demand(price, ref, product_id, season_id)

    def sample_reward(self, price, ref, product_id, season_id, noise):
        q = self.expected_demand(price, ref, product_id, season_id)
        if noise > 0:
            q = q * np.exp(self.rng.normal(0.0, noise))
        return price * q

    def next_ref(self, price, ref):
        return (1 - self.cfg.eta) * ref + self.cfg.eta * price

    def next_season(self, season_id):
        return (int(season_id) + 1) % self.n_seasons

    def ref_to_bin(self, ref):
        return int(np.argmin(np.abs(self.ref_grid - ref)))

    def obs(self, ref, t, product_id=0, season_id=0):
        ref_s = (ref - self.cfg.p_min) / (self.cfg.p_max - self.cfg.p_min)
        t_s = t / max(1, self.H - 1)
        prod = np.zeros(self.n_products, dtype=np.float32)
        prod[int(product_id)] = 1.0
        return np.concatenate(
            [
                np.array([ref_s, t_s], dtype=np.float32),
                self.season_encoding[int(season_id)].astype(np.float32),
                prod,
            ]
        ).astype(np.float32)

    def decode_obs(self, obs):
        ref = self.cfg.p_min + obs[0] * (self.cfg.p_max - self.cfg.p_min)
        b = self.ref_to_bin(ref)
        t = int(round(obs[1] * max(1, self.H - 1)))
        season_xy = np.asarray(obs[2:4], dtype=float)
        s = int(np.argmin(np.linalg.norm(self.season_encoding - season_xy, axis=1)))
        p = int(np.argmax(obs[4:]))
        return b, t, p, s

    def _build_dynamics(self):
        A, B = self.cfg.n_prices, self.cfg.n_ref_bins
        P, S, H = self.n_products, self.n_seasons, self.H
        self.N = np.zeros((A, B), dtype=int)
        self.R = np.zeros((H, P, S, A, B), dtype=float)
        for a, price in enumerate(self.prices):
            for b, ref in enumerate(self.ref_grid):
                self.N[a, b] = self.ref_to_bin(self.next_ref(price, ref))
                for t in range(H):
                    for p in range(P):
                        for s in range(S):
                            self.R[t, p, s, a, b] = self.expected_reward(price, ref, p, s)

    def solve_optimal(self):
        A, B, H = self.cfg.n_prices, self.cfg.n_ref_bins, self.H
        P, S = self.n_products, self.n_seasons
        Q = np.zeros((H, P, S, B, A), dtype=float)
        V = np.zeros((H + 1, P, S, B), dtype=float)
        for t in reversed(range(H)):
            for p in range(P):
                for s in range(S):
                    ns = self.next_season(s)
                    for b in range(B):
                        q = self.R[t, p, s, :, b] + V[t + 1, p, ns, self.N[:, b]]
                        Q[t, p, s, b] = q
                        V[t, p, s, b] = q.max()
        self.Qstar, self.Vstar = Q, V
        self.pistar = Q.argmax(axis=4)
        return Q, V, self.pistar

    def initial_states(self, n):
        b = self.rng.integers(0, self.cfg.n_ref_bins, size=n)
        p = self.rng.integers(0, self.n_products, size=n)
        s = self.rng.integers(0, self.n_seasons, size=n)
        return np.column_stack([b, p, s]).astype(int)

    def reward_from_obs_action(self, obs, action):
        b, t, p, s = self.decode_obs(obs)
        return float(self.R[t, p, s, int(action), b])

    def myopic_policy(self, obs):
        b, t, p, s = self.decode_obs(obs)
        return int(self.R[t, p, s, :, b].argmax())

    def evaluate_policy_fn(self, policy_fn, init_states):
        returns = []
        for b0, p, s0 in init_states:
            b, s = int(b0), int(s0)
            total = 0.0
            for t in range(self.H):
                ref = self.ref_grid[b]
                obs = self.obs(ref, t, p, s)
                a = int(policy_fn(obs))
                total += self.R[t, p, s, a, b]
                b = int(self.N[a, b])
                s = self.next_season(s)
            returns.append(total)
        return float(np.mean(returns)), np.asarray(returns, dtype=float)


class ContextRegionSpecialised:
    """Complementary reference-region logger with context-aware optimal actions."""

    def __init__(self, mdp, low_region, rng, eps=0.05, expert_q=1.0):
        self.mdp = mdp
        self.low_region = bool(low_region)
        self.rng = rng
        self.eps = float(eps)
        self.expert_q = float(expert_q)
        if not hasattr(mdp, "pistar"):
            mdp.solve_optimal()

    def action(self, ref_bin, t, product_id, season_id):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.mdp.cfg.n_prices))
        mid = self.mdp.cfg.n_ref_bins // 2
        in_region = (ref_bin < mid) if self.low_region else (ref_bin >= mid)
        a_myopic = int(self.mdp.R[t, product_id, season_id, :, ref_bin].argmax())
        if in_region:
            a_opt = int(self.mdp.pistar[t, product_id, season_id, ref_bin])
            return int(round((1 - self.expert_q) * a_myopic + self.expert_q * a_opt))
        return a_myopic


def rollout_context(mdp, policy, n_traj, noise, rng, seg=0):
    trajs = []
    for _ in range(n_traj):
        b = int(rng.integers(0, mdp.cfg.n_ref_bins))
        product_id = int(rng.integers(0, mdp.n_products))
        season_id = int(rng.integers(0, mdp.n_seasons))
        obs, rbins, acts, rews, prods, seasons = [], [], [], [], [], []
        for t in range(mdp.H):
            ref = mdp.ref_grid[b]
            a = policy.action(b, t, product_id, season_id)
            obs.append(mdp.obs(ref, t, product_id, season_id))
            rbins.append(b)
            acts.append(a)
            prods.append(product_id)
            seasons.append(season_id)
            rews.append(mdp.sample_reward(mdp.prices[a], ref, product_id, season_id, noise))
            b = int(mdp.N[a, b])
            season_id = mdp.next_season(season_id)
        trajs.append(
            ContextTrajectory(
                np.asarray(obs, np.float32),
                np.asarray(rbins, dtype=int),
                np.asarray(acts, dtype=int),
                np.asarray(rews, np.float32),
                np.asarray(prods, dtype=int),
                np.asarray(seasons, dtype=int),
                seg,
            )
        )
    return trajs


def make_context_stitching_necessary(mdp, n_traj, noise, seed, expert_q=1.0):
    rng = np.random.default_rng(seed)
    lo = ContextRegionSpecialised(mdp, low_region=True, rng=rng, expert_q=expert_q)
    hi = ContextRegionSpecialised(mdp, low_region=False, rng=rng, expert_q=expert_q)
    return (
        rollout_context(mdp, lo, n_traj // 2, noise, rng)
        + rollout_context(mdp, hi, n_traj - n_traj // 2, noise, rng)
    )


def context_relabel_one(tr, demand_model, mdp, lam=1.0, device=None):
    device = resolve_device(device, demand_model)
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    logged = tr.rtg
    relabelled = np.zeros(mdp.H, np.float32)

    demand_model.eval()
    with torch.no_grad():
        for t in range(mdp.H):
            b = int(tr.ref_bins[t])
            product_id = int(tr.product_ids[t])
            season_id = int(tr.season_ids[t])
            ref = mdp.ref_grid[b]

            a_t = int(tr.actions[t])
            s_t = torch.tensor(
                mdp.obs(ref, t, product_id, season_id),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            dem_t = demand_model(prices[a_t].unsqueeze(0), s_t)
            total = float(prices[a_t] * dem_t.squeeze())

            b = int(mdp.N[a_t, b])
            season_id = mdp.next_season(season_id)

            for k in range(t + 1, mdp.H):
                ref = mdp.ref_grid[b]
                s_k = torch.tensor(
                    mdp.obs(ref, k, product_id, season_id),
                    dtype=torch.float32,
                    device=device,
                )
                s_rep = s_k.unsqueeze(0).repeat(len(prices), 1)
                dem = demand_model(prices, s_rep)
                rev = prices * dem
                a_best = int(torch.argmax(rev).item())
                total += float(rev[a_best].item())
                b = int(mdp.N[a_best, b])
                season_id = mdp.next_season(season_id)

            relabelled[t] = total
    return (1 - lam) * logged + lam * relabelled


def context_relabel_dataset(trajs, demand_model, mdp, lam=1.0):
    return [context_relabel_one(tr, demand_model, mdp, lam=lam) for tr in trajs]


def make_context_dt_policy(model, mdp, target_return, device=None):
    device = resolve_device(device, model)
    state = {"rtg": [], "s": [], "a": [], "t": []}

    def policy_fn(obs):
        _, t, _, _ = mdp.decode_obs(obs)
        if t == 0:
            state["rtg"] = [float(target_return)]
            state["s"] = []
            state["a"] = []
            state["t"] = []
        else:
            state["rtg"].append(state["rtg"][-1])
        state["s"].append(obs.astype(np.float32))
        state["t"].append(t)
        a_hist = state["a"] + [0]
        logits = _step_logits(model, state["rtg"], state["s"], a_hist, state["t"], device)
        a = int(torch.argmax(logits).item())
        state["a"].append(a)
        state["rtg"][-1] = state["rtg"][-1] - mdp.reward_from_obs_action(obs, a)
        return a

    return policy_fn


def eval_dt(model, mdp, init, train_rtg):
    target = float(np.quantile(np.concatenate(train_rtg), 0.95))
    pol = make_context_dt_policy(model, mdp, target)
    v, _ = mdp.evaluate_policy_fn(pol, init)
    return v, target


def plan_with_dm(dm, mdp, device=None):
    device = resolve_device(device, dm)
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    P, S = mdp.n_products, mdp.n_seasons
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    Rhat = np.zeros((H, P, S, A, B), dtype=float)
    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for p in range(P):
                for s in range(S):
                    for b, ref in enumerate(mdp.ref_grid):
                        obs = torch.tensor(mdp.obs(ref, t, p, s), dtype=torch.float32, device=device)
                        obs_rep = obs.unsqueeze(0).repeat(A, 1)
                        dem = dm(prices, obs_rep).detach().cpu().numpy()
                        Rhat[t, p, s, :, b] = mdp.prices * dem

    V = np.zeros((H + 1, P, S, B), dtype=float)
    pi = np.zeros((H, P, S, B), dtype=int)
    for t in reversed(range(H)):
        for p in range(P):
            for s in range(S):
                ns = mdp.next_season(s)
                for b in range(B):
                    q = Rhat[t, p, s, :, b] + V[t + 1, p, ns, mdp.N[:, b]]
                    pi[t, p, s, b] = int(q.argmax())
                    V[t, p, s, b] = q.max()
    return pi, V


def tabular_policy_fn(pi, mdp):
    def fn(obs):
        b, t, p, s = mdp.decode_obs(obs)
        return int(pi[t, p, s, b])

    return fn


def setup_context(cfg, mode, noise, seed, products, seasons, season_amp):
    sc = copy.deepcopy(cfg.sim)
    sc.demand_noise = noise
    sc.seed = seed
    use_product = mode in {"product", "season_product"}
    use_season = mode in {"seasonal", "season_product"}
    mdp = CommercialContextPricingMDP(
        sc,
        n_products=products,
        n_seasons=seasons,
        use_product=use_product,
        use_season=use_season,
        season_amp=season_amp,
    )
    mdp.solve_optimal()
    init = mdp.initial_states(cfg.data.n_eval_episodes)
    v_opt = float(np.mean([mdp.Vstar[0, p, s, b] for b, p, s in init]))
    v_beh, _ = mdp.evaluate_policy_fn(mdp.myopic_policy, init)
    return mdp, init, v_opt, v_beh


def parse_modes(text):
    aliases = {"none": "baseline", "base": "baseline", "context": "season_product"}
    allowed = {"baseline", "seasonal", "product", "season_product"}
    modes = []
    for raw in text.split(","):
        item = raw.strip().lower().replace("-", "_")
        if not item:
            continue
        item = aliases.get(item, item)
        if item not in allowed:
            raise ValueError(f"Unknown mode '{raw}'. Expected one of {sorted(allowed)}.")
        modes.append(item)
    return modes or ["baseline", "season_product"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument(
        "--modes",
        default="baseline,season_product",
        help="Comma-separated: baseline, seasonal, product, season_product.",
    )
    ap.add_argument("--products", type=int, default=3)
    ap.add_argument("--seasons", type=int, default=4)
    ap.add_argument("--season-amp", type=float, default=0.35)
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    modes = parse_modes(args.modes)
    n_traj = min(cfg.exp.data_sizes)
    noise = max(cfg.exp.noise_levels)
    rows = []

    print(
        "commercial context diagnostic: "
        f"N={n_traj} noise={noise} modes={modes} seeds={cfg.exp.seeds}"
    )
    for mode in modes:
        for seed in cfg.exp.seeds:
            _seed(seed)
            mdp, init, v_opt, v_beh = setup_context(
                cfg,
                mode,
                noise,
                seed,
                args.products,
                args.seasons,
                args.season_amp,
            )
            trajs = make_context_stitching_necessary(
                mdp,
                n_traj,
                noise,
                seed,
                cfg.data.expert_q,
            )

            dm_struct = StructuredDemandModel(
                mdp.obs_dim,
                cfg.model.elasticity_lo,
                cfg.model.elasticity_hi,
            )
            fit_demand_model(dm_struct, trajs, mdp, cfg.model.demand_epochs)
            dm_uncon = UnconstrainedDemandModel(mdp.obs_dim)
            fit_demand_model(dm_uncon, trajs, mdp, cfg.model.demand_epochs)

            rtg_vanilla = logged_rtg(trajs)
            rtg_struct = context_relabel_dataset(
                trajs,
                dm_struct,
                mdp,
                cfg.model.relabel_lambda,
            )

            dt_v = train_dt(
                D.pack_dt(trajs, rtg_vanilla),
                mdp.obs_dim,
                mdp.cfg.n_prices,
                cfg.model,
                seed=seed,
            )
            v_dt, _ = eval_dt(dt_v, mdp, init, rtg_vanilla)
            dt_s = train_dt(
                D.pack_dt(trajs, rtg_struct),
                mdp.obs_dim,
                mdp.cfg.n_prices,
                cfg.model,
                seed=seed,
            )
            v_sdt, _ = eval_dt(dt_s, mdp, init, rtg_struct)

            row = dict(
                mode=mode,
                seed=seed,
                use_season=mode in {"seasonal", "season_product"},
                use_product=mode in {"product", "season_product"},
                n_products=mdp.n_products,
                n_seasons=mdp.n_seasons,
                v_opt=round(v_opt, 3),
                v_behaviour=round(v_beh, 3),
                vanillaDT=round(M.normalised_value(v_dt, v_beh, v_opt), 3),
                structuredDT_context=round(M.normalised_value(v_sdt, v_beh, v_opt), 3),
            )

            for tag, dm in (("structured", dm_struct), ("unconstrained", dm_uncon)):
                pi, Vhat = plan_with_dm(dm, mdp)
                v_true, _ = mdp.evaluate_policy_fn(tabular_policy_fn(pi, mdp), init)
                inmodel_value = float(np.mean([Vhat[0, p, s, b] for b, p, s in init]))
                row[f"EtO_{tag}"] = round(M.normalised_value(v_true, v_beh, v_opt), 3)
                row[f"inmodel_{tag}"] = round(inmodel_value, 3)
                row[f"truegap_{tag}"] = round(inmodel_value - v_true, 3)

            rows.append(row)
            print(
                f"{mode} seed {seed}: "
                f"vanilla={row['vanillaDT']:.3f} "
                f"structDT={row['structuredDT_context']:.3f} "
                f"EtO_struct={row['EtO_structured']:.3f} "
                f"EtO_uncon={row['EtO_unconstrained']:.3f}"
            )

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    # Record the commit, device and library versions this run actually used; the
    # parameters alone do not pin a result. See REPRODUCE.md.
    provenance.stamp(args.outdir, replace=True)
    out = os.path.join(args.outdir, "commercial_context.csv")
    df.to_csv(out, index=False)

    tests = []
    for mode, g in df.groupby("mode"):
        med, p = M.paired_test(g["structuredDT_context"], g["EtO_structured"])
        tests.append(
            dict(
                mode=mode,
                comparison="structuredDT_context - EtO_structured",
                median_diff=round(med, 3),
                p_value=round(p, 6) if np.isfinite(p) else np.nan,
            )
        )
        med, p = M.paired_test(g["structuredDT_context"], g["vanillaDT"])
        tests.append(
            dict(
                mode=mode,
                comparison="structuredDT_context - vanillaDT",
                median_diff=round(med, 3),
                p_value=round(p, 6) if np.isfinite(p) else np.nan,
            )
        )
    test_df = pd.DataFrame(tests)
    test_out = os.path.join(args.outdir, "commercial_context_tests.csv")
    test_df.to_csv(test_out, index=False)

    print(f"\nWrote {out}")
    print(f"Wrote {test_out}")
    print("\n=== commercial context summary ===")
    summary = df.groupby("mode").mean(numeric_only=True).reset_index()
    for _, r in summary.iterrows():
        print(
            f"{r['mode']}: vanilla={r.vanillaDT:.3f} "
            f"structDT={r.structuredDT_context:.3f} "
            f"EtO_struct={r.EtO_structured:.3f} "
            f"EtO_uncon={r.EtO_unconstrained:.3f} "
            f"truegap_struct={r.truegap_structured:+.1f}"
        )


if __name__ == "__main__":
    main()
