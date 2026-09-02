"""Reference-price pricing MDP: the controlled testbed (E0).

State for ground truth = (reference-price bin, timestep). The price chosen at t
shifts the reference price, which changes demand at t+1, so optimal pricing is
intertemporal and stitching across sub-trajectories is genuinely required.

Because the data-generating process is known, we obtain EXACT ground truth:
  - the optimal policy / value by finite-horizon backward induction
  - the value of ANY policy by deterministic expected rollout (no MC noise)

Demand-noise corrupts only the *logged* rewards (the training signal); policy
*evaluation* uses expected reward, so high noise hurts methods that condition on
luck-inflated realised returns -- exactly the stochasticity failure mode.

References:
  - Gallego and van Ryzin (1994), finite-horizon stochastic dynamic pricing.
  - Popescu and Wu (2007), dynamic pricing with reference-price effects.
  - Hausenblas et al. (2025), recent offline-RL framing for dynamic pricing.

Implements: environment E1 of §3.2.1. The exact dynamic programming here supplies both
normalisation anchors of §3.5.1.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: PricingMDP(), solve_optimal(), evaluate_policy_fn().
#
# Reference-price pricing MDP: today's price shifts tomorrow's reference point, and the
# whole MDP is small enough that backward induction gives the exact optimum.
#
# Implements or follows:
#   - Popescu, I. and Wu, Y. (2007) 'Dynamic Pricing Strategies with Reference Effects',
#     Operations Research, 55(3).
#   - Train, K.E. (2009) Discrete Choice Methods with Simulation. 2nd edn. Cambridge
#     University Press.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
from pricing_dt.core.config import SimConfig


class PricingMDP:
    """Environment E1: finite-horizon pricing with a reference-price state.

    Demand is u = alpha - beta*p + delta*(ref - p) with E[q] = M*sigmoid(u), and today's
    price moves tomorrow's reference by ref' = (1 - eta)*ref + eta*p. That transition is
    what makes the problem sequential rather than a bandit, and it is the feature the
    myopic prior cannot represent.

    The price and reference grids are small enough for exact backward induction, so
    `solve_optimal` gives the sequential optimum that anchors 1 on the normalised scale
    of Section 3.5.1; the oracle myopic policy anchoring 0 is built from `self.R` in
    `experiments._setup`.
    """
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.prices = np.linspace(cfg.p_min, cfg.p_max, cfg.n_prices)
        # reference price lives on the same range as prices
        self.ref_grid = np.linspace(cfg.p_min, cfg.p_max, cfg.n_ref_bins)
        self.H = cfg.horizon
        self.rng = np.random.default_rng(cfg.seed)
        self._build_dynamics()

    # ------- core demand / reward / transition (the true DGP) -------
    def expected_demand(self, price, ref):
        """E[demand] under the true logit model (vectorised over price/ref)."""
        c = self.cfg
        u = c.alpha - c.beta * price + c.delta * (ref - price)
        return c.market_size * _sigmoid(u)

    def expected_reward(self, price, ref):
        return price * self.expected_demand(price, ref)

    def sample_reward(self, price, ref, noise):
        q = self.expected_demand(price, ref)
        if noise > 0:
            # MEDIAN-one, not mean-one: exp(N(0, s)) has median 1 and mean exp(s^2/2), so
            # at the highest sweep level (s = 0.5) a logged reward sits 13.3% above the
            # expected reward on average. Anchors and every policy value are computed
            # from expected_reward and carry none of this; the shift lands only on the
            # logged training signal. Left uncorrected deliberately, because a real log
            # is a noisy realisation rather than a mean-preserving perturbation, but
            # named here so it is not read as mean-one. For mean-one use
            # exp(N(-s^2/2, s)).
            q = q * np.exp(self.rng.normal(0.0, noise))
        return price * q

    def next_ref(self, price, ref):
        return (1 - self.cfg.eta) * ref + self.cfg.eta * price

    def ref_to_bin(self, ref):
        return int(np.argmin(np.abs(self.ref_grid - ref)))

    def ref_to_bin_batch(self, refs):
        """Vectorised `ref_to_bin` over an array of reference prices."""
        refs = np.asarray(refs)
        return np.argmin(np.abs(self.ref_grid[None, :] - refs[:, None]), axis=1)

    # ------- observation for learned models -------
    def obs(self, ref, t):
        """Continuous observation fed to BC/DT/CQL: [ref_scaled, t_scaled]."""
        c = self.cfg
        ref_s = (ref - c.p_min) / (c.p_max - c.p_min)
        return np.array([ref_s, t / max(1, self.H - 1)], dtype=np.float32)

    def obs_batch(self, refs, t):
        """`obs` for a whole batch of reference prices at a shared timestep t."""
        c = self.cfg
        ref_s = (np.asarray(refs) - c.p_min) / (c.p_max - c.p_min)
        t_s = np.full(ref_s.shape, t / max(1, self.H - 1))
        return np.stack([ref_s, t_s], axis=1).astype(np.float32)

    def obs_to_bins(self, obs):
        """Recover reference bins from a batch of observations."""
        obs = np.asarray(obs)
        refs = self.cfg.p_min + obs[:, 0] * (self.cfg.p_max - self.cfg.p_min)
        return self.ref_to_bin_batch(refs)

    def decode_obs(self, obs):
        """Inverse of obs(): recover (ref_bin, t) from an observation vector."""
        ref = self.cfg.p_min + obs[0] * (self.cfg.p_max - self.cfg.p_min)
        t = int(round(obs[1] * max(1, self.H - 1)))
        return self.ref_to_bin(ref), t

    # ------- exact dynamic programming on the discretised MDP -------
    def _build_dynamics(self):
        """Precompute expected reward R[a, ref_bin] and next-bin index N[a, ref_bin]."""
        A, B = self.cfg.n_prices, self.cfg.n_ref_bins
        self.R = np.zeros((A, B))
        self.N = np.zeros((A, B), dtype=int)
        for a, p in enumerate(self.prices):
            for b, ref in enumerate(self.ref_grid):
                self.R[a, b] = self.expected_reward(p, ref)
                self.N[a, b] = self.ref_to_bin(self.next_ref(p, ref))

    def solve_optimal(self):
        """Backward induction. Returns Qstar[t,b,a], Vstar[t,b], pistar[t,b]."""
        A, B, H = self.cfg.n_prices, self.cfg.n_ref_bins, self.H
        Q = np.zeros((H, B, A))
        V = np.zeros((H + 1, B))
        for t in reversed(range(H)):
            for b in range(B):
                q_ta = self.R[:, b] + V[t + 1, self.N[:, b]]
                Q[t, b] = q_ta
                V[t, b] = q_ta.max()
        pistar = Q.argmax(axis=2)        # [H, B]
        self.Qstar, self.Vstar, self.pistar = Q, V, pistar
        return Q, V, pistar

    def evaluate_tabular_policy(self, pi):
        """Exact value of a tabular policy pi[t,b] via policy evaluation (expected)."""
        A, B, H = self.cfg.n_prices, self.cfg.n_ref_bins, self.H
        V = np.zeros((H + 1, B))
        for t in reversed(range(H)):
            for b in range(B):
                a = pi[t, b]
                V[t, b] = self.R[a, b] + V[t + 1, self.N[a, b]]
        return V

    # ------- exact evaluation of an arbitrary (continuous-obs) policy -------
    def evaluate_policy_fn(self, policy_fn, init_bins):
        """Deterministic expected rollout from each initial reference bin.

        policy_fn(obs)->action_index. Returns mean exact return over init_bins.
        Transition uses the binned next-ref so it matches the DP ground truth.

        The rollout is inherently lock-step: every episode is at the same
        timestep t at the same moment, and the transition is deterministic given
        the action. So if the policy exposes a `.batched(obs_batch, t)` callable
        (the neural policies do), all initial bins are advanced together and the
        network sees one [B, ...] forward per timestep instead of B separate
        batch-1 forwards. That is what makes GPU execution worthwhile here: the
        scalar path is dominated by per-call launch overhead, not by arithmetic.
        """
        batched = getattr(policy_fn, "batched", None)
        if batched is not None:
            return self._evaluate_policy_batched(batched, init_bins)
        returns = []
        for b0 in init_bins:
            ref = self.ref_grid[b0]
            total = 0.0
            for t in range(self.H):
                a = int(policy_fn(self.obs(ref, t)))
                b = self.ref_to_bin(ref)
                total += self.R[a, b]
                ref = self.ref_grid[self.N[a, b]]
            returns.append(total)
        return float(np.mean(returns)), np.array(returns)

    def _evaluate_policy_batched(self, batched_fn, init_bins):
        """Lock-step vectorised twin of `evaluate_policy_fn`.

        Exactly equivalent to the scalar loop: `ref` there is always a grid
        point (it starts at `ref_grid[b0]` and is reassigned from `ref_grid[N]`),
        so `ref_to_bin(ref)` is just the bin index we carry directly.
        """
        bins = np.asarray(init_bins, dtype=int).copy()
        if bins.size == 0:
            return float("nan"), np.array([])
        total = np.zeros(bins.shape[0], dtype=float)
        for t in range(self.H):
            actions = np.asarray(batched_fn(self.obs_batch(self.ref_grid[bins], t), t),
                                 dtype=int)
            total += self.R[actions, bins]
            bins = self.N[actions, bins]
        return float(np.mean(total)), total

    def initial_bins(self, n):
        """Sample initial reference bins (uniform over the grid)."""
        return self.rng.integers(0, self.cfg.n_ref_bins, size=n)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
