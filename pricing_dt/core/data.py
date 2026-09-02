"""Logging policies and dataset generation.

Provides:
  - behaviour policies (epsilon-greedy-around-suboptimal, region-specialised,
    Q-snapshot loggers for non-stationary data)
  - trajectory generation with noisy logged rewards
  - stitching-necessary datasets (no single trajectory is globally good)
  - non-stationary K-segment logs with controllable drift (E3)
  - packing into Decision-Transformer tensors

Implements: the logging policies of §3.2.1, including the two region specialists whose
logs make stitching necessary.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: Trajectory(), pack_dt(), RegionSpecialised(), make_standard().
#
# Trajectory container, Decision-Transformer packing, and the logging policies that
# generate the offline datasets.
#
# Implements or follows:
#   - Chen, L. et al. (2021) 'Decision Transformer: Reinforcement Learning via Sequence
#     Modeling', NeurIPS 34. arXiv:2106.01345.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
from dataclasses import dataclass
import numpy as np
from pricing_dt.core.simulator import PricingMDP


@dataclass
class Trajectory:
    obs: np.ndarray         # [H, obs_dim]
    ref_bins: np.ndarray    # [H] int   (true reference bin at each step; for ground-truth/relabelling)
    actions: np.ndarray     # [H] int   (price index)
    rewards: np.ndarray     # [H] float (NOISY logged reward)
    seg: int = 0            # segment id (non-stationary logs)

    @property
    def rtg(self):
        """logged return-to-go from noisy rewards (what vanilla DT conditions on)."""
        return np.cumsum(self.rewards[::-1])[::-1].copy()

    @property
    def total_return(self):
        return float(self.rewards.sum())


# ----------------------- behaviour policies -----------------------
class EpsGreedyAroundSuboptimal:
    """Greedy w.r.t. a *myopic* (one-step) reward -> intentionally sub-optimal
    because it ignores the reference-price dynamics. Eps-explores."""
    def __init__(self, mdp: PricingMDP, epsilon, rng):
        self.mdp, self.eps, self.rng = mdp, epsilon, rng

    def action(self, ref_bin, t):
        if self.rng.random() < self.eps:
            return self.rng.integers(0, self.mdp.cfg.n_prices)
        return int(self.mdp.R[:, ref_bin].argmax())   # myopic greedy => suboptimal


class RegionSpecialised:
    """Boundedly-rational specialist: good in one half of reference space, poor in
    the other. Mixing two complementary specialists makes stitching necessary.

    In its region the specialist prices PARTWAY from the myopic action toward the
    intertemporal optimum, controlled by `expert_q` in [0, 1]:
      expert_q = 1.0  -> plays the exact DP optimum in-region. The two specialists
                         then COLLECTIVELY contain the optimal action for every state
                         (each in its own region), so a return-conditioned DT can
                         STITCH those pieces; this is the validated sequential
                         stitching regime used by the main diagnostics.
      expert_q < 1.0  -> a boundedly-rational expert. This was tried as a way to lower
                         the per-start logged ceiling (make single trajectories beatable),
                         but it BACKFIRES: it removes the optimal actions from the data
                         entirely, and an imitation-based DT cannot reinvent actions it
                         never observed even when relabelled to aim high — so every
                         variant degrades (nv collapses toward / below behaviour). The
                         empirical finding is that 1.0 is best; the knob is retained for
                         transparency and future non-imitation methods.
    Out of region it is myopic (ignores the dynamics entirely)."""
    def __init__(self, mdp: PricingMDP, low_region: bool, rng, eps=0.05, expert_q=1.0):
        self.mdp, self.low, self.rng, self.eps = mdp, low_region, rng, eps
        self.expert_q = expert_q
        mdp.solve_optimal()
        self.pistar = mdp.pistar

    def action(self, ref_bin, t):
        mid = self.mdp.cfg.n_ref_bins // 2
        in_region = (ref_bin < mid) if self.low else (ref_bin >= mid)
        a_myopic = int(self.mdp.R[:, ref_bin].argmax())
        if in_region and self.rng.random() > self.eps:
            a_opt = int(self.pistar[t, ref_bin])
            # boundedly-rational: price index partway from myopic toward optimal
            return int(round((1 - self.expert_q) * a_myopic + self.expert_q * a_opt))
        return a_myopic                                  # myopic elsewhere


class SoftmaxQSnapshot:
    """Logger defined by a (possibly partially trained) Q table, used for
    non-stationary logs. temp controls stochasticity; train_frac in [0,1]
    interpolates Q from a MYOPIC (one-step-reward) table toward the optimal Q,
    modelling a still-learning agent that begins greedy on immediate revenue and
    gradually internalises the reference-price dynamics.

    Crucially this is a PREFERENCE drift, not merely a sharpness change: because
    the reference-price coupling makes the myopic-optimal and intertemporal-optimal
    prices genuinely different (the C0 gap), early- and late-training snapshots
    prefer different actions. Interpolating from a flat (uniform) Q instead would
    leave every snapshot ranking actions by the same Qstar order and varying only
    varying in softmax temperature — too small a logging drift for the C3 pooled-
    vs-segmented bias to surface. Myopic->optimal gives a real total-variation
    sweep across segments."""
    def __init__(self, mdp: PricingMDP, train_frac, temp, rng):
        mdp.solve_optimal()
        self.mdp, self.temp, self.rng = mdp, max(temp, 1e-3), rng
        # interpolate between a MYOPIC Q (one-step reward, broadcast over time) and optimal Q
        q_myopic = np.repeat(mdp.R.T[None, :, :], mdp.H, axis=0)   # [H, B, A]
        self.Q = (1 - train_frac) * q_myopic + train_frac * mdp.Qstar

    def probs(self, ref_bin, t):
        q = self.Q[t, ref_bin] / self.temp
        q = q - q.max()
        p = np.exp(q)
        return p / p.sum()

    def action(self, ref_bin, t):
        return int(self.rng.choice(self.mdp.cfg.n_prices, p=self.probs(ref_bin, t)))


# ----------------------- generation -----------------------
def rollout(mdp: PricingMDP, policy, n_traj, noise, rng, seg=0):
    trajs = []
    for _ in range(n_traj):
        b0 = int(rng.integers(0, mdp.cfg.n_ref_bins))
        ref = mdp.ref_grid[b0]
        obs, rbins, acts, rews = [], [], [], []
        for t in range(mdp.H):
            b = mdp.ref_to_bin(ref)
            a = policy.action(b, t)
            obs.append(mdp.obs(ref, t)); rbins.append(b); acts.append(a)
            rews.append(mdp.sample_reward(mdp.prices[a], ref, noise))
            ref = mdp.ref_grid[mdp.N[a, b]]
        trajs.append(Trajectory(np.array(obs, np.float32), np.array(rbins),
                                np.array(acts), np.array(rews, np.float32), seg))
    return trajs


def make_standard(mdp, n_traj, noise, epsilon, seed):
    """The default offline log: epsilon-greedy around a deliberately sub-optimal price.

    `epsilon` sets how much of the price grid the log ever visits, and so how much
    support any mask has to work with; `noise` corrupts the logged reward only, leaving
    the transition exact. Contrast `make_stitching_necessary`, whose region specialists make no
    single trajectory globally good.
    """
    rng = np.random.default_rng(seed)
    pol = EpsGreedyAroundSuboptimal(mdp, epsilon, rng)
    return rollout(mdp, pol, n_traj, noise, rng)


def make_stitching_necessary(mdp, n_traj, noise, seed, expert_q=1.0):
    """Half the trajectories from each complementary specialist. With expert_q=1.0
    the two specialists jointly contain the optimal action for every state (each in
    its region) but no single trajectory is globally optimal across the drift, so
    stitching is required. See RegionSpecialised for the expert_q knob."""
    rng = np.random.default_rng(seed)
    lo = RegionSpecialised(mdp, low_region=True, rng=rng, expert_q=expert_q)
    hi = RegionSpecialised(mdp, low_region=False, rng=rng, expert_q=expert_q)
    a = rollout(mdp, lo, n_traj // 2, noise, rng)
    b = rollout(mdp, hi, n_traj - n_traj // 2, noise, rng)
    return a + b


def make_nonstationary(mdp, n_segments, traj_per_seg, noise, temp, seed, spread=1.0):
    """K segments logged by Q snapshots. `spread` in [0,1] controls how different
    the segment loggers are: spread=0 => all segments use the same snapshot (no
    drift, pooled estimator unbiased); spread=1 => snapshots span a wide training
    range (strong drift, pooled estimator biased). Returns (trajs, loggers)."""
    rng = np.random.default_rng(seed)
    half = 0.4 * spread
    fracs = np.linspace(0.5 - half, 0.5 + half, n_segments)
    trajs, loggers = [], []
    for s, fr in enumerate(fracs):
        lg = SoftmaxQSnapshot(mdp, train_frac=float(np.clip(fr, 0.02, 1.0)),
                              temp=max(temp, 0.05), rng=rng)
        loggers.append(lg)
        trajs += rollout(mdp, lg, traj_per_seg, noise, rng, seg=s)
    return trajs, loggers


# ----------------------- DT tensor packing -----------------------
def pack_dt(trajs, rtg_list):
    """Stack trajectories into DT tensors. rtg_list supplies the (possibly
    relabelled) return-to-go per trajectory, so vanilla/structured/Q-DT differ
    only in this column. Returns dict of np arrays [N, H, ...]."""
    N = len(trajs); H = trajs[0].obs.shape[0]; od = trajs[0].obs.shape[1]
    S = np.zeros((N, H, od), np.float32)
    A = np.zeros((N, H), np.int64)
    G = np.zeros((N, H), np.float32)
    T = np.zeros((N, H), np.int64)
    for i, (tr, g) in enumerate(zip(trajs, rtg_list)):
        S[i] = tr.obs; A[i] = tr.actions; G[i] = g; T[i] = np.arange(H)
    return dict(states=S, actions=A, rtg=G, timesteps=T)
