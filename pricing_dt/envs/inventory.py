"""Second environment: lost-sales inventory control with censored, overdispersed demand.

PURPOSE. The channel result (a domain model is safe in the goal channel and toxic in
the action channel) currently rests on one testbed. Replicating it needs an environment
that reproduces BOTH halves, and the toxic half is the hard one. From the pricing
analysis, three conditions must hold:

  (a) genuine sequential coupling, so a greedy roll-forward compounds the model's
      optimism over the horizon;
  (b) a structural prior that is systematically OPTIMISTIC off the logged support --
      without this the planner has nothing to exploit and there is no collapse;
  (c) a planner able to reach that region.

WHY PLAIN NEWSVENDOR FAILS (b). Under lost-sales censoring a naively fitted demand
model UNDER-estimates demand. An under-estimating model also predicts low profit, so
its in-model value is low too: that is ordinary estimation error, not the optimizer's
curse, whose signature is in-model value >> true value.

THE MECHANISM THAT DOES WORK: under-estimated demand VARIANCE.
True demand here is overdispersed (a mixture with occasional spikes). The logging
policy orders conservatively, so the upper tail is censored. The structural prior is
**Poisson**, whose variance is identically equal to its mean and which therefore
*cannot represent overdispersion at all* -- the exact analogue of the bounded-elasticity
prior, which could not represent a steep enough demand decline. A planner using it
believes stockouts are rare, so it under-orders safety stock: in-model profit looks
high (little holding cost, few stockouts believed) while true profit collapses through
lost sales. Poisson demand is also a textbook-standard inventory model, so this is a
realistic prior rather than an engineered strawman.

Sequential coupling (a) comes from inventory carry-over: x' = max(0, x + q - D).

EXACT GROUND TRUTH. State and demand are discretised, so the transition kernel
P[a, x, x'] and expected reward R[a, x] are computed exactly, and the value of any
MARKOV policy comes from exact dynamic programming / exact forward propagation of the
state distribution -- no Monte-Carlo noise. A return-conditioned controller is NOT Markov
(its action depends on the running return-to-go, which is path-dependent), so its value
here is a Monte-Carlo estimate; the anchors stay exact either way.
Note the transition here is STOCHASTIC (the pricing MDP's was deterministic), so
evaluation propagates a distribution over states rather than a single trajectory.

References:
  - The Poisson/lost-sales prior is a standard inventory-control modelling
    choice; it is used here as a defensible structural prior, not as a new
    inventory algorithm.
  - Hausenblas et al. (2025) motivates the offline-RL setting for revenue and
    pricing decisions where online exploration is costly.

Implements: environment E2 of §3.2.2, including the censoring that makes the log record
sales rather than demand.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: InventoryMDP(), solve_optimal(), OrderUpTo().
#
# Lost-sales inventory MDP with censored demand: the log records sales, not demand,
# because stockouts hide the rest. Poisson prior cannot represent the true
# overdispersion.
#
# Implements or follows:
#   - Talluri, K.T. and van Ryzin, G.J. (2004) The Theory and Practice of Revenue
#     Management. Springer.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
from dataclasses import dataclass
import numpy as np


@dataclass
class InvConfig:
    horizon: int = 8               # H
    max_inventory: int = 20        # inventory grid 0..max_inventory
    max_order: int = 10            # action grid 0..max_order
    max_demand: int = 30           # demand support 0..max_demand
    price: float = 5.0             # revenue per unit sold
    order_cost: float = 2.0        # cost per unit ordered
    hold_cost: float = 0.5         # holding cost per unit of ending inventory
    stockout_penalty: float = 8.0  # goodwill/backorder cost per unit of UNMET demand.
                                   # Load-bearing for condition (b): without a penalty on
                                   # the bad tail, under-stating demand VARIANCE does not
                                   # inflate in-model value -- the model simply fails to
                                   # predict sales it also fails to make, so its value
                                   # estimate falls with the truth instead of above it.
    # overdispersed demand: mixture of a base rate and an occasional spike
    mu_base: float = 4.0
    mu_spike: float = 14.0
    p_spike: float = 0.20
    seed: int = 0


def _poisson_pmf(mu, kmax):
    k = np.arange(kmax + 1)
    logp = -mu + k * np.log(max(mu, 1e-9)) - np.array(
        [np.sum(np.log(np.arange(1, i + 1))) if i > 0 else 0.0 for i in k])
    p = np.exp(logp)
    p[-1] += max(0.0, 1.0 - p.sum())      # fold the tail onto the last atom
    return p / p.sum()


class InventoryMDP:
    """Lost-sales inventory MDP with exact DP on a discrete grid."""

    def __init__(self, cfg: InvConfig):
        self.cfg = cfg
        self.H = cfg.horizon
        self.n_states = cfg.max_inventory + 1          # inventory levels
        self.n_actions = cfg.max_order + 1             # order quantities
        self.rng = np.random.default_rng(cfg.seed)
        self.demand_pmf = self._true_demand_pmf()
        self._build_dynamics()

    # ---------------- true demand ----------------
    def _true_demand_pmf(self):
        c = self.cfg
        lo = _poisson_pmf(c.mu_base, c.max_demand)
        hi = _poisson_pmf(c.mu_spike, c.max_demand)
        return (1 - c.p_spike) * lo + c.p_spike * hi

    def demand_mean(self):
        return float(np.dot(np.arange(self.cfg.max_demand + 1), self.demand_pmf))

    def demand_var(self):
        k = np.arange(self.cfg.max_demand + 1)
        m = self.demand_mean()
        return float(np.dot((k - m) ** 2, self.demand_pmf))

    # ---------------- dynamics ----------------
    def _step_arrays(self, pmf):
        """R[a,x] and P[a,x,x'] under an arbitrary demand pmf (true or fitted)."""
        c = self.cfg
        A, S, Dm = self.n_actions, self.n_states, c.max_demand
        R = np.zeros((A, S))
        P = np.zeros((A, S, S))
        d = np.arange(Dm + 1)
        for a in range(A):
            for x in range(S):
                avail = min(x + a, c.max_inventory)
                sales = np.minimum(d, avail)
                end = avail - sales
                unmet = np.maximum(d - avail, 0)
                rew = (c.price * sales - c.order_cost * a - c.hold_cost * end
                       - c.stockout_penalty * unmet)
                R[a, x] = float(np.dot(rew, pmf))
                for dd in range(Dm + 1):
                    P[a, x, int(end[dd])] += pmf[dd]
        return R, P

    def _build_dynamics(self):
        self.R, self.P = self._step_arrays(self.demand_pmf)

    # ---------------- exact DP ----------------
    def solve_optimal(self, R=None, P=None):
        """Backward induction over the exact (inventory, period) grid.

        Called on the true dynamics it gives the sequential optimum anchoring 1 on E2's
        normalised scale. `R`/`P` are overridable so the same routine can be run on a
        model's believed dynamics, which is how the in-model gap of Equation (3.4) is
        measured.
        """
        R = self.R if R is None else R
        P = self.P if P is None else P
        A, S, H = self.n_actions, self.n_states, self.H
        Q = np.zeros((H, S, A))
        V = np.zeros((H + 1, S))
        for t in reversed(range(H)):
            for a in range(A):
                Q[t, :, a] = R[a] + P[a] @ V[t + 1]
            V[t] = Q[t].max(1)
        pi = Q.argmax(2)
        self.Qstar, self.Vstar, self.pistar = Q, V, pi
        return Q, V, pi

    def evaluate_tabular_policy(self, pi, R=None, P=None):
        """Exact value of pi[t,x] under the TRUE dynamics (unless overridden)."""
        R = self.R if R is None else R
        P = self.P if P is None else P
        S, H = self.n_states, self.H
        V = np.zeros((H + 1, S))
        for t in reversed(range(H)):
            for x in range(S):
                a = int(pi[t, x])
                V[t, x] = R[a, x] + P[a, x] @ V[t + 1]
        return V

    def evaluate_policy_fn(self, policy_fn, init_dist):
        """Exact expected return of an arbitrary obs->action policy, by forward
        propagation of the state distribution (no sampling)."""
        S, H = self.n_states, self.H
        dist = np.asarray(init_dist, float)
        dist = dist / dist.sum()
        total = 0.0
        for t in range(H):
            nxt = np.zeros(S)
            for x in range(S):
                if dist[x] <= 0:
                    continue
                a = int(policy_fn(self.obs(x, t)))
                total += dist[x] * self.R[a, x]
                nxt += dist[x] * self.P[a, x]
            dist = nxt
        return float(total)

    # ---------------- observations ----------------
    def obs(self, x, t):
        return np.array([x / max(1, self.cfg.max_inventory),
                         t / max(1, self.H - 1)], dtype=np.float32)

    def decode_obs(self, o):
        x = int(round(float(o[0]) * self.cfg.max_inventory))
        t = int(round(float(o[1]) * max(1, self.H - 1)))
        return x, t

    # ---------------- sampling (data generation) ----------------
    def sample_demand(self, rng):
        return int(rng.choice(self.cfg.max_demand + 1, p=self.demand_pmf))

    def step(self, x, a, rng):
        """Returns (reward, next_x, observed_sales, censored_flag).

        `observed_sales = min(demand, available)` is what a logged dataset records;
        when it equals availability the true demand is CENSORED and unobservable."""
        c = self.cfg
        avail = min(x + a, c.max_inventory)
        d = self.sample_demand(rng)
        sales = min(d, avail)
        end = avail - sales
        unmet = max(d - avail, 0)
        rew = (c.price * sales - c.order_cost * a - c.hold_cost * end
               - c.stockout_penalty * unmet)
        return float(rew), int(end), int(sales), bool(d >= avail)


# ---------------- logging policies ----------------
class OrderUpTo:
    """Conservative order-up-to-S logger. Low S keeps inventory (and therefore
    availability) low, which is exactly what censors the demand upper tail."""

    def __init__(self, mdp: InventoryMDP, level, rng, epsilon=0.1):
        self.mdp, self.level, self.rng, self.eps = mdp, level, rng, epsilon

    def action(self, x, t):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(0, self.mdp.n_actions))
        return int(np.clip(self.level - x, 0, self.mdp.cfg.max_order))
