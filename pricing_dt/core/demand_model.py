"""Demand models used to supply the relabelling signal (the methodological core).

StructuredDemandModel encodes the economic prior:
  - monotonicity: expected demand is non-increasing in price (enforced exactly
    by parameterising the price coefficient as strictly positive)
  - bounded elasticity: the price coefficient is squashed into a range implied
    by [elasticity_lo, elasticity_hi], so the model cannot produce economically
    absurd extrapolations in sparsely-covered regions.

UnconstrainedDemandModel is the ablation (B): no economic prior. NOTE the capacities
are NOT matched -- the structured model is two one-hidden-layer heads (514 parameters),
the unconstrained model one two-hidden-layer network (4481). The gap is conservative
wherever the structured model wins, and must be stated wherever the unconstrained one
does.

Both predict E[demand] given (price, state). Fit by regression on logged
(price, state) -> realised demand. (demand = reward / price.)

References:
  - Ban and Keskin (2021), machine-learning dynamic pricing with heterogeneous
    elasticity.
  - Train (2009), discrete-choice demand modelling background.

Implements: the structured demand prior of §3.2.1 and its unconstrained and
misspecified comparators; panel C of Figure C.1 gives the parameterisation.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: StructuredDemandModel(), UnconstrainedDemandModel().
#
# The two demand models the study contrasts: a structured prior enforcing monotone,
# bounded-elasticity demand, and an unconstrained tabular/parametric fit.
#
# Implements or follows:
#   - Train, K.E. (2009) Discrete Choice Methods with Simulation. 2nd edn. Cambridge
#     University Press.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from pricing_dt.core.torch_utils import align_inputs, resolve_device


class StructuredDemandModel(nn.Module):
    r"""log-demand = g(s) - beta(s) * price, with beta(s) > 0 and bounded.

    g(s): unconstrained MLP (baseline log-demand level).
    beta(s): the log-demand slope in the price LEVEL, squashed to [e_lo, e_hi] via a
             sigmoid. What is bounded is the SEMI-elasticity: -d log q / d p = beta(s).
             The price elasticity is -d log q / d log p = beta(s) * p, so it equals beta
             only at p = 1 and spans [0.5*beta, 2*beta] over the price grid [0.5, 2.0].
             The monotonicity guarantee, d E[demand]/d price < 0, follows from beta > 0
             and is unaffected by the distinction.
    """
    def __init__(self, obs_dim, e_lo, e_hi, hidden=64):
        super().__init__()
        self.e_lo, self.e_hi = e_lo, e_hi
        self.g = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))
        self.b = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))

    def beta(self, s):
        s = align_inputs(self, s)
        return self.e_lo + (self.e_hi - self.e_lo) * torch.sigmoid(self.b(s))

    def log_demand(self, price, s):
        price, s = align_inputs(self, price, s)
        return self.g(s).squeeze(-1) - self.beta(s).squeeze(-1) * price

    def forward(self, price, s):
        return torch.exp(self.log_demand(price, s))   # E[demand] >= 0, monotone decreasing


class UnconstrainedDemandModel(nn.Module):
    """Plain MLP on [s, price] -> demand. No monotonicity, no elasticity bound."""
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Softplus())

    def forward(self, price, s):
        price, s = align_inputs(self, price, s)
        x = torch.cat([s, price.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


def fit_demand_model(model, trajs, mdp, epochs, lr=5e-3, device=None, misspecify=False,
                     batch_size=256):
    """Fit on logged (state, price, realised demand). `misspecify` perturbs the
    targets to emulate a wrong prior (ablation C is driven via elasticity bounds,
    but this flag also allows label perturbation if desired).

    The objective is a LOG-SPACE regression  (log q_hat - log q)^2  with
    minibatching. Demand is multiplicative/log-normal in this testbed (see
    SimConfig.demand_noise), so the natural, well-conditioned target is log-demand;
    a plain MSE-through-exp loss has vanishing gradients while predictions start
    tiny, which left the structured model badly underfit (its relabelled RTG came
    out ~60x too small). Log-space fitting recovers the correct revenue scale fast.
    """
    auto_device = device is None
    device = resolve_device(device, model)
    model.to(device)
    S, P, D = [], [], []
    for tr in trajs:
        S.append(tr.obs)
        P.append(mdp.prices[tr.actions])
        D.append(tr.rewards / np.maximum(mdp.prices[tr.actions], 1e-6))  # demand = reward/price
    S = torch.tensor(np.concatenate(S), dtype=torch.float32, device=device)
    P = torch.tensor(np.concatenate(P), dtype=torch.float32, device=device)
    D = torch.tensor(np.concatenate(D), dtype=torch.float32, device=device)
    if misspecify:
        D = D * 1.5
    logD = torch.log(torch.clamp(D, min=1e-3))
    n = S.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(P[idx], S[idx])
            log_pred = torch.log(torch.clamp(pred, min=1e-3))
            loss = ((log_pred - logD[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    if auto_device:
        model.to(device)
    return model


class MisspecifiedStructuredDemandModel(nn.Module):
    r"""A structured prior corrupted along a single 'severity' axis, for the
    formal misspecification scan.

    log-demand = g(s) - beta_eff * price + severity * kappa * price

    where beta_eff uses elasticity bounds shifted away from the truth by `severity`,
    and the extra +severity*kappa*price term progressively cancels and then flips
    the raw log-demand slope. At severity 0 the model is the correct
    monotone-decreasing prior; as severity grows the prior becomes wrong and can
    become NON-MONOTONE in raw log-demand. Very high severities may also saturate
    the forward() clamp, so diagnostics should inspect log_demand() directly.
    This gives an interpretable 'how wrong is the prior' knob to scan.
    """
    def __init__(self, obs_dim, e_lo, e_hi, severity=0.0, kappa=1.0, hidden=64):
        super().__init__()
        # bounds pushed wrong in proportion to severity (shifted upward, away from truth)
        self.e_lo = e_lo + severity * 0.5 * (e_hi - e_lo)
        self.e_hi = e_hi + severity * 1.0 * (e_hi - e_lo)
        self.severity, self.kappa = severity, kappa
        self.g = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))
        self.b = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                               nn.Linear(hidden, 1))

    def beta(self, s):
        s = align_inputs(self, s)
        return self.e_lo + (self.e_hi - self.e_lo) * torch.sigmoid(self.b(s))

    def log_demand(self, price, s):
        price, s = align_inputs(self, price, s)
        base = self.g(s).squeeze(-1) - self.beta(s).squeeze(-1) * price
        # corruption term scaled by the elasticity range so that at high severity
        # it exceeds the fitted slope and flips monotonicity (Veblen-like prior).
        bump = self.severity * self.kappa * (self.e_hi - self.e_lo) * price
        return base + bump

    def forward(self, price, s):
        return torch.exp(torch.clamp(self.log_demand(price, s), max=10.0))
