"""Non-DT baselines.

Implemented fully (discrete actions, short horizon -> tractable on CPU):
  - BehaviourCloning
  - DiscreteCQL   (Q-learning + conservative log-sum-exp penalty)
  - DiscreteIQL   (expectile value + advantage-weighted regression)

The study's claims hinge on structured-DT vs {vanilla DT, Q-DT}, which ARE fully
implemented; EDT is provided in dt.py. CQL/IQL are supporting value baselines.

References:
  - Kumar et al. (2020), Conservative Q-Learning.
  - Kostrikov, Nair and Levine (2022), Implicit Q-Learning.
  - Bhargava et al. (2024), modern DT/CQL/BC comparison context.

Implements: the comparator families of §3.4 -- behaviour cloning, discrete IQL, the
offline bandits of Appendix F.1, and the estimate-then-optimise planner.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: train_iql(), train_cql(), train_bc().
#
# Value-based and imitation baselines: implicit Q-learning, conservative Q-learning and
# behaviour cloning, all on the same data and evaluation protocol.
#
# Implements or follows:
#   - Kostrikov, I., Nair, A. and Levine, S. (2022) 'Offline Reinforcement Learning with
#     Implicit Q-Learning'. arXiv:2110.06169.
#   - Kumar, A., Zhou, A., Tucker, G. and Levine, S. (2020) 'Conservative Q-Learning for
#     Offline Reinforcement Learning', NeurIPS 33.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from pricing_dt.core.dt import _supported_actions
from pricing_dt.core.torch_utils import (align_inputs, dataloader_permutation,
                                          resolve_device)


class MLP(nn.Module):
    def __init__(self, i, o, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))

    def forward(self, x):
        return self.net(align_inputs(self, x))


# ----------------------- behaviour cloning -----------------------
def train_bc(trajs, obs_dim, n_actions, cfg, device=None, seed=0):
    torch.manual_seed(seed)
    auto_device = device is None
    device = resolve_device(device)
    S = np.concatenate([t.obs for t in trajs])
    A = np.concatenate([t.actions for t in trajs])
    model = MLP(obs_dim, n_actions, cfg.q_hidden).to(device)
    # Same device-resident minibatching as train_dt: one torch.randperm per epoch
    # chunked by batch_size reproduces DataLoader(shuffle=True) exactly, so the
    # fitted weights are unchanged for a given seed.
    S_d = torch.as_tensor(S, device=device)
    A_d = torch.as_tensor(A, device=device)
    n = S_d.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs * 2):
        perm = dataloader_permutation(n).to(device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad(); lossf(model(S_d[idx]), A_d[idx]).backward(); opt.step()
    model.eval()
    if auto_device:
        model.to(device)
    return model


def policy_from_qnet(model, device=None):
    device = resolve_device(device, model)

    @torch.no_grad()
    def fn(obs):
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return int(model(s).argmax().item())

    @torch.no_grad()
    def batched(obs, t=None):
        s = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
        return model(s).argmax(dim=-1).cpu().numpy()

    fn.batched = batched          # picked up by PricingMDP.evaluate_policy_fn
    return fn


def make_support_masked_qnet_policy(model, mdp, counts, min_count=None,
                                    topk=None, device=None):
    """`policy_from_qnet` with the logged-support mask the DT arms already use.

    A Q-net scores every action from the state alone, so the mask is a pure
    re-ranking of the network's own output: training, targets and the fitted
    value function are untouched. This exists so the Q-DT and IQL arms can carry
    the *identical* constraint the structured and vanilla DT arms carry. Without
    it the support comparison is bare-against-masked, which is not a comparison
    of targets at all.
    """
    device = resolve_device(device, model)

    def _mask_for(obs_batch, t):
        bins = mdp.obs_to_bins(np.asarray(obs_batch, dtype=np.float32))
        return np.stack([_supported_actions(counts, int(t), int(b),
                                            mdp.cfg.n_prices,
                                            min_count=min_count, topk=topk)
                         for b in bins])

    def _argmax(q, obs_batch, t):
        mask = torch.as_tensor(_mask_for(obs_batch, t), dtype=torch.bool,
                               device=q.device)
        return q.masked_fill(~mask, float("-inf")).argmax(dim=-1)

    @torch.no_grad()
    def fn(obs):
        t = int(round(float(obs[1]) * (mdp.H - 1)))
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return int(_argmax(model(s), np.asarray([obs]), t).item())

    @torch.no_grad()
    def batched(obs, t=None):
        s = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
        return _argmax(model(s), obs, t).cpu().numpy()

    fn.batched = batched
    return fn


@torch.no_grad()
def qnet_action_probs(model, obs, device=None, temp=1.0):
    device = resolve_device(device, model)
    s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    return torch.softmax(model(s)[0] / temp, dim=-1).cpu().numpy()


# ----------------------- discrete CQL -----------------------
def train_cql(trajs, mdp, obs_dim, n_actions, cfg, device=None, seed=0):
    """One-step-bootstrapped conservative Q-learning on logged transitions."""
    torch.manual_seed(seed)
    auto_device = device is None
    device = resolve_device(device)
    S, A, R, S2, NT = [], [], [], [], []
    for tr in trajs:
        H = len(tr.actions)
        for t in range(H):
            S.append(tr.obs[t]); A.append(tr.actions[t]); R.append(tr.rewards[t])
            if t + 1 < H:
                S2.append(tr.obs[t + 1]); NT.append(1.0)
            else:
                S2.append(tr.obs[t]); NT.append(0.0)
    S = torch.tensor(np.array(S), dtype=torch.float32, device=device)
    A = torch.tensor(np.array(A), dtype=torch.long, device=device)
    R = torch.tensor(np.array(R), dtype=torch.float32, device=device)
    S2 = torch.tensor(np.array(S2), dtype=torch.float32, device=device)
    NT = torch.tensor(np.array(NT), dtype=torch.float32, device=device)

    q = MLP(obs_dim, n_actions, cfg.q_hidden).to(device)
    qt = MLP(obs_dim, n_actions, cfg.q_hidden).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
    for it in range(cfg.q_epochs * 20):
        with torch.no_grad():
            target = R + cfg.gamma * NT * qt(S2).max(1).values
        qa = q(S)
        pred = qa.gather(1, A.unsqueeze(1)).squeeze(1)
        td = ((pred - target) ** 2).mean()
        # conservative penalty: push down logsumexp over actions, pull up data action
        cons = (torch.logsumexp(qa, dim=1) - pred).mean()
        loss = td + cfg.cql_alpha * cons
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 25 == 0:
            qt.load_state_dict(q.state_dict())
    q.eval()
    if auto_device:
        q.to(device)
    return q


# ----------------------- discrete IQL -----------------------
def _flatten_transitions(trajs):
    S, A, R, S2, NT = [], [], [], [], []
    for tr in trajs:
        H = len(tr.actions)
        for t in range(H):
            S.append(tr.obs[t])
            A.append(tr.actions[t])
            R.append(tr.rewards[t])
            if t + 1 < H:
                S2.append(tr.obs[t + 1])
                NT.append(1.0)
            else:
                S2.append(tr.obs[t])
                NT.append(0.0)
    return (
        np.array(S, dtype=np.float32),
        np.array(A, dtype=np.int64),
        np.array(R, dtype=np.float32),
        np.array(S2, dtype=np.float32),
        np.array(NT, dtype=np.float32),
    )


def _expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def train_iql(trajs, mdp, obs_dim, n_actions, cfg, device=None, seed=0,
              expectile=0.7, awr_beta=3.0, weight_clip=20.0,
              updates=None, batch_size=None, lr=None):
    """Discrete IQL for finite-horizon pricing logs.

    Returns a policy-logit network that can be passed to ``policy_from_qnet``.
    The observation already contains timestep, so separate time-indexed critics
    are not needed for this small MDP.
    """
    del mdp  # interface mirrors train_cql; dynamics are not used by IQL.
    torch.manual_seed(seed)
    auto_device = device is None
    device = resolve_device(device)
    S, A, R, S2, NT = _flatten_transitions(trajs)
    S = torch.tensor(S, dtype=torch.float32, device=device)
    A = torch.tensor(A, dtype=torch.long, device=device)
    R = torch.tensor(R, dtype=torch.float32, device=device)
    S2 = torch.tensor(S2, dtype=torch.float32, device=device)
    NT = torch.tensor(NT, dtype=torch.float32, device=device)

    q = MLP(obs_dim, n_actions, cfg.q_hidden).to(device)
    v = MLP(obs_dim, 1, cfg.q_hidden).to(device)
    pi = MLP(obs_dim, n_actions, cfg.q_hidden).to(device)
    lr = cfg.lr if lr is None else lr
    opt_q = torch.optim.Adam(q.parameters(), lr=lr)
    opt_v = torch.optim.Adam(v.parameters(), lr=lr)
    opt_pi = torch.optim.Adam(pi.parameters(), lr=lr)

    n = S.shape[0]
    updates = cfg.q_epochs * 20 if updates is None else int(updates)
    batch_size = n if batch_size is None else min(int(batch_size), n)
    q_losses, v_losses, pi_losses, w_means = [], [], [], []

    for _ in range(updates):
        if batch_size < n:
            idx = torch.randint(0, n, (batch_size,), device=device)
            Sb, Ab, Rb, S2b, NTb = S[idx], A[idx], R[idx], S2[idx], NT[idx]
        else:
            Sb, Ab, Rb, S2b, NTb = S, A, R, S2, NT

        with torch.no_grad():
            q_data = q(Sb).gather(1, Ab.unsqueeze(1)).squeeze(1)
        v_pred = v(Sb).squeeze(1)
        v_loss = _expectile_loss(q_data - v_pred, expectile)
        opt_v.zero_grad()
        v_loss.backward()
        opt_v.step()

        with torch.no_grad():
            target = Rb + cfg.gamma * NTb * v(S2b).squeeze(1)
        pred = q(Sb).gather(1, Ab.unsqueeze(1)).squeeze(1)
        q_loss = ((pred - target) ** 2).mean()
        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        with torch.no_grad():
            adv = q(Sb).gather(1, Ab.unsqueeze(1)).squeeze(1) - v(Sb).squeeze(1)
            weights = torch.exp(awr_beta * adv).clamp(max=weight_clip)
        ce = nn.functional.cross_entropy(pi(Sb), Ab, reduction="none")
        pi_loss = (weights * ce).mean()
        opt_pi.zero_grad()
        pi_loss.backward()
        opt_pi.step()

        q_losses.append(float(q_loss.detach().cpu()))
        v_losses.append(float(v_loss.detach().cpu()))
        pi_losses.append(float(pi_loss.detach().cpu()))
        w_means.append(float(weights.mean().detach().cpu()))

    pi.eval()
    pi.iql_diagnostics = {
        "iql_updates": updates,
        "iql_expectile": float(expectile),
        "iql_awr_beta": float(awr_beta),
        "iql_weight_clip": float(weight_clip),
        "iql_final_q_loss": float(np.mean(q_losses[-20:])),
        "iql_final_v_loss": float(np.mean(v_losses[-20:])),
        "iql_final_pi_loss": float(np.mean(pi_losses[-20:])),
        "iql_mean_adv_weight": float(np.mean(w_means[-20:])),
    }
    if auto_device:
        pi.to(device)
    return pi


def train_iql_with_diagnostics(*args, **kwargs):
    pi = train_iql(*args, **kwargs)
    return pi, dict(pi.iql_diagnostics)
