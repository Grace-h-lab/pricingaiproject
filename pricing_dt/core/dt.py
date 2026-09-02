"""Minimal Decision Transformer for discrete pricing actions.

The three DT variants in the study (vanilla / Q-DT / structured) share this exact
architecture and training code; they differ ONLY in the return-to-go column of
the dataset (logged / value-relabelled / structure-relabelled). This isolates the
relabelling mechanism as the sole independent variable.

Also provides an EDT-style inference that searches over candidate target returns
at each step (a faithful-but-simplified take on Elastic DT's test-time stitching).

References:
  - Chen et al. (2021), Decision Transformer.
  - Wu et al. (2023), Elastic Decision Transformer.
  - Bhargava et al. (2024), empirical guidance on DT vs CQL/BC.
  - Hu et al. (2024), Gao et al. (2024), and Kim et al. (2024) for current
    value-/advantage-aided conditional sequence modelling variants.

Implements: the sequence model of §3.4 (Table G.4 gives the settings) and the
logged-support mask Supp_k of Equation (3.2), applied at inference in
`_supported_actions`.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: DecisionTransformer(), _supported_actions(), train_dt().
#
# Decision Transformer over (return-to-go, state, action) tokens, plus the
# inference-time logged-support mask that is this study's main intervention.
#
# Implements or follows:
#   - Chen, L. et al. (2021) 'Decision Transformer: Reinforcement Learning via Sequence
#     Modeling', NeurIPS 34. arXiv:2106.01345.
#   - Fujimoto, S., Conti, E., Ghavamzadeh, M. and Pineau, J. (2019) 'Benchmarking Batch
#     Deep Reinforcement Learning Algorithms', eq. 17. arXiv:1910.01708.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
from pricing_dt.core.torch_utils import (align_inputs, dataloader_permutation,
                                          resolve_device)


class CausalBlock(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(p))

    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask,
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Tokens per timestep: (return-to-go, state, action). Predicts next action."""
    def __init__(self, obs_dim, n_actions, cfg, max_T):
        super().__init__()
        d = cfg.d_model
        self.n_actions = n_actions
        self.d = d
        # RTG normalisation stats (set in train_dt from the training RTG column).
        # Raw returns here span ~480 (vanilla) to ~5000 (Q-DT) to single digits if a
        # relabeller is mis-scaled; feeding those magnitudes straight into a Linear(1,d)
        # swamps the conditioning signal and the DT collapses to behaviour cloning
        # (vanilla and Q-DT then learn the SAME policy). Standardising the return token
        # makes the model genuinely return-conditioned, which is what stitching needs.
        self.register_buffer("rtg_mean", torch.zeros(1))
        self.register_buffer("rtg_std", torch.ones(1))
        self.embed_rtg = nn.Linear(1, d)
        self.embed_state = nn.Linear(obs_dim, d)
        self.embed_action = nn.Embedding(n_actions, d)
        self.embed_time = nn.Embedding(max_T + 1, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([CausalBlock(d, cfg.n_head, cfg.dropout)
                                     for _ in range(cfg.n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_actions)

    def forward(self, rtg, states, actions, timesteps):
        rtg, states, actions, timesteps = align_inputs(self, rtg, states, actions, timesteps)
        B, T = states.shape[:2]
        te = self.embed_time(timesteps)
        rtg_n = (rtg - self.rtg_mean) / self.rtg_std
        r = self.embed_rtg(rtg_n.unsqueeze(-1)) + te
        s = self.embed_state(states) + te
        a = self.embed_action(actions) + te
        # interleave to (r0,s0,a0, r1,s1,a1, ...) -> length 3T
        tok = torch.stack([r, s, a], dim=2).reshape(B, 3 * T, self.d)
        tok = self.drop(tok)
        L = 3 * T
        mask = torch.triu(torch.full((L, L), float("-inf"), device=tok.device), 1)
        for blk in self.blocks:
            tok = blk(tok, mask)
        tok = self.ln(tok)
        # action logits are read from the STATE token positions (index 1 of each triple)
        state_tok = tok[:, 1::3, :]            # [B, T, d]
        return self.head(state_tok)            # [B, T, n_actions]


def train_dt(data, obs_dim, n_actions, cfg, device=None, seed=0):
    """Train the Decision Transformer on one relabelled dataset.

    The three conditioning arms differ only in `data["rtg"]`, so the return token is
    standardised against this dataset's own RTG column: on a shared scale the arms would
    differ in how far their targets sit from the conditioning range as well as in what
    those targets say, and the comparison would not isolate the target.
    """
    torch.manual_seed(seed)
    auto_device = device is None
    device = resolve_device(device)
    model = DecisionTransformer(obs_dim, n_actions, cfg, max_T=data["timesteps"].max() + 1).to(device)
    # Standardise the return token using THIS dataset's RTG column (the only thing
    # that differs across the three variants), so conditioning is informative.
    rtg_all = torch.tensor(data["rtg"], dtype=torch.float32)
    model.rtg_mean.copy_(rtg_all.mean().reshape(1))
    model.rtg_std.copy_(rtg_all.std().clamp(min=1e-6).reshape(1))
    # The whole dataset is a few hundred short trajectories, so it lives on the
    # device for the duration of training. This replaces a DataLoader's per-sample
    # Python collation and per-batch host-to-device copy with plain index_select.
    # `DataLoader(shuffle=True)` draws exactly one `torch.randperm(n)` per epoch
    # from the global RNG and chunks it by batch_size, so reproducing that here
    # keeps the minibatch composition — and hence the trained weights — identical
    # to the previous implementation for a given seed.
    rtg_all_d, s_all, a_all, t_all = (
        torch.as_tensor(data["rtg"], device=device),
        torch.as_tensor(data["states"], device=device),
        torch.as_tensor(data["actions"], device=device),
        torch.as_tensor(data["timesteps"], device=device),
    )
    n = s_all.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg.epochs):
        perm = dataloader_permutation(n).to(device)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            rtg, s, a, t = rtg_all_d[idx], s_all[idx], a_all[idx], t_all[idx]
            logits = model(rtg, s, a, t)                # [B,T,A]
            loss = lossf(logits.reshape(-1, n_actions), a.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    if auto_device:
        model.to(device)
    return model


@torch.no_grad()
def _step_logits(model, rtg_hist, s_hist, a_hist, t_hist, device):
    rtg = torch.tensor(np.array(rtg_hist), dtype=torch.float32, device=device).unsqueeze(0)
    s = torch.tensor(np.array(s_hist), dtype=torch.float32, device=device).unsqueeze(0)
    a = torch.tensor(np.array(a_hist), dtype=torch.long, device=device).unsqueeze(0)
    t = torch.tensor(np.array(t_hist), dtype=torch.long, device=device).unsqueeze(0)
    return model(rtg, s, a, t)[0, -1]      # logits at the latest state token


class _BatchedRollout:
    """Lock-step history buffer for evaluating a DT from many starts at once.

    Mirrors the scalar controller's bookkeeping exactly, including its quirks:
      * the return token at the current step is seeded from the previous step's
        already-decremented value (`rtg.append(rtg[-1])`), and
      * the action history is padded with a dummy 0 for the not-yet-chosen action.
    The running return is held in float64 and cast to float32 only at the model
    boundary, which is what the scalar path does when it rebuilds the tensor from
    a Python float list on every step.
    """

    def __init__(self, model, device, target_return):
        self.model, self.device = model, device
        self.target = float(target_return)
        self.rtg = self.s = self.a = self.t = None

    def step_logits(self, obs, t):
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        B = obs_t.shape[0]
        if t == 0:
            self.rtg = torch.full((B, 1), self.target, dtype=torch.float64, device=self.device)
            self.s = obs_t.unsqueeze(1)
            self.a = torch.zeros((B, 0), dtype=torch.long, device=self.device)
            self.t = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        else:
            self.rtg = torch.cat([self.rtg, self.rtg[:, -1:]], dim=1)
            self.s = torch.cat([self.s, obs_t.unsqueeze(1)], dim=1)
            self.t = torch.cat([self.t, torch.full((B, 1), t, dtype=torch.long,
                                                   device=self.device)], dim=1)
        a_hist = torch.cat([self.a, torch.zeros((B, 1), dtype=torch.long,
                                                device=self.device)], dim=1)
        return self.model(self.rtg.float(), self.s, a_hist, self.t)[:, -1]

    def commit(self, actions, obs, mdp):
        """Record the chosen actions and decrement the target by expected reward."""
        self.a = torch.cat([self.a, torch.as_tensor(actions, dtype=torch.long,
                                                    device=self.device).unsqueeze(1)], dim=1)
        bins = mdp.obs_to_bins(obs)
        self.rtg[:, -1] -= torch.as_tensor(mdp.R[actions, bins], dtype=torch.float64,
                                           device=self.device)
        return actions


def make_dt_policy(model, mdp, target_return, device=None):
    """Greedy DT controller. Maintains history within an episode and decrements
    the target return by realised EXPECTED reward at each step. Returns a
    policy_fn(obs)->action plus an episode reset, suitable for exact evaluation.

    The returned callable also carries a `.batched(obs_batch, t)` attribute that
    advances every evaluation start in lock-step through one [B, ...] forward per
    timestep; `PricingMDP.evaluate_policy_fn` picks it up automatically."""
    device = resolve_device(device, model)
    state = {"reset": True, "rtg": [], "s": [], "a": [], "t": [], "ref": None}

    def reset(ref):
        state.update(reset=False, rtg=[], s=[], a=[], t=[], ref=ref)

    def policy_fn(obs):
        # detect episode start by t==0 in the obs encoding
        t = int(round(obs[1] * (mdp.H - 1)))
        if t == 0:
            reset(None)
            state["rtg"] = [float(target_return)]
        else:
            state["rtg"].append(state["rtg"][-1])  # placeholder; updated below post-hoc
        state["s"].append(obs.astype(np.float32))
        # pad action history with a dummy 0 for the current (unknown) action
        a_hist = state["a"] + [0]
        state["t"].append(t)
        logits = _step_logits(model, state["rtg"], state["s"], a_hist, state["t"], device)
        a = int(torch.argmax(logits).item())
        state["a"].append(a)
        # decrement target by expected reward of the chosen action at current ref
        b = mdp.ref_to_bin(_obs_to_ref(obs, mdp))
        state["rtg"][-1] = state["rtg"][-1] - float(mdp.R[a, b])
        return a

    roll = _BatchedRollout(model, device, target_return)

    @torch.no_grad()
    def batched(obs, t):
        actions = roll.step_logits(obs, t).argmax(dim=-1).cpu().numpy()
        return roll.commit(actions, obs, mdp)

    policy_fn.batched = batched
    return policy_fn


def _supported_actions(counts, t, b, n_actions, min_count=None, topk=None):
    """The logged-support mask Supp_k(s) of Equation (3.2), shared by every masked arm.

    Ranks the actions available at (t, b) by how often THE LOG played each one there:
    `topk=k` keeps the k most-played with a non-zero count, `min_count=c` keeps every
    action played at least c times. Given both, topk decides.

    The ranking is over logged counts, which is what makes this the support mask. The
    trust region Top_m of Equation (3.1) ranks the same actions by the DT's own
    probability instead, and lives in `diagnostics.diag_trust_region.make_topm_policy`;
    Appendix A records that the two are easily confused. Read `topk` here as a depth
    into the log, never as a depth into the policy.
    """
    supported = np.ones(n_actions, dtype=bool)
    if min_count is not None:
        supported = counts[t, b] >= min_count
    if topk is not None:
        supported = np.zeros(n_actions, dtype=bool)
        top = np.argsort(counts[t, b])[-int(topk):]
        supported[top] = counts[t, b, top] > 0
    if not supported.any():
        # Nothing admissible. If the log has been at (t, b) at all, fall back to its
        # most frequent action there; if it has never been, the log says nothing about
        # this state and the mask has no basis to restrict, so leave the action set
        # alone rather than collapsing to an arbitrary index. (Reachability note: a
        # masked policy started from a logged bin never enters an unvisited state,
        # because logged actions lead to logged bins, so this branch is unreachable in
        # the reported evaluations -- it is defensive, not load-bearing.)
        if counts[t, b].sum() > 0:
            supported[int(np.argmax(counts[t, b]))] = True
        else:
            supported[:] = True
    return supported


def make_support_masked_dt_policy(model, mdp, target_return, counts,
                                  min_count=None, topk=None, device=None):
    """Greedy DT controller with an explicit logged-action support mask.

    The mask is applied only at inference: training data and RTG relabelling are
    unchanged, so this isolates whether support constraints alone explain gains.
    """
    device = resolve_device(device, model)
    state = {"reset": True, "rtg": [], "s": [], "a": [], "t": [], "ref": None}

    def reset(ref):
        state.update(reset=False, rtg=[], s=[], a=[], t=[], ref=ref)

    def policy_fn(obs):
        t = int(round(obs[1] * (mdp.H - 1)))
        if t == 0:
            reset(None)
            state["rtg"] = [float(target_return)]
        else:
            state["rtg"].append(state["rtg"][-1])
        state["s"].append(obs.astype(np.float32))
        a_hist = state["a"] + [0]
        state["t"].append(t)
        logits = _step_logits(model, state["rtg"], state["s"], a_hist,
                              state["t"], device).clone()
        b = mdp.ref_to_bin(_obs_to_ref(obs, mdp))
        supported = _supported_actions(counts, t, b, mdp.cfg.n_prices,
                                       min_count=min_count, topk=topk)
        mask = torch.tensor(supported, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(~mask, float("-inf"))
        a = int(torch.argmax(logits).item())
        state["a"].append(a)
        state["rtg"][-1] = state["rtg"][-1] - float(mdp.R[a, b])
        return a

    roll = _BatchedRollout(model, device, target_return)

    @torch.no_grad()
    def batched(obs, t):
        logits = roll.step_logits(obs, t)
        bins = mdp.obs_to_bins(obs)
        supported = np.stack([_supported_actions(counts, t, int(b), mdp.cfg.n_prices,
                                                 min_count=min_count, topk=topk)
                              for b in bins])
        mask = torch.as_tensor(supported, dtype=torch.bool, device=logits.device)
        actions = logits.masked_fill(~mask, float("-inf")).argmax(dim=-1).cpu().numpy()
        return roll.commit(actions, obs, mdp)

    policy_fn.batched = batched
    return policy_fn


def _obs_to_ref(obs, mdp):
    return mdp.cfg.p_min + obs[0] * (mdp.cfg.p_max - mdp.cfg.p_min)


@torch.no_grad()
def dt_action_probs(model, mdp, target_return, obs, device=None, temp=1.0):
    """pi_target(a | s) for OPE: query the model at the target return for a
    single state with empty history (contextual approximation)."""
    device = resolve_device(device, model)
    t = int(round(obs[1] * (mdp.H - 1)))
    logits = _step_logits(model, [float(target_return)], [obs.astype(np.float32)],
                          [0], [t], device)
    p = torch.softmax(logits / temp, dim=-1).cpu().numpy()
    return p


def edt_policy(model, mdp, candidate_returns, device=None):
    """EDT-style inference: at each step pick the action under the candidate target
    return that yields the most confident high-return action (simplified: choose
    the candidate maximising the top action logit)."""
    device = resolve_device(device, model)
    base = {"s": [], "a": [], "t": [], "rtg": []}

    def policy_fn(obs):
        t = int(round(obs[1] * (mdp.H - 1)))
        if t == 0:
            base.update(s=[], a=[], t=[], rtg=[])
        base["s"].append(obs.astype(np.float32)); base["t"].append(t)
        best_a, best_score = 0, -1e9
        for R in candidate_returns:
            rtg_hist = base["rtg"] + [float(R)]
            logits = _step_logits(model, rtg_hist, base["s"], base["a"] + [0],
                                  base["t"], device)
            score = float(logits.max().item())
            if score > best_score:
                best_score, best_a = score, int(torch.argmax(logits).item())
                chosenR = float(R)
        base["a"].append(best_a); base["rtg"].append(chosenR)
        return best_a

    return policy_fn
