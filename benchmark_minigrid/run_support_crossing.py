"""Support-mask crossing on a public discrete-action offline benchmark.

Tests the contraction account of Ch4 4.7.4 / paper 7 outside the two environments
built for this study, using the Farama Minari distribution of the D4RL MiniGrid
datasets. Two logs of the SAME environment differing almost only in the logging
policy: an expert log that uses 3 of 7 actions, and a random log that is uniform
over all 7. The account predicts the mask contracts hard against the first and
barely at all against the second (PREREGISTRATION_SUPPORT_D4RL.md, P3).

Everything here is scale-free; no exact optimum is required, which is why this
claim ports and the channel/Q* claims do not (D4RL_FEASIBILITY.md).

Run inside the benchmark venv (needs minari, minigrid, gymnasium, h5py).
Outputs go to results_minigrid_support_*/ and are never merged with the pricing
or inventory lineages.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: support_sets(), train_bc(), train_iql(), train_cql(), train_dt().
#
# The pre-registered external replication: the same support-mask crossing on a public
# discrete-action offline benchmark, with both this study's top-k operator and the
# published discrete BCQ threshold.
#
# Implements or follows:
#   - Chevalier-Boisvert, M. et al. (2023) 'Minigrid & Miniworld', NeurIPS 36.
#     arXiv:2306.13831.
#   - Younis, O.G. et al. (2024) Minari [Software]. Zenodo. DOI: 10.5281/zenodo.13767625.
#   - Fu, J., Kumar, A., Nachum, O., Tucker, G. and Levine, S. (2020) 'D4RL: Datasets for
#     Deep Data-Driven Reinforcement Learning'. arXiv:2004.07219.
#   - Fujimoto, S., Conti, E., Ghavamzadeh, M. and Pineau, J. (2019) 'Benchmarking Batch
#     Deep Reinforcement Learning Algorithms', eq. 17. arXiv:1910.01708.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATASETS = {
    "expert": "D4RL/minigrid/fourrooms-v0",
    "random": "D4RL/minigrid/fourrooms-random-v0",
}
N_ACTIONS = 7
GAMMA = 0.99
TOPK = 3          # identical to the E1/E2 mask
BCQ_TAU = 0.3     # discrete-BCQ threshold, reported as a robustness form


# --------------------------------------------------------------------------- data

def encode(image, direction):
    """MiniGrid dict observation -> flat float vector.

    image is (7,7,3) of small integer codes; direction is one of four headings.
    Kept deliberately simple: the experiment is about the mask, not the encoder.
    """
    img = np.asarray(image, dtype=np.float32).reshape(len(image), -1) / 10.0
    d = np.asarray(direction, dtype=np.int64)
    onehot = np.zeros((len(d), 4), dtype=np.float32)
    onehot[np.arange(len(d)), d] = 1.0
    return np.concatenate([img, onehot], axis=1)


def load(name, max_steps=None, seed=0):
    import minari
    ds = minari.load_dataset(DATASETS[name], download=True)
    eps = list(ds.iterate_episodes())
    rng = np.random.default_rng(seed)
    if max_steps is not None:
        order = rng.permutation(len(eps))
        keep, total = [], 0
        for i in order:
            keep.append(eps[i])
            total += len(eps[i].actions)
            if total >= max_steps:
                break
        eps = keep
    S, A, R, S2, D, EP, RTG = [], [], [], [], [], [], []
    for k, e in enumerate(eps):
        obs = encode(e.observations["image"], e.observations["direction"])
        a = np.asarray(e.actions, dtype=np.int64)
        r = np.asarray(e.rewards, dtype=np.float32)
        T = len(a)
        term = np.zeros(T, dtype=np.float32)
        if bool(np.asarray(e.terminations)[-1]):
            term[-1] = 1.0
        S.append(obs[:T]); S2.append(obs[1:T + 1]); A.append(a); R.append(r)
        D.append(term); EP.append(np.full(T, k, dtype=np.int64))
        RTG.append(np.cumsum(r[::-1])[::-1].copy())      # undiscounted, as in the study
    out = dict(
        s=np.concatenate(S), a=np.concatenate(A), r=np.concatenate(R),
        s2=np.concatenate(S2), done=np.concatenate(D), ep=np.concatenate(EP),
        rtg=np.concatenate(RTG),
    )
    out["ep_returns"] = np.array([e.rewards.sum() for e in eps], dtype=np.float32)
    return ds, out


# ------------------------------------------------------------------------- models

def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                         nn.Linear(h, o))


class Batcher:
    def __init__(self, d, device, seed):
        self.n = len(d["a"])
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.t = {k: torch.as_tensor(v).to(device) for k, v in d.items()
                  if k in ("s", "a", "r", "s2", "done", "rtg")}

    def sample(self, bs):
        idx = torch.randint(0, self.n, (bs,), generator=self.g)
        idx = idx.to(self.t["a"].device)
        return {k: v[idx] for k, v in self.t.items()}


def train_bc(b, obs_dim, device, updates, bs, lr, seed):
    torch.manual_seed(seed)
    net = mlp(obs_dim, N_ACTIONS).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(updates):
        z = b.sample(bs)
        loss = F.cross_entropy(net(z["s"]), z["a"])
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


def train_cql(b, obs_dim, device, updates, bs, lr, seed, alpha=1.0):
    torch.manual_seed(seed + 1)
    q, qt = mlp(obs_dim, N_ACTIONS).to(device), mlp(obs_dim, N_ACTIONS).to(device)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    for i in range(updates):
        z = b.sample(bs)
        with torch.no_grad():
            tgt = z["r"] + GAMMA * (1 - z["done"]) * qt(z["s2"]).max(1).values
        qa = q(z["s"]).gather(1, z["a"][:, None]).squeeze(1)
        td = F.mse_loss(qa, tgt)
        cons = (torch.logsumexp(q(z["s"]), dim=1) - qa).mean()
        loss = td + alpha * cons
        opt.zero_grad(); loss.backward(); opt.step()
        if i % 200 == 0:
            qt.load_state_dict(q.state_dict())
    return q.eval()


def train_iql(b, obs_dim, device, updates, bs, lr, seed, expectile=0.7, beta=3.0):
    torch.manual_seed(seed + 2)
    q, qt = mlp(obs_dim, N_ACTIONS).to(device), mlp(obs_dim, N_ACTIONS).to(device)
    qt.load_state_dict(q.state_dict())
    v, pi = mlp(obs_dim, 1).to(device), mlp(obs_dim, N_ACTIONS).to(device)
    oq = torch.optim.Adam(q.parameters(), lr=lr)
    ov = torch.optim.Adam(v.parameters(), lr=lr)
    op = torch.optim.Adam(pi.parameters(), lr=lr)
    for i in range(updates):
        z = b.sample(bs)
        with torch.no_grad():
            qsa = qt(z["s"]).gather(1, z["a"][:, None]).squeeze(1)
        vs = v(z["s"]).squeeze(1)
        diff = qsa - vs
        w = torch.where(diff > 0, expectile, 1.0 - expectile)
        lv = (w * diff.pow(2)).mean()
        ov.zero_grad(); lv.backward(); ov.step()
        with torch.no_grad():
            tgt = z["r"] + GAMMA * (1 - z["done"]) * v(z["s2"]).squeeze(1)
        qa = q(z["s"]).gather(1, z["a"][:, None]).squeeze(1)
        lq = F.mse_loss(qa, tgt)
        oq.zero_grad(); lq.backward(); oq.step()
        with torch.no_grad():
            adv = (qt(z["s"]).gather(1, z["a"][:, None]).squeeze(1)
                   - v(z["s"]).squeeze(1))
            wt = torch.clamp(torch.exp(beta * adv), max=20.0)
        lp = (wt * F.cross_entropy(pi(z["s"]), z["a"], reduction="none")).mean()
        op.zero_grad(); lp.backward(); op.step()
        if i % 200 == 0:
            qt.load_state_dict(q.state_dict())
    return pi.eval()


# ------------------------------------------------------------------ decision transformer

class DT(nn.Module):
    """Same shape as pricing_dt/core/dt.py: d=64, 3 blocks, 2 heads, causal, pre-LN."""

    def __init__(self, obs_dim, d=64, n_layer=3, n_head=2, K=20, dropout=0.1):
        super().__init__()
        self.K, self.d = K, d
        self.es, self.ea = nn.Linear(obs_dim, d), nn.Embedding(N_ACTIONS, d)
        self.er = nn.Linear(1, d)
        self.pos = nn.Embedding(K, d)
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.tr = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, N_ACTIONS)

    def forward(self, rtg, s, a):
        B, T = a.shape
        p = self.pos(torch.arange(T, device=a.device))[None]
        tok = torch.stack([self.er(rtg[..., None]) + p, self.es(s) + p,
                           self.ea(a) + p], dim=2).reshape(B, 3 * T, self.d)
        m = torch.triu(torch.ones(3 * T, 3 * T, device=a.device, dtype=torch.bool), 1)
        h = self.tr(tok, mask=m)
        return self.head(h[:, 1::3])          # read actions off the state tokens


def dt_sequences(d, rtg, K):
    """Slice trajectories into fixed-length windows, right-padded."""
    S, A, R, M = [], [], [], []
    for e in np.unique(d["ep"]):
        i = np.flatnonzero(d["ep"] == e)
        for st in range(0, len(i), K):
            j = i[st:st + K]
            pad = K - len(j)
            S.append(np.pad(d["s"][j], ((0, pad), (0, 0))))
            A.append(np.pad(d["a"][j], (0, pad)))
            R.append(np.pad(rtg[j], (0, pad)))
            M.append(np.pad(np.ones(len(j), np.float32), (0, pad)))
    return (np.stack(S).astype(np.float32), np.stack(A), np.stack(R).astype(np.float32),
            np.stack(M))


def train_dt(d, rtg, obs_dim, device, epochs, bs, lr, seed, K=20):
    torch.manual_seed(seed + 3)
    S, A, R, M = dt_sequences(d, rtg, K)
    mu, sd = R[M > 0].mean(), R[M > 0].std() + 1e-6
    S = torch.as_tensor(S).to(device); A = torch.as_tensor(A).to(device)
    R = torch.as_tensor((R - mu) / sd).to(device); M = torch.as_tensor(M).to(device)
    net = DT(obs_dim, K=K).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(S)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for k in range(0, n, bs):
            i = perm[k:k + bs]
            logits = net(R[i], S[i], A[i])
            loss = (F.cross_entropy(logits.reshape(-1, N_ACTIONS), A[i].reshape(-1),
                                    reduction="none") * M[i].reshape(-1)).sum() / M[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net, float(mu), float(sd)


# ------------------------------------------------------------------------ policies

def support_sets(logits, topk=TOPK, tau=None):
    """Boolean mask of admissible actions from the behaviour-cloning estimate."""
    if tau is None:
        idx = logits.topk(topk, dim=-1).indices
        m = torch.zeros_like(logits, dtype=torch.bool)
        return m.scatter_(-1, idx, True)
    p = logits.softmax(-1)
    return p / p.max(-1, keepdim=True).values >= tau


class Policy:
    """Wraps a scorer with an optional inference-time support mask.

    Acts on a BATCH of states: evaluation runs every episode in lockstep so each
    timestep costs one GPU call instead of one per environment. Both admissible
    sets are computed on every step regardless of which is enforced, so
    off-support rates are measured against a fixed reference.
    """

    def __init__(self, score, bc, mask):
        self.score, self.bc, self.mask = score, bc, mask
        self.n = 0
        self.off = {"top3": 0, "bcq_tau": 0}
        self.size = {"top3": 0, "bcq_tau": 0}

    def reset(self, n_env, device):
        pass

    def sets(self, S):
        logits = self.bc(S)
        return {"top3": support_sets(logits, topk=TOPK),
                "bcq_tau": support_sets(logits, tau=BCQ_TAU)}

    def record(self, allowed, bare, live):
        k = int(live.sum())
        if not k:
            return
        self.n += k
        for name, m in allowed.items():
            self.size[name] += int(m[live].sum())
            hit = m[live].gather(1, bare[live][:, None]).squeeze(1)
            self.off[name] += int((~hit).sum())

    def choose(self, q, allowed, bare):
        if self.mask == "bare":
            return bare
        m = allowed[self.mask]
        picked = q.masked_fill(~m, -float("inf")).argmax(-1)
        return torch.where(m.any(-1), picked, bare)      # empty set -> unmasked

    def act(self, S, live):
        with torch.no_grad():
            q = self.score(S)
            allowed = self.sets(S)
            bare = q.argmax(-1)
            self.record(allowed, bare, live)
            return self.choose(q, allowed, bare)


class DTPolicy(Policy):
    """Return-conditioned policy with a per-environment history buffer."""

    def __init__(self, net, mu, sd, target, bc, mask, K=20, horizon=128):
        super().__init__(None, bc, mask)
        self.net, self.mu, self.sd, self.target = net, mu, sd, target
        self.K, self.horizon = K, horizon

    def reset(self, n_env, device):
        self.t = 0
        self.rtg = torch.full((n_env,), float(self.target), device=device)
        self.Sbuf = None
        self.Abuf = torch.zeros(n_env, self.horizon, dtype=torch.long, device=device)

    def act(self, S, live):
        with torch.no_grad():
            if self.Sbuf is None:
                self.Sbuf = torch.zeros(len(S), self.horizon, S.shape[1],
                                        device=S.device)
            t = min(self.t, self.horizon - 1)
            self.Sbuf[:, t] = S
            lo = max(0, t - self.K + 1)
            Sw, Aw = self.Sbuf[:, lo:t + 1], self.Abuf[:, lo:t + 1]
            R = ((self.rtg - self.mu) / self.sd)[:, None].expand(-1, Sw.shape[1])
            q = self.net(R, Sw, Aw)[:, -1]
            allowed = self.sets(S)
            bare = q.argmax(-1)
            self.record(allowed, bare, live)
            a = self.choose(q, allowed, bare)
            self.Abuf[:, t] = a
            self.t += 1
            return a

    def observe_reward(self, r):
        self.rtg -= r


class RandomPolicy(Policy):
    """The no-learner floor: uniform choice, carrying the identical mask."""

    def __init__(self, bc, mask, seed):
        super().__init__(None, bc, mask)
        self.seed = seed

    def reset(self, n_env, device):
        self.g = torch.Generator(device=device).manual_seed(self.seed)

    def act(self, S, live):
        with torch.no_grad():
            allowed = self.sets(S)
            n = len(S)
            bare = torch.randint(N_ACTIONS, (n,), generator=self.g, device=S.device)
            self.record(allowed, bare, live)
            if self.mask == "bare":
                return bare
            m = allowed[self.mask].float()
            m = torch.where(m.sum(-1, keepdim=True) > 0, m, torch.ones_like(m))
            return torch.multinomial(m, 1, generator=self.g).squeeze(1)


# ---------------------------------------------------------------------- evaluation

def evaluate(make_env, policy, device, n_episodes, base_seed, cache={}):
    """Run every evaluation episode in lockstep.

    One batched forward per timestep instead of one per environment: the tiny
    networks here are latency-bound, not compute-bound, so stepping the episodes
    together is what actually puts the GPU to work. Terminated episodes are
    frozen by a liveness mask rather than removed, which keeps the batch shape
    and the per-episode seeds fixed.
    """
    key = (id(make_env), n_episodes)
    if key not in cache:
        cache[key] = [make_env() for _ in range(n_episodes)]
    envs = cache[key]
    obs = [e.reset(seed=base_seed + k)[0] for k, e in enumerate(envs)]
    live = np.ones(n_episodes, dtype=bool)
    total = np.zeros(n_episodes, dtype=np.float64)
    policy.reset(n_episodes, device)
    horizon = getattr(envs[0].unwrapped, "max_steps", 100)

    for _ in range(int(horizon) + 1):
        if not live.any():
            break
        S = torch.as_tensor(
            encode(np.stack([o["image"] for o in obs]),
                   np.array([o["direction"] for o in obs]))
        ).to(device)
        live_t = torch.as_tensor(live, device=device)
        acts = policy.act(S, live_t).detach().cpu().numpy()
        rewards = np.zeros(n_episodes, dtype=np.float32)
        for i, e in enumerate(envs):
            if not live[i]:
                continue
            o, r, term, trunc, _ = e.step(int(acts[i]))
            obs[i] = o
            rewards[i] = float(r)
            total[i] += float(r)
            if term or trunc:
                live[i] = False
        if hasattr(policy, "observe_reward"):
            policy.observe_reward(torch.as_tensor(rewards, device=device))
    return float(total.mean()), float(total.std())


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_minigrid_support_20260825")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--dt-epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--datasets", default="expert,random")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.eval_episodes, a.updates, a.dt_epochs = 1, 5, 200, 1
        a.max_steps, a.outdir = 5000, a.outdir + "_smoke"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA not available; this run is GPU-only by request.")
    print(f"device={torch.cuda.get_device_name(0)}")
    os.makedirs(a.outdir, exist_ok=True)
    rows = []

    for tag in a.datasets.split(","):
        ds, d = load(tag, max_steps=a.max_steps)
        make_env = ds.recover_environment
        obs_dim = d["s"].shape[1]
        target = float(d["ep_returns"].max())
        print(f"\n[{tag}] {len(d['a'])} transitions, obs_dim={obs_dim}, "
              f"DT target={target:.3f}")

        for seed in range(a.seeds):
            t0 = time.time()
            b = Batcher(d, device, seed)
            bc = train_bc(b, obs_dim, device, a.updates // 2, a.batch, a.lr, seed)
            cql = train_cql(b, obs_dim, device, a.updates, a.batch, a.lr, seed)
            iql = train_iql(b, obs_dim, device, a.updates, a.batch, a.lr, seed)
            dt, mu, sd = train_dt(d, d["rtg"], obs_dim, device, a.dt_epochs,
                                  64, a.lr, seed)
            with torch.no_grad():                       # Q-relabelled target (Q-DT td)
                S = torch.as_tensor(d["s"]).to(device)
                S2 = torch.as_tensor(d["s2"]).to(device)
                v2 = torch.cat([cql(S2[i:i + 8192]).max(1).values
                                for i in range(0, len(S2), 8192)]).cpu().numpy()
            qrtg = d["r"] + GAMMA * (1 - d["done"]) * v2
            qdt, qmu, qsd = train_dt(d, qrtg.astype(np.float32), obs_dim, device,
                                     a.dt_epochs, 64, a.lr, seed)
            qtarget = float(np.percentile(qrtg, 95))

            arms = {
                "BC": lambda m: Policy(bc, bc, m),
                "CQL": lambda m: Policy(cql, bc, m),
                "IQL": lambda m: Policy(iql, bc, m),
                "DT": lambda m: DTPolicy(dt, mu, sd, target, bc, m),
                "Q-DT": lambda m: DTPolicy(qdt, qmu, qsd, qtarget, bc, m),
                "random (no learner)": lambda m: RandomPolicy(bc, m, seed),
            }
            for arm, make in arms.items():
                # off-support is a property of the UNMASKED arm: a masked rollout
                # visits different states, so its rate is not comparable. Measure
                # it once on the bare run and carry it, as E1/E2 do.
                carried = {}
                for mask in ("bare", "top3", "bcq_tau"):
                    p = make(mask)
                    mean, sdv = evaluate(make_env, p, device, a.eval_episodes,
                                         10_000 + 1000 * seed)
                    if mask == "bare":
                        carried = {f"off_bare_{k}": v / max(p.n, 1)
                                   for k, v in p.off.items()}
                    rows.append(dict(dataset=tag, seed=seed, arm=arm, mask=mask,
                                     ret_mean=mean, ret_sd=sdv, **carried,
                                     size_top3=p.size["top3"] / max(p.n, 1),
                                     size_tau=p.size["bcq_tau"] / max(p.n, 1)))
                    print(f"  {tag} s{seed} {arm:20s} {mask:8s} ret={mean:.4f} "
                          f"off3={carried['off_bare_top3']:.3f} "
                          f"offT={carried['off_bare_bcq_tau']:.3f} "
                          f"|A3|={p.size['top3'] / max(p.n, 1):.2f} "
                          f"|AT|={p.size['bcq_tau'] / max(p.n, 1):.2f}")
            print(f"  [{tag} seed {seed}] {time.time() - t0:.0f}s")

    import csv
    path = os.path.join(a.outdir, "minigrid_support_raw.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump(vars(a), open(os.path.join(a.outdir, "protocol.json"), "w"), indent=1)
    print(f"\nwrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
