"""Bootstrapped-value relabelling: critic read-outs in the Q-DT slot.

WHAT THIS IS NOT. Q-DT (Yamagata, Khalil and Santos-Rodriguez, 2023, Algorithm 1)
relabels by a backward recursion,

    R_{t-1} = r_{t-1} + max(R_t, V_hat(s_t)),

whose running maximum against the logged return-to-go means relabelling can only
raise a target, never lower it. None of the read-outs below does that: each reads the
critic at one step, with no recursion and no maximum against the logged return. They
are QDT-INSPIRED critic read-outs, not that algorithm, and nothing here should be
read as a reproduction of its published numbers.

WHAT IT IS FOR. This study needs an unstructured comparator occupying the SAME
relabelling slot as the structured mechanism: one that replaces logged return-to-go
with a bootstrapped value rather than with economic structure. Each read-out below
does exactly that, and the comparison RQ4 rests on is between relabelling signals in
one pipeline, not between published implementations.

WHY THE DEFAULT IS ACTION-DEPENDENT. Relabelling with the STATE value
V(s_t) = max_a Q(s_t, a) does NOT depend on the logged action a_t. That is the
defect Appendix E.1 records and corrects for the structured relabeller
(`relabel.achievable_rtg`): an action-independent target makes every action at a
state carry the SAME return token, so the DT learns to pair a high target with the
sub-optimal actions actually present in the data, and the return token stops
discriminating actions. An action-independent comparator set against an
action-dependent structured relabeller would also confound the C2 comparison,
which is about the relabelling signal rather than about action dependence.

The default is therefore the action-dependent analogue, matching `achievable_rtg`
term-by-term:

    structured :  R_hat_t = r_hat(s_t, a_t) + sum_{k>t} max_p r_hat(s_k, p)
    Q-DT (td)  :  R_hat_t = r_t             + V(s_{t+1})

`mode="state_value"` selects the action-independent read-out and is retained so that
effect can be reported. It is defective FOR THIS STUDY, whose RQ3 is about
within-state discrimination, not defective as a value read-out: the state value is
also what Algorithm 1 feeds into its maximum. What Algorithm 1 adds, and what none of
these read-outs has, is the backward recursion around it.

References:
  - Yamagata, Khalil and Santos-Rodriguez (2023), Q-learning Decision
    Transformer.
  - Hu et al. (2024), Gao et al. (2024), and Kim et al. (2024) for closely
    related modern Q-/advantage-aided Decision Transformer variants.

Implements: the conservative critic of §3.4.1 and its five relabelling read-outs
(`state_value`, `q_sa`, `td`, `td+denoise`, `oracle`); panel E of Figure C.1
tabulates them.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: value_relabel().
#
# Q-learning Decision Transformer: relabel the conditioning return with a bootstrapped
# value estimate rather than the realised return.
#
# Implements or follows:
#   - Yamagata, T., Khalil, A. and Santos-Rodriguez, R. (2023) 'Q-learning Decision
#     Transformer', ICML.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import numpy as np
import torch
from pricing_dt.core.baselines import train_cql
from pricing_dt.core.torch_utils import resolve_device


def value_relabel(trajs, mdp, obs_dim, n_actions, cfg, device=None, seed=0,
                  mode="td", denoise=False, q=None):
    """Fit a value (reuse the CQL Q-net) and relabel each trajectory's RTG.

    mode="td" (DEFAULT; action-dependent):
        R_hat_t = r_t + V(s_{t+1}),  V(s) = max_a Q(s, a),  V(s_H) = 0.
        Action-dependent through the realised reward of the action taken; the
        continuation is the bootstrapped optimum. Direct analogue of the
        structured target.
    mode="q_sa":
        R_hat_t = Q(s_t, a_t). The other faithful action-dependent form, taking
        both terms from the Q-net rather than using the logged reward.
    mode="state_value" (the earlier default; action-INDEPENDENT, so it carries no
                        within-state signal -- retained to report that effect):
        R_hat_t = V(s_t) = max_a Q(s_t, a).

    denoise=True replaces the realised r_t with the de-noised expected reward
    mdp.R[a_t, b_t] in mode="td", matching the structured relabeller's use of
    expected (not noise-realised) revenue for the current step.

    `q` lets a caller reuse an already-fitted Q-net across modes (the modes differ
    only in how the same Q is read out, so refitting would add noise, not rigour).

    Returns (list of [H] float32 arrays, fitted Q-net).
    """
    if q is None:
        q = train_cql(trajs, mdp, obs_dim, n_actions, cfg, device=device, seed=seed)
    device = resolve_device(device, q)
    q.to(device)

    # One forward over every logged state in the dataset. The Q-net is a plain
    # MLP, so rows are independent and stacking the trajectories is equivalent to
    # the old per-trajectory loop — but it replaces len(trajs) batch-of-H calls
    # with a single batched call, which is the difference between being launch-
    # overhead bound and being arithmetic bound on a GPU.
    lengths = [len(tr.obs) for tr in trajs]
    with torch.no_grad():
        all_s = torch.as_tensor(np.concatenate([tr.obs for tr in trajs]),
                                dtype=torch.float32, device=device)
        all_qa = q(all_s)
    qa_by_traj = torch.split(all_qa, lengths)

    relabelled = []
    with torch.no_grad():
        for tr, qa in zip(trajs, qa_by_traj):
            v = qa.max(1).values.cpu().numpy().astype(np.float32)   # V(s_t) per step
            H = len(tr.actions)

            if mode == "state_value":                        # legacy, action-independent
                relabelled.append(v)
                continue

            if mode == "q_sa":
                a = torch.tensor(tr.actions, dtype=torch.long, device=device)
                g = qa.gather(1, a.unsqueeze(1)).squeeze(1).cpu().numpy().astype(np.float32)
                relabelled.append(g)
                continue

            if mode != "td":
                raise ValueError(f"unknown relabel mode {mode!r}")

            if denoise:
                r = np.array([mdp.R[int(tr.actions[t]), int(tr.ref_bins[t])]
                              for t in range(H)], np.float32)
            else:
                r = tr.rewards.astype(np.float32)
            g = np.zeros(H, np.float32)
            g[:H - 1] = r[:H - 1] + v[1:]                    # r_t + V(s_{t+1})
            g[H - 1] = r[H - 1]                              # V(terminal) = 0
            relabelled.append(g)

    return relabelled, q
