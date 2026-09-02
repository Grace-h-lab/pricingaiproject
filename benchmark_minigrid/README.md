# External-validity track: support masking on a public benchmark

This directory is **separate from the pricing and inventory lineages** and its results are
never merged with them (Ch3 §3.5.5). It exists to answer one question that Ch5 §5.5 and the
paper's §9 both name as the study's largest unfilled gap:

> the support-constraint results are masked-versus-bare comparisons, they need no exact
> optimum, they could be run on a standard discrete-action offline benchmark, and they were
> not.

`D4RL_FEASIBILITY.md` explains why the channel and `Q*` results cannot be ported — they are
stated in units of a true optimality gap. That argument does not cover the contraction
claim, which is scale-free. This track closes the gap the argument leaves open.

The hypotheses, statistics and decision rule were fixed in advance in
[`../PREREGISTRATION_SUPPORT_D4RL.md`](../PREREGISTRATION_SUPPORT_D4RL.md), before any
policy was trained.

## Benchmark

Farama Minari distribution of the D4RL MiniGrid datasets — two logs of the **same**
environment (`MiniGrid-FourRooms-v0`, `Discrete(7)`), differing almost only in the logging
policy:

| dataset | episodes | steps | mean return | success | logged actions |
|---|---:|---:|---:|---:|---|
| `D4RL/minigrid/fourrooms-v0` | 590 | 10,010 | 0.847 | 100% | 3 of 7 ever played |
| `D4RL/minigrid/fourrooms-random-v0` | 10,174 | 1,000,070 | 0.018 | 3.1% | uniform over all 7 |

That pairing is the point: the contraction account says a mask can only contract toward a
band it actually binds to, so it must act hard on the first log and barely at all on the
second.

## Arms

Five learners mirroring the five E1 families — behaviour cloning, discrete CQL, discrete IQL
(expectile 0.7, AWR β = 3), a return-conditioned Decision Transformer on logged
return-to-go, and a Q-relabelled Decision Transformer using the CQL critic (the analogue of
Q-DT `td`) — plus a uniform random policy as the no-learner floor. Network shapes follow
`pricing_dt/core/config.py` (DT: d = 64, 3 blocks, 2 heads; Q-nets: hidden 128).

Behaviour cloning doubles as an **assertion arm**: because the mask is defined from `π_β`
and BC's policy *is* `π_β`, its off-support rate must be exactly 0.000 and its mask gain
exactly 0.0000. A run in which it is not has a bug.

## Two mask forms, both declared in advance

- **top-*k*, k = 3** — identical to E1/E2.
- **discrete-BCQ threshold**, `A_τ(s) = {a : π_β(a|s) / max_a' π_β(a'|s) ≥ τ}`, τ = 0.3.

Because MiniGrid states are not tabular, logged probability comes from a behaviour-cloning
estimate rather than per-state counts. The mask is applied at inference to an already-trained
policy; no arm is retrained under it.

**A finding from the pilot, before the confirmatory run:** top-3 is *vacuous* on the expert
log. That logger only plays 3 of its 7 actions, so the top-3 set is those 3 actions at every
state (`|A₃| = 3.00`, off-support 0.000 for every learner) and the mask removes nothing. The
E1/E2 operator's *k* has to be chosen relative to the logging policy's entropy rather than as
an absolute; the τ form is what binds here. This only became visible outside the environments
built for this study.

## Reproducing

Needs a Python environment with `minari`, `minigrid`, `gymnasium`, `h5py` **in addition to**
the repository's own requirements. Keep them out of the main environment — create a venv
with `--system-site-packages` so the existing PyTorch build is reused rather than resolved
again:

```sh
python -m venv --system-site-packages .venv_bench
.venv_bench/Scripts/python -m pip install -r benchmark_minigrid/requirements-benchmark.txt
.venv_bench/Scripts/python -u benchmark_minigrid/run_support_crossing.py \
    --seeds 3 --updates 20000 --dt-epochs 20 --eval-episodes 100 \
    --max-steps 200000 --outdir results_minigrid_support_20260825
.venv_bench/Scripts/python benchmark_minigrid/analyse.py \
    results_minigrid_support_20260825/minigrid_support_raw.csv
```

`--smoke` runs the same code paths in about 15 s and its numbers are not comparable with the
reported ones. The run is GPU-only by request and exits if CUDA is unavailable; it was
executed on an RTX 5060 Laptop (8.5 GB, sm_120) under torch 2.11.0+cu128.

`analyse.py` computes only the four pre-registered statistics and prints the declared
decision-rule verdict. With three training seeds these are directional outcomes against
pre-declared predictions, not significance tests.
