# Appendix: Does the Logged-Support Constraint Contain an LLM?

## The question changed

The first version of this appendix asked *"can a prompt-only LLM act as a
competitive sequential pricing policy?"* and reported that `Qwen2.5-1.5B-Instruct`
scored `-3.757`. That question is replaced, for three reasons given in
`pricing_dt/diagnostics/diag_appendix_llm.py`, and the old result is **withdrawn**
(see "Withdrawn result" below).

The replacement question is on-thesis and answerable:

> An LLM choosing prices directly is the most extreme case of the **action
> channel** — a model with no representation of logged support, handed an argmax
> over the whole price grid. Ch4 §4.6.1 shows the support constraint rescues
> every other family in proportion to how far it strays (Spearman `+1.000` over
> eight arms, from vanilla DT at `+0.052` to a bandit learner at `+2.997`).
> **Does it rescue an LLM too, and how much of what survives is the LLM rather
> than the constraint?**

Framed this way the model's size matters far less: the claim is about the channel,
not about LLM capability, so a small open-weight model is an acceptable instance.

## Withdrawn result

The previously reported table (`Qwen2.5-1.5B-Instruct`, mean NV `-3.757`, run on a
Colab T4) is withdrawn on two grounds.

1. **No artefacts.** `results_appendix_llm_colab_qwen15_full/` is not in the
   repository. The only `appendix_llm_*` outputs present were produced by the
   `heuristic` provider, whose own `protocol.json` states it "is not evidence about
   real LLM performance". The reported numbers cannot be reproduced or audited.
2. **The reported figures indicate a degenerate output, not a measurement.**
   `match_optimal_rate` and `match_myopic_rate` were both exactly `0.000` at a
   `valid_output_rate` of `1.000`. The control run below measures the chance floor:
   a uniform random policy matches the myopic action **`0.088`** of the time
   (`1/11 = 0.091`). Scoring exactly `0.000` on 168 states is *below chance* and is
   the signature of a near-constant emitted index. Because the action table was not
   archived, this cannot be checked.

A third defect affected the protocol rather than the result: `main()` built the
state table once from `seeds[0]` and evaluated it against seeds 0–2, so the
reported "3 seeds" were one policy transferred to three MDPs, not three runs.

## What the harness now does

| Change | Why |
|---|---|
| **Two information conditions** (`--info-modes`) | The old prompt carried `immediate_revenue`, `expected_demand` and `next_reference_price` **from the true simulator**, so the model never estimated demand — only arithmetic and lookahead remained. `oracle_info` retains that as a planning probe; **`log_only`** gives only what the log contains in that (period, reference-bin) cell — times each price was tried and its mean logged revenue, `null` if never tried. `log_only` is the arm that belongs in the dose-response table, because it is the information DT, Q-DT, IQL and the bandit learners get. |
| **Ranked output** | The DT/Q-net mask sets unsupported logits to `-inf` and takes the argmax, i.e. the model's *highest-ranked supported* action. A single forced choice cannot express that. The model now returns an ordered `ranking`, from which the free arm (`ranking[0]`) and the masked arm (first supported entry) are both read off **one** response. |
| **Chance and floor controls** | A `Random policy` arm (chance floor) and a **`Random policy + mask`** arm. The second is the important one: it is what the support set buys with *zero* intelligence, so `(LLM + mask) − (random + mask)` is the part attributable to the model. |
| **Per-seed prompting** | Records, support counts and prompts are rebuilt per seed. |
| **Degeneracy diagnostics** | `modal_action_share`, `distinct_actions_used`, `action_entropy_normalised`. |
| **Rollout-weighted off-support** | `off_support_rate` is now measured *along the evaluation rollout*, matching `diag_gate2_pricing._support_metrics`. The state-table mean is kept separately as `off_support_rate_statetable` — the two differ a lot (DP oracle: `0.027` vs `0.522`), and only the rollout figure is comparable to the other families. |

The mask is `dt._supported_actions`, literally the same helper the DT and Q-net
arms use. Protocol: hardest cell (`N=100`, `noise=0.5`) x 10 seeds x 168 states,
matching `diag_channel_ladder` / `diag_heldout_protocol` / `diag_pricing_constraints`.

## Control results (in repository, no LLM required)

`results_appendix_llm_controls_20260821/`, full config, 10 seeds. Anchors verify
exactly: DP oracle `1.0000`, myopic one-step optimiser `0.0000`.

| Arm | Mean nv | Off-support | Myopic-match | Modal action share |
|---|---:|---:|---:|---:|
| DP oracle | `1.000` | `0.027` | — | 0.238 |
| DP oracle + support top3 | `0.949` | `0.000` | — | 0.429 |
| **Random policy + support top3** | **`0.448`** | `0.000` | `0.389` | 0.429 |
| Heuristic proxy + support top3 | `0.145` | `0.000` | `0.494` | 0.429 |
| Heuristic proxy (not LLM) | `0.088` | `0.164` | `0.911` | 0.232 |
| Myopic one-step optimiser | `0.000` | `0.099` | `1.000` | 0.238 |
| **Random policy** | **`-1.856`** | `0.866` | **`0.088`** | 0.140 |

Two things follow, and the second is not confined to this appendix.

**The chance floor is established.** Uniform choice matches the myopic action
`0.088` of the time against `1/11 = 0.091`. Any arm reporting a myopic-match rate
of `0.000` over 168 states is emitting a near-constant index, not deciding.

**Masking a random policy is worth `+2.304` and lands it at `+0.448`.** The support
set alone — with no learner at all — recovers 45% of the myopic-to-optimum gap.
This is the floor every masked number in the study should be read against.
Comparing at the same cell, seeds and anchors:

| Masked arm | nv | Over `random + mask` | $p$ (n=10) |
|---|---:|---:|---:|
| Vanilla DT + mask | `0.527` | **`+0.080`** | **`0.106`** |
| Q-DT `td` + mask | `0.716` | `+0.269` | `0.002` |
| Structured DT + mask | `0.722` | `+0.275` | `0.002` |
| Q-DT `q_sa` + mask | `0.760` | `+0.312` | `0.002` |
| IQL + mask | `0.834` | `+0.386` | `0.002` |

**Support-masked vanilla DT is not statistically distinguishable from random
selection inside the same support set.** The Ch4 finding that masking improves
every family (+0.052 for vanilla DT, $p=4.7\times10^{-8}$ over 90 runs) stands, but
this floor bounds its interpretation: for a weak learner the mask does essentially
all of the work. The four value-carrying arms clear the floor comfortably.

## Running the LLM arms

No LLM row exists yet — the harness is ready, the model call is not runnable in a
sandbox without credentials or a weight download.

```bash
# local open-weight model on a Colab GPU (log_only is the arm that matters)
python -m pricing_dt.diagnostics.diag_appendix_llm \
    --provider transformers --model <a current instruct model> \
    --info-modes log_only,oracle_info --max-tokens 160 \
    --outdir results_appendix_llm_<model>_20260821

# hosted route (needs HF_TOKEN / HUGGINGFACE_TOKEN / HUGGINGFACEHUB_API_TOKEN)
python -m pricing_dt.diagnostics.diag_appendix_llm \
    --provider huggingface --model <repo/model> --info-modes log_only \
    --outdir results_appendix_llm_hosted_20260821
```

Cost: 168 states x 10 seeds x 2 information conditions = 3,360 calls
(1,680 for `log_only` alone). Hosted-provider reliability was poor in the earlier
attempt — Together and Groq were blocked by Cloudflare, DeepInfra did not serve the
tested model, Novita exhausted its included credits — so the local route is the
safer one.

**Use a current model.** `Qwen2.5-1.5B-Instruct` dates from 2024 and is roughly two
generations behind as of August 2026; a negative result from it supports no claim
about LLMs in general. Under the channel framing the model matters less, but a
dated model is an avoidable weakness.

## Reading the result, whatever it turns out to be

Check in this order, and do not quote a normalised value until the first two pass.

1. **`modal_action_share`.** Near `1.0` means a constant index; the nv is then not
   a capability measurement and only the degeneracy itself is reportable.
2. **`match_myopic_rate` against `0.088`.** Below chance is a red flag, not a result.
3. **`off_support_rate` (bare).** Expected near `1.0`. This is the arm's position on
   the dose-response axis and the whole point of the appendix.
4. **`(LLM + mask) − 0.448`.** This, not the raw masked nv, is what the LLM
   contributes over the constraint.

The publishable outcome is a ninth point on the dose-response curve of §4.6.1,
whichever way it falls. If the mask rescues the LLM by a large margin and it still
lands near the random-masked floor, the reading is that an LLM behaves like every
other unconstrained action channel and contributes little once contained. If it
clears the floor, that is a positive result about prompt-only pricing under a
support constraint. Both are reportable; neither depends on the LLM being good.
