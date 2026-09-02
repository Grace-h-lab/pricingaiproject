# D4RL External-Validity Feasibility Check

Date: 2026-08-14

## Status

D4RL is not integrated into this mainline repository. The blocker is both
methodological and environmental, not just a missing command.

## Why It Is Out Of Scope Here

**The original reason is now obsolete, and a stronger one has replaced it.**

The first version of this document argued that D4RL was out of scope because "the
pricing structured demand prior is not portable to D4RL environments." That is still
true, but it no longer matters: the structured prior is not the contribution. Under a
fair comparator and a symmetric support constraint it is measurably worse than two
alternatives (-0.110 against masked Q-DT `td`, p = 2.4e-11). What the study now
claims is the **support constraint** itself, which is entirely domain-agnostic --
`dt._supported_actions` only counts logged actions per state.

The binding obstacle is therefore different, and harder:

> **D4RL's headline tasks are continuous-action** (halfcheetah, hopper, walker2d,
> maze2d, antmaze), and a top-k logged-support mask **has no direct definition over a
> continuous action space.**

Porting the claim would require first inventing a continuous analogue -- a k-nearest
neighbour restriction to logged actions, or a BCQ-style perturbation model around
them -- and then showing that analogue is the same intervention. That is a research
contribution in its own right, not a replication, and it would confound the very
thing being tested: any result would be as much about the choice of continuous
relaxation as about the constraint.

Three further requirements make it a separate project rather than an extension:

1. A continuous-action Decision Transformer policy head and loss.
2. D4RL dataset loading and normalised-score evaluation, with no exact optimum
   available -- so the value normalisation this study depends on (nv = 0 at the
   perfect-information myopic policy, nv = 1 at the optimum) cannot be computed.
3. Matched continuous-action baselines re-tuned per task over the same seeds.

Point 2 is the deepest of the three. Every mechanism claim in this study is stated in
units of the true optimality gap, which only an exactly-solvable environment supplies.
A D4RL run would produce a ranking, not a mechanism.

## Update, 2026-08-25: the support claim WAS taken to a public benchmark, and failed

The argument below — that a second exactly-solvable environment was the better test — is
retained as the reasoning of the time, but it is no longer the whole story. D4RL's *MiniGrid*
datasets are discrete-action and are distributed through Minari, so the support claim could
be tested on public data after all without inventing a continuous relaxation.

It was, under a pre-registered decision rule (`PREREGISTRATION_SUPPORT_D4RL.md`,
`benchmark_minigrid/`), and **it did not replicate**. The top-3 operator proved not to be
portable: on the expert log the behaviour policy plays only three of seven actions, so the
admissible set is the whole logged action set and the mask does nothing; on the random log it
binds and every gain is negative. The no-learner floor replicated; the contraction account
did not. The dissertation restricts that claim to E1 and E2 by name (Appendix F.3.1).

Note what this does *not* change: the channel and `Q*` results still cannot be ported, for
exactly the reason given below — they are stated in units of a true optimality gap.

## What was done instead, and why it is the better test

Rather than change action space and benchmark stack at once, the support crossing was
replicated in the **second exactly-solvable discrete environment already built for
this study** (`diag_env2_support`). Lost-sales inventory has 11 order quantities
against pricing's 11 prices, exact dynamic-programming anchors, and a structurally
different prior, so the identical mask applies unchanged and the result is stated in
the same units.

That answers the question a reader actually has -- does the constraint generalise
beyond pricing -- at a fraction of the cost, and it tests the claim the study makes
rather than the one it withdrew.

## What The Mainline Uses Instead

This repository now contains lower-dependency boundary checks aligned with the
corrected mechanism:

- `pricing_dt/envs/inventory.py`, `pricing_dt/diagnostics/diag_env2_probe.py`, and `pricing_dt/diagnostics/diag_env2_channels.py` test a
  second exact discrete environment: lost-sales inventory with exact dynamic
  programming anchors.
- `pricing_dt/diagnostics/diag_gate3_real_ope.py` adds descriptive Online Retail II OPE sensitivity:
  DM/IPS/SNIPS/DR, clipping, bootstrap intervals, ESS, and action-overlap
  diagnostics.

These checks do not establish broad continuous-control validity. They do give a
cleaner near-term boundary for the current discrete-action mechanism without
mixing in a new action space and benchmark stack.

## Future D4RL Branch

A proper D4RL study should live in a separate branch or repository. Suggested
smoke tasks for that branch are `maze2d-umaze-v1`, `halfcheetah-medium-v2`,
`hopper-medium-v2`, and `walker2d-medium-v2`.
