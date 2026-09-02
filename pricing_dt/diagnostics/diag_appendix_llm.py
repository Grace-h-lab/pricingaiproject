"""Appendix: does the logged-support constraint contain a prompt-only LLM?

THE QUESTION. Not whether a prompt-only LLM is a competitive sequential pricing
policy, which is not answerable at the scale available here and is off-thesis, but
the one that puts the LLM on the same axis as every other family:

    An LLM choosing prices directly is the most extreme case of the ACTION channel:
    a model with no representation of logged support, given an argmax over the whole
    price grid. Section 4.6.1 shows that the support constraint rescues every
    other family in proportion to how far it strays (Spearman +1.000 over eight
    arms, from vanilla DT at +0.052 to a bandit at +2.997). Does it rescue an LLM
    too, and how much of what remains is the LLM rather than the constraint?

FOUR CONDITIONS THE MEASUREMENT HAS TO MEET. Each is a way the arm can be
made to look better or worse than it is, and each is controlled for here.

1. **The prompt must not solve the hard part.** Carrying `immediate_revenue`,
   `expected_demand` and `next_reference_price` from the TRUE simulator leaves the
   model no demand to estimate, only arithmetic and lookahead. That measures numeric
   reasoning rather than pricing, and it is not the information every other arm gets.
   `--info-modes` offers two conditions:
       oracle_info  the full-information case, kept as a planning-ability probe;
       log_only     what the LOG shows in this (period, reference-bin) cell — how
                    many times each price was tried and the mean logged revenue,
                    "not observed" otherwise. This is the arm that belongs in the
                    dose-response table, because it is the same information DT,
                    Q-DT, IQL and the bandit learners see.

2. **The mask needs an analogue.** For DT and Q-net arms the mask sets unsupported
   logits to -inf and takes the argmax, selecting the model's HIGHEST-RANKED
   SUPPORTED action. A single forced choice cannot express that, so the model is
   asked for an ordered `ranking` of action indices, best first, from which
       free   = ranking[0]
       masked = first entry of ranking that survives the top-k logged-support mask
   are both read off ONE response. Same semantics as the other families, no second
   prompting pass.

3. **Chance has to be established.** A myopic-match rate of exactly 0.000 at a
   valid-output rate of 1.000 reads as a capability until chance is drawn: under 11
   actions uniform choice matches the myopic action about 9% of the time, so 0.000
   is BELOW chance and indicates a near-constant output. Two controls make this
   legible: a `random` arm (the chance floor) and a
   `random + mask` arm. The latter is the more important of the two — it is what the
   support set alone buys with zero intelligence, so

       (LLM + mask) - (random + mask)

   is the part attributable to the LLM. No other family in this study carries that
   control, and without it a masked score is uninterpretable.

4. **Prompts have to be per seed.** Building the state table from `seeds[0]` and
   evaluating that one action table against every seed reports n seeds while running
   one policy against n MDPs. Records, support counts and prompts are rebuilt per
   seed.

DEGENERACY DIAGNOSTICS. `modal_action_share`, `distinct_actions_used` and
`action_entropy_normalised` are recorded per arm. A modal share near 1.0 means the
model emitted a constant index and its normalised value says nothing about pricing;
this must be checked before any number here is quoted.

PROTOCOL. Hardest cell (`N=100`, `noise=0.5`) x 10 seeds by default, matching
`diag_channel_ladder`, `diag_heldout_protocol`, `diag_e2ab_fixed` and
`diag_pricing_constraints`, so `off_support_rate` and the mask gain drop straight
into the dose-response table. The mask is `dt._supported_actions` — literally the
same function the DT and Q-net arms use.

Outputs: appendix_llm_{protocol.json, state_prompts.jsonl, actions.csv, raw.csv,
summary.csv, dose_response.csv}
"""
import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as Metric
from pricing_dt.core.dt import _supported_actions
from pricing_dt.experiments.experiments import _setup, _seed, _traj_start_bins


SYSTEM_PROMPT = (
    "You are a pricing decision assistant. You will be given a pricing state and a "
    "list of allowed action_index values. Rank ALL allowed action_index values from "
    "best to worst for maximising expected total revenue over the current AND "
    "remaining periods, not just immediate revenue. Return compact JSON only."
)


def _parse_seeds(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_cells(text):
    cells = []
    for chunk in text.split(","):
        n_s, noise_s = chunk.split(":")
        cells.append((int(n_s), float(noise_s)))
    return cells


def _support_counts(trajs, mdp):
    counts = np.zeros((mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=float)
    revenue = np.zeros_like(counts)
    for tr in trajs:
        for t, a in enumerate(tr.actions):
            b = int(tr.ref_bins[t])
            counts[t, b, int(a)] += 1.0
            revenue[t, b, int(a)] += float(tr.rewards[t])
    return counts, revenue


# ----------------------------- state records -----------------------------
def _state_record(mdp, t, b, counts, revenue, info_mode):
    """One prompt-able state.

    `oracle_info` exposes the true one-step revenue and transition; `log_only`
    exposes only what the log contains in this cell, which is the information the
    DT / Q-DT / IQL / bandit arms are given.
    """
    ref = float(mdp.ref_grid[b])
    candidates = []
    for a, price in enumerate(mdp.prices):
        next_b = int(mdp.N[a, b])
        cand = {
            "action_index": int(a),
            "price": round(float(price), 4),
            # the transition is KNOWN structure in this testbed and is granted to
            # every method, so it stays in both information conditions.
            "next_reference_price": round(float(mdp.ref_grid[next_b]), 4),
        }
        if info_mode == "oracle_info":
            cand["expected_demand"] = round(float(mdp.expected_demand(price, ref)), 4)
            cand["immediate_revenue"] = round(float(mdp.R[a, b]), 4)
        else:
            n_obs = float(counts[t, b, a])
            cand["times_observed_in_log"] = int(n_obs)
            cand["mean_logged_revenue"] = (
                round(float(revenue[t, b, a] / n_obs), 4) if n_obs > 0 else None)
        candidates.append(cand)
    return {
        "info_mode": info_mode,
        "period": int(t),
        "horizon": int(mdp.H),
        "periods_remaining_including_current": int(mdp.H - t),
        "reference_bin": int(b),
        "reference_price": round(ref, 4),
        "allowed_actions": candidates,
        "myopic_action": int(mdp.R[:, b].argmax()),
        "optimal_action": int(mdp.pistar[t, b]),
    }


def _prompt_for(record):
    if record["info_mode"] == "oracle_info":
        instruction = (
            "Rank every action_index from best to worst for total expected revenue "
            "over the current and remaining periods. Each action lists its exact "
            "immediate revenue and the next reference price it induces. Higher "
            "prices raise the next reference price but cut current demand."
        )
    else:
        instruction = (
            "Rank every action_index from best to worst for total expected revenue "
            "over the current and remaining periods. You are given only what the "
            "historical log contains for this state: how many times each price was "
            "tried and the mean revenue observed when it was. A null "
            "mean_logged_revenue means that price was never tried here, so its "
            "revenue is unknown and must be inferred. Higher prices raise the next "
            "reference price but cut current demand."
        )
    payload = {
        "task": "finite_horizon_dynamic_pricing",
        "instruction": instruction,
        "state": {
            "period": record["period"],
            "horizon": record["horizon"],
            "periods_remaining_including_current":
                record["periods_remaining_including_current"],
            "reference_price": record["reference_price"],
        },
        "allowed_actions": record["allowed_actions"],
        "output_schema": {
            "ranking": "array of ALL allowed action_index values, best first",
            "action_index": "your single best action_index (= ranking[0])",
            "reason": "one short sentence",
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _all_state_records(mdp, counts, revenue, info_mode):
    records = []
    for t in range(mdp.H):
        for b in range(mdp.cfg.n_ref_bins):
            rec = _state_record(mdp, t, b, counts, revenue, info_mode)
            rec["prompt"] = _prompt_for(rec)
            records.append(rec)
    return records


# ----------------------------- parsing -----------------------------
def _clip_action(action, n_actions):
    if action is None:
        return None
    try:
        return int(np.clip(int(action), 0, n_actions - 1))
    except (TypeError, ValueError):
        return None


def _parse_llm_ranking(text, n_actions):
    """Return (ranking, ok, note). `ranking` is a de-duplicated action order.

    A response that yields only a single action still produces a usable length-1
    ranking; the masked arm then falls back to the most-logged supported action,
    which is recorded in `parse_note` so those cases stay visible.
    """
    if text is None or not str(text).strip():
        return [], False, "empty_response"
    raw = str(text).strip()

    obj = None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                obj = None

    def _dedup(seq):
        out = []
        for x in seq:
            v = _clip_action(x, n_actions)
            if v is not None and v not in out:
                out.append(v)
        return out

    if isinstance(obj, dict):
        if isinstance(obj.get("ranking"), list):
            order = _dedup(obj["ranking"])
            if order:
                return order, True, "json_ranking"
        for key in ("action_index", "action", "price_index"):
            if key in obj:
                v = _clip_action(obj[key], n_actions)
                if v is not None:
                    return [v], True, f"json_{key}_only"

    m = re.search(r"ranking[^\[]*\[([0-9,\s]+)\]", raw, flags=re.I)
    if m:
        order = _dedup(m.group(1).split(","))
        if order:
            return order, True, "regex_ranking"
    m = re.search(r"action[_\s-]*index[^0-9-]*(-?\d+)", raw, flags=re.I)
    if m:
        v = _clip_action(m.group(1), n_actions)
        if v is not None:
            return [v], True, "regex_action_index_only"
    return [], False, "unparseable"


def _apply_mask(ranking, counts, t, b, n_actions, topk):
    """The DT mask, expressed on a ranking: highest-ranked SUPPORTED action.

    Identical semantics to `masked_fill(-inf).argmax()` on network logits, and it
    uses the same `_supported_actions` helper, so the constraint is literally the
    one the other families carry.
    """
    supported = _supported_actions(counts, t, b, n_actions, topk=topk)
    for a in ranking:
        if supported[a]:
            return int(a), "ranked_supported"
    return int(np.argmax(counts[t, b])), "fallback_most_logged"


# ----------------------------- reference policies -----------------------------
def _heuristic_ranking(record, mdp):
    """Deterministic prompt-style proxy, not an LLM. Ranks by a qualitative rule."""
    cands = record["allowed_actions"]
    ref = float(record["reference_price"])
    periods_after = max(0, int(record["periods_remaining_including_current"]) - 1)
    next_ref = np.array([c["next_reference_price"] for c in cands], float)
    if record["info_mode"] == "oracle_info":
        rev = np.array([c["immediate_revenue"] for c in cands], float)
    else:
        rev = np.array([c["mean_logged_revenue"] or 0.0 for c in cands], float)
    rev_score = rev / max(float(rev.max()), 1e-9)
    lift = (next_ref - ref) / max(float(mdp.cfg.p_max - mdp.cfg.p_min), 1e-9)
    score = rev_score + (0.35 * periods_after / max(1, mdp.H - 1)) * lift
    return [int(a) for a in np.argsort(-score)]


# ----------------------------- providers (unchanged plumbing) --------------
ENDPOINTS = {
    # OpenAI-compatible chat-completions endpoints. DeepSeek is used in preference
    # to a closed frontier API because its weights are MIT-licensed and its
    # checkpoints are dated, so an API result here is reproducible in principle by
    # anyone who downloads the same checkpoint. That is the standard §3.5.5 sets for
    # every other number in this study, and a closed endpoint cannot meet it.
    "huggingface": "https://router.huggingface.co/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
}

KEY_VARS = {
    "huggingface": ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_TOKEN"),
}


def _api_key(provider):
    for name in KEY_VARS.get(provider, ()):
        tok = os.environ.get(name)
        if tok:
            return tok
    return None


def _hf_token():
    return _api_key("huggingface")


def _endpoint(args):
    return args.api_url or os.environ.get("LLM_CHAT_COMPLETIONS_URL")         or ENDPOINTS.get(args.provider)


SERVED_MODELS = set()   # what the endpoint actually served, for the protocol record


def _call_chat_api(prompt, args):
    """One OpenAI-compatible chat-completions call. Shared by every hosted provider.

    Two provider behaviours are handled explicitly because both silently destroy a
    run otherwise.

    Reasoning models return their tokens in `reasoning_content` and leave `content`
    empty. On DeepSeek V4 the bare model ids (deepseek-v4-pro, deepseek-v4-flash)
    enable thinking while the `deepseek-chat` alias does not, so an explicit id plus
    a modest max_tokens yields finish_reason='length' and an empty string for every
    call -- a complete run of empty responses that the harness then converts into
    myopic fallbacks. `--reasoning-effort none` disables it; the response is checked
    and a truncated-reasoning reply is raised rather than returned as empty.

    The served model id is also recorded, because these ids are aliases: no dated
    checkpoint is exposed by the API, so the served string plus the run date is the
    strongest provenance available (§3.5.5).
    """
    key = _api_key(args.provider)
    if not key:
        raise RuntimeError(f"no API key for provider '{args.provider}'; set one of "
                           f"{', '.join(KEY_VARS.get(args.provider, ()))}")
    if not args.model:
        raise RuntimeError(f"--model is required for --provider {args.provider}")
    body = {"model": args.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": prompt}],
            "temperature": args.temperature, "max_tokens": args.max_tokens}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    req = urllib.request.Request(
        _endpoint(args), data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("model"):
        SERVED_MODELS.add(str(payload["model"]))
    choice = payload["choices"][0]
    content = choice["message"].get("content") or ""
    if not content.strip():
        why = choice.get("finish_reason")
        rt = (payload.get("usage", {}).get("completion_tokens_details", {})
              .get("reasoning_tokens", 0))
        raise RuntimeError(
            f"empty content (finish_reason={why}, reasoning_tokens={rt}). "
            f"This model is reasoning and never emitted an answer; pass "
            f"--reasoning-effort none, or raise --max-tokens well above {rt}.")
    return content


def _format_api_exception(exc):
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        msg = f"HTTP Error {exc.code}: {exc.reason}"
        return f"{msg}; body={body[:1000]}" if body else msg
    return str(exc)


def _make_transformers_chat(args):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("--provider transformers requires: "
                           "pip install transformers accelerate") from exc
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    model.eval()

    def chat(prompt):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        if hasattr(tok, "apply_chat_template"):
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        else:
            text = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                                 do_sample=args.temperature > 0,
                                 temperature=max(args.temperature, 1e-6),
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:],
                          skip_special_tokens=True).strip()

    return chat


def _llm_label(args):
    if args.provider in ENDPOINTS:
        return f"LLM {args.provider} {args.model}"
    if args.provider == "transformers":
        return f"LLM local {args.model}"
    return None


# ----------------------------- arm construction -----------------------------
def _build_arms(mdp, records, counts, args, seed):
    """One response per state -> a free arm and a masked arm, for every source."""
    rows = []
    n_actions = mdp.cfg.n_prices
    rng = np.random.default_rng(1000 + seed)

    def emit(tag, rec, ranking, valid, note, raw=""):
        t, b = rec["period"], rec["reference_bin"]
        if ranking:
            free, free_note = int(ranking[0]), note
        else:
            free, free_note = int(mdp.R[:, b].argmax()), "fallback_myopic"
        masked, mask_note = _apply_mask(ranking or [free], counts, t, b,
                                        n_actions, args.support_topk)
        supported = _supported_actions(counts, t, b, n_actions,
                                       topk=args.support_topk)
        for method, action, mask_name, nt in (
                (tag, free, "none", free_note),
                (f"{tag} + support top{args.support_topk}", masked,
                 f"top{args.support_topk}", mask_note)):
            rows.append({
                "seed": int(seed), "info_mode": rec["info_mode"], "method": method,
                "support_mask": mask_name, "period": int(t),
                "reference_bin": int(b), "action_index": int(action),
                "price": round(float(mdp.prices[action]), 4),
                "optimal_action": int(rec["optimal_action"]),
                "myopic_action": int(rec["myopic_action"]),
                "matches_optimal": bool(action == rec["optimal_action"]),
                "matches_myopic": bool(action == rec["myopic_action"]),
                "off_support": bool(not supported[action]),
                "ranking_len": int(len(ranking)),
                "valid_output": bool(valid), "parse_note": nt,
                "raw_response": raw[:400],
            })

    for rec in records:
        emit("DP oracle", rec, [rec["optimal_action"]], True, "deterministic")
        emit("Myopic one-step optimizer", rec, [rec["myopic_action"]], True,
             "deterministic")
        emit("Heuristic proxy (not LLM)", rec, _heuristic_ranking(rec, mdp), True,
             "deterministic")
        emit("Random policy", rec,
             [int(x) for x in rng.permutation(n_actions)], True, "chance_floor")

    label = _llm_label(args)
    if label is None:
        return pd.DataFrame(rows)

    chat, load_error = None, None
    if args.provider == "transformers":
        try:
            chat = _make_transformers_chat(args)
        except RuntimeError as exc:
            load_error = str(exc)

    for i, rec in enumerate(records, start=1):
        try:
            if args.provider in ENDPOINTS:
                raw = _call_chat_api(rec["prompt"], args)
            elif chat is None:
                raw, ranking, ok, note = load_error, [], False, "load_error"
                emit(label, rec, ranking, ok, note, raw or "")
                continue
            else:
                raw = chat(rec["prompt"])
            ranking, ok, note = _parse_llm_ranking(raw, n_actions)
        except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raw, ranking, ok, note = _format_api_exception(exc), [], False, "api_error"
        emit(label, rec, ranking, ok, note, raw)
        if args.sleep_seconds > 0 and i < len(records):
            time.sleep(args.sleep_seconds)
    return pd.DataFrame(rows)


def _policy_from_actions(sub, mdp):
    table = {(int(r.period), int(r.reference_bin)): int(r.action_index)
             for r in sub.itertuples(index=False)}

    def policy(obs):
        b, t = mdp.decode_obs(obs)
        return table.get((t, b), int(mdp.R[:, b].argmax()))

    return policy


def _rollout_support(policy, mdp, init_bins, counts, topk):
    """Off-support rate measured ALONG THE EVALUATION ROLLOUT.

    `diag_gate2_pricing._support_metrics` measures it this way, so the state-table
    mean is not comparable to the numbers in the dose-response table: a policy can
    be wildly off-support in states its rollout never reaches. Both are reported;
    this is the one that lines up with the other families.
    """
    unseen, logged = [], []
    for b0 in init_bins:
        b = int(b0)
        for t in range(mdp.H):
            a = int(policy(mdp.obs(mdp.ref_grid[b], t)))
            supported = _supported_actions(counts, t, b, mdp.cfg.n_prices, topk=topk)
            unseen.append(not bool(supported[a]))
            logged.append(float(counts[t, b, a]) <= 0.0)
            b = int(mdp.N[a, b])
    return (float(np.mean(unseen)), float(np.mean(logged)))


def _degeneracy(sub, n_actions):
    a = sub["action_index"].to_numpy()
    freq = np.bincount(a, minlength=n_actions) / max(len(a), 1)
    nz = freq[freq > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(n_actions)) if len(nz) > 1 else 0.0
    return {"modal_action_share": float(freq.max()),
            "distinct_actions_used": int((freq > 0).sum()),
            "action_entropy_normalised": ent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results_appendix_llm")
    ap.add_argument("--provider",
                    choices=["heuristic", "huggingface", "deepseek",
                             "transformers"],
                    default="heuristic")
    ap.add_argument("--model", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--cells", default=None, help="Comma-separated N:noise cells.")
    ap.add_argument("--info-modes", default="log_only,oracle_info",
                    help="Comma-separated: log_only and/or oracle_info.")
    ap.add_argument("--support-topk", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--sleep-seconds", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default=None,
                    help="Passed through to the provider, e.g. 'none' to disable "
                         "thinking. Required for DeepSeek V4 explicit model ids, "
                         "which otherwise spend the whole token budget reasoning "
                         "and return empty content.")
    ap.add_argument("--api-url", default=None,
                    help="Override the chat-completions endpoint for a hosted provider.")
    args = ap.parse_args()
    if args.provider != "heuristic" and not args.model:
        ap.error(f"--model is required for --provider {args.provider}")
    if args.provider in ENDPOINTS and not _api_key(args.provider):
        ap.error(f"--provider {args.provider} requires one of "
                 f"{', '.join(KEY_VARS[args.provider])} in the environment")

    cfg = C.smoke() if args.smoke else C.full()
    seeds = _parse_seeds(args.seeds) if args.seeds else ([0] if args.smoke
                                                         else cfg.exp.seeds)
    cells = (_parse_cells(args.cells) if args.cells
             else [(min(cfg.exp.data_sizes), max(cfg.exp.noise_levels))])
    info_modes = [m.strip() for m in args.info_modes.split(",") if m.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    all_actions, raw_rows = [], []
    prompts_path = os.path.join(args.outdir, "appendix_llm_state_prompts.jsonl")
    pf = open(prompts_path, "w", encoding="utf-8")

    for n, noise in cells:
        cell_id = f"N{int(n)}_noise{float(noise):g}"
        for seed in seeds:
            # Records, support counts and prompts are rebuilt PER SEED; the previous
            # version built them once from seeds[0] and transferred the table.
            _seed(seed)
            mdp, _, _, _ = _setup(cfg, {"demand_noise": float(noise)}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, int(n), float(noise), seed,
                                               cfg.data.expert_q)
            counts, revenue = _support_counts(trajs, mdp)
            init_bins = _traj_start_bins(trajs)
            v_opt = float(mdp.Vstar[0, init_bins].mean())
            myopic = lambda o: int(mdp.R[:, mdp.decode_obs(o)[0]].argmax())
            v_beh, _ = mdp.evaluate_policy_fn(myopic, init_bins)

            for info_mode in info_modes:
                print(f"\n=== appendix LLM cell={cell_id} seed={seed} "
                      f"info={info_mode} ===", flush=True)
                records = _all_state_records(mdp, counts, revenue, info_mode)
                for rec in records:
                    pf.write(json.dumps({
                        "seed": seed, "info_mode": info_mode,
                        "period": rec["period"],
                        "reference_bin": rec["reference_bin"],
                        "prompt": rec["prompt"],
                        "optimal_action": rec["optimal_action"],
                        "myopic_action": rec["myopic_action"]},
                        ensure_ascii=True) + "\n")
                acts = _build_arms(mdp, records, counts, args, seed)
                acts["cell_id"] = cell_id
                all_actions.append(acts)

                for method, sub in acts.groupby("method"):
                    pol = _policy_from_actions(sub, mdp)
                    value, _ = mdp.evaluate_policy_fn(pol, init_bins)
                    roll_off, roll_unlogged = _rollout_support(
                        pol, mdp, init_bins, counts, args.support_topk)
                    row = {"cell_id": cell_id, "N": int(n), "noise": float(noise),
                           "seed": int(seed), "info_mode": info_mode,
                           "provider": args.provider, "model": args.model or "",
                           "method": method,
                           "support_mask": sub["support_mask"].iloc[0],
                           "value": float(value),
                           "v_behaviour_myopic": float(v_beh),
                           "v_optimal": float(v_opt),
                           "nv": float(Metric.normalised_value(value, v_beh, v_opt)),
                           "regret": float(Metric.regret(value, v_opt)),
                           "valid_output_rate": float(sub["valid_output"].mean()),
                           "off_support_rate": roll_off,
                           "unlogged_action_rate": roll_unlogged,
                           "off_support_rate_statetable":
                               float(sub["off_support"].mean()),
                           "match_optimal_rate": float(sub["matches_optimal"].mean()),
                           "match_myopic_rate": float(sub["matches_myopic"].mean()),
                           "mean_ranking_len": float(sub["ranking_len"].mean())}
                    row.update(_degeneracy(sub, mdp.cfg.n_prices))
                    raw_rows.append(row)
    pf.close()

    actions = pd.concat(all_actions, ignore_index=True)
    actions.to_csv(os.path.join(args.outdir, "appendix_llm_actions.csv"), index=False)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(os.path.join(args.outdir, "appendix_llm_raw.csv"), index=False)

    summary = (raw.groupby(["method", "info_mode", "support_mask", "provider",
                            "model"], dropna=False)
               .agg(n_runs=("nv", "size"), mean_nv=("nv", "mean"),
                    median_nv=("nv", "median"), mean_regret=("regret", "mean"),
                    valid_output_rate=("valid_output_rate", "mean"),
                    off_support_rate=("off_support_rate", "mean"),
                    unlogged_action_rate=("unlogged_action_rate", "mean"),
                    match_optimal_rate=("match_optimal_rate", "mean"),
                    match_myopic_rate=("match_myopic_rate", "mean"),
                    modal_action_share=("modal_action_share", "mean"),
                    distinct_actions_used=("distinct_actions_used", "mean"),
                    action_entropy_normalised=("action_entropy_normalised", "mean"))
               .reset_index().sort_values("mean_nv", ascending=False))
    summary.to_csv(os.path.join(args.outdir, "appendix_llm_summary.csv"), index=False)

    # dose-response row: bare off-support rate vs mask gain, paired over seeds
    dose = []
    mask_tag = f" + support top{args.support_topk}"
    for (info_mode, base), sub in raw[raw.support_mask == "none"].groupby(
            ["info_mode", "method"]):
        m = raw[(raw.info_mode == info_mode) & (raw.method == base + mask_tag)]
        if m.empty:
            continue
        j = sub[["seed", "nv"]].merge(m[["seed", "nv"]], on="seed",
                                      suffixes=("_bare", "_masked"))
        dose.append({"info_mode": info_mode, "arm": base, "n_seeds": len(j),
                     "bare_off_support": float(sub["off_support_rate"].mean()),
                     "bare_nv": float(j["nv_bare"].mean()),
                     "masked_nv": float(j["nv_masked"].mean()),
                     "mask_gain": float((j["nv_masked"] - j["nv_bare"]).mean())})
    dose = pd.DataFrame(dose).sort_values(["info_mode", "bare_off_support"])
    dose.to_csv(os.path.join(args.outdir, "appendix_llm_dose_response.csv"),
                index=False)

    with open(os.path.join(args.outdir, "appendix_llm_protocol.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "stage": "Appendix: does the logged-support constraint contain an LLM?",
            "question": ("Not 'can an LLM price'. An LLM choosing prices directly is "
                         "the extreme of the action channel; this asks whether the "
                         "same top-k logged-support mask that rescues DT, Q-DT, IQL "
                         "and the bandit learners also rescues it, and how much of "
                         "what remains is the LLM rather than the constraint."),
            "preset": "smoke" if args.smoke else "full",
            "provider": args.provider, "model": args.model,
            # recorded so a hosted run is auditable: which endpoint, which model
            # string, and at what sampling temperature. For an open-weight model
            # served over an API (e.g. DeepSeek V4, MIT-licensed with dated
            # checkpoints) these three fields are what make the run reproducible
            # from public weights rather than only from the vendor's endpoint.
            "endpoint": _endpoint(args) if args.provider in ENDPOINTS else None,
            "reasoning_effort": args.reasoning_effort,
            "served_models": sorted(SERVED_MODELS),
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "determinism_note": (
                "Hosted inference is not bit-deterministic even at temperature 0, "
                "so a hosted arm carries run-to-run variation the exactly-solvable "
                "arms do not. Treat it as one sample, not as a fixed quantity."),
            "seeds": seeds, "cells": cells, "info_modes": info_modes,
            "support_topk": args.support_topk,
            "mask": ("dt._supported_actions, the identical helper the DT and Q-net "
                     "arms use; applied to the model's own ranking, so the masked "
                     "arm is its highest-ranked SUPPORTED action"),
            "controls": {
                "Random policy": "chance floor; establishes the myopic-match rate under uniform choice",
                "Random policy + mask": "what the support set alone buys with zero intelligence; "
                                        "(LLM+mask) - (random+mask) is the part attributable to the LLM",
            },
            "read_this_first": ("Check modal_action_share before quoting any nv. A "
                                "value near 1.0 means the model emitted a constant "
                                "index and the nv is not a capability measurement."),
            "heuristic_note": ("The heuristic provider is an offline proxy for testing "
                               "the harness. It is not evidence about LLM performance."),
        }, f, indent=2)

    # A hosted run whose calls all fail still produces a full results table,
    # because every unparseable response falls back to the myopic action. That
    # failure is therefore SILENT and looks like a real arm scoring nv = 0. Refuse
    # to let it pass unremarked.
    _label = _llm_label(args)
    if _label is not None:
        _sub = raw[raw.method.str.startswith(_label)]
        for _, r in _sub.iterrows():
            if r["valid_output_rate"] < 0.5:
                print(f"\n*** WARNING: {r['method']} ({r['info_mode']}, seed "
                      f"{int(r['seed'])}) parsed only {r['valid_output_rate']:.1%} "
                      f"of responses. Unparsed states fall back to the myopic "
                      f"action, so this arm's normalised value is NOT a "
                      f"measurement of the model. Inspect parse_note in "
                      f"appendix_llm_actions.csv before using it.")
        if not _sub.empty and _sub["modal_action_share"].mean() > 0.9:
            print(f"\n*** WARNING: {_label} emitted the same action for "
                  f"{_sub['modal_action_share'].mean():.1%} of states. That is a "
                  f"degenerate output, not a policy; its normalised value measures "
                  f"nothing. Report the degeneracy instead.")

    print("\n===== Appendix LLM summary =====")
    cols = ["method", "info_mode", "mean_nv", "off_support_rate",
            "match_myopic_rate", "modal_action_share", "distinct_actions_used"]
    print(summary[cols].to_string(index=False))
    if not dose.empty:
        print("\n===== dose-response (bare off-support vs mask gain) =====")
        print(dose.to_string(index=False))
    print(f"\nWrote: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
