"""Gate 2 pricing-MDP diagnostics with the corrected Q-DT baseline.

This runner provides:

  * a stronger value-guided DT comparator,
  * discrete IQL as a support-aware offline-RL baseline,
  * explicit support-masked DT inference.

The mainline Q-DT comparator uses the action-dependent target
qdt.value_relabel(..., mode="td") rather than the action-independent state value
V(s_t), and also reports the alternate action-dependent mode="q_sa".
"""
import argparse
import copy
import json
import os

import numpy as np
import pandas as pd

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.baselines import (make_support_masked_qnet_policy,
                                       policy_from_qnet, train_cql,
                                       train_iql_with_diagnostics)
from pricing_dt.core.demand_model import StructuredDemandModel, fit_demand_model
from pricing_dt.core.dt import make_dt_policy, make_support_masked_dt_policy, train_dt
from pricing_dt.experiments.experiments import _seed, _setup, _traj_start_bins
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.relabel import logged_rtg, relabel_dataset


METHODS = [
    "Structured DT",
    "Structured DT support count>=1",
    "Structured DT support top3",
    "Vanilla DT",
    "Vanilla DT support top3",
    "Q-DT fixed td denoised",
    "Q-DT fixed td denoised support top3",
    "Q-DT fixed q_sa",
    "Q-DT fixed q_sa support top3",
    "IQL expectile0.7 beta3",
    "IQL expectile0.7 beta3 support top3",
]


def _parse_seeds(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_cells(text):
    cells = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("cells must be comma-separated N:noise pairs")
        n, noise = item.split(":", 1)
        cells.append((int(n.strip()), float(noise.strip())))
    return cells


def _default_cells(cfg):
    return [(int(n), float(noise))
            for n in cfg.exp.data_sizes
            for noise in cfg.exp.noise_levels]


def _cell_id(n, noise):
    return f"N{int(n)}_noise{float(noise):g}".replace(".", "p")


def _completed_keys(raw):
    if raw.empty:
        return set()
    return set(zip(raw["cell_id"].astype(str),
                   raw["seed"].astype(int),
                   raw["method"].astype(str)))


def _load_existing(outdir):
    raw_path = os.path.join(outdir, "gate2_pricing_raw.csv")
    qlog_path = os.path.join(outdir, "gate2_pricing_qdt_log.csv")
    raw = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
    qlog = pd.read_csv(qlog_path) if os.path.exists(qlog_path) else pd.DataFrame()
    return raw.to_dict("records"), qlog.to_dict("records"), _completed_keys(raw)


def _support_counts(trajs, mdp):
    counts = np.zeros((mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=float)
    for tr in trajs:
        for t, a in enumerate(tr.actions):
            counts[t, int(tr.ref_bins[t]), int(a)] += 1.0
    return counts, counts.sum(axis=2)


def _safe_mean(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _support_metrics(policy_fn, mdp, init_bins, counts, totals):
    unseen, behaviour_probs, logged_counts = [], [], []
    for b0 in init_bins:
        ref = mdp.ref_grid[int(b0)]
        for t in range(mdp.H):
            b = mdp.ref_to_bin(ref)
            a = int(policy_fn(mdp.obs(ref, t)))
            c = float(counts[t, b, a])
            total = float(totals[t, b])
            unseen.append(c <= 0.0)
            behaviour_probs.append(c / total if total > 0.0 else np.nan)
            logged_counts.append(c)
            ref = mdp.ref_grid[mdp.N[a, b]]
    return {
        "selected_unseen_rate": float(np.mean(unseen)),
        "mean_behavior_prob": _safe_mean(behaviour_probs),
        "mean_logged_count": _safe_mean(logged_counts),
    }


def _evaluate_method(raw_rows, *, gate, cell_id, n, noise, seed, method, family,
                     policy_fn, mdp, init_bins, v_beh, v_opt, counts, totals,
                     extra=None):
    v, _ = mdp.evaluate_policy_fn(policy_fn, init_bins)
    support = _support_metrics(policy_fn, mdp, init_bins, counts, totals)
    row = {
        "gate": gate,
        "cell_id": cell_id,
        "N": int(n),
        "noise": float(noise),
        "seed": int(seed),
        "method": method,
        "family": family,
        "v_policy": float(v),
        "v_behaviour_expected": float(v_beh),
        "v_optimal_same_start": float(v_opt),
        "nv": float(M.normalised_value(v, v_beh, v_opt)),
        **support,
    }
    if extra:
        row.update(extra)
    raw_rows.append(row)
    return row


def _summarise(raw):
    if raw.empty:
        return pd.DataFrame()
    return (raw.groupby(["cell_id", "N", "noise", "method", "family"],
                        dropna=False)
            .agg(n_runs=("nv", "size"),
                 mean_nv=("nv", "mean"),
                 std_nv=("nv", "std"),
                 median_nv=("nv", "median"),
                 mean_value=("v_policy", "mean"),
                 mean_unseen_rate=("selected_unseen_rate", "mean"),
                 mean_behavior_prob=("mean_behavior_prob", "mean"))
            .reset_index()
            .sort_values(["cell_id", "mean_nv"], ascending=[True, False]))


def _make_pairwise(raw):
    rows = []
    if raw.empty:
        return pd.DataFrame(rows)

    comparisons = []
    for method in sorted(raw["method"].dropna().unique()):
        if method != "Structured DT":
            label = "structured_minus_" + method.lower().replace(" ", "_").replace(">", "gte")
            comparisons.append(("Structured DT", method, label))
    comparisons.extend([
        ("Structured DT support count>=1", "Structured DT",
         "support_count1_minus_structured"),
        ("Structured DT support top3", "Structured DT",
         "support_top3_minus_structured"),
        ("Vanilla DT support top3", "Vanilla DT",
         "vanilla_support_top3_minus_vanilla"),
        ("Q-DT fixed td denoised", "Q-DT fixed q_sa",
         "qdt_td_minus_qsa"),
    ])

    seen = set()
    for a, b, label in comparisons:
        if (a, b, label) in seen:
            continue
        seen.add((a, b, label))
        sub = raw[raw["method"].isin([a, b])]
        piv = sub.pivot_table(index=["cell_id", "seed"], columns="method",
                              values="nv", aggfunc="mean").dropna()
        if a not in piv or b not in piv or piv.empty:
            continue
        median_diff, p_value = M.paired_test(piv[a].values, piv[b].values)
        rows.append({
            "comparison": label,
            "method_a": a,
            "method_b": b,
            "n_pairs": int(len(piv)),
            "mean_diff_a_minus_b": float((piv[a] - piv[b]).mean()),
            "median_diff_a_minus_b": float(median_diff),
            "wilcoxon_p": float(p_value),
            "a_win_rate": float((piv[a] > piv[b]).mean()),
        })
    return pd.DataFrame(rows)


def _make_method_means(raw):
    if raw.empty:
        return pd.DataFrame()
    return (raw.groupby(["method", "family"], dropna=False)
            .agg(n_runs=("nv", "size"),
                 mean_nv=("nv", "mean"),
                 median_nv=("nv", "median"),
                 mean_value=("v_policy", "mean"),
                 mean_unseen_rate=("selected_unseen_rate", "mean"),
                 mean_behavior_prob=("mean_behavior_prob", "mean"))
            .reset_index()
            .sort_values("mean_nv", ascending=False))


def _write_outputs(outdir, raw_rows, qlog_rows):
    os.makedirs(outdir, exist_ok=True)
    raw = pd.DataFrame(raw_rows)
    if not raw.empty:
        raw = raw.drop_duplicates(["cell_id", "seed", "method"], keep="last")
    raw_path = os.path.join(outdir, "gate2_pricing_raw.csv")
    raw.to_csv(raw_path, index=False)

    summary = _summarise(raw)
    summary_path = os.path.join(outdir, "gate2_pricing_summary.csv")
    summary.to_csv(summary_path, index=False)

    pairwise = _make_pairwise(raw)
    pairwise.to_csv(os.path.join(outdir, "gate2_pricing_pairwise.csv"),
                    index=False)

    means = _make_method_means(raw)
    means.to_csv(os.path.join(outdir, "gate2_pricing_method_means.csv"),
                 index=False)

    qlog = pd.DataFrame(qlog_rows)
    if not qlog.empty:
        qlog = qlog.drop_duplicates(["cell_id", "seed", "config"], keep="last")
    qlog.to_csv(os.path.join(outdir, "gate2_pricing_qdt_log.csv"), index=False)
    return raw_path, summary_path


def _qdt_model_cfg(cfg, smoke):
    model_cfg = copy.deepcopy(cfg.model)
    updates = 300 if smoke else 2000
    model_cfg.cql_alpha = 0.1
    model_cfg.lr = 3e-4
    model_cfg.q_epochs = max(1, int(round(updates / 20)))
    return model_cfg, updates


def _target_q95(rtg):
    return float(np.quantile(np.concatenate(rtg), 0.95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results_gate2_pricing")
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds. Default full: config seeds; smoke: 0,1.")
    ap.add_argument("--cells", default=None,
                    help="Comma-separated N:noise cells, for example 400:0.2,1600:0.5.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim, n_actions = 2, cfg.sim.n_prices
    seeds = _parse_seeds(args.seeds) if args.seeds else cfg.exp.seeds
    cells = _parse_cells(args.cells) if args.cells else (
        [(min(cfg.exp.data_sizes), max(cfg.exp.noise_levels))]
        if args.smoke else _default_cells(cfg)
    )
    qdt_cfg, qdt_updates = _qdt_model_cfg(cfg, args.smoke)
    iql_updates = 300 if args.smoke else 2000

    if args.overwrite:
        raw_rows, qlog_rows, done = [], [], set()
    else:
        raw_rows, qlog_rows, done = _load_existing(args.outdir)

    os.makedirs(args.outdir, exist_ok=True)
    protocol = {
        "date": "2026-08-14",
        "stage": "Gate 2 pricing-MDP diagnostics, corrected mainline",
        "preset": "smoke" if args.smoke else "full",
        "cells": cells,
        "seeds": seeds,
        "methods": METHODS,
        "qdt_note": (
            "Q-DT targets use qdt.value_relabel with action-dependent modes "
            "td and q_sa. The legacy state_value target is intentionally absent "
            "from this mainline runner."
        ),
        "qdt_config": {
            "cql_updates": qdt_updates,
            "cql_alpha": qdt_cfg.cql_alpha,
            "lr": qdt_cfg.lr,
            "q_hidden": qdt_cfg.q_hidden,
        },
        "iql_config": {
            "updates": iql_updates,
            "expectile": 0.7,
            "awr_beta": 3.0,
            "weight_clip": 20.0,
            "lr": 3e-4,
        },
        "support_masks": [
            "count>=1 keeps actions observed in the same timestep/reference-bin cell",
            "top3 keeps up to three most observed actions in that cell",
            "top3 is applied to EVERY family (structured, vanilla, Q-DT, IQL) at "
            "inference only, so target choice and support constraint are crossed "
            "rather than confounded",
        ],
    }
    with open(os.path.join(args.outdir, "gate2_pricing_protocol.json"),
              "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    def is_done(cell, seed, method):
        return (cell, int(seed), method) in done

    def mark_done(cell, seed, method):
        done.add((cell, int(seed), method))

    for n, noise in cells:
        cell = _cell_id(n, noise)
        for seed in seeds:
            print(f"\n=== Gate2 pricing fixed-QDT cell={cell} seed={seed} ===",
                  flush=True)
            _seed(seed)
            mdp, _, _, _ = _setup(cfg, {"demand_noise": float(noise)},
                                  seed=seed)
            trajs = D.make_stitching_necessary(
                mdp, int(n), float(noise), seed, cfg.data.expert_q)
            init_bins = _traj_start_bins(trajs)
            v_opt = float(mdp.Vstar[0, init_bins].mean())

            def myopic(obs):
                b, _ = mdp.decode_obs(obs)
                return int(mdp.R[:, b].argmax())

            v_beh, _ = mdp.evaluate_policy_fn(myopic, init_bins)
            counts, totals = _support_counts(trajs, mdp)

            dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo,
                                       cfg.model.elasticity_hi)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            rtg_struct = relabel_dataset(trajs, dm, mdp,
                                         cfg.model.relabel_lambda)

            need_struct = any(not is_done(cell, seed, m) for m in [
                "Structured DT",
                "Structured DT support count>=1",
                "Structured DT support top3",
            ])
            if need_struct:
                model_struct = train_dt(D.pack_dt(trajs, rtg_struct), obs_dim,
                                        n_actions, cfg.model, seed=seed)
                target_struct = _target_q95(rtg_struct)
                struct_specs = [
                    ("Structured DT", make_dt_policy, {}, "none"),
                    ("Structured DT support count>=1",
                     make_support_masked_dt_policy,
                     {"counts": counts, "min_count": 1}, "count>=1"),
                    ("Structured DT support top3", make_support_masked_dt_policy,
                     {"counts": counts, "topk": 3}, "top3"),
                ]
                for method, factory, kwargs, mask_name in struct_specs:
                    if is_done(cell, seed, method):
                        continue
                    policy = factory(model_struct, mdp, target_struct, **kwargs)
                    _evaluate_method(
                        raw_rows, gate="gate2C_support_mask", cell_id=cell,
                        n=n, noise=noise, seed=seed, method=method,
                        family="DT", policy_fn=policy, mdp=mdp,
                        init_bins=init_bins, v_beh=v_beh, v_opt=v_opt,
                        counts=counts, totals=totals,
                        extra={"target_q95": target_struct,
                               "support_mask": mask_name})
                    mark_done(cell, seed, method)
                    print(f"  {method}: done", flush=True)
                    _write_outputs(args.outdir, raw_rows, qlog_rows)

            rtg_vanilla = logged_rtg(trajs)
            need_vanilla = any(not is_done(cell, seed, m) for m in [
                "Vanilla DT",
                "Vanilla DT support top3",
            ])
            if need_vanilla:
                model_vanilla = train_dt(D.pack_dt(trajs, rtg_vanilla),
                                         obs_dim, n_actions, cfg.model,
                                         seed=seed)
                target_vanilla = _target_q95(rtg_vanilla)
                vanilla_specs = [
                    ("Vanilla DT", make_dt_policy, {}, "none"),
                    ("Vanilla DT support top3", make_support_masked_dt_policy,
                     {"counts": counts, "topk": 3}, "top3"),
                ]
                for method, factory, kwargs, mask_name in vanilla_specs:
                    if is_done(cell, seed, method):
                        continue
                    policy = factory(model_vanilla, mdp, target_vanilla,
                                     **kwargs)
                    _evaluate_method(
                        raw_rows, gate="gate2C_support_mask", cell_id=cell,
                        n=n, noise=noise, seed=seed, method=method,
                        family="DT", policy_fn=policy, mdp=mdp,
                        init_bins=init_bins, v_beh=v_beh, v_opt=v_opt,
                        counts=counts, totals=totals,
                        extra={"target_q95": target_vanilla,
                               "support_mask": mask_name})
                    mark_done(cell, seed, method)
                    print(f"  {method}: done", flush=True)
                    _write_outputs(args.outdir, raw_rows, qlog_rows)

            qdt_methods = [
                ("Q-DT fixed td denoised", "td", True),
                ("Q-DT fixed q_sa", "q_sa", False),
            ]
            # The support mask is applied at INFERENCE only, so it re-ranks the
            # logits of an already-trained model. Each Q-DT target therefore gets
            # the same top-3 mask the structured and vanilla arms get, reusing the
            # SAME trained DT: the extra arm costs one evaluation, not a retrain.
            # Without these cells the support comparison is bare-against-masked,
            # so no "which target wins" statement is well posed.
            qdt_variants = [
                ("", make_dt_policy, {}, "none"),
                (" support top3", make_support_masked_dt_policy,
                 {"topk": 3}, "top3"),
            ]
            qdt_names = [m + sfx for m, _, _ in qdt_methods
                         for sfx, _, _, _ in qdt_variants]
            need_qdt = any(not is_done(cell, seed, m) for m in qdt_names)
            if need_qdt:
                q = train_cql(trajs, mdp, obs_dim, n_actions, qdt_cfg,
                              seed=seed)
                qlog_rows.append({
                    "cell_id": cell,
                    "N": int(n),
                    "noise": float(noise),
                    "seed": int(seed),
                    "config": "fixed_cql_alpha0.1",
                    "cql_updates": int(qdt_updates),
                    "cql_alpha": float(qdt_cfg.cql_alpha),
                    "lr": float(qdt_cfg.lr),
                })
                for method, mode, denoise in qdt_methods:
                    if all(is_done(cell, seed, method + sfx)
                           for sfx, _, _, _ in qdt_variants):
                        continue
                    rtg_q, _ = value_relabel(
                        trajs, mdp, obs_dim, n_actions, qdt_cfg, seed=seed,
                        mode=mode, denoise=denoise, q=q)
                    model_q = train_dt(D.pack_dt(trajs, rtg_q), obs_dim,
                                       n_actions, cfg.model, seed=seed)
                    target_q = _target_q95(rtg_q)
                    for sfx, factory, kwargs, mask_name in qdt_variants:
                        name = method + sfx
                        if is_done(cell, seed, name):
                            continue
                        kw = dict(kwargs)
                        if mask_name != "none":
                            kw["counts"] = counts
                        _evaluate_method(
                            raw_rows, gate="gate2A_fixed_value_guided_dt",
                            cell_id=cell, n=n, noise=noise, seed=seed,
                            method=name, family="Q-DT",
                            policy_fn=factory(model_q, mdp, target_q, **kw),
                            mdp=mdp, init_bins=init_bins, v_beh=v_beh,
                            v_opt=v_opt, counts=counts, totals=totals,
                            extra={"target_q95": target_q,
                                   "qdt_mode": mode,
                                   "qdt_denoise": bool(denoise),
                                   "support_mask": mask_name,
                                   "cql_updates": int(qdt_updates),
                                   "cql_alpha": float(qdt_cfg.cql_alpha)})
                        mark_done(cell, seed, name)
                        print(f"  {name}: done", flush=True)
                        _write_outputs(args.outdir, raw_rows, qlog_rows)

            # IQL gets the same treatment for the same reason: its policy is a
            # Q-net, so the mask is again a pure re-ranking of its own scores.
            iql_variants = [
                ("", lambda pi: policy_from_qnet(pi), "none"),
                (" support top3",
                 lambda pi: make_support_masked_qnet_policy(pi, mdp, counts,
                                                            topk=3), "top3"),
            ]
            iql_base = "IQL expectile0.7 beta3"
            if any(not is_done(cell, seed, iql_base + sfx)
                   for sfx, _, _ in iql_variants):
                iql_batch = min(512, int(n) * cfg.sim.horizon)
                pi_iql, idiag = train_iql_with_diagnostics(
                    trajs, mdp, obs_dim, n_actions, cfg.model, seed=seed,
                    expectile=0.7, awr_beta=3.0, weight_clip=20.0,
                    updates=iql_updates, batch_size=iql_batch, lr=3e-4)
                for sfx, make_pi, mask_name in iql_variants:
                    name = iql_base + sfx
                    if is_done(cell, seed, name):
                        continue
                    _evaluate_method(
                        raw_rows, gate="gate2B_iql", cell_id=cell, n=n,
                        noise=noise, seed=seed, method=name, family="IQL",
                        policy_fn=make_pi(pi_iql), mdp=mdp,
                        init_bins=init_bins, v_beh=v_beh, v_opt=v_opt,
                        counts=counts, totals=totals,
                        extra={**idiag, "support_mask": mask_name})
                    mark_done(cell, seed, name)
                    print(f"  {name}: done", flush=True)
                    _write_outputs(args.outdir, raw_rows, qlog_rows)

    raw_path, summary_path = _write_outputs(args.outdir, raw_rows, qlog_rows)
    print(f"\nWrote {raw_path}")
    print(f"Wrote {summary_path}")
    means_path = os.path.join(args.outdir, "gate2_pricing_method_means.csv")
    if os.path.exists(means_path):
        means = pd.read_csv(means_path)
        cols = ["method", "n_runs", "mean_nv", "median_nv",
                "mean_unseen_rate", "mean_behavior_prob"]
        print(means[cols].to_string(index=False))


if __name__ == "__main__":
    main()
