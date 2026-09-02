"""Demand-curve diagnostic for optimizer error amplification.

This diagnostic turns the main channel-mechanism claim into inspectable curves.
It fits the same demand models used by the estimate-then-optimize baseline, then
compares, state by state:

  - true demand and fitted demand across the discrete price grid;
  - true dynamic Q(p) and fitted dynamic Qhat(p);
  - the true optimal price and the price selected by the fitted optimizer.

The key quantity is not only demand prediction error. It is whether a fitted
demand curve, when placed in the action channel and optimized by argmax, selects
a price whose true dynamic value is poor. That is the optimizer-amplification
mechanism behind the project's "safer in support/goal/channel-constrained
positions" conclusion.

Output:
  - demand_curve_state_summary.csv
  - demand_curve_points.csv
  - demand_curve_seed_summary.csv
  - demand_curve_top_states.png
  - demand_curve_summary.png

References:
  - Smith and Winkler (2006), the optimizer's curse.
  - Elmachtoub and Grigas (2022), Smart Predict-then-Optimize.
  - Dong et al. (2023), model-based offline RL under local misspecification.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.demand_model import (
    StructuredDemandModel,
    UnconstrainedDemandModel,
    fit_demand_model,
)
from pricing_dt.experiments.experiments import _seed, _setup


def fitted_surfaces(dm, mdp, device="cpu"):
    """Return fitted demand, one-step revenue, dynamic Qhat, Vhat and pi."""
    A, B, H = mdp.cfg.n_prices, mdp.cfg.n_ref_bins, mdp.H
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    demand_hat = np.zeros((H, B, A), dtype=float)
    reward_hat = np.zeros((H, B, A), dtype=float)

    dm.eval()
    with torch.no_grad():
        for t in range(H):
            for b, ref in enumerate(mdp.ref_grid):
                obs = torch.tensor(
                    mdp.obs(ref, t),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0).repeat(A, 1)
                pred = dm(prices, obs).detach().cpu().numpy()
                demand_hat[t, b] = pred
                reward_hat[t, b] = mdp.prices * pred

    Vhat = np.zeros((H + 1, B), dtype=float)
    Qhat = np.zeros((H, B, A), dtype=float)
    pi = np.zeros((H, B), dtype=int)
    for t in reversed(range(H)):
        for b in range(B):
            q = reward_hat[t, b] + Vhat[t + 1, mdp.N[:, b]]
            Qhat[t, b] = q
            pi[t, b] = int(np.argmax(q))
            Vhat[t, b] = float(q[pi[t, b]])
    return demand_hat, reward_hat, Qhat, Vhat, pi


def true_policy_value(pi, mdp):
    """Exact true value for all states if the tabular policy pi is followed."""
    B, H = mdp.cfg.n_ref_bins, mdp.H
    V = np.zeros((H + 1, B), dtype=float)
    for t in reversed(range(H)):
        for b in range(B):
            a = int(pi[t, b])
            V[t, b] = mdp.R[a, b] + V[t + 1, mdp.N[a, b]]
    return V


def support_counts(trajs, mdp):
    counts = np.zeros((mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=int)
    for tr in trajs:
        for t, (b, a) in enumerate(zip(tr.ref_bins, tr.actions)):
            counts[t, int(b), int(a)] += 1
    return counts


def make_rows(seed, model_name, mdp, init, v_beh, v_opt, trajs, demand_hat, reward_hat, Qhat, Vhat, pi, top_k):
    counts = support_counts(trajs, mdp)
    visits = counts.sum(axis=2)
    Vpi = true_policy_value(pi, mdp)
    state_rows = []
    point_rows = []
    true_dem_by_b = np.zeros((mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=float)
    for b, ref in enumerate(mdp.ref_grid):
        true_dem_by_b[b] = mdp.expected_demand(mdp.prices, ref)

    for t in range(mdp.H):
        for b, ref in enumerate(mdp.ref_grid):
            true_a = int(mdp.pistar[t, b])
            chosen_a = int(pi[t, b])
            support = counts[t, b]
            support_order = np.argsort(-support)
            selected_count = int(support[chosen_a])
            state_visit_count = int(visits[t, b])
            selected_share = selected_count / max(1, state_visit_count)
            log_err = np.log(np.clip(demand_hat[t, b], 1e-3, None)) - np.log(
                np.clip(true_dem_by_b[b], 1e-3, None)
            )
            action_regret = float(mdp.Qstar[t, b, true_a] - mdp.Qstar[t, b, chosen_a])
            policy_regret = float(mdp.Vstar[t, b] - Vpi[t, b])
            state_rows.append(
                dict(
                    seed=seed,
                    model=model_name,
                    t=t,
                    ref_bin=b,
                    ref_price=round(float(ref), 4),
                    true_action=true_a,
                    chosen_action=chosen_a,
                    true_price=round(float(mdp.prices[true_a]), 4),
                    chosen_price=round(float(mdp.prices[chosen_a]), 4),
                    price_index_error=abs(chosen_a - true_a),
                    wrong_action=int(chosen_a != true_a),
                    action_regret=action_regret,
                    policy_regret=policy_regret,
                    inmodel_optimism=float(Vhat[t, b] - Vpi[t, b]),
                    demand_log_rmse=float(np.sqrt(np.mean(log_err ** 2))),
                    demand_log_error_at_chosen=float(log_err[chosen_a]),
                    state_visit_count=state_visit_count,
                    selected_support_count=selected_count,
                    selected_support_share=selected_share,
                    selected_in_topk_support=int(chosen_a in set(support_order[:top_k])),
                )
            )
            for a, price in enumerate(mdp.prices):
                point_rows.append(
                    dict(
                        seed=seed,
                        model=model_name,
                        t=t,
                        ref_bin=b,
                        action=a,
                        price=round(float(price), 4),
                        true_demand=float(true_dem_by_b[b, a]),
                        estimated_demand=float(demand_hat[t, b, a]),
                        true_revenue=float(mdp.R[a, b]),
                        estimated_revenue=float(reward_hat[t, b, a]),
                        true_dynamic_q=float(mdp.Qstar[t, b, a]),
                        estimated_dynamic_q=float(Qhat[t, b, a]),
                    )
                )

    init_value_true = float(np.mean(Vpi[0, init]))
    init_value_hat = float(np.mean(Vhat[0, init]))
    seed_row = dict(
        seed=seed,
        model=model_name,
        policy_value=init_value_true,
        policy_nv=M.normalised_value(init_value_true, v_beh, v_opt),
        inmodel_value=init_value_hat,
        init_inmodel_gap=init_value_hat - init_value_true,
        wrong_action_share=float(np.mean([r["wrong_action"] for r in state_rows])),
        mean_price_index_error=float(np.mean([r["price_index_error"] for r in state_rows])),
        mean_policy_regret=float(np.mean([r["policy_regret"] for r in state_rows])),
        p90_policy_regret=float(np.quantile([r["policy_regret"] for r in state_rows], 0.90)),
        mean_action_regret=float(np.mean([r["action_regret"] for r in state_rows])),
        mean_demand_log_rmse=float(np.mean([r["demand_log_rmse"] for r in state_rows])),
        unsupported_choice_share=float(
            np.mean([int(r["selected_support_count"] == 0) for r in state_rows])
        ),
        outside_topk_support_share=float(
            np.mean([int(r["selected_in_topk_support"] == 0) for r in state_rows])
        ),
    )
    return state_rows, point_rows, seed_row


def _series_color(i):
    return ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"][i % 6]


def plot_top_states(points_df, state_df, outpath, model, top_n):
    top = (
        state_df[state_df["model"] == model]
        .sort_values(["policy_regret", "action_regret"], ascending=False)
        .head(top_n)
    )
    if top.empty:
        return

    fig, axes = plt.subplots(
        len(top),
        2,
        figsize=(12.5, max(3.0, 2.9 * len(top))),
        squeeze=False,
    )
    for row_idx, (_, row) in enumerate(top.iterrows()):
        mask = (
            (points_df["seed"] == row.seed)
            & (points_df["model"] == row.model)
            & (points_df["t"] == row.t)
            & (points_df["ref_bin"] == row.ref_bin)
        )
        pts = points_df[mask].sort_values("price")
        ax_d = axes[row_idx, 0]
        ax_q = axes[row_idx, 1]
        x = pts["price"].values
        ax_d.plot(x, pts["true_demand"].values, color=_series_color(0), marker="o", label="True demand")
        ax_d.plot(x, pts["estimated_demand"].values, color=_series_color(1), marker="s", label="Fitted demand")
        ax_q.plot(x, pts["true_dynamic_q"].values, color=_series_color(0), marker="o", label="True dynamic value")
        ax_q.plot(x, pts["estimated_dynamic_q"].values, color=_series_color(1), marker="s", label="Fitted dynamic value")

        for ax in (ax_d, ax_q):
            ax.axvline(row.true_price, color=_series_color(2), linestyle="--", linewidth=1.4, label="True optimal price")
            ax.axvline(row.chosen_price, color=_series_color(3), linestyle=":", linewidth=1.8, label="Optimizer choice")
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("Price")
        ax_d.set_ylabel("Expected demand")
        ax_q.set_ylabel("Dynamic objective")
        title = (
            f"seed={int(row.seed)}, t={int(row.t)}, ref={row.ref_price:.2f}, "
            f"regret={row.policy_regret:.1f}, chosen={row.chosen_price:.2f}, true={row.true_price:.2f}"
        )
        ax_d.set_title(title)
        ax_q.set_title("Argmax surface")
        if row_idx == 0:
            ax_d.legend(loc="best", fontsize=8)
            ax_q.legend(loc="best", fontsize=8)

    fig.suptitle(f"Optimizer amplification: top {len(top)} {model} states", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_summary(state_df, seed_df, outpath, top_k):
    models = list(seed_df["model"].drop_duplicates())
    colors = {m: _series_color(i) for i, m in enumerate(models)}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))

    ax = axes[0, 0]
    means = seed_df.groupby("model")["policy_nv"].mean().reindex(models)
    ax.bar(models, means.values, color=[colors[m] for m in models], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Normalized value")
    ax.set_title("Action-channel EtO value")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    data = [state_df[state_df["model"] == m]["policy_regret"].values for m in models]
    try:
        ax.boxplot(data, tick_labels=models, showfliers=False)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(data, labels=models, showfliers=False)
    ax.set_ylabel("True policy regret")
    ax.set_title("State-level regret after argmax")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    for m in models:
        g = state_df[state_df["model"] == m]
        ax.scatter(
            g["demand_log_rmse"],
            g["policy_regret"],
            s=16,
            alpha=0.45,
            label=m,
            color=colors[m],
        )
    ax.set_xlabel("Demand log-RMSE at state")
    ax.set_ylabel("True policy regret")
    ax.set_title("Prediction error becomes action regret")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    support = seed_df.groupby("model")[
        ["unsupported_choice_share", "outside_topk_support_share"]
    ].mean().reindex(models)
    x = np.arange(len(models))
    width = 0.36
    ax.bar(
        x - width / 2,
        support["unsupported_choice_share"].values,
        width=width,
        color=_series_color(4),
        label="No exact support",
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        support["outside_topk_support_share"].values,
        width=width,
        color=_series_color(5),
        label=f"Outside top-{top_k} support",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of states")
    ax.set_title("Optimizer-selected action support")
    ax.legend(fontsize=8)

    fig.suptitle("Demand model in the action channel: error amplification summary", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_models(text):
    aliases = {"struct": "structured", "mlp": "unconstrained", "uncon": "unconstrained"}
    allowed = {"structured", "unconstrained"}
    models = []
    for raw in text.split(","):
        item = raw.strip().lower().replace("-", "_")
        if not item:
            continue
        item = aliases.get(item, item)
        if item not in allowed:
            raise ValueError(f"Unknown model '{raw}'. Expected one of {sorted(allowed)}.")
        models.append(item)
    return models or ["structured", "unconstrained"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", default="structured,unconstrained")
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--support-top-k", type=int, default=3)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    n_traj = min(cfg.exp.data_sizes)
    noise = max(cfg.exp.noise_levels)
    models = parse_models(args.models)
    print(
        "demand-curve amplification diagnostic: "
        f"N={n_traj} noise={noise} seeds={cfg.exp.seeds} models={models}"
    )

    all_state_rows = []
    all_point_rows = []
    all_seed_rows = []
    for seed in cfg.exp.seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, n_traj, noise, seed, cfg.data.expert_q)

        demand_models = {}
        if "structured" in models:
            dm = StructuredDemandModel(2, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            demand_models["structured"] = dm
        if "unconstrained" in models:
            dm = UnconstrainedDemandModel(2)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            demand_models["unconstrained"] = dm

        for name, dm in demand_models.items():
            demand_hat, reward_hat, Qhat, Vhat, pi = fitted_surfaces(dm, mdp)
            state_rows, point_rows, seed_row = make_rows(
                seed,
                name,
                mdp,
                init,
                v_beh,
                v_opt,
                trajs,
                demand_hat,
                reward_hat,
                Qhat,
                Vhat,
                pi,
                args.support_top_k,
            )
            all_state_rows.extend(state_rows)
            all_point_rows.extend(point_rows)
            all_seed_rows.append(seed_row)
            print(
                f"seed {seed} {name}: nv={seed_row['policy_nv']:.3f} "
                f"inmodel_gap={seed_row['init_inmodel_gap']:+.1f} "
                f"wrong_action={seed_row['wrong_action_share']:.2f} "
                f"mean_regret={seed_row['mean_policy_regret']:.1f}"
            )

    os.makedirs(args.outdir, exist_ok=True)
    state_df = pd.DataFrame(all_state_rows)
    points_df = pd.DataFrame(all_point_rows)
    seed_df = pd.DataFrame(all_seed_rows)

    state_out = os.path.join(args.outdir, "demand_curve_state_summary.csv")
    points_out = os.path.join(args.outdir, "demand_curve_points.csv")
    seed_out = os.path.join(args.outdir, "demand_curve_seed_summary.csv")
    state_df.to_csv(state_out, index=False)
    points_df.to_csv(points_out, index=False)
    seed_df.to_csv(seed_out, index=False)

    if not args.no_plots:
        plot_summary(
            state_df,
            seed_df,
            os.path.join(args.outdir, "demand_curve_summary.png"),
            args.support_top_k,
        )
        plot_model = "structured" if "structured" in models else models[0]
        plot_top_states(
            points_df,
            state_df,
            os.path.join(args.outdir, "demand_curve_top_states.png"),
            plot_model,
            args.top_n,
        )

    print(f"\nWrote {state_out}")
    print(f"Wrote {points_out}")
    print(f"Wrote {seed_out}")
    if not args.no_plots:
        print(f"Wrote {os.path.join(args.outdir, 'demand_curve_summary.png')}")
        print(f"Wrote {os.path.join(args.outdir, 'demand_curve_top_states.png')}")

    print("\n=== demand-curve amplification summary ===")
    summary = seed_df.groupby("model").mean(numeric_only=True).reset_index()
    for _, r in summary.iterrows():
        print(
            f"{r['model']}: nv={r.policy_nv:.3f} "
            f"inmodel_gap={r.init_inmodel_gap:+.1f} "
            f"wrong_action={r.wrong_action_share:.2f} "
            f"mean_regret={r.mean_policy_regret:.1f} "
            f"unsupported={r.unsupported_choice_share:.2f} "
            f"outside_top{args.support_top_k}={r.outside_topk_support_share:.2f}"
        )


if __name__ == "__main__":
    main()
