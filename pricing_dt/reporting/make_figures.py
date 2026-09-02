"""Turn experiment CSVs into dissertation-ready figures.

Reads the CSVs written by the main runners/diagnostics into --indir and writes
PNGs to --outdir (default: <indir>/figures). Each figure is skipped gracefully if
its CSV is absent, so this works after running a single experiment or all of them.

Usage:
    python make_figures.py --indir results
    python make_figures.py --indir results --outdir results/figures
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless / no display needed
import matplotlib.pyplot as plt


def _read(indir, name):
    path = os.path.join(indir, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def fig_c0(indir, outdir):
    df = _read(indir, "c0_sequential_necessity.csv")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["ref_strength_delta"], df["v_intertemporal_optimal"],
            "o-", label="Intertemporal optimal")
    ax.plot(df["ref_strength_delta"], df["v_myopic_greedy"],
            "s--", label="Myopic per-period greedy")
    ax.fill_between(df["ref_strength_delta"], df["v_myopic_greedy"],
                    df["v_intertemporal_optimal"], alpha=0.15, label="Sequential gap")
    ax.set_xlabel("Reference-price strength (delta)")
    ax.set_ylabel("Expected value")
    ax.set_title("C0: sequential structure grows with reference-price coupling")
    ax.legend()
    return _save(fig, outdir, "c0_sequential_necessity.png")


def fig_e3_bias(indir, outdir):
    summ = _read(indir, "e3_ope_summary.csv")
    raw = _read(indir, "e3_ope.csv")
    if summ is None:
        return None
    x = summ["logger_tv"] if "logger_tv" in summ.columns else summ["drift"]
    xlabel = "Logging drift (mean TV between segment loggers)" if "logger_tv" in summ.columns else "Logging drift"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, summ["mean_abs_bias_pooled"], "o-", color="#de2d26", label="Pooled DR (weak q̂)")
    ax.plot(x, summ["mean_abs_bias_segmented"], "s-", color="#2c7fb8", label="Segmented DR (weak q̂)")
    if raw is not None:
        for est in ("bias_pooled", "bias_segmented"):
            g = raw.groupby("drift")[est].std()
            base = summ["mean_abs_bias_pooled"] if "pooled" in est else summ["mean_abs_bias_segmented"]
            ax.fill_between(x, base - g.values, base + g.values, alpha=0.12)
    # masked comparison: a capable (state-dependent) q_hat hides the effect
    if "mean_abs_bias_pooled_strong" in summ.columns:
        ax.plot(x, summ["mean_abs_bias_pooled_strong"], "o--", color="#fc9272",
                alpha=0.8, label="Pooled DR (strong q̂, masked)")
        ax.plot(x, summ["mean_abs_bias_segmented_strong"], "s--", color="#9ecae1",
                alpha=0.8, label="Segmented DR (strong q̂, masked)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean |bias| vs known true value")
    ax.set_title("C3: pooled DR bias exceeds segmented under drift\n(visible only when q̂ leans on importance weights)")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "c3_bias_vs_drift.png")


def fig_mis(indir, outdir):
    df = _read(indir, "mis_scan_summary.csv")
    if df is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["severity"], df["mean_nv_structured"], "o-", label="Structured (corrupted prior)")
    ax.plot(df["severity"], df["mean_nv_QDT"], "s--", label="Q-DT floor (no prior)")
    # Mark where the prior diagnostic first becomes invalid: either genuinely
    # non-monotone in raw log-demand, or degenerate after forward() clipping.
    if "prior_monotone" in df:
        invalid = ~df["prior_monotone"].astype(bool)
        if "prior_degenerate_after_clamp" in df:
            invalid = invalid | df["prior_degenerate_after_clamp"].astype(bool)
        bad = df[invalid]
        if len(bad):
            ax.axvline(bad["severity"].min(), color="grey", ls=":",
                       label="prior diagnostic invalid")
    ax.set_xlabel("Misspecification severity")
    ax.set_ylabel("Normalised value")
    ax.set_title("Misspecification scan: gross prior error erodes the advantage")
    ax.legend()
    return _save(fig, outdir, "misspecification_scan.png")


def fig_e1(indir, outdir):
    df = _read(indir, "e1_vanilla_failure.csv")
    if df is None:
        return None
    g = df.groupby("noise").agg(
        v_behaviour=("v_behaviour", "mean"),
        v_vanillaDT=("v_vanillaDT", "mean"),
        v_optimal=("v_optimal", "mean"),
        stitch_margin=("stitch_avg_margin_vanillaDT", "mean")).reset_index()
    x = np.arange(len(g)); w = 0.25
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax.bar(x - w, g["v_behaviour"], w, label="Logged behaviour")
    ax.bar(x, g["v_vanillaDT"], w, label="Vanilla DT")
    ax.bar(x + w, g["v_optimal"], w, label="Optimal (same starts)")
    ax.set_xticks(x); ax.set_xticklabels([f"noise={n}" for n in g["noise"]])
    ax.set_ylabel("Expected value (from logged starts)")
    ax.set_title("E1/C1: vanilla DT value")
    ax.legend(fontsize=8)
    ax2.bar(x, g["stitch_margin"], color="#de2d26")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels([f"noise={n}" for n in g["noise"]])
    ax2.set_ylabel("Avg per-start stitching margin")
    ax2.set_title("Vanilla DT vs best logged-from-same-start\n(negative => cannot stitch)")
    return _save(fig, outdir, "e1_vanilla_failure.png")


def fig_optimism_frontier(indir, outdir):
    """③ The anchor-weight (λ) frontier as a core mechanism figure: net
    policy value rises monotonically as the structured optimistic target is mixed
    in (λ 0→1), i.e. a tunable optimism knob rather than an inherited side-effect."""
    df = _read(indir, "optimism_frontier.csv")
    if df is None:
        return None
    lams = sorted(df["lam"].unique())
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for seed, g in df.groupby("seed"):
        ax.plot(g["lam"], g["nv"], color="grey", alpha=0.25, lw=0.8)
    m = df.groupby("lam")["nv"].mean().reindex(lams)
    se = df.groupby("lam")["nv"].sem().reindex(lams)
    ax.errorbar(lams, m.values, yerr=se.values, fmt="o-", color="#2c7fb8",
                lw=2.2, capsize=4, label="mean over seeds")
    for lam in lams:
        ax.annotate(f"{m[lam]:.2f}", (lam, m[lam]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Optimism weight λ  (0 = logged target, 1 = fully structured)")
    ax.set_ylabel("Normalised policy value")
    ax.set_title("Optimism knob: net value rises monotonically with λ\n"
                 "(structured target shaping as a tunable, demand-model-free dial)")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "optimism_frontier.png")


def fig_target_inflation(indir, outdir):
    """⑤a The load-bearing mechanism, visualised: the structured prior (A) inflates
    its relabel target above the true optimum while the better-fitting unconstrained
    model (B) stays calibrated (≈1.0×), and the gap opens up in the sequential regime."""
    df = _read(indir, "structure_verdict_scan.csv")
    if df is None:
        return None
    deltas = sorted(df["delta"].unique())
    pos = np.arange(len(deltas))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    Adata = [df[df.delta == d]["inflation_A"].values for d in deltas]
    Bdata = [df[df.delta == d]["inflation_B"].values for d in deltas]
    bA = ax.boxplot(Adata, positions=pos - 0.18, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor="#2c7fb8", alpha=0.6),
                    medianprops=dict(color="black"))
    bB = ax.boxplot(Bdata, positions=pos + 0.18, widths=0.3, patch_artist=True,
                    boxprops=dict(facecolor="#fec44f", alpha=0.7),
                    medianprops=dict(color="black"))
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="calibrated (target = true optimum)")
    ax.set_xticks(pos); ax.set_xticklabels(deltas)
    ax.set_xlabel("Reference-price strength δ  (sequential coupling)")
    ax.set_ylabel("Relabel target ÷ true optimum")
    ax.legend([bA["boxes"][0], bB["boxes"][0]],
              ["A: structured prior", "B: unconstrained (better fit)"], fontsize=8)
    ax.set_title("Target inflation by coupling δ (per-seed): in the sequential regime\n"
                 "(δ≥2) structured A inflates to ≈1.5× while better-fitting B stays ≈1.0×\n"
                 "(at δ=1 both over-extrapolate in the low-signal regime, B more than A)")
    return _save(fig, outdir, "target_inflation.png")


def fig_e2ab_fixed(indir, outdir):
    """Fixed-QDT prior-isolation ablation."""
    df = _read(indir, "e2ab_fixed_summary.csv")
    if df is None:
        return None
    df = df.sort_values("heldout")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    cols = ["crimson" if a == "A_full_prior"
            else "lightgrey" if "broken" in a
            else "steelblue" if a.startswith("D_bootstrap")
            else "seagreen" if a == "C_misspecified"
            else "tan" for a in df["arm"]]
    ax.barh(range(len(df)), df["heldout"], xerr=df.get("sd"), capsize=3, color=cols)
    if "default_rule" in df.columns:
        ax.scatter(df["default_rule"], range(len(df)), marker="x", color="black",
                   label="default 0.95 target")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["arm"], fontsize=8)
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("Held-out selected normalised value")
    ax.set_title("Prior-isolation ablation with corrected Q-DT")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, outdir, "e2ab_fixed.png")


def fig_elasticity_fixed(indir, outdir):
    """Elasticity-band diagnostic against the action-dependent and action-independent
    Q-DT read-outs."""
    df = _read(indir, "elasticity_fixed_summary.csv")
    if df is None:
        return None
    df = df.sort_values("beta")
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(df["beta"], df["adv_vs_fixed"], "o-", color="steelblue",
            label="structured - corrected Q-DT")
    if "adv_vs_broken" in df.columns:
        ax.plot(df["beta"], df["adv_vs_broken"], "s--", color="lightcoral",
                label="structured - broken Q-DT")
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(0.95, 1.83, color="green", alpha=0.08, label="real-data IQR (0.95-1.83)")
    ax.axvline(1.37, color="green", ls="--", lw=1.0, label="real-data median (1.37)")
    ax.set_xlabel("True own-price sensitivity beta")
    ax.set_ylabel("Normalised-value advantage")
    ax.set_title("Elasticity-band check after the Q-DT baseline fix")
    ax.legend(fontsize=8, loc="lower left")
    return _save(fig, outdir, "elasticity_fixed.png")


def fig_c3_shrinkage(indir, outdir):
    """⑥ C3 absolute: under ESTIMATED propensities, naive per-segment estimation
    worsens bias, but a partially-pooled (shrinkage) propensity recovers — with the
    kappa sweep tracing the bias-variance tradeoff between naive and pooled."""
    summ = _read(indir, "c3_shrinkage_summary.csv")
    kdf = _read(indir, "c3_shrinkage_kappa.csv")
    if summ is None:
        return None
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = summ["logger_tv"] if "logger_tv" in summ.columns else summ["drift"]
    ax.plot(x, summ["mean_abs_bias_pooled"], "o-", color="#636363", label="Pooled (1 propensity)")
    ax.plot(x, summ["mean_abs_bias_naive_seg"], "s--", color="#de2d26",
            label="Segmented, naive per-segment π̂")
    ax.plot(x, summ["mean_abs_bias_shrunk_seg"], "D-", color="#2c7fb8",
            label="Segmented, shrinkage π̂ (proposed)")
    ax.set_xlabel("Logging drift (mean TV between segment loggers)")
    ax.set_ylabel("Mean |bias| vs known true value")
    ax.set_title("C3 under estimated propensities: shrinkage only marginally\n"
                 "eases the naive-segmentation penalty — pooling stays best")
    ax.legend(fontsize=8)
    if kdf is not None:
        kk = kdf["kappa"].astype(float).values
        # plot kappa on a symlog-ish index (0 ... large), label ends
        idx = np.arange(len(kk))
        ax2.errorbar(idx, kdf["mean_abs_bias_shrunk_seg"], yerr=kdf.get("se"),
                     fmt="o-", color="#2c7fb8", capsize=3)
        labels = ["0\n(naive)"] + [f"{int(k)}" for k in kk[1:-1]] + ["∞\n(pooled)"]
        ax2.set_xticks(idx); ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_xlabel("Shrinkage strength κ  (0 = naive segment fit, ∞ = pooled)")
        ax2.set_ylabel("Mean |bias| at highest drift")
        ax2.set_title("Bias monotonically decreases toward pooling (κ→∞):\nno interior κ beats the pooled estimator")
    return _save(fig, outdir, "c3_shrinkage.png")


def fig_c3_mis(indir, outdir):
    """⑥ C3 deployable: a marginalised IS (occupancy-ratio) estimator, fit from data,
    achieves far lower bias than pooled per-step DR and stays low across the drift
    sweep — the result shrinkage could not deliver. Right panel reports the
    target-occupancy coverage (the support limitation, shown not hidden)."""
    summ = _read(indir, "c3_mis_summary.csv")
    if summ is None:
        return None
    e3 = _read(indir, "e3_ope_summary.csv")          # oracle segmented DR floor
    x = summ["logger_tv"] if "logger_tv" in summ.columns else summ["drift"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(x, summ["mean_abs_bias_pooled"], "o-", color="#636363",
            label="Pooled per-step DR (est. π̂)")
    ax.plot(x, summ["mean_abs_bias_mis"], "D-", color="#2c7fb8", lw=2.2,
            label="Marginalised IS (proposed, data-driven)")
    if e3 is not None and "mean_abs_bias_segmented" in e3.columns:
        xe = e3["logger_tv"] if "logger_tv" in e3.columns else e3["drift"]
        ax.plot(xe, e3["mean_abs_bias_segmented"], ":", color="#31a354",
                label="Oracle segmented DR (unattainable floor)")
    ax.set_xlabel("Logging drift (mean TV between segment loggers)")
    ax.set_ylabel("Mean |bias| vs known true value")
    ax.set_title("Marginalised IS vs per-step DR: opposite drift-dependence.\n"
                 "MIS is coverage-limited at low drift but overtakes pooled DR at high drift")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    ax2.plot(x, summ["mean_coverage"], "o-", color="#2c7fb8")
    ax2.set_xlabel("Logging drift (mean TV between segment loggers)")
    ax2.set_ylabel("Target-occupancy coverage (fraction supported)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Why: coverage rises with drift\n(more diverse loggers visit more of the target's cells)")
    return _save(fig, outdir, "c3_mis.png")


def fig_trust_region(indir, outdir):
    """The channel-interpolation scissors: one axis from imitation (m=1) to
    planning (m=A), with an accurate-model control."""
    df = _read(indir, "trust_region_summary.csv")
    if df is None:
        return None
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for mdl, c, lab in (("structured", "crimson", "Fitted structured demand model"),
                        ("oracle", "steelblue", "Exact $Q^*$ (control)")):
        s = df[df.model == mdl].sort_values("m")
        ax.plot(s["m"], s["nv"], "o-", color=c, label=lab)
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.annotate("structured DT\n(goal channel)", xy=(1, 0.67), xytext=(2.2, -1.6),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("estimate-then-optimize\n(action channel)",
                xy=(df.m.max(), df[df.model == "structured"].nv.min()),
                xytext=(4.5, -3.6), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("Trust-region width $m$ (actions admitted from the DT's ranking)")
    ax.set_ylabel("True normalised value")
    ax.set_title("Widening the trust region is harmful\nonly when the model is wrong")
    ax.legend(fontsize=8, loc="lower left")

    s = df[df.model == "structured"].sort_values("m")
    ax2.plot(s["m"], s["v_inmodel"], "s-", color="darkorange", label="Value the model believes in")
    ax2.plot(s["m"], s["curse_gap"], "^--", color="purple", label="Optimizer's-curse gap")
    ax2.set_xlabel("Trust-region width $m$")
    ax2.set_ylabel("Value (revenue units)")
    ax2.set_title("As the trust region widens, the model's\noptimism is increasingly cashed in")
    ax2.legend(fontsize=8)
    ax3 = ax2.twinx()
    ax3.plot(s["m"], s["support"], "v:", color="green", lw=1,
             label="Logged support of chosen actions")
    ax3.set_ylabel("Mean logged frequency of chosen action", fontsize=8)
    ax3.legend(fontsize=7, loc="center right")
    return _save(fig, outdir, "trust_region_scissors.png")


def fig_c2_fixed(indir, outdir):
    """C2 along the data-size axis, against the action-independent and action-dependent
    baselines."""
    df = _read(indir, "c2_fixed_summary.csv")
    if df is None:
        return None
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(df["N"], df["adv_vs_legacy"], "o-", color="lightcoral",
            label="vs the legacy, defective baseline")
    ax.plot(df["N"], df["adv_vs_fixed"], "s-", color="crimson",
            label="vs the fair baseline")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks(df["N"]); ax.set_xticklabels(df["N"])
    ax.set_xlabel("Logged trajectories $N$")
    ax.set_ylabel("Structured DT advantage over Q-DT")
    ax.set_title("The 'advantage grows with N' trend is an\nartefact of the defective comparator")
    ax.legend(fontsize=8)
    for col, c, lab in (("structuredDT", "crimson", "Structured DT"),
                        ("QDT_td", "steelblue", "Q-DT (td)"),
                        ("QDT_qsa", "cadetblue", "Q-DT (q_sa)"),
                        ("QDT_legacy", "lightgrey", "Q-DT (legacy, defective)"),
                        ("vanillaDT", "darkseagreen", "Vanilla DT")):
        if col in df.columns:
            ax2.plot(df["N"], df[col], "o-", color=c, label=lab)
    ax2.set_xscale("log"); ax2.set_xticks(df["N"]); ax2.set_xticklabels(df["N"])
    ax2.set_xlabel("Logged trajectories $N$")
    ax2.set_ylabel("Mean normalised value")
    ax2.set_title("A correct value baseline improves with data;\nthe defective one degrades")
    ax2.legend(fontsize=8)
    return _save(fig, outdir, "c2_fixed_baseline.png")


def fig_channel_ladder(indir, outdir):
    """Relabelling arms including the baseline fix and the oracle rung."""
    df = _read(indir, "channel_ladder.csv")
    if df is None:
        return None
    arms = ["A_structured", "QDT_legacy", "QDT_td", "QDT_td_dn", "QDT_qsa",
            "oracle_Qstar", "oracle_matchA"]
    arms = [a for a in arms if a in df.columns]
    means = [df[a].mean() for a in arms]
    sds = [df[a].std() for a in arms]
    cols = ["crimson" if a == "A_structured"
            else "lightgrey" if a == "QDT_legacy"
            else "steelblue" if a.startswith("QDT")
            else "seagreen" for a in arms]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(arms)), means, yerr=sds, capsize=3, color=cols)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(arms, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean normalised value")
    ax.set_title("Goal-channel arms: the baseline fix (grey -> blue) and the oracle ceiling (green)")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=7)
    return _save(fig, outdir, "channel_ladder.png")


def fig_target_decomp(indir, outdir):
    """Which term of the structured target carries the effect."""
    path = os.path.join(indir, "target_decomp_matrix.csv")
    if not os.path.exists(path):
        return None
    piv = pd.read_csv(path, index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_xlabel("POTENTIAL term source (aspiration field)")
    ax.set_ylabel("ACTION term source (discrimination)")
    ax.set_title("Target decomposition: normalised value")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, outdir, "target_decomp.png")


def fig_data_channel(indir, outdir):
    """The channel matrix: action vs data vs goal vs selection."""
    df = _read(indir, "data_channel_summary.csv")
    if df is None:
        return None
    df = df.sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = ["steelblue" if a.startswith("dyna") else
            "seagreen" if a.startswith("filtBC") else "lightgrey" for a in df["arm"]]
    ax.barh(range(len(df)), df["mean"], xerr=df["std"], capsize=2, color=cols)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df["arm"], fontsize=8)
    ax.axvline(0.670, color="crimson", ls="--", lw=1.2, label="goal channel (structured relabel)")
    ax.axvline(-4.543, color="black", ls=":", lw=1.2, label="action channel (estimate-then-optimize)")
    ax.axvline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("Mean normalised value")
    ax.set_title("Data channel (blue) and data-selection channel (green)\nagainst the other two channels")
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, outdir, "data_channel.png")


def fig_conditioning(indir, outdir):
    """Is the arm ranking an artefact of the 0.95-quantile conditioning rule?"""
    df = _read(indir, "conditioning_sweep.csv")
    if df is None:
        return None
    piv = df.groupby(["arm", "target_rule"]).nv.mean().unstack()
    order = [c for c in ["q0.5", "q0.75", "q0.9", "q0.95", "q0.99", "q1.0",
                         "max x1.25", "max x1.5", "max x2.0"] if c in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for arm in piv.index:
        ax.plot(range(len(order)), piv.loc[arm], "o-", label=arm)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    ax.axvline(order.index("q0.95"), color="grey", ls=":", lw=1)
    ax.text(order.index("q0.95"), ax.get_ylim()[1], " default rule", fontsize=7, va="top")
    ax.set_xlabel("Conditioning target")
    ax.set_ylabel("Mean normalised value")
    ax.set_title("Sensitivity of each relabelling arm to the conditioning target")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "conditioning_sweep.png")


def fig_env2(indir, outdir):
    """Second environment: the channel contrast and the trust-region sweep replicated."""
    ch = _read(indir, "env2_channels_summary.csv")
    tr = _read(indir, "env2_trust.csv")
    if ch is None or tr is None:
        return None
    ch = ch.rename(columns={ch.columns[0]: "arm"}).sort_values("mean")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = ["crimson" if a == "structured" else "grey" for a in ch["arm"]]
    ax.barh(range(len(ch)), ch["mean"], xerr=ch["std"], capsize=3, color=cols)
    ax.set_yticks(range(len(ch))); ax.set_yticklabels(ch["arm"], fontsize=9)
    ax.axvline(0.0, color="k", lw=0.8)
    ax.axvline(-3.246, color="black", ls=":", lw=1.2,
               label="action channel (plan on the same model)")
    ax.set_xlabel("Mean normalised value")
    ax.set_title("Inventory: goal channel is SAFE but not USEFUL\n"
                 "(structured sits below vanilla and BC)")
    ax.legend(fontsize=8, loc="lower right")

    piv = tr.groupby(["model", "m"]).nv.mean().unstack().T
    for mdl, c, lab in (("structured", "crimson", "Fitted Poisson prior"),
                        ("oracle", "steelblue", "Exact $Q^*$ (control)")):
        if mdl in piv.columns:
            ax2.plot(piv.index, piv[mdl], "o-", color=c, label=lab)
    ax2.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax2.set_xlabel("Trust-region width $m$")
    ax2.set_ylabel("True normalised value")
    ax2.set_title("Inventory: the scissors replicate\n"
                  "(same knob, same control)")
    ax2.legend(fontsize=8, loc="center left")
    return _save(fig, outdir, "env2_channels.png")


def fig_gate2_pricing(indir, outdir):
    """Gate2: the integrated pricing comparison, and the support/value relationship.

    Reads the raw table rather than the summary so the right-hand panel can pair
    each method's achieved value with how often it selects an action that never
    appears in the logs at that (t, reference bin).
    """
    df = _read(indir, "gate2_pricing_raw.csv")
    if df is None:
        return None
    agg = (df.groupby("method")
             .agg(nv=("nv", "mean"), sd=("nv", "std"),
                  off=("selected_unseen_rate", "mean"))
             .sort_values("nv"))
    masked = [("support" in m) for m in agg.index]
    fam = ["Q-DT" if m.startswith("Q-DT") else "IQL" if m.startswith("IQL")
           else "structured" if m.startswith("Structured") else "vanilla"
           for m in agg.index]
    base = {"structured": "crimson", "Q-DT": "steelblue",
            "IQL": "goldenrod", "vanilla": "darkseagreen"}
    cols = [base[f] for f in fam]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    y = range(len(agg))
    ax.barh(list(y), agg["nv"], xerr=agg["sd"], capsize=3, color=cols,
            hatch=["//" if m else "" for m in masked], edgecolor="white")
    ax.set_yticks(list(y))
    ax.set_yticklabels(agg.index, fontsize=8)
    ax.set_xlabel("Mean normalised value (90 runs each)")
    ax.set_title("Support-masked arms (hatched) lead every family\n"
                 "9 cells x 10 seeds, shared exact evaluation anchors")
    for i, (v, sd) in enumerate(zip(agg["nv"], agg["sd"])):
        ax.text(v + sd + 0.015, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0, float((agg["nv"] + agg["sd"]).max()) * 1.22)

    # The two support-masking rules pick the same actions here, so their points
    # coincide exactly; collapse them into one label instead of overplotting.
    seen = {}
    for (m, row), c, mk in zip(agg.iterrows(), cols, masked):
        key = (round(row["off"], 6), round(row["nv"], 6))
        ax2.scatter(row["off"], row["nv"], s=90, color=c,
                    marker="s" if mk else "o",
                    edgecolor="black" if mk else "none", zorder=3)
        if key in seen:
            seen[key].append(m)
            continue
        seen[key] = [m]
    for (off, nv), names in seen.items():
        if len(names) > 1 and all(n.startswith("Structured") for n in names):
            label = "Structured DT + support mask"
        elif len(names) > 1:
            label = names[0] + f" (+{len(names)-1} identical)"
        else:
            label = names[0]
        ax2.annotate(label, (off, nv), fontsize=7,
                     xytext=(6, -10), textcoords="offset points")
    ax2.margins(x=0.18)
    ax2.set_xlabel("Fraction of chosen actions never logged at that state")
    ax2.set_ylabel("Mean normalised value")
    ax2.set_title("Leaving logged support tracks losing value\n"
                  "(squares = explicit support mask)")
    ax2.grid(alpha=0.25, ls=":")
    return _save(fig, outdir, "gate2_pricing.png")


def _save(fig, outdir, name):
    fig.tight_layout()
    path = os.path.join(outdir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_support_contraction(indir, outdir):
    """The support mask is a CONTRACTION toward the log's own value, not an
    improvement operator.

    Reads the two support-crossing runs by explicit path rather than from --indir,
    because they are separate result lineages (Section 3.5.5) and must not be merged
    with the CPU tree. Each arm is an arrow from its bare to its masked value, so the
    compression -- and the arms whose arrows point DOWN -- are visible without a table.
    """
    e1p = "results_bandit_20260821/combined_raw.csv"
    e2p = "results_env2_support_n30_20260824/env2_support_raw.csv"
    if not (os.path.exists(e1p) and os.path.exists(e2p)):
        return None
    gp = pd.read_csv(e1p).pivot_table(index=["N", "noise", "seed"],
                                      columns="method", values="nv")
    e1arms = {"Vanilla DT": "vanilla", "Structured DT": "structured",
              "Q-DT fixed td denoised": "Q-DT td", "Q-DT fixed q_sa": "Q-DT q_sa",
              "IQL expectile0.7 beta3": "IQL", "Bandit DM": "bandit DM",
              "Bandit IPS": "bandit IPS", "Bandit DR": "bandit DR"}
    E1 = pd.DataFrame({"arm": list(e1arms.values()),
                       "bare": [gp[k].mean() for k in e1arms],
                       "masked": [gp[k + " support top3"].mean() for k in e1arms]})
    pv = pd.read_csv(e2p).pivot_table(index=["seed", "arm"], columns="mask", values="nv")
    E2 = pv.groupby("arm").agg(bare=("none", "mean"), masked=("top3", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, df, title, floor in ((axes[0], E1.sort_values("bare"), "E1  pricing", 0.448),
                                 (axes[1], E2.sort_values("bare"), "E2  inventory", -1.811)):
        y = np.arange(len(df))
        for i, r in enumerate(df.itertuples(index=False)):
            up = r.masked >= r.bare
            ax.annotate("", xy=(r.masked, i), xytext=(r.bare, i),
                        arrowprops=dict(arrowstyle="->", lw=1.6,
                                        color="steelblue" if up else "crimson"))
        ax.plot(df.bare, y, "o", color="grey", ms=5, label="bare", zorder=3)
        ax.plot(df.masked, y, "D", color="black", ms=5, label="masked", zorder=3)
        ax.axvline(floor, color="darkorange", lw=1.4, ls="--",
                   label="random + mask (%+.2f)" % floor)
        ax.set_yticks(y)
        ax.set_yticklabels(df.arm, fontsize=8)
        # symlog keeps the region of interest (-1..1) linear while compressing the
        # long negative tail of the unmasked random control, which would otherwise
        # squash every learner arm into the right-hand margin.
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("True normalised value (symlog outside $\pm$1)")
        sb, sm = df.bare.std(), df.masked.std()
        ax.set_title("%s\nspread compressed %.1fx  (sd %.2f -> %.2f)" % (title, sb / sm, sb, sm),
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Masking to logged support compresses the spread across arms, but does "
                 "not move them all the same way\n(red arrows: the mask makes that arm "
                 "WORSE)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(outdir, "support_contraction.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_masked_floor(indir, outdir):
    """Masked arms against the no-learner floor, E1, including the LLM appendix arm.

    A masked score is evidence about a method only in the part that clears the same
    mask applied to a policy containing no learner at all.
    """
    gpth = "results_bandit_20260821/combined_raw.csv"
    cpth = "results_appendix_llm_controls_20260821/appendix_llm_raw.csv"
    lpth = "results_appendix_llm_deepseek_20260824/appendix_llm_raw.csv"
    if not all(os.path.exists(x) for x in (gpth, cpth, lpth)):
        return None
    g = pd.read_csv(gpth)
    g = g[(g.N == 100) & (g.noise == 0.5)]
    arms = {"IQL expectile0.7 beta3 support top3": "IQL",
            "Q-DT fixed q_sa support top3": "Q-DT q_sa",
            "Q-DT fixed td denoised support top3": "Q-DT td",
            "Structured DT support top3": "structured DT",
            "Vanilla DT support top3": "vanilla DT",
            "Bandit DM support top3": "bandit DM"}
    vals = {lab: g[g.method == k].nv.mean() for k, lab in arms.items()}
    c = pd.read_csv(cpth)
    c = c[c.info_mode == "oracle_info"]
    floor = c[c.method == "Random policy + support top3"].nv.mean()
    lf = pd.read_csv(lpth)
    vals["LLM (v4-pro)"] = lf[(lf.method.str.startswith("LLM")) &
                              (lf.support_mask != "none")].nv.mean()
    ser = pd.Series(vals).sort_values()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    col = ["crimson" if v < floor else "steelblue" for v in ser.values]
    ax.barh(range(len(ser)), ser.values, color=col, alpha=0.85)
    ax.axvline(floor, color="darkorange", lw=1.8, ls="--")
    ax.annotate("no-learner floor\n(random + mask, %+.3f)" % floor,
                xy=(floor, len(ser) - 1.3), xytext=(floor + 0.06, len(ser) - 2.4),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_yticks(range(len(ser)))
    ax.set_yticklabels(ser.index, fontsize=9)
    ax.set_xlabel("True normalised value (every arm support-masked)")
    ax.set_title("With the constraint applied even-handedly, only the part above the\n"
                 "no-learner floor is attributable to the method", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(outdir, "masked_floor.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.indir, "figures")
    os.makedirs(outdir, exist_ok=True)

    made = []
    for fn in (fig_c0, fig_e1, fig_mis, fig_e3_bias,
               fig_optimism_frontier, fig_target_inflation,
               fig_c3_shrinkage, fig_c3_mis,
               fig_trust_region, fig_c2_fixed, fig_e2ab_fixed,
               fig_elasticity_fixed, fig_channel_ladder, fig_target_decomp,
               fig_data_channel, fig_conditioning, fig_env2,
               fig_gate2_pricing,
               fig_support_contraction, fig_masked_floor):
        p = fn(args.indir, outdir)
        if p:
            made.append(p)
    if made:
        print("Wrote figures:")
        for p in made:
            print("  ", p)
    else:
        print(f"No matching CSVs found in {args.indir}. Run experiments first "
              f"(e.g. python run.py --exp all --preset full --outdir {args.indir}).")


if __name__ == "__main__":
    main()
