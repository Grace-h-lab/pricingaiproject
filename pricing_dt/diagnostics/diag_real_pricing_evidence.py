"""Real-data pricing evidence: elasticity and support diagnostics.

Online Retail II cannot reveal the ground-truth optimal price, because only the
chosen price is observed at each product-time state. This diagnostic strengthens
the descriptive real-data layer without making a causal claim:

  - transaction-level per-product log-log elasticity calibration;
  - weekly product-panel elasticity with product fixed effects and calendar
    controls;
  - price-bin support / coverage diagnostics showing how many counterfactual
    prices are unobserved in real logs.

The results are calibration and support evidence, not proof of a real-data
optimal-pricing policy.

Outputs:
  - real_pricing_evidence_summary.csv
  - real_elasticity_by_product.csv
  - real_price_support_by_product.csv
  - real_pricing_evidence_protocol.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from pricing_dt.core import realdata as RD


def _resolve_data_path(path):
    if path:
        return path
    for candidate in ("online_retail_II.xlsx", "online_retail_II.csv"):
        if os.path.exists(candidate):
            return candidate
    return None


def _weekly_panel(df):
    work = df.copy()
    work["stock"] = work["stock"].astype(str)
    work["week"] = work["date"].dt.to_period("W").apply(lambda p: p.start_time)
    work["revenue"] = work["qty"] * work["price"]
    g = (
        work.groupby(["stock", "week"], observed=True)
        .agg(qty=("qty", "sum"), revenue=("revenue", "sum"), rows=("qty", "size"))
        .reset_index()
    )
    g = g[(g["qty"] > 0) & (g["revenue"] > 0)].copy()
    g["price"] = g["revenue"] / g["qty"]
    g = g[(g["price"] > 0) & np.isfinite(g["price"])].copy()
    g["log_qty"] = np.log(g["qty"].clip(lower=1e-9))
    g["log_price"] = np.log(g["price"].clip(lower=1e-9))
    week_no = g["week"].dt.isocalendar().week.astype(int).to_numpy()
    g["week_sin"] = np.sin(2.0 * np.pi * week_no / 52.0)
    g["week_cos"] = np.cos(2.0 * np.pi * week_no / 52.0)
    week_idx = (g["week"] - g["week"].min()).dt.days / 7.0
    denom = max(float(week_idx.max() - week_idx.min()), 1.0)
    g["time_trend"] = (week_idx - week_idx.min()) / denom
    return g.sort_values(["stock", "week"]).reset_index(drop=True)


def _ols(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[keep], X[keep]
    if len(y) <= X.shape[1] + 1:
        return {"n": int(len(y)), "coef": [], "se": [], "r2": float("nan")}
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    try:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid @ resid) / sst) if sst > 0 else float("nan")
    return {
        "n": int(len(y)),
        "coef": beta.tolist(),
        "se": se.tolist(),
        "r2": float(r2),
    }


def _demean_by_stock(panel, cols):
    out = panel[cols].astype(float).copy()
    for c in cols:
        out[c] = out[c] - panel.groupby("stock")[c].transform("mean")
    return out


def _panel_elasticity(panel, min_weeks, min_price_bins, n_price_bins):
    support = _price_support(panel, n_price_bins)
    eligible = support[
        (support["n_weeks"] >= min_weeks)
        & (support["distinct_price_bins"] >= min_price_bins)
    ]["stock"]
    p = panel[panel["stock"].isin(eligible)].copy()
    if p.empty:
        return {}, pd.DataFrame()

    pooled_X = np.column_stack(
        [
            np.ones(len(p)),
            p["log_price"].to_numpy(),
            p["week_sin"].to_numpy(),
            p["week_cos"].to_numpy(),
            p["time_trend"].to_numpy(),
        ]
    )
    pooled = _ols(p["log_qty"].to_numpy(), pooled_X)

    dm = _demean_by_stock(
        p,
        ["log_qty", "log_price", "week_sin", "week_cos", "time_trend"],
    )
    fe_X = dm[["log_price", "week_sin", "week_cos", "time_trend"]].to_numpy()
    fe = _ols(dm["log_qty"].to_numpy(), fe_X)

    slopes = []
    for stock, g in p.groupby("stock", observed=True):
        if len(g) < min_weeks or g["log_price"].std() < 1e-8:
            continue
        b = np.polyfit(g["log_price"], g["log_qty"], 1)[0]
        if np.isfinite(b):
            slopes.append(
                {
                    "stock": stock,
                    "n_weeks": int(len(g)),
                    "weekly_loglog_slope": float(b),
                    "weekly_abs_elasticity": float(abs(b)),
                    "downward_sloping": bool(b < 0),
                }
            )
    slopes_df = pd.DataFrame(slopes)
    summary = {
        "panel_rows_used": int(len(p)),
        "panel_products_used": int(p["stock"].nunique()),
        "pooled_weekly_slope": float(pooled["coef"][1]) if pooled.get("coef") else float("nan"),
        "pooled_weekly_elasticity_abs": float(abs(pooled["coef"][1])) if pooled.get("coef") else float("nan"),
        "pooled_weekly_slope_se": float(pooled["se"][1]) if pooled.get("se") else float("nan"),
        "pooled_weekly_r2": pooled.get("r2", float("nan")),
        "fe_weekly_slope": float(fe["coef"][0]) if fe.get("coef") else float("nan"),
        "fe_weekly_elasticity_abs": float(abs(fe["coef"][0])) if fe.get("coef") else float("nan"),
        "fe_weekly_slope_se": float(fe["se"][0]) if fe.get("se") else float("nan"),
        "fe_weekly_r2_within": fe.get("r2", float("nan")),
    }
    if not slopes_df.empty:
        summary.update(
            {
                "product_weekly_abs_elasticity_median": float(slopes_df["weekly_abs_elasticity"].median()),
                "product_weekly_abs_elasticity_p25": float(slopes_df["weekly_abs_elasticity"].quantile(0.25)),
                "product_weekly_abs_elasticity_p75": float(slopes_df["weekly_abs_elasticity"].quantile(0.75)),
                "product_downward_slope_share": float(slopes_df["downward_sloping"].mean()),
            }
        )
    return summary, slopes_df


def _price_support(panel, n_price_bins):
    rows = []
    edges = np.linspace(0.0, 1.0, n_price_bins + 1)
    for stock, g in panel.groupby("stock", observed=True):
        prices = g["price"].to_numpy(dtype=float)
        if len(prices) == 0:
            continue
        p_min, p_max = float(np.min(prices)), float(np.max(prices))
        if p_max - p_min <= 1e-9:
            bins = np.zeros(len(prices), dtype=int)
            scaled = np.zeros(len(prices), dtype=float)
        else:
            scaled = (prices - p_min) / (p_max - p_min)
            bins = np.clip(np.digitize(scaled, edges, right=True), 0, n_price_bins - 1)
        counts = pd.Series(bins).value_counts()
        rows.append(
            {
                "stock": stock,
                "n_weeks": int(len(g)),
                "total_qty": float(g["qty"].sum()),
                "distinct_prices": int(pd.Series(prices).round(6).nunique()),
                "distinct_price_bins": int(counts.size),
                "bin_coverage_share": float(counts.size / n_price_bins),
                "main_bin_share": float(counts.max() / max(len(g), 1)),
                "price_min": p_min,
                "price_median": float(np.median(prices)),
                "price_max": p_max,
                "price_cv": float(np.std(prices) / max(np.mean(prices), 1e-9)),
                "scaled_price_sd": float(np.std(scaled)),
            }
        )
    return pd.DataFrame(rows)


def _support_summary(support, n_price_bins):
    if support.empty:
        return {}
    return {
        "support_products": int(len(support)),
        "median_price_bins_per_product": float(support["distinct_price_bins"].median()),
        "mean_price_bins_per_product": float(support["distinct_price_bins"].mean()),
        "p25_price_bins_per_product": float(support["distinct_price_bins"].quantile(0.25)),
        "p75_price_bins_per_product": float(support["distinct_price_bins"].quantile(0.75)),
        "mean_bin_coverage_share": float(support["bin_coverage_share"].mean()),
        "median_main_bin_share": float(support["main_bin_share"].median()),
        "share_products_ge_2_bins": float((support["distinct_price_bins"] >= 2).mean()),
        "share_products_ge_4_bins": float((support["distinct_price_bins"] >= 4).mean()),
        "share_products_ge_6_bins": float((support["distinct_price_bins"] >= 6).mean()),
        "decision_level_unobserved_price_share": float((n_price_bins - 1) / n_price_bins),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-data", default=None)
    ap.add_argument("--outdir", default="results_real_pricing_evidence")
    ap.add_argument("--n-price-bins", type=int, default=11)
    ap.add_argument("--min-weeks", type=int, default=8)
    ap.add_argument("--min-price-bins", type=int, default=3)
    args = ap.parse_args()

    path = _resolve_data_path(args.real_data)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"loading real data from: {path or 'ucimlrepo fallback'}")
    df = RD.load_online_retail_ii(path)
    panel = _weekly_panel(df)
    support = _price_support(panel, args.n_price_bins)
    panel_summary, slopes = _panel_elasticity(
        panel,
        args.min_weeks,
        args.min_price_bins,
        args.n_price_bins,
    )
    tx_calib = RD.calibrate_elasticity(df)

    support_path = os.path.join(args.outdir, "real_price_support_by_product.csv")
    support.to_csv(support_path, index=False)
    slopes_path = os.path.join(args.outdir, "real_elasticity_by_product.csv")
    slopes.to_csv(slopes_path, index=False)

    summary = {
        "clean_transactions": int(len(df)),
        "transaction_products": int(df["stock"].astype(str).nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "weekly_panel_rows": int(len(panel)),
        "weekly_panel_products": int(panel["stock"].nunique()),
        "transaction_products_used_for_elasticity": int(tx_calib["n_products_used"]),
        "transaction_median_abs_elasticity": float(tx_calib["median_elasticity"]),
        "transaction_abs_elasticity_p25": float(tx_calib["p25"]),
        "transaction_abs_elasticity_p75": float(tx_calib["p75"]),
        **panel_summary,
        **_support_summary(support, args.n_price_bins),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(args.outdir, "real_pricing_evidence_summary.csv")
    summary_df.round(6).to_csv(summary_path, index=False)

    protocol = {
        "stage": "Real-data elasticity and support diagnostics",
        "data_path": path,
        "n_price_bins": args.n_price_bins,
        "min_weeks": args.min_weeks,
        "min_price_bins": args.min_price_bins,
        "interpretation": (
            "Elasticities are descriptive calibration estimates, not causal price "
            "effects. Support diagnostics show why real logs cannot identify "
            "ground-truth optimal prices without experiments or stronger causal "
            "assumptions."
        ),
    }
    protocol_path = os.path.join(args.outdir, "real_pricing_evidence_protocol.json")
    with open(protocol_path, "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print(f"Wrote {summary_path}")
    print(f"Wrote {slopes_path}")
    print(f"Wrote {support_path}")
    print(f"Wrote {protocol_path}")
    print("\n=== real pricing evidence summary ===")
    print(
        f"transactions={summary['clean_transactions']} "
        f"products={summary['transaction_products']} "
        f"transaction median |elasticity|={summary['transaction_median_abs_elasticity']:.3f}"
    )
    print(
        f"weekly FE |elasticity|={summary['fe_weekly_elasticity_abs']:.3f} "
        f"(slope={summary['fe_weekly_slope']:.3f}, se={summary['fe_weekly_slope_se']:.3f})"
    )
    print(
        f"median price bins/product={summary['median_price_bins_per_product']:.1f}/"
        f"{args.n_price_bins}; share products >=6 bins="
        f"{summary['share_products_ge_6_bins']:.3f}"
    )
    print(
        "At a product-time decision, only one realised price is observed; "
        f"{summary['decision_level_unobserved_price_share']:.3f} of the discrete "
        "price grid is counterfactual."
    )


if __name__ == "__main__":
    main()
