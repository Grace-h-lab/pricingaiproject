"""Real retail data track: Online Retail II (UCI id=502, CC BY 4.0).

Two roles, kept separate throughout:
  1. CALIBRATION  -- estimate price elasticity from real transactions and map it
     onto the simulator's parameters, so the (ground-truth) simulator experiments
     run at a realistic elasticity. This is the main, clean use: it keeps every
     C1/C2/C3 result valid (the simulator still provides exact truth) while
     anchoring the regime to reality.
  2. REALISM CHECK -- build (state, price, demand) trajectories from the real
     logs and run vanilla-DT / BC + OPE on a time-based held-out split. There is
     NO ground-truth policy value on real data, so only OPE estimates and the
     fitted (interpretable) demand curve are reported here.

Loading:
  - Preferred: pass a local CSV/XLSX (UCI download or a Kaggle mirror such as
    'online_retail_II.csv'); column-name variants are handled.
  - Best-effort fallback: load_online_retail_ii() with no path tries ucimlrepo,
    but UCI id=502 is not always exposed through the Python import API.

Reference:
  - Chen, D. (2012), Online Retail II, UCI Machine Learning Repository,
    DOI: 10.24432/C5CG6D.
"""
import warnings

import numpy as np
import pandas as pd
from pricing_dt.core.data import Trajectory


# ----------------------- load & clean -----------------------
_COLMAP = {
    "InvoiceNo": "invoice", "Invoice": "invoice",
    "StockCode": "stock", "Description": "desc",
    "Quantity": "qty",
    "InvoiceDate": "date",
    "UnitPrice": "price", "Price": "price",
    "CustomerID": "cust", "Customer ID": "cust",
    "Country": "country",
}


def load_online_retail_ii(path=None):
    """Return a cleaned tidy DataFrame with columns [stock, date, qty, price]."""
    if path is None:
        try:
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=502)
        except Exception as exc:
            raise RuntimeError(
                "Online Retail II is not reliably available through ucimlrepo. "
                "Download the CSV/XLSX manually and pass it with --real-data."
            ) from exc
        df = ds.data.features.copy()
    elif str(path).lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in _COLMAP.items() if k in df.columns})
    df["date"] = pd.to_datetime(df["date"])
    # Guard against the single-year extract. Online Retail II (UCI id=502) is two
    # sheets covering 2009-12-01 to 2011-12-09; the widely mirrored single-year file
    # is the second sheet alone, and its row count is within one of that sheet's, so
    # a size check would not catch the substitution -- the span is what separates them.
    span_days = (df["date"].max() - df["date"].min()).days
    if span_days < 600:
        warnings.warn(
            f"Loaded retail data spans only {span_days} days "
            f"({df['date'].min().date()} to {df['date'].max().date()}). Online Retail II "
            "covers about 738 days across two sheets; this looks like the single-year "
            "extract (UCI id=352), which calibrates a different elasticity band.",
            RuntimeWarning, stacklevel=2)
    # remove cancellations (invoice starting 'C'), non-positive qty/price, missing product
    df = df[~df["invoice"].astype(str).str.upper().str.startswith("C")]
    df = df[(df["qty"] > 0) & (df["price"] > 0) & df["stock"].notna()]
    return df[["stock", "date", "qty", "price"]].reset_index(drop=True)


# ----------------------- elasticity calibration -----------------------
def calibrate_elasticity(df, min_prices=6, min_rows=40):
    """Per-product log-log OLS of log(qty) on log(price); return the median slope
    across products with enough price variation. Demand is downward-sloping, so
    the slope is negative; |slope| is the elasticity estimate."""
    slopes = []
    for stock, g in df.groupby("stock"):
        if len(g) < min_rows or g["price"].nunique() < min_prices:
            continue
        x = np.log(g["price"].values); y = np.log(g["qty"].values)
        if x.std() < 1e-6:
            continue
        b = np.polyfit(x, y, 1)[0]
        if np.isfinite(b):
            slopes.append(b)
    slopes = np.array(slopes)
    elasticity = float(np.median(np.abs(slopes))) if len(slopes) else float("nan")
    return dict(n_products_used=len(slopes),
                median_elasticity=round(elasticity, 3),
                p25=round(float(np.percentile(np.abs(slopes), 25)), 3) if len(slopes) else None,
                p75=round(float(np.percentile(np.abs(slopes), 75)), 3) if len(slopes) else None)


def suggest_sim_params(elasticity):
    """Map an empirical elasticity onto SimConfig.beta and the structured prior's bounds.

    `elasticity` comes from a log-log fit, so it is a price elasticity. `beta` multiplies
    the price LEVEL, so it is a semi-elasticity, and the two agree only at p = 1: the
    simulator's own elasticity is beta * p, which over the price grid [0.5, 2.0] spans
    exactly the [0.5 * beta, 2 * beta] the bounds below are set to. The mapping is
    therefore a centring on the empirical estimate at mid-grid rather than an equality,
    and is used to place the simulator in a plausible region, not to claim the two
    quantities are the same.
    """
    return dict(beta=round(max(elasticity, 0.5), 2),
                elasticity_lo=round(max(elasticity * 0.5, 0.2), 2),
                elasticity_hi=round(elasticity * 2.0, 2))


# ----------------------- trajectory construction -----------------------
def build_real_trajectories(df, n_prices=11, horizon=8, freq="W",
                            top_k_products=200, split=0.8, seed=0):
    """Aggregate to regular time steps per product and cut into fixed-length
    episodes. State = [scaled reference price, scaled time]; action = price bin
    (price scaled within product); reward = scaled price * scaled demand
    (per-product scaling keeps revenue comparable across products).

    Returns (train_trajs, test_trajs, price_bins) by a TIME-BASED split.
    """
    rng = np.random.default_rng(seed)
    top = df["stock"].value_counts().head(top_k_products).index
    edges = np.linspace(0.0, 1.0, n_prices)            # shared bins on within-product scaled price
    train, test = [], []

    for stock in top:
        g = df[df["stock"] == stock].set_index("date").sort_index()
        # weekly: quantity-weighted mean price, total demand
        wk = g.resample(freq).apply(lambda x: pd.Series({
            "price": np.average(x["price"], weights=x["qty"]) if len(x) else np.nan,
            "qty": x["qty"].sum()}))
        wk = wk.dropna()
        if len(wk) < horizon + 1:
            continue
        p = wk["price"].values; q = wk["qty"].values
        if p.max() - p.min() < 1e-6 or q.max() <= 0:
            continue
        ps = (p - p.min()) / (p.max() - p.min())       # scaled price in [0,1]
        qs = q / q.max()                                # scaled demand in [0,1]
        a = np.clip(np.digitize(ps, edges) - 1, 0, n_prices - 1)
        n_windows = len(wk) // horizon
        n_train = int(n_windows * split)               # earliest windows -> train (no leakage)
        for w in range(n_windows):
            sl = slice(w * horizon, (w + 1) * horizon)
            ps_w, a_w, qs_w = ps[sl], a[sl], qs[sl]
            ref = np.concatenate([[ps_w[0]], 0.5 * ps_w[:-1] + 0.5 * np.cumsum(ps_w)[:-1] /
                                  np.arange(1, horizon)])  # trailing reference price
            obs = np.stack([ref, np.arange(horizon) / (horizon - 1)], 1).astype(np.float32)
            rew = (ps_w * qs_w).astype(np.float32)         # scaled revenue
            tr = Trajectory(obs, np.zeros(horizon, int), a_w.astype(int), rew, seg=0)
            (train if w < n_train else test).append((tr, ps_w.astype(np.float32)))
    return train, test, edges
