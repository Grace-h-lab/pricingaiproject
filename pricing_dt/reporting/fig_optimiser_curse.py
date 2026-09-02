"""One state, drawn twice: what the fitted model believes and what is true.

The optimiser's curse is stated in Chapter 4 as a property of an argmax over an imperfect
model. This figure shows it happening at the single worst state of the pricing environment,
because the statement is easier to accept once the reader has seen the shape that produces it:
the fitted demand curve climbs steeply as the price falls, the true one is almost flat, and the
planner walks straight to the price where the gap is widest.

Data: results_demand_curves_20260818/demand_curve_points.csv, written by
pricing_dt/diagnostics/diag_demand_curve_amplification.py. Nothing is recomputed here.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results_demand_curves_20260818/demand_curve_points.csv"
STATE = "results_demand_curves_20260818/demand_curve_state_summary.csv"
OUT = "results/figures/optimiser_curse_state.png"


def worst_state():
    """The state carrying the largest policy regret, which is the one Chapter 4 quotes."""
    rows = list(csv.DictReader(open(STATE, encoding="utf-8")))
    w = max(rows, key=lambda r: float(r["policy_regret"]))
    return w, float(w["policy_regret"])


def main():
    if not os.path.exists(SRC):
        print("missing %s" % SRC)
        return 1
    w, regret = worst_state()
    key = (w["seed"], w["model"], w["t"], w["ref_bin"])
    pts = [r for r in csv.DictReader(open(SRC, encoding="utf-8"))
           if (r["seed"], r["model"], r["t"], r["ref_bin"]) == key]
    pts.sort(key=lambda r: float(r["price"]))
    p = [float(r["price"]) for r in pts]
    chosen, true_p = float(w["chosen_price"]), float(w["true_price"])

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for a, (tk, ek, lab) in zip(ax, [("true_demand", "estimated_demand", "Expected demand"),
                                     ("true_dynamic_q", "estimated_dynamic_q", "Dynamic value")]):
        a.plot(p, [float(r[tk]) for r in pts], "o-", color="#1f77b4", label="true")
        a.plot(p, [float(r[ek]) for r in pts], "s-", color="#ff7f0e", label="fitted model")
        a.axvline(true_p, ls="--", color="#2ca02c", label="true optimum")
        a.axvline(chosen, ls=":", color="#d62728", label="planner's choice")
        a.set_xlabel("Price")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("The optimiser walks to where the model is most wrong "
                 "(state: t=%s, reference price %s; policy regret %.1f)"
                 % (w["t"], w["ref_price"], regret), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT, dpi=150)
    print("wrote %s  (seed=%s, t=%s, ref=%s, chosen=%s, true=%s, policy regret=%.1f)"
          % (OUT, w["seed"], w["t"], w["ref_price"], chosen, true_p, regret))
    return 0


if __name__ == "__main__":
    sys.exit(main())
