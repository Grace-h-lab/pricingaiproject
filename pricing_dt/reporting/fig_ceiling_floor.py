"""What the log fixes, drawn against how good the logging policy was.

Three quantities share the informative band between zero and one, and a fourth, the floor
without the mask, sits near -1.95 throughout. Drawing all four on one scale would push the
first three into the top fifth of the axes, so the axis is broken: the reader still sees the
gap the mask closes, at its true size, without losing the comparison that carries the finding.

Data: results_expertq_sweep_20260825/expertq_sweep.csv (15 configurations per quality level),
the same file verify_claims.py recomputes the quoted ceiling and floor from.
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "results_expertq_sweep_20260825/expertq_sweep.csv"
OUT = "results/figures/ceiling_floor_by_logger.png"


def main():
    if not os.path.exists(SRC):
        print("missing %s" % SRC)
        return 1
    g = defaultdict(list)
    for x in csv.DictReader(open(SRC, encoding="utf-8")):
        g[float(x["expert_q"])].append(x)
    q = sorted(g)
    mean = lambda c: [st.mean(float(y[c]) for y in g[k]) for k in q]

    fig, (hi, lo) = plt.subplots(2, 1, sharex=True, figsize=(7.2, 4.6),
                                 gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    for c, lab, sty in [("nv_ceiling", "ceiling inside the mask", dict(marker="o", color="#1f77b4")),
                        ("nv_floor_masked", "no-learner floor, masked", dict(marker="s", color="#ff7f0e")),
                        ("nv_logger", "the logging policy itself", dict(marker="^", color="#7f7f7f", ls="--"))]:
        hi.plot(q, mean(c), label=lab, **sty)
    lo.plot(q, mean("nv_floor_bare"), marker="v", color="#d62728",
            label="no-learner floor, no mask")

    hi.set_ylim(-0.12, 1.05)
    lo.set_ylim(-2.05, -1.88)
    hi.spines["bottom"].set_visible(False)
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False)
    for a in (hi, lo):
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper left")
    lo.set_xlabel("Logging-policy quality")
    hi.set_ylabel("Normalised value")
    hi.set_title("The log fixes what is attainable: the ceiling moves from 0.00 to 0.99,\n"
                 "and the floor tracks the logging policy rather than the method", fontsize=10)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("wrote %s" % OUT)
    for k, c, f, b, l in zip(q, mean("nv_ceiling"), mean("nv_floor_masked"),
                             mean("nv_floor_bare"), mean("nv_logger")):
        print("  q=%.2f  ceiling %+.3f  floor %+.3f  bare %+.3f  logger %+.3f" % (k, c, f, b, l))
    return 0


if __name__ == "__main__":
    sys.exit(main())
