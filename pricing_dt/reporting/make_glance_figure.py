"""Draw Appendix B as one page: the study's question, design, findings and boundaries.

The abstract, Chapter 1 and Chapter 6 each give this account in prose. A figure adds what
prose cannot: the channel contrast, the sweep that turns it into an intervention, and the
band the support constraint fixes are three different shapes, and only as shapes can they
be compared at a glance.

Five commitments are deliberate.

  * Nothing is encoded by colour alone. Each sweep carries a line style and a marker as well
    as a hue, environments are named in text, and every panel is labelled, so the page
    survives greyscale printing.
  * The two normalisation anchors are stated on the page. nv = 0 is the oracle myopic policy
    in E1 and the logging policy in E2 (Section 3.5.1), so the two value columns are not a
    common scale and the figure says so rather than letting adjacency imply it.
  * The two ceilings are kept apart. 0.9936 is the top-3 ceiling under the main protocol's
    logged start distribution; the competence sweep runs to 0.986 under a uniform one.
  * The no-learner floor is drawn as a reference level, never as a lower bound: three arms in
    Table 4.11 fall below it, and the panel says so.
  * The trust-region sweep is plotted at its integer widths with visible markers, because it
    is a graded control over eleven actions and not a continuous one.

Every number is cross-referenced in Chapters 4 and 5. The two sweeps and the competence sweep
are read from the run files rather than typed in, so the curves cannot drift from the tables.

Text is placed by hand on a figure-fraction grid, which cannot reflow, so every string is
declared with the width it has to fit in and `fit()` reports any line that would overrun its
box. A silent overlap is the one defect a reader would notice before anything else.

Run from anywhere:  python pricing_dt/reporting/make_glance_figure.py
"""
import collections
import csv
import io
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------- palette
# Each hue means one thing on this page and nothing else. Green is reserved for the
# accurate-model control, so the data channel is drawn neutral: its value degrades with
# rollout horizon (+0.383 at h = 1 to -3.826 at h = 8, Section 4.3) and a green card would
# read as an endorsement the results do not support.
PRIMARY = "#0F3557"     # structure, section bars
E1C = "#1F6FB2"         # pricing
E2C = "#6B4E9B"         # inventory
EXACT = "#2E7D4F"       # exact-Q* control, valid mechanism
FAIL = "#B4342A"        # fitted-model failure
SUPPORT = "#B5741C"     # support, floor, ceiling
NEUTRAL = "#5A6B78"     # data channel, neutral notes
PANEL = "#F3F6F9"
RULE = "#C3D0DB"
MUTE = "#46545F"
CORRECT = "#FAEDEB"
SCALEBG = "#FDF5E9"
GOODBG = "#F1F6F2"

PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
CHAR_EM, LINE_SP = 0.52, 1.45          # DejaVu Sans mean advance for mixed-case prose

HERE = os.path.dirname(os.path.abspath(__file__))
# Two levels up: this script sits in pricing_dt/reporting/, not beside the results tree.
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "results", "figures", "study_at_a_glance.png")

_OVERRUN = []


# ---------------------------------------------------------------- data
def _rows(path):
    with io.open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def e1_sweep():
    """E1 trust-region sweep: one row per width, already averaged over seeds."""
    out = {}
    for row in _rows("results/trust_region_summary.csv"):
        out.setdefault(row["model"], {})[int(row["m"])] = float(row["nv"])
    return out["oracle"], out["structured"]


def e2_sweep():
    """E2 trust-region sweep: one row per seed and width."""
    agg = collections.defaultdict(list)
    for row in _rows("results/env2_trust.csv"):
        agg[(row["model"], int(row["m"]))].append(float(row["nv"]))
    out = {}
    for (model, m), vals in agg.items():
        out.setdefault(model, {})[m] = statistics.mean(vals)
    return out["oracle"], out["structured"]


def competence_sweep():
    """Ceiling inside the mask as the logging specialists' in-region quality is swept."""
    agg = collections.defaultdict(list)
    for row in _rows("results_expertq_sweep_20260825/expertq_sweep.csv"):
        agg[float(row["expert_q"])].append(float(row["nv_ceiling"]))
    return sorted((q, statistics.mean(v)) for q, v in agg.items())


# ---------------------------------------------------------------- drawing
def fit(s, size, maxw, where):
    """Record any line that will not fit the width it was placed in."""
    per = (size * CHAR_EM / 72.0) / PAGE_W_IN
    for line in s.split("\n"):
        plain = line.replace("$", "").replace("\\", "")
        if len(plain) * per > maxw:
            _OVERRUN.append("%-26s wide %5.3f > %5.3f  %r"
                            % (where, len(plain) * per, maxw, line[:46]))
    return s


def drop(s, size, y, floor, where):
    """Record a text block whose last line would fall below the box holding it."""
    lines = s.count("\n") + 1
    bottom = y - lines * (size * LINE_SP / 72.0) / PAGE_H_IN
    if bottom < floor:
        _OVERRUN.append("%-26s deep %5.3f < %5.3f  %r"
                        % (where, bottom, floor, s.split("\n")[0][:46]))


def box(fig, x, y, w, h, face=PANEL, edge=RULE, lw=0.8):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.001,rounding_size=0.005",
        transform=fig.transFigure, facecolor=face, edgecolor=edge,
        linewidth=lw, zorder=1))


def bar(fig, x, y, w, h, text, size=8.5):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0", transform=fig.transFigure,
        facecolor=PRIMARY, edgecolor="none", zorder=2))
    fig.text(x + 0.009, y + h / 2.0, text, ha="left", va="center",
             fontsize=size, color="white", weight="bold", zorder=3)


def txt(fig, x, y, s, size=6.0, color=MUTE, weight="normal", ha="left",
        style="normal", maxw=None, floor=None, where=""):
    if maxw is not None:
        fit(s, size, maxw, where)
    if floor is not None:
        drop(s, size, y, floor, where)
    fig.text(x, y, s, ha=ha, va="top", fontsize=size, color=color,
             weight=weight, style=style, zorder=4, linespacing=1.45)


def rule(fig, x0, x1, y, color=RULE, lw=0.7):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              color=color, linewidth=lw, zorder=3))


def inset(fig, x, y, w, h):
    ax = fig.add_axes([x, y, w, h], zorder=5)
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(labelsize=4.6, colors=MUTE, length=2, width=0.6, pad=1.2)
    return ax


def numdot(fig, x, y, n, col):
    fig.patches.append(plt.Circle((x, y), 0.0082, transform=fig.transFigure,
                                  facecolor=col, edgecolor="none", zorder=6))
    fig.text(x, y, n, ha="center", va="center", fontsize=6.8,
             color="white", weight="bold", zorder=7)


# ---------------------------------------------------------------- page
def draw():
    fig = plt.figure(figsize=(PAGE_W_IN, 11.69))
    fig.patch.set_facecolor("white")

    # ---- header -------------------------------------------------------
    # The page sits under the appendix heading in the document, so it does not repeat it.
    txt(fig, 0.030, 0.9910, "What a Pricing Log Can Support",
        size=18.0, color=PRIMARY, weight="bold")
    rule(fig, 0.030, 0.970, 0.9585, color=PRIMARY, lw=1.3)

    bar(fig, 0.030, 0.9230, 0.940, 0.0300,
        "One fitted model, three places to put it.  "
        "How does its position change the cost of model error?", size=10.0)

    # ---- Section A ----------------------------------------------------
    bar(fig, 0.030, 0.8940, 0.940, 0.0200, "A   STUDY DESIGN")

    txt(fig, 0.030, 0.8935, "One fitted demand model, held fixed, routed three ways",
        size=6.6, color=PRIMARY, weight="bold", maxw=0.462, where="A1 head")
    for i, (name, col, body, gloss, tag) in enumerate([
            ("ACTION CHANNEL", FAIL,
             "Scores every action;\nthe policy plays the\nargmax.",
             "The model selects\nthe action.", "fit, then optimise"),
            ("DATA CHANNEL", NEUTRAL,
             "Generates synthetic\ntransitions appended\nto training.",
             "The model changes the\ntraining distribution.", "augmentation"),
            ("GOAL CHANNEL", E2C,
             "Sets only the\nconditioning target\n(return-to-go).",
             "The model never does\nthe deployment argmax.", "return-conditioned")]):
        x = 0.030 + i * 0.1580
        box(fig, x, 0.8010, 0.1460, 0.0800, face="white", edge=col, lw=1.1)
        txt(fig, x + 0.007, 0.8770, name, size=5.8, color=col, weight="bold",
            maxw=0.132, where="chan %d name" % i)
        txt(fig, x + 0.007, 0.8650, body, size=5.8, color=MUTE,
            maxw=0.132, where="chan %d body" % i)
        txt(fig, x + 0.007, 0.8290, gloss, size=5.0, color=MUTE, style="italic",
            maxw=0.132, where="chan %d gloss" % i)
        txt(fig, x + 0.007, 0.8060, tag, size=5.0, color=col, weight="bold",
            maxw=0.132, where="chan %d tag" % i)

    box(fig, 0.030, 0.7570, 0.4620, 0.0400, face=PANEL, edge=RULE)
    txt(fig, 0.038, 0.7880,
        "The fitted model, the logged data and the auxiliary knowledge are\n"
        "held fixed. Only the route into the policy changes.",
        size=5.6, color=PRIMARY, weight="bold", maxw=0.446, floor=0.7570, where="A1 note")

    txt(fig, 0.5180, 0.8935,
        "Two exactly-solvable environments, priors wrong in different ways",
        size=6.6, color=PRIMARY, weight="bold", maxw=0.452, where="A2 head")
    for i, (name, col, pairs) in enumerate([
            ("E1   REFERENCE-PRICE PRICING", E1C,
             [("Decision", "11 prices, horizon 8"),
              ("Prior", "monotone, bounded elasticity"),
              ("Fails by", "cannot represent a steep enough\noff-support decline"),
              ("Calibrated", "1.07 M real retail transactions")]),
            ("E2   LOST-SALES INVENTORY", E2C,
             [("Decision", "11 order levels, horizon 8"),
              ("Prior", "Poisson demand"),
              ("Fails by", "variance is identically the mean,\nso no overdispersion"),
              ("Calibrated", "not to data; a different prior")])]):
        x = 0.5180 + i * 0.2310
        box(fig, x, 0.8010, 0.2210, 0.0800, face="white", edge=col, lw=1.1)
        txt(fig, x + 0.007, 0.8770, name, size=5.8, color=col, weight="bold",
            maxw=0.207, where="env %d name" % i)
        yy = 0.8645
        for key, val in pairs:
            txt(fig, x + 0.007, yy, key, size=5.0, color=col, weight="bold",
                maxw=0.048, where="env %d key" % i)
            txt(fig, x + 0.059, yy, val, size=5.0, color=MUTE,
                maxw=0.160, where="env %d val" % i)
            yy -= 0.0185 if "\n" in val else 0.0110

    box(fig, 0.5180, 0.7570, 0.2210, 0.0400, face=SCALEBG, edge=SUPPORT)
    txt(fig, 0.5250, 0.7930, "SCALE", size=5.0, color=SUPPORT, weight="bold")
    txt(fig, 0.5250, 0.7845,
        "nv = 1 is the exact sequential optimum in both.\n"
        "nv = 0 is the oracle-myopic policy in E1, the\n"
        "logging policy in E2: these are not one scale.",
        size=4.8, color=MUTE, maxw=0.207, floor=0.7570, where="scale")

    box(fig, 0.7490, 0.7570, 0.2210, 0.0400, face=PANEL, edge=RULE)
    txt(fig, 0.7560, 0.7930, "PROTOCOL", size=5.0, color=PRIMARY, weight="bold")
    txt(fig, 0.7560, 0.7845,
        "10 learned arms plus a no-learner control, one\n"
        "support constraint, 10 seeds x 3x3 cells = 90\n"
        "evals per arm. Wilcoxon + Holm; unit = seed.",
        size=4.8, color=MUTE, maxw=0.207, floor=0.7570, where="protocol")

    # ---- Section B ----------------------------------------------------
    bar(fig, 0.030, 0.7300, 0.940, 0.0200, "B   FIVE MAIN FINDINGS")

    # Finding 1 --------------------------------------------------------
    box(fig, 0.030, 0.6320, 0.940, 0.0950, face="white", edge=RULE)
    numdot(fig, 0.0470, 0.7175, "1", FAIL)
    txt(fig, 0.0600, 0.7228, "Model placement produces large value differences.",
        size=8.6, color=PRIMARY, weight="bold", maxw=0.900, where="F1 title")
    txt(fig, 0.9560, 0.7222, "RQ1", size=5.4, color=FAIL, weight="bold", ha="right")
    txt(fig, 0.0420, 0.6970,
        "The same fitted model, the same logs,\n"
        "the same auxiliary knowledge. Only the\n"
        "route into the policy changes.",
        size=6.0, color=MUTE, maxw=0.250, floor=0.6320, where="F1 body")
    txt(fig, 0.0420, 0.6610,
        "Same direction and order of magnitude in both.\n"
        "The magnitudes are not comparable: each sits on\n"
        "its own environment's scale.",
        size=5.0, color=MUTE, style="italic", maxw=0.250, floor=0.6320, where="F1 note")

    for cx, cname, ccol in ((0.5900, "E1  Pricing", E1C),
                            (0.7350, "E2  Inventory", E2C)):
        txt(fig, cx, 0.7080, cname, size=6.0, color=ccol, weight="bold", ha="right")
    rule(fig, 0.3050, 0.7350, 0.7000)
    for j, (label, v1, v2, strong) in enumerate([
            ("Action channel, plan against the model", "-4.543", "-3.246", True),
            ("What the planner believed it earned", "1800.3", "94.8", False),
            ("against a true optimum of", "480.0", "80.9", False),
            ("Goal channel, relabel the target only", "+0.670", "+0.169", True)]):
        yy = 0.6935 - j * 0.0150
        txt(fig, 0.3050 + (0.012 if not strong and j == 2 else 0.0), yy, label,
            size=6.0, color=PRIMARY if strong else MUTE,
            weight="bold" if strong else "normal", maxw=0.225, where="F1 row %d" % j)
        for cx, v in ((0.5900, v1), (0.7350, v2)):
            txt(fig, cx, yy + (0.0008 if strong else 0.0), v,
                size=7.4 if strong else 6.0,
                color=(FAIL if v.startswith("-") else EXACT) if strong else MUTE,
                weight="bold" if strong else "normal", ha="right")

    box(fig, 0.7750, 0.6420, 0.1850, 0.0620, face=PANEL, edge=RULE)
    txt(fig, 0.7820, 0.6980, "SWING IN nv", size=5.0, color=MUTE, weight="bold")
    for k, (val, env, col) in enumerate((("5.2", "units, E1", E1C),
                                         ("3.4", "units, E2", E2C))):
        yy = 0.6870 - k * 0.0175
        txt(fig, 0.7820, yy, val, size=13.0, color=PRIMARY, weight="bold")
        txt(fig, 0.8300, yy - 0.0025, env, size=5.0, color=col, weight="bold")
    txt(fig, 0.7820, 0.6510, "produced entirely by the channel",
        size=4.8, color=MUTE, style="italic", maxw=0.171, floor=0.6420, where="F1 swing")

    # Finding 2 --------------------------------------------------------
    box(fig, 0.030, 0.4810, 0.940, 0.1440, face="white", edge=RULE)
    numdot(fig, 0.0470, 0.6155, "2", PRIMARY)
    txt(fig, 0.0600, 0.6208, "Greater off-support freedom exposes model error.",
        size=8.6, color=PRIMARY, weight="bold", maxw=0.900, where="F2 title")
    txt(fig, 0.9560, 0.6202, "RQ2", size=5.4, color=PRIMARY, weight="bold", ha="right")
    txt(fig, 0.0420, 0.6040,
        "The trust region of Equation (3.1) is\n"
        "widened in graded integer steps. Under\n"
        "the fitted model true value deteriorates;\n"
        "under the exact-$Q^*$ control it improves\n"
        "monotonically to the optimum.\n\n"
        "The harm is therefore exploitation of\n"
        "model error, not departure from imitation.",
        size=6.0, color=MUTE, maxw=0.222, floor=0.4810, where="F2 body")
    txt(fig, 0.0420, 0.5090,
        "E2 endpoint  -3.646, sd 1.664 over ten seeds,\n"
        "95% CI [-4.836, -2.456]. E2 learned-policy\n"
        "values are Monte-Carlo (Section 3.5.1).",
        size=5.0, color=MUTE, style="italic", maxw=0.222, floor=0.4810, where="F2 note")

    e1o, e1s = e1_sweep()
    e2o, e2s = e2_sweep()
    for k, (ax_x, title, tcol, orc, fit_) in enumerate([
            (0.2950, "E1  Pricing", E1C, e1o, e1s),
            (0.6450, "E2  Inventory", E2C, e2o, e2s)]):
        ax = inset(fig, ax_x, 0.5100, 0.2900, 0.0800)
        ms = sorted(fit_)
        ax.plot(ms, [orc[m] for m in ms], color=EXACT, lw=1.4, ls="-",
                marker="^", ms=3.0, label="exact $Q^*$ control")
        ax.plot(ms, [fit_[m] for m in ms], color=FAIL, lw=1.4, ls="-.",
                marker="x", ms=3.4, mew=1.1, label="fitted model")
        ax.axhline(0.0, color=RULE, lw=0.6, ls=":")
        ax.set_xticks(ms)
        ax.set_xlim(min(ms) - 0.4, max(ms) + 1.4)
        ax.set_xlabel("trust-region width $m$", fontsize=4.8, color=MUTE, labelpad=0.8)
        ax.set_title(title, fontsize=6.0, color=tcol, weight="bold", pad=2.5)
        if k == 0:
            ax.set_ylabel("true nv", fontsize=4.8, color=MUTE, labelpad=0.8)
        ax.legend(fontsize=4.6, frameon=False, loc="lower left",
                  handlelength=2.2, borderpad=0.1, labelspacing=0.2)
        last = ms[-1]
        ax.annotate("%+.3f" % orc[last], xy=(last, orc[last]), xytext=(3, -1),
                    textcoords="offset points", fontsize=5.0, color=EXACT,
                    weight="bold", ha="left", va="center")
        ax.annotate("%+.3f" % fit_[last], xy=(last, fit_[last]), xytext=(3, -1),
                    textcoords="offset points", fontsize=5.0, color=FAIL,
                    weight="bold", ha="left", va="center")

    # Finding 3 --------------------------------------------------------
    box(fig, 0.030, 0.3240, 0.4620, 0.1500, face="white", edge=RULE)
    numdot(fig, 0.0470, 0.4635, "3", EXACT)
    txt(fig, 0.0600, 0.4690, "Target accuracy alone does not explain",
        size=8.2, color=PRIMARY, weight="bold", maxw=0.410, where="F3 title a")
    txt(fig, 0.0600, 0.4560, "return-conditioned performance.",
        size=8.2, color=PRIMARY, weight="bold", maxw=0.410, where="F3 title b")
    txt(fig, 0.4780, 0.4694, "RQ3", size=5.4, color=EXACT, weight="bold", ha="right")
    txt(fig, 0.0420, 0.4320,
        "An exact value function can be a worse conditioning target: it folds\n"
        "in the optimal continuation, so it scores poor logged actions highly\n"
        "and flattens the differences a return-conditioned policy consumes.",
        size=6.0, color=MUTE, maxw=0.440, floor=0.3240, where="F3 body")

    txt(fig, 0.0420, 0.3950, "E1  PRE-REGISTERED REPLICATION, 60 FRESH SEEDS",
        size=5.2, color=PRIMARY, weight="bold", maxw=0.440, where="F3 prereg head")
    for i, (lab, val, p) in enumerate([
            ("structured - exact $Q^*$", "+0.0738", "Holm $p$ = .00073"),
            ("structured - exact potential", "+0.0947", "Holm $p$ = .00040")]):
        x = 0.0420 + i * 0.2160
        box(fig, x, 0.3560, 0.1980, 0.0300, face=PANEL, edge=RULE)
        txt(fig, x + 0.007, 0.3840, lab, size=5.0, color=MUTE,
            maxw=0.184, where="F3 card %d" % i)
        txt(fig, x + 0.007, 0.3745, val, size=10.5, color=EXACT, weight="bold")
        txt(fig, x + 0.100, 0.3760, p, size=5.0, color=MUTE)

    box(fig, 0.0420, 0.3255, 0.4140, 0.0275, face="#F5F2F9", edge=E2C, lw=0.9)
    txt(fig, 0.0490, 0.3495,
        "E2 bounds it   structured - exact $Q^*$ = -0.051, unadjusted $p$ = 0.0059; there the\n"
        "exact function does discriminate. The performance ordering was pre-registered; the\n"
        "within-state account stays exploratory, since realised noise also creates spread.",
        size=4.4, color=MUTE, maxw=0.400, floor=0.3260, where="F3 e2")

    # Finding 4 --------------------------------------------------------
    box(fig, 0.5080, 0.3240, 0.4620, 0.1500, face="white", edge=RULE)
    numdot(fig, 0.5250, 0.4635, "4", SUPPORT)
    txt(fig, 0.5380, 0.4690, "Support restriction changes the feasible",
        size=8.2, color=PRIMARY, weight="bold", maxw=0.410, where="F4 title a")
    txt(fig, 0.9560, 0.4694, "boundary result", size=5.4, color=SUPPORT,
        weight="bold", ha="right")
    txt(fig, 0.5380, 0.4560, "set and the value attainable inside it.",
        size=8.2, color=PRIMARY, weight="bold", maxw=0.410, where="F4 title b")
    txt(fig, 0.5200, 0.4430,
        "Uniform choice inside the same top-3 mask already recovers 45%\n"
        "of the gap, yet learned arms can still fall below it.",
        size=6.0, color=MUTE, maxw=0.440, floor=0.3240, where="F4 body")

    for i, (head, big, sub, note) in enumerate([
            ("NO-LEARNER FLOOR", "+0.448", "E1      E2  -1.811",
             "a reference level, not a lower\nbound: three arms fall below it"),
            ("CEILING IN THE MASK", "0.9936", "E1, top-3 logged set",
             "a true upper bound: no policy\nin that set can exceed it")]):
        x = 0.5200 + i * 0.1520
        box(fig, x, 0.3690, 0.1420, 0.0510, face=SCALEBG, edge=SUPPORT, lw=0.9)
        txt(fig, x + 0.006, 0.4160, head, size=4.8, color=SUPPORT, weight="bold",
            maxw=0.130, where="F4 card %d head" % i)
        txt(fig, x + 0.006, 0.4060, big, size=11.5, color=PRIMARY, weight="bold")
        txt(fig, x + 0.006, 0.3915, sub, size=4.8, color=MUTE, weight="bold",
            maxw=0.130, where="F4 card %d sub" % i)
        txt(fig, x + 0.006, 0.3850, note, size=4.6, color=MUTE, style="italic",
            maxw=0.130, where="F4 card %d note" % i)

    ax = inset(fig, 0.8420, 0.3830, 0.1130, 0.0330)
    sweep = competence_sweep()
    ax.plot([q for q, _ in sweep], [v for _, v in sweep], color=SUPPORT,
            lw=1.3, ls="-", marker="s", ms=2.6)
    ax.set_xlabel("logging-policy competence", fontsize=4.4, color=MUTE, labelpad=0.8)
    ax.set_title("ceiling inside the mask", fontsize=4.8, color=SUPPORT,
                 weight="bold", pad=2.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.18, 1.22)
    ax.annotate("0.986", xy=(1.0, sweep[-1][1]), xytext=(-2, -7),
                textcoords="offset points", fontsize=4.8, color=SUPPORT,
                weight="bold", ha="right")
    ax.annotate("0.000", xy=(0.0, 0.0), xytext=(1, 4), textcoords="offset points",
                fontsize=4.8, color=SUPPORT, weight="bold")

    txt(fig, 0.5200, 0.3620,
        "Sweeping the logging specialists' in-region quality moves the ceiling from 0.000\n"
        "to 0.986, a different quantity from the 0.9936 beside it: uniform start\n"
        "distribution there, logged here. The dip at 0.50 is a rounding artefact.",
        size=4.8, color=MUTE, style="italic", maxw=0.440, floor=0.3240, where="F4 note")
    txt(fig, 0.5200, 0.3370,
        "The log fixes which action opportunities exist; the learner "
        "decides how well they are used.",
        size=5.4, color=PRIMARY, weight="bold", maxw=0.440, floor=0.3240, where="F4 kicker")

    # Finding 5 --------------------------------------------------------
    box(fig, 0.030, 0.2290, 0.940, 0.0870, face=CORRECT, edge=FAIL, lw=0.9)
    numdot(fig, 0.0470, 0.3050, "5", FAIL)
    txt(fig, 0.0600, 0.3105,
        "Protocol correction: the original method-level advantage does not survive.",
        size=8.6, color=PRIMARY, weight="bold", maxw=0.900, where="F5 title")
    txt(fig, 0.9560, 0.3099, "RQ4, reported as a null",
        size=5.4, color=FAIL, weight="bold", ha="right")
    txt(fig, 0.0420, 0.2860,
        "This project set out to validate structured\n"
        "demand-model relabelling. Under a fair\n"
        "comparator and one symmetric constraint,\n"
        "its advantage becomes a loss.",
        size=6.0, color=MUTE, maxw=0.270, floor=0.2290, where="F5 body")

    txt(fig, 0.3250, 0.2860, "Structured relabel + mask\nQ-DT $q_{sa}$ + mask",
        size=6.0, color=MUTE, maxw=0.145, floor=0.2290, where="F5 arms")
    txt(fig, 0.5150, 0.2860, "0.740\n0.859", size=6.0, color=MUTE, ha="right")
    txt(fig, 0.3250, 0.2600, "-0.119", size=15.0, color=FAIL, weight="bold")
    txt(fig, 0.4250, 0.2585, "E1, the same top-3 mask,\n90 evaluations per arm",
        size=5.0, color=MUTE, style="italic", maxw=0.130, floor=0.2290, where="F5 gloss")

    txt(fig, 0.5450, 0.2930,
        "THREE PROTOCOL ASYMMETRIES PRODUCED THE EARLIER POSITIVE RESULT",
        size=5.2, color=PRIMARY, weight="bold", maxw=0.415, where="F5 head")
    txt(fig, 0.5450, 0.2820,
        "1   a comparator whose target carried no within-state information\n"
        "2   an evaluation rule that asked different arms for different things\n"
        "3   a constraint applied to the proposal but not to its comparators",
        size=6.0, color=MUTE, maxw=0.415, floor=0.2290, where="F5 list")
    txt(fig, 0.5450, 0.2420, "Corrected analysis in Section 4.6.1 and Appendix E.",
        size=5.0, color=MUTE, style="italic", maxw=0.415, where="F5 ref")

    # ---- Section C ----------------------------------------------------
    bar(fig, 0.030, 0.2000, 0.940, 0.0200, "C   EVIDENCE BOUNDARIES")

    for i, (head, body) in enumerate([
            ("1   Two simulated environments",
             "The mechanism evidence comes from two controlled\n"
             "environments with known ground truth.\n"
             "Exactness was bought with realism."),
            ("2   The contraction account did not replicate",
             "Mask-gain rank correlation +1.000 (E1) and +0.714 (E2)\n"
             "is 0.00 or negative on a public discrete-action\n"
             "benchmark. The floor itself held; the account\n"
             "explaining it did not, so it is restricted to E1 and E2."),
            ("3   Anchors exact, E2 values Monte-Carlo",
             "Transition kernels, optimal policies and both anchors\n"
             "are exact by dynamic programming. E2 transitions are\n"
             "stochastic, so learned-policy values are estimated by\n"
             "Monte-Carlo with common random seeds.")]):
        x = 0.030 + i * 0.3160
        box(fig, x, 0.1330, 0.3020, 0.0620, face="white", edge=RULE)
        txt(fig, x + 0.008, 0.1900, head, size=6.0, color=PRIMARY, weight="bold",
            maxw=0.286, where="bound %d head" % i)
        txt(fig, x + 0.008, 0.1775, body, size=5.0, color=MUTE,
            maxw=0.286, where="bound %d body" % i)

    # ---- footer -------------------------------------------------------
    box(fig, 0.030, 0.0250, 0.5800, 0.0950, face=GOODBG, edge=EXACT, lw=0.9)
    txt(fig, 0.0380, 0.1130, "WHY IT MATTERS OUTSIDE THE STUDY",
        size=5.4, color=EXACT, weight="bold")
    for i, (big, small) in enumerate([
            ("-1.449", "direct fitted-model optimisation in E1, in its one-step\n"
                       "form an offline direct-method contextual bandit"),
            ("0 of 90", "runs beat a perfect-information myopic policy"),
            ("96.6%", "of the time the true myopic-optimal price is already\n"
                      "sitting in the log")]):
        yy = 0.1010 - i * 0.0215
        txt(fig, 0.0380, yy, big, size=8.4, color=FAIL, weight="bold")
        txt(fig, 0.1150, yy - 0.0010, small, size=5.2, color=MUTE,
            maxw=0.460, where="foot %d" % i)
    txt(fig, 0.0380, 0.0400,
        "Not missing data: regression extrapolation followed by an argmax.",
        size=6.2, color=PRIMARY, weight="bold", maxw=0.560, where="foot kicker")

    box(fig, 0.6300, 0.0250, 0.3400, 0.0950, face=GOODBG, edge=EXACT, lw=0.9)
    txt(fig, 0.6380, 0.1130, "TWO INTERVENTIONS CONTAINED IT HERE",
        size=5.4, color=EXACT, weight="bold")
    txt(fig, 0.6380, 0.1010,
        "1   Move the model to the conditioning target.\n"
        "2   Leave it in place and restrict the optimiser\n"
        "      to the logged action set.",
        size=5.6, color=MUTE, maxw=0.324, floor=0.0250, where="foot routes")
    txt(fig, 0.6380, 0.0620, "-4.54   to   +0.47",
        size=12.5, color=EXACT, weight="bold")
    txt(fig, 0.6380, 0.0435,
        "the same planner, with no retraining (Table F.2).\n"
        "Neither route is claimed as a deployment rule.",
        size=5.0, color=MUTE, style="italic", maxw=0.324, floor=0.0250, where="foot note")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)

    if _OVERRUN:
        print("TEXT OVERRUNS (%d):" % len(_OVERRUN))
        for line in _OVERRUN:
            print("   " + line)
    else:
        print("no text overruns")
    print("saved %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    draw()
