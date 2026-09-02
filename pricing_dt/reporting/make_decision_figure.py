"""Draw the deployment procedure as an ordered decision, with the study's own numbers on it.

Table 5.1 lists eight questions and what this study measured for each. What it cannot show is
the order they have to be asked in, or where a "no" ends the enquiry, and those are half the
result: coverage binds before freedom, freedom before the floor, and the floor before any
comparison between methods. The figure carries that ordering; the table carries the detail.
Each decision here names the table row it draws on, so the two are one device in two forms
rather than two overlapping ones.

Three choices are deliberate.

  * Every decision box carries a measured quantity. A procedure diagram without numbers is a
    recommendation, and Section 5.6 says explicitly that these steps are not that.
  * Step 4 is drawn as an action, not a decision. It has no branch: the floor and the ceiling
    are measured, and step 5 is where the answer changes what happens next.
  * Step 3's affirmative branch is labelled as rarely reachable. Off-support behaviour is not
    establishable from an offline log, which is the premise of the setting, so a diagram that
    offered "verified: widen freedom" as an ordinary option would contradict the finding.

Run from anywhere:  python pricing_dt/reporting/make_decision_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

PRIMARY = "#0F3557"
ACCENT = "#A85218"
PANEL = "#EDF2F7"
RULE = "#C3D0DB"
MUTE = "#52616E"
STOPFACE = "#FFFFFF"

W, XL = 0.44, 0.02          # decision column
XR, WR = 0.56, 0.42         # outcome column


def rounded(ax, x, y, w, h, edge=RULE, face=PANEL, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=lw, edgecolor=edge, facecolor=face,
                                linestyle=ls, zorder=2))


def step(ax, y, h, num, title, measure, row, decision=True):
    rounded(ax, XL, y, W, h, edge=PRIMARY if decision else RULE,
            lw=1.4 if decision else 1.0)
    ax.text(XL + 0.022, y + h * 0.68, num, ha="left", va="center", fontsize=9.4,
            weight="bold", color=MUTE, zorder=3)
    ax.text(XL + 0.055, y + h * 0.68, title, ha="left", va="center", fontsize=8.8,
            weight="bold", color=PRIMARY, zorder=3)
    ax.text(XL + 0.055, y + h * 0.30, measure, ha="left", va="center", fontsize=7.0,
            color=MUTE, zorder=3, linespacing=1.5)
    ax.text(XL + W - 0.014, y + h - 0.018, row, ha="right", va="top", fontsize=6.4,
            color=RULE, zorder=3, style="italic")


def outcome(ax, y, h, title, body, stop=True):
    rounded(ax, XR, y, WR, h, edge=ACCENT if stop else RULE, face=STOPFACE,
            lw=1.3 if stop else 1.0, ls="-" if stop else (0, (3, 2)))
    ax.text(XR + 0.018, y + h * 0.68, title, ha="left", va="center", fontsize=8.4,
            weight="bold", color=ACCENT if stop else PRIMARY, zorder=3)
    ax.text(XR + 0.018, y + h * 0.28, body, ha="left", va="center", fontsize=7.0,
            color=MUTE, zorder=3, linespacing=1.5)


def arrow(ax, p, q, color=MUTE, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(10.4, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.075, 1)
    ax.axis("off")

    H = 0.108                      # step height
    GAP = 0.038
    ys = [0.885 - i * (H + GAP) for i in range(6)]

    step(ax, ys[0], H, "1", "Are long-run effects material?",
         "the intertemporal optimum separates from the myopic one\nas the coupling strengthens (§4.1)",
         "§4.1")
    outcome(ax, ys[0], H, "Myopic methods may suffice",
            "the channel question does not arise; still map coverage")

    step(ax, ys[1], H, "2", "Does the log cover the candidate actions?",
         "at $m=11$ the chosen action's logged support is $0.030$;\nunder thinned coverage every family reaches about $-1.9$",
         "row 7")
    outcome(ax, ys[1], H, "Coverage is the binding limit",
            "no method recovers it; report the ceiling the log fixes")

    step(ax, ys[2], H, "3", "Is off-support behaviour verified?",
         "normally it is not; where it was not, the action channel\nreached $-4.54$ (E1) and $-3.25$ (E2)",
         "rows 1-2")
    outcome(ax, ys[2], H, "Confine the model's freedom",
            "goal channel $+0.67$ / $+0.17$, or a support mask:\nthe same planner moves $-4.54 \\rightarrow +0.47$")

    step(ax, ys[3], H, "4", "Measure what the constraint alone gives",
         "no-learner floor $+0.448$ (E1), $-1.811$ (E2);\nceiling inside the top-3 mask $0.9936$",
         "row 5", decision=False)

    step(ax, ys[4], H, "5", "Does the learner clear that floor?",
         "masked value minus floor, on each environment's own scale",
         "rows 5, 8")
    outcome(ax, ys[4], H, "Uniform choice inside the set already reaches it",
            "report it as the constraint's contribution, not the method's")

    step(ax, ys[5], H, "6", "Compare methods, support held fixed",
         "$0.165$ to $0.960$ across families under one mask;\nno universal ranking is claimed",
         "row 6")

    # the spine, and the branches that leave it
    for i in range(5):
        arrow(ax, (XL + W / 2, ys[i]), (XL + W / 2, ys[i + 1] + H), color=PRIMARY)
        ax.text(XL + W / 2 + 0.012, ys[i] - GAP / 2,
                ["yes", "yes, or partly", "no", "", "yes"][i],
                ha="left", va="center", fontsize=6.8, color=PRIMARY)
    for i in (0, 1, 2, 4):
        arrow(ax, (XL + W, ys[i] + H / 2), (XR, ys[i] + H / 2), color=ACCENT)
        ax.text((XL + W + XR) / 2, ys[i] + H / 2 + 0.014,
                ["no", "no", "yes (rare)", "no"][[0, 1, 2, 4].index(i)],
                ha="center", va="bottom", fontsize=6.8, color=ACCENT)

    # the real-log check hangs off the end rather than floating beside it
    yc = ys[5] - 0.152
    rounded(ax, XL, yc, XR + WR - XL, 0.086, edge=RULE, face=PANEL, ls=(0, (3, 2)))
    ax.text(XL + 0.022, yc + 0.058, "If a real log is used to rank", ha="left", va="center",
            fontsize=8.4, weight="bold", color=PRIMARY, zorder=3)
    ax.text(XL + 0.022, yc + 0.024,
            "seven policies could not be ranked on this study's own log:\n"
            "check overlap and effective sample size before believing one (§5.5)",
            ha="left", va="center", fontsize=7.0, color=MUTE, zorder=3, linespacing=1.5)
    arrow(ax, (XL + W / 2, ys[5]), (XL + W / 2, yc + 0.086), color=MUTE, ls=(0, (3, 2)))

    ax.text(0.02, -0.028,
            "Row references are to Table 5.1, which carries the full measurement for each "
            "step. Values are normalised value (Equation 3.3) unless stated.",
            ha="left", va="center", fontsize=7.0, color=MUTE)
    ax.text(0.02, -0.052,
            "Solid boxes on the left are decisions; box 4 is a measurement and has no branch. "
            "Boxes on the right have no outgoing arrow: they end the procedure.",
            ha="left", va="center", fontsize=7.0, color=MUTE)

    # Two levels up: this script sits in pricing_dt/reporting/, not beside the results tree.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "results", "figures", "deployment_procedure.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote " + os.path.normpath(out))


if __name__ == "__main__":
    main()
