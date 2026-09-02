"""Draw the Chapter 3 pipeline: what is held fixed and what varies.

The three relabelling channels are compared against one another, and the fitted model is the
*same object* in all three. That single fact is what makes the comparison controlled, and a
diagram carries it better than a paragraph.

Run from anywhere:  python pricing_dt/reporting/make_pipeline_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PRIMARY = "#0F3557"
ACCENT = "#A85218"
PANEL = "#EDF2F7"
RULE = "#C3D0DB"
MUTE = "#52616E"


def box(ax, x, y, w, h, title, body="", face=PANEL, edge=RULE, lw=1.0):
    """Title and body are placed as fractions of the box height, so a box can be resized
    without the text drifting outside it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))
    ty = y + h * (0.70 if body else 0.50)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=8.6,
            weight="bold", color=PRIMARY, zorder=3)
    if body:
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=7.1,
                color=MUTE, zorder=3, linespacing=1.5)


def arrow(ax, p, q, color=MUTE, lw=1.1, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, linewidth=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- row 1: where the data comes from -----------------------------------------
    y1, h1 = 0.845, 0.135
    box(ax, 0.015, y1, 0.29, h1, "Exactly-solvable environment",
        "E1 pricing  ·  E2 inventory\nevery policy's value exact")
    box(ax, 0.355, y1, 0.29, h1, "Logging policy  $\\pi_\\beta$",
        "competence swept from\nmyopic to optimal")
    box(ax, 0.695, y1, 0.29, h1, "Logged trajectories  $\\mathcal{D}$",
        "$(s_t, a_t, r_t, s_{t+1})$ only —\nno counterfactual outcome")
    arrow(ax, (0.305, y1 + h1 / 2), (0.355, y1 + h1 / 2))
    arrow(ax, (0.645, y1 + h1 / 2), (0.695, y1 + h1 / 2))

    # ---- row 2: the object held fixed ---------------------------------------------
    y2, h2 = 0.655, 0.115
    box(ax, 0.325, y2, 0.35, h2, "One fitted domain model  $\\hat{M}$",
        "the same object in all three channels", face="#FFFFFF", edge=PRIMARY, lw=1.7)
    arrow(ax, (0.84, y1), (0.675, y2 + h2 * 0.8), color=PRIMARY)
    ax.text(0.985, y2 + h2 * 0.30,
            "identical fit, data and auxiliary\nknowledge in every channel",
            ha="right", va="center", fontsize=7.1, color=PRIMARY, style="italic",
            linespacing=1.5)

    # ---- row 3: the three channels, in order of increasing m ----------------------
    # The trust-region axis is drawn behind the boxes, which cover it: what shows
    # between them reads as the three channels sitting on one axis, which is the claim.
    y3, h3 = 0.395, 0.155
    ax.annotate('', xy=(1.03, y3 + h3 / 2), xytext=(-0.03, y3 + h3 / 2),
                arrowprops=dict(arrowstyle='<|-|>', color=PRIMARY, lw=1.4,
                                mutation_scale=11), zorder=0)
    box(ax, 0.015, y3, 0.29, h3, 'Goal channel',
        '$\hat{M}$ supplies only the' + chr(10) + 'conditioning target')
    box(ax, 0.355, y3, 0.29, h3, 'Data channel',
        '$\hat{M}$ generates synthetic' + chr(10) + 'transitions for training')
    box(ax, 0.695, y3, 0.29, h3, 'Action channel',
        'a planner takes an' + chr(10) + 'argmax against $\hat{M}$', edge=ACCENT, lw=1.6)
    for x, ls in ((0.16, (0, (4, 2))), (0.50, (0, (4, 2))), (0.84, '-')):
        arrow(ax, (0.50, y2), (x, y3 + h3),
              color=ACCENT if x == 0.84 else MUTE, ls=ls)
    ax.text(0.50, y3 - 0.022, 'one axis, not three options:  trust-region width $m$ runs from $m = 1$ at the goal channel to $m = |\mathcal{A}|$ at the planner   (Eq. 3.1)',
            ha='center', va='top', fontsize=7.4, color=PRIMARY)

    # ---- row 4: constraint, policy, evaluation ---------------------------------------
    y4, h4 = 0.115, 0.125
    box(ax, 0.015, y4, 0.29, h4, 'Support mask  $\mathrm{Supp}_k$',
        'applied at inference,' + chr(10) + 'no retraining')
    box(ax, 0.355, y4, 0.29, h4, 'Learned policy  $\pi$', 'one policy per channel')
    box(ax, 0.695, y4, 0.29, h4, 'Exact evaluation',
        r'backward induction $\rightarrow$ $\mathrm{nv}(\pi)$' + chr(10) + '(Eq. 3.3)',
        face='#FFFFFF', edge=PRIMARY, lw=1.7)
    for x in (0.16, 0.50, 0.84):
        arrow(ax, (x, y3 - 0.075), (0.50 if x == 0.50 else x, y4 + h4),
              color=MUTE, ls='-' if x == 0.50 else (0, (4, 2)))
    arrow(ax, (0.305, y4 + h4 / 2), (0.355, y4 + h4 / 2))
    arrow(ax, (0.645, y4 + h4 / 2), (0.695, y4 + h4 / 2), color=PRIMARY, lw=1.5)
    ax.text(0.015, 0.035,
            "Solid: the model selects actions.   Dashed: it reaches the policy without "
            "selecting them.   Every path is evaluated on the same scale.",
            ha="left", va="center", fontsize=7.1, color=MUTE)

    # Two levels up: this script sits in pricing_dt/reporting/, not beside the results tree.
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "results", "figures", "pipeline_overview.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print("wrote " + os.path.normpath(out))


if __name__ == "__main__":
    main()
