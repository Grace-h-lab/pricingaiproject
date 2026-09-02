"""Draw Figure 3.1: the dynamics of each testbed and the limits of each prior.

Section 3.2 introduces two environments in prose. A reader meeting them for the first time
has to hold two things at once: that the action moves a state variable the next period
inherits, which is what makes each problem sequential rather than a bandit, and that the
structural prior is unable to represent one specific feature of it, which is the object the
whole study puts under test. Figure C.1 gives the same two environments as a parameter
sheet; this figure gives the transitions and where each prior gives out.

Two things govern the layout. Every constant and every transition is read from the shipped
code at draw time, so the figure cannot drift from the environments it describes. And
the document places every figure at 5.8 inches, so the canvas is kept narrow and the text
short: a wide canvas is a small typeface once the page scales it, and the arithmetic is
printed by --check so the trade is visible rather than assumed.

Run from anywhere:  python pricing_dt/reporting/make_env_dynamics_figure.py [--check]
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# the Figure 3.2 palette, plus the two environment hues Figure C.1 already uses
PRIMARY = "#0F3557"
E1C = "#1F6FB2"
E2C = "#6B4E9B"
FAIL = "#B4342A"
PANEL = "#EDF2F7"
RULE = "#C3D0DB"
MUTE = "#52616E"

# type scale, chosen so that nothing lands under 6 pt once the page scales the figure
FS_BAND, FS_TITLE, FS_BODY, FS_EDGE, FS_NOTE = 9.8, 8.8, 7.5, 7.8, 7.6

FIG_W, FIG_H = 7.8, 5.35
DOCX_W = 5.8                 # the width every figure is placed at in the document
CHAR_EM, CHAR_EM_B = 0.50, 0.58      # DejaVu Sans regular and bold

HERE = os.path.dirname(os.path.abspath(__file__))
# Two levels up: this script sits in pricing_dt/reporting/, not beside the results tree.
ROOT = os.path.dirname(os.path.dirname(HERE))
_WIDE = []


def shipped():
    """Read the constants this figure prints from the code that runs the experiments."""
    sys.path.insert(0, ROOT)
    from pricing_dt.core.config import SimConfig
    from pricing_dt.envs.inventory import InvConfig, InventoryMDP

    sim, inv = SimConfig(), InvConfig()
    env2 = InventoryMDP(inv)
    return sim, inv, env2.demand_mean(), env2.demand_var()


def watch(artist, maxw_axes, where):
    """Remember a text artist and the room it was given, for measurement after the draw."""
    _WIDE.append((artist, maxw_axes, where))


def audit(fig, ax):
    """Measure what was actually rendered. An em-factor estimate keeps mis-sizing bold
    faces and maths, and a title that overruns its box lands on whatever sits beside it;
    the canvas knows the real extents once it has been drawn."""
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for artist, maxw, where in _WIDE:
        bb = artist.get_window_extent(fig.canvas.get_renderer())
        x0, _ = inv.transform((bb.x0, bb.y0))
        x1, _ = inv.transform((bb.x1, bb.y1))
        w = abs(x1 - x0)
        if w > maxw + 1e-9:
            bad.append("%-16s %5.3f > %5.3f  %r" % (where, w, maxw, artist.get_text()[:44]))
    return bad


def box(ax, x, y, w, h, title, body="", face=PANEL, edge=RULE, tcol=PRIMARY, where="",
        ty=None, by=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2))
    t = ax.text(x + w / 2, y + h * (ty if ty is not None else (0.68 if body else 0.50)),
                title, ha="center", va="center", fontsize=FS_TITLE, weight="bold",
                color=tcol, zorder=3, linespacing=1.35)
    watch(t, w - 0.010, where + " title")
    if body:
        b = ax.text(x + w / 2, y + h * (by if by is not None else 0.28), body, ha="center",
                    va="center", fontsize=FS_BODY, color=MUTE, zorder=3, linespacing=1.5)
        watch(b, w - 0.010, where + " body")


def arrow(ax, p, q, color=MUTE, lw=1.3, ls="-", ms=11):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


def label(ax, x, y, s, color=MUTE, size=FS_BODY, weight="normal", ha="center",
          style="normal", maxw=None, where=""):
    t = ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color, weight=weight,
                style=style, zorder=5)
    if maxw is not None:
        watch(t, maxw, where)
    return t


def loop(ax, x_from, x_to, y_boxes, y_drop, text, color, where):
    """The return path, drawn as three segments rather than a connectionstyle so the
    crossbar lands where the label is not. This edge is the point of the figure."""
    ax.plot([x_from, x_from], [y_boxes, y_drop], color=color, lw=1.4, zorder=4)
    ax.plot([x_from, x_to], [y_drop, y_drop], color=color, lw=1.4, zorder=4)
    arrow(ax, (x_to, y_drop), (x_to, y_boxes - 0.004), color=color, lw=1.4)
    label(ax, (x_from + x_to) / 2, y_drop - 0.046, text, color=color, size=FS_NOTE,
          weight="bold", maxw=0.62, where=where)


def band(ax, y_top, tag, name, qualifier, colour, chain, dashed_second=False):
    """One environment: a heading, a three-node chain, and the boxes on it."""
    ax.add_patch(FancyBboxPatch((0.0, y_top - 0.418), 0.995, 0.418,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=0.9, edgecolor=RULE, facecolor="white", zorder=0))
    ax.plot([0.016, 0.028], [y_top - 0.048, y_top - 0.048], color=colour, lw=3.0,
            solid_capstyle="butt", zorder=3)
    label(ax, 0.038, y_top - 0.048, "%s   %s" % (tag, name), color=colour, size=FS_BAND,
          weight="bold", ha="left", maxw=0.66, where=tag + " head")
    label(ax, 0.978, y_top - 0.048, qualifier, color=MUTE, size=FS_NOTE, ha="right",
          style="italic")

    yb, hb = y_top - 0.252, 0.118
    for x, w, title, body in chain:
        box(ax, x, yb, w, hb, title, body, tcol=colour, where=tag)
    arrow(ax, (0.180, yb + hb / 2), (0.216, yb + hb / 2), color=colour)
    arrow(ax, (0.447, yb + hb / 2), (0.483, yb + hb / 2), color=colour,
          ls=(0, (3, 2)) if dashed_second else "-")
    return yb, hb


def main(check=False):
    sim, inv, d_mean, d_var = shipped()
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-0.010, 1.005)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- E1 -----------------------------------------------------------------------
    yb, hb = band(ax, 0.950, "E1", "Reference-price pricing", "exactly solvable", E1C,
                  [(0.022, 0.150, "state", "$(\\mathrm{ref}_t,\\ t)$"),
                   (0.224, 0.215, "action: a price",
                    "$p_t$, one of %d levels" % sim.n_prices),
                   (0.491, 0.215, "what is logged",
                    "$r_t = p_t\\,\\mathbb{E}[q_t]\\,e^{\\varepsilon}$")])
    label(ax, 0.198, yb + hb / 2 + 0.032, "$a_t$", color=E1C, size=FS_EDGE, weight="bold",
          maxw=0.050, where="E1 edge1")
    label(ax, 0.465, yb + hb / 2 + 0.032, "$\\mathbb{E}[q_t]$", color=E1C, size=FS_EDGE,
          weight="bold", maxw=0.050, where="E1 edge2")
    loop(ax, 0.332, 0.097, yb, yb - 0.058,
         "$\\mathrm{ref}_{t+1} = (1-\\eta)\\,\\mathrm{ref}_t + \\eta\\,p_t$", E1C, "E1 loop")
    box(ax, 0.740, yb - 0.050, 0.243, hb + 0.100,
        "structural\nmisspecification",
        "monotone, bounded\nelasticity: it cannot fit\na steep off-support fall",
        face="#FBF0EE", edge=FAIL, tcol=FAIL, where="E1 prior", ty=0.78, by=0.30)
    label(ax, 0.028, yb - 0.140,
          "$\\eta = %.1f$ moves the state with the price, $\\delta = %.1f$ moves demand with "
          "the state; at $\\delta = 0$ both optima coincide."
          % (sim.eta, sim.delta),
          color=PRIMARY, size=FS_NOTE, ha="left", maxw=0.98, where="E1 note")

    # ---- E2 -----------------------------------------------------------------------
    yb, hb = band(ax, 0.472, "E2", "Lost-sales inventory with censored demand",
                  "overdispersed demand", E2C,
                  [(0.022, 0.150, "state", "$(x_t,\\ t)$"),
                   (0.224, 0.215, "action: an order",
                    "avail$_t$ = min$(x_t{+}a_t,\\,%d)$" % inv.max_inventory),
                   (0.491, 0.215, "what is logged",
                    "sales$_t$ = min$(D_t,\\,$avail$_t)$")],
                  dashed_second=True)
    label(ax, 0.198, yb + hb / 2 + 0.032, "$a_t$", color=E2C, size=FS_EDGE, weight="bold",
          maxw=0.050, where="E2 edge1")
    label(ax, 0.465, yb + hb / 2 + 0.032, "$D_t$", color=E2C, size=FS_EDGE,
          weight="bold", maxw=0.050, where="E2 edge2")
    loop(ax, 0.332, 0.097, yb, yb - 0.058,
         "$x_{t+1} = \\max(0,\\ \\mathrm{avail}_t - D_t)$", E2C, "E2 loop")
    box(ax, 0.740, yb - 0.050, 0.243, hb + 0.100,
        "structural\nmisspecification",
        "Poisson forces variance\n$=$ mean, but the mixture\nhas a variance-to-mean\n"
        "ratio of %.2f" % (d_var / d_mean),
        face="#FBF0EE", edge=FAIL, tcol=FAIL, where="E2 prior", ty=0.80, by=0.32)
    label(ax, 0.028, yb - 0.140,
          "Latent $D_t$ above availability is never seen, so the log is censored: it "
          "trains on sales, not on demand.",
          color=FAIL, size=FS_NOTE, ha="left", maxw=0.98, where="E2 note")

    out = os.path.join(ROOT, "results", "figures", "env_dynamics.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")

    bad = audit(fig, ax)
    if bad:
        print("TEXT TOO WIDE (%d):" % len(bad))
        for w in bad:
            print("   " + w)
    else:
        print("no text overruns")
    if check:
        from PIL import Image
        w_in = Image.open(out).size[0] / 220.0
        k = DOCX_W / w_in
        print("printed size: canvas %.2f in -> %.2f in on the page, scale %.3f" % (w_in, DOCX_W, k))
        for name, pt in (("band heading", FS_BAND), ("box title", FS_TITLE),
                         ("box body", FS_BODY), ("notes", FS_NOTE)):
            print("   %-14s %4.1f pt  ->  %4.1f pt%s"
                  % (name, pt, pt * k, "   UNDER 6 pt" if pt * k < 6 else ""))
    plt.close(fig)
    print("wrote " + os.path.normpath(out))


if __name__ == "__main__":
    main(check="--check" in sys.argv)
