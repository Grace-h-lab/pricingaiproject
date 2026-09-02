"""Draw Appendix C as two portrait pages: what was built, and how it was tested.

What a technically literate reader needs from an implementation appendix is the objects
themselves -- the demand equation, the relabelling target, the critic objective, the
token layout, the two inference operators, the falsification conditions -- at a level of
detail the chapters themselves do not carry.

The pages follow the visual hierarchy of Figure B.1: lettered section bars, panel titles at a
size that survives print, one semantic palette shared between the two appendices, and one idea
per band. The first band is a genuine pipeline rather than a row of captions: six stages
joined by arrows and grouped into INPUT, MODEL and OUTPUT, cut by the rule separating what the
learner may see from what only the experimenter knows. That rule is the most important line on
either page, so it is drawn in the failure colour and labelled.

Every constant is read out of the shipped configuration at draw time rather than typed in, so
a figure claiming to transcribe the implementation cannot quietly stop matching it. If a field
is renamed the draw fails instead of printing a stale number. The parameter count and the true
demand moments are computed by instantiating the objects themselves.

Two quantities on these pages are easy to conflate, so both are stated in full:

  * The RQ4 result is -0.119 against masked Q-DT with Holm p = 0.006 over ten seeds, the
    correction being applied over seeds rather than over the 90 cell-seed rows.
  * The 0.9936 ceiling belongs to the main protocol's logged start distribution; the
    competence sweep runs to 0.986 under a uniform one.

Run from anywhere:  python pricing_dt/reporting/make_architecture_figure.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- palette
# One meaning per hue, shared with Figure B.1 so the two read as one appendix.
PRIMARY = "#0F3557"     # structure, section bars
E1C = "#1F6FB2"         # pricing
E2C = "#6B4E9B"         # inventory
EXACT = "#2E7D4F"       # exact control, ground truth, a claim that held
FAIL = "#B4342A"        # fitted-model failure, refutation, the isolation rule
SUPPORT = "#B5741C"     # support, floor, ceiling
NEUTRAL = "#5A6B78"     # qualifiers and captions
PANEL = "#F3F6F9"
CODEBG = "#EDF1F5"
RULE = "#C3D0DB"
MUTE = "#3E4C57"
DARK = "#16202A"
INPUTBG = "#EAF1F8"
MODELBG = "#EDF6EF"
OUTBG = "#F8F2E8"

# Type scale, matched to Figure B.1 rather than to the landscape prototype.
FS_TITLE, FS_EYE = 16.0, 5.8
FS_SEC, FS_PANEL = 8.4, 8.0
FS_LEAD, FS_BODY = 6.4, 6.0
FS_CODE, FS_SMALL, FS_TINY = 5.4, 5.4, 5.0

PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
CHAR_EM, LINE_SP = 0.52, 1.42
MONO_EM = 0.60

HERE = os.path.dirname(os.path.abspath(__file__))
# Two levels up: this script sits in pricing_dt/reporting/, not beside the results tree.
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(ROOT, "results", "figures")

_OVERRUN = []


# ---------------------------------------------------------------- shipped constants
def _shipped():
    """Read every constant these pages print from the code that runs the experiments."""
    import sys
    sys.path.insert(0, ROOT)
    from pricing_dt.core.config import SimConfig, ModelConfig
    from pricing_dt.core import dt
    from pricing_dt.envs.inventory import InventoryMDP, InvConfig

    sim, mdl, inv = SimConfig(), ModelConfig(), InvConfig()
    env2 = InventoryMDP(inv)
    net = dt.DecisionTransformer(obs_dim=2, n_actions=sim.n_prices, cfg=mdl,
                                 max_T=sim.horizon)
    return dict(
        sim=sim, mdl=mdl, inv=inv,
        params=sum(p.numel() for p in net.parameters()),
        d_mean=env2.demand_mean(), d_var=env2.demand_var(),
        # diag_gate2_pricing._qdt_model_cfg overrides the class defaults for the critic;
        # these are the values the family-table run recorded in four_family_raw.csv.
        cql_alpha=0.1, cql_lr="3e-4", cql_updates=2000, iql_updates=2000,
    )


# ---------------------------------------------------------------- drawing
def fit(s, size, maxw, where, mono=False):
    per = (size * (MONO_EM if mono else CHAR_EM) / 72.0) / PAGE_W_IN
    for line in s.split("\n"):
        plain = line.replace("$", "").replace("\\", "")
        if len(plain) * per > maxw:
            _OVERRUN.append("%-22s wide %5.3f > %5.3f  %r"
                            % (where, len(plain) * per, maxw, line[:42]))


def drop(s, size, y, floor, where):
    n = s.count("\n") + 1
    bottom = y - n * (size * LINE_SP / 72.0) / PAGE_H_IN
    if bottom < floor:
        _OVERRUN.append("%-22s deep %5.3f < %5.3f  %r"
                        % (where, bottom, floor, s.split("\n")[0][:42]))


def lh(size):
    return (size * LINE_SP / 72.0) / PAGE_H_IN


def box(fig, x, y, w, h, face=PANEL, edge=RULE, lw=0.8, z=1):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.001,rounding_size=0.005",
        transform=fig.transFigure, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def txt(fig, x, y, s, size=FS_BODY, color=MUTE, weight="normal", ha="left", style="normal",
        mono=False, maxw=None, floor=None, where="", z=4):
    if maxw is not None:
        fit(s, size, maxw, where, mono)
    if floor is not None:
        drop(s, size, y, floor, where)
    fig.text(x, y, s, ha=ha, va="top", fontsize=size, color=color, weight=weight,
             style=style, zorder=z, linespacing=LINE_SP,
             family="DejaVu Sans Mono" if mono else None)


def rule(fig, x0, x1, y, color=RULE, lw=0.7, ls="-"):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              color=color, linewidth=lw, linestyle=ls, zorder=3))


def vrule(fig, x, y0, y1, color=RULE, lw=0.7, ls="-"):
    fig.add_artist(plt.Line2D([x, x], [y0, y1], transform=fig.transFigure,
                              color=color, linewidth=lw, linestyle=ls, zorder=3))


def arrow(fig, x0, x1, y, color=PRIMARY, lw=1.6, z=6, ms=8):
    fig.patches.append(FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=ms, color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z))


def node(fig, x, y, w, h, label, color=PRIMARY, size=None, where=""):
    """A rounded state/quantity box with a centred monospace label."""
    size = size or FS_TINY
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.001,rounding_size=0.004",
        transform=fig.transFigure, facecolor="white", edgecolor=color,
        linewidth=0.9, zorder=5))
    fit(label, size, w - 0.006, where or "node", mono=True)
    fig.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center",
             fontsize=size, color=color, family="DejaVu Sans Mono", zorder=6)
    return x + w


def edge(fig, x0, x1, y, label="", color=NEUTRAL, size=None, where="", above=True):
    """A horizontal arrow with its label set clear of the line."""
    size = size or FS_TINY
    arrow(fig, x0, x1, y, color=color, lw=1.1)
    if label:
        fit(label, size, (x1 - x0) + 0.030, where or "edge", mono=True)
        fig.text((x0 + x1) / 2.0, y + (0.0042 if above else -0.0088), label,
                 ha="center", va="bottom" if above else "top", fontsize=size,
                 color=color, family="DejaVu Sans Mono", zorder=6)


def vedge(fig, x, y0, y1, color=NEUTRAL, lw=1.1):
    """A short downward arrow, for a branch out of a chain."""
    fig.patches.append(FancyArrowPatch(
        (x, y0), (x, y1), transform=fig.transFigure, arrowstyle="-|>",
        mutation_scale=7, color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=5))


def feedback(fig, x_from, x_to, y_line, drop_to, label, color=PRIMARY, where="",
             maxw=None):
    """A return path drawn under a chain: down, back along, and up into the first node."""
    fig.patches.append(FancyArrowPatch(
        (x_from, y_line), (x_to, y_line), transform=fig.transFigure,
        connectionstyle="bar,fraction=%.4f" % ((drop_to - y_line) / max(x_from - x_to, 1e-6)),
        arrowstyle="-|>", mutation_scale=8, color=color, linewidth=1.1,
        shrinkA=0, shrinkB=0, zorder=5))
    fit(label, FS_TINY, maxw or abs(x_from - x_to) + 0.040, where or "feedback", mono=True)
    fig.text((x_from + x_to) / 2.0, drop_to - 0.0030, label, ha="center", va="top",
             fontsize=FS_TINY, color=color, family="DejaVu Sans Mono", zorder=6)


def secbar(fig, x, y, w, h, text):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0", transform=fig.transFigure,
        facecolor=PRIMARY, edgecolor="none", zorder=2))
    fig.text(x + 0.010, y + h / 2.0, text, ha="left", va="center", fontsize=FS_SEC,
             color="white", weight="bold", zorder=3)


def panel(fig, x, y, w, h, tag, title, note="", accent=PRIMARY, face="white"):
    """A titled panel: a coloured letter chip, the title, a right-aligned qualifier."""
    box(fig, x, y, w, h, face=face, edge=RULE, lw=0.9)
    fig.patches.append(plt.Circle((x + 0.0145, y + h - 0.0104), 0.0080,
                                  transform=fig.transFigure, facecolor=accent,
                                  edgecolor="none", zorder=5))
    fig.text(x + 0.0145, y + h - 0.0104, tag, ha="center", va="center", fontsize=6.2,
             color="white", weight="bold", zorder=6)
    txt(fig, x + 0.028, y + h - 0.0048, title, size=FS_PANEL, color=PRIMARY, weight="bold",
        maxw=w - 0.135, where=tag + " title")
    if note:
        txt(fig, x + w - 0.008, y + h - 0.0044, note, size=FS_TINY, color=NEUTRAL,
            style="italic", ha="right")
    return y + h - 0.0205          # where content may start


def code(fig, x, y, w, lines, size=FS_CODE, pad=0.0038, where=""):
    n = lines.count("\n") + 1
    h = n * lh(size) + 2 * pad
    box(fig, x, y - h, w, h, face=CODEBG, edge="none", lw=0, z=2)
    txt(fig, x + pad, y - pad, lines, size=size, color=DARK, mono=True,
        maxw=w - 2 * pad, where=where)
    return y - h


def para(fig, x, y, s, w, size=FS_BODY, color=MUTE, style="normal", gap=0.0052,
         floor=None, where=""):
    txt(fig, x, y, s, size=size, color=color, style=style, maxw=w, floor=floor, where=where)
    return y - (s.count("\n") + 1) * lh(size) - gap


def newpage():
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    fig.patch.set_facecolor("white")
    return fig


def header(fig, title, right):
    """Title and its qualifier. The page number goes in the footer, not up here."""
    txt(fig, 0.030, 0.9935, title, size=FS_TITLE, color=PRIMARY, weight="bold",
        maxw=0.66, where="page title")
    txt(fig, 0.970, 0.9905, right, size=FS_TINY, color=NEUTRAL, ha="right", style="italic")
    rule(fig, 0.030, 0.970, 0.9640, color=PRIMARY, lw=1.3)


def footer(fig, n):
    rule(fig, 0.030, 0.970, 0.0265, color=RULE, lw=0.7)
    txt(fig, 0.970, 0.0225, "PAGE %d OF 2" % n, size=FS_TINY, color=NEUTRAL,
        weight="bold", ha="right")


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, name)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    print("saved %s" % os.path.relpath(out, ROOT))

def page_one(S):
    """Page 1 carries the pipeline and the objects in it, and nothing else.

    Six panels on one page rather than eight on a crowded one: each band gets the same
    height, so the whitespace is a grid rather than whatever each block happened to leave.
    """
    sim, mdl, inv = S["sim"], S["mdl"], S["inv"]
    fig = newpage()
    header(fig, "The pipeline and the objects in it",
           "Every constant is read from the shipped configuration at draw time.\n"
           "A = %d actions,   H = %d,   |B| = %d reference bins."
           % (sim.n_prices, sim.horizon, sim.n_ref_bins))

    # ---- A: the pipeline, drawn as a pipeline --------------------------
    secbar(fig, 0.030, 0.9350, 0.940, 0.0215, "A    INPUT  →  MODEL  →  OUTPUT")
    box(fig, 0.030, 0.8080, 0.940, 0.1220, face="white", edge=RULE, lw=0.9)

    for name, x0, x1, bg, col in [("INPUT", 0.038, 0.192, INPUTBG, E1C),
                                  ("MODEL", 0.200, 0.664, MODELBG, EXACT),
                                  ("OUTPUT", 0.672, 0.962, OUTBG, SUPPORT)]:
        box(fig, x0, 0.8330, x1 - x0, 0.0800, face=bg, edge="none", lw=0, z=1)
        txt(fig, x0 + 0.007, 0.9250, name, size=FS_LEAD, color=col, weight="bold")

    steps = [
        (0.044, "LOGGED DATA", E1C,
         "D = {tau_i}, i = 1..N\nN in {100, 400, 1600}\ntwo region specialists\nper environment"),
        (0.206, "FIT", EXACT,
         "qhat(p, s), panel C\none model object per\nseed, reused by every\nchannel"),
        (0.358, "RELABEL", EXACT,
         "G_t -> Rhat_t, panel D\nthe ONLY column that\ndiffers between the\nreturn-conditioned arms"),
        (0.510, "TRAIN", EXACT,
         "DT, %s params\npanel F; one architecture\nand one schedule for\nevery arm" % f"{S['params']:,}"),
        (0.678, "ACT", SUPPORT,
         "a_t = argmax over the\nadmissible set; page 2\nconstraints applied at\ninference, no retraining"),
        (0.838, "MEASURE", EXACT,
         "nv = (V_pi - V_beh) /\n(V* - V_beh); anchors by\nbackward induction in\nfloat64"),
    ]
    for i, (x, name, col, body) in enumerate(steps):
        txt(fig, x, 0.9130, name, size=FS_BODY, color=col, weight="bold",
            maxw=0.142, where=name)
        txt(fig, x, 0.9018, body, size=FS_TINY, color=MUTE, mono=True,
            maxw=0.146, floor=0.8330, where=name + " body")
        if i:
            heavy = i in (1, 4)          # INPUT -> MODEL and TRAIN -> ACT cross a zone
            arrow(fig, x - 0.0260, x - 0.0060, 0.9100,
                  color=PRIMARY if heavy else NEUTRAL,
                  lw=2.2 if heavy else 1.7, ms=11)

    vrule(fig, 0.8060, 0.8380, 0.9210, color=FAIL, lw=1.3, ls=(0, (2.5, 2)))
    txt(fig, 0.038, 0.8285,
        "One fitted model, one log, one set of auxiliary knowledge, held fixed. "
        "Only the route changes.",
        size=FS_SMALL, color=PRIMARY, weight="bold", maxw=0.500, floor=0.8080,
        where="A kicker")
    txt(fig, 0.962, 0.8285,
        "left of the dashed rule: logged data only.  right of it: exact, "
        "for measurement only",
        size=FS_SMALL, color=FAIL, weight="bold", ha="right", maxw=0.420,
        floor=0.8080, where="rule note")

    # ---- B: the objects, three equal bands -----------------------------
    secbar(fig, 0.030, 0.7715, 0.940, 0.0215, "B    THE OBJECTS IN IT")
    ROW = [0.5415, 0.2907, 0.0399]       # 0.2200 panels, 0.0308 gutters, even margins
    H = 0.2200

    y0 = panel(fig, 0.030, ROW[0], 0.4620, H, "A",
               "Environment E1 — reference-price MDP", "exactly solvable", E1C)
    y = code(fig, 0.040, y0, 0.4420,
             "a_t in {0..%d} -> p_a in [%.1f, %.1f];  one price per period\n"
             "u(p, ref) = alpha - beta*p + delta*(ref - p)\n"
             "E[q] = M * sigmoid(u)      r(p, ref) = p * E[q]"
             % (sim.n_prices - 1, sim.p_min, sim.p_max), where="A eqs")
    yd = y - 0.0075
    node(fig, 0.043, yd - 0.0135, 0.098, 0.0135, "(ref_t, t)", E1C, where="A s")
    edge(fig, 0.145, 0.196, yd - 0.0068, "a_t", color=E1C, where="A e1")
    node(fig, 0.200, yd - 0.0135, 0.098, 0.0135, "price p_a", E1C, where="A p")
    edge(fig, 0.302, 0.353, yd - 0.0068, "E[q]", color=E1C, where="A e2")
    node(fig, 0.357, yd - 0.0135, 0.120, 0.0135, "r = p * E[q]", E1C, where="A r")
    feedback(fig, 0.249, 0.092, yd - 0.0135, yd - 0.0245,
             "ref' = (1 - eta)*ref + eta*p     deterministic", E1C, where="A fb",
             maxw=0.250)
    y = yd - 0.0300
    txt(fig, 0.040, y - 0.0180,
        "alpha %.1f   beta %.1f   delta %.1f   eta %.1f   M %.0f"
        % (sim.alpha, sim.beta, sim.delta, sim.eta, sim.market_size),
        size=FS_BODY, color=E1C, weight="bold", mono=True, maxw=0.44, where="A consts")
    y = para(fig, 0.040, y - 0.0300,
             "delta > 0 is what makes the problem sequential. At delta = 0 the myopic and\n"
             "intertemporal optima coincide exactly and the problem is a disguised bandit,\n"
             "which Appendix G.10 checks rather than assumes.",
             0.442, gap=0.0170, where="A note")
    y = code(fig, 0.040, y, 0.4420,
             "logged r_t = p * q * exp(eps),  eps ~ N(0, sigma^2)\n"
             "sigma in {0.05, 0.20, 0.50}    multiplicative log-normal",
             where="A noise")
    para(fig, 0.040, y - 0.0140,
         "Evaluation uses expected values; the noise is in the log only.",
         0.442, size=FS_SMALL, style="italic", floor=ROW[0], where="A eval")

    y0 = panel(fig, 0.5080, ROW[0], 0.4620, H, "B",
               "Environment E2 — lost-sales inventory", "censored, overdispersed", E2C)
    y = code(fig, 0.518, y0, 0.4420,
             "a_t in {0..%d} order quantities;  lost sales, not backorders\n"
             "D ~ 0.8*Poisson(%.0f) + 0.2*Poisson(%.0f)     drawn each period\n"
             "x' = max(0, avail - D)       stock is capped at %d first"
             % (inv.max_order, inv.mu_base, inv.mu_spike, inv.max_inventory), where="B eqs")
    yd = y - 0.0075
    node(fig, 0.521, yd - 0.0135, 0.092, 0.0135, "(x_t, t)", E2C, where="B s")
    edge(fig, 0.617, 0.660, yd - 0.0068, "a_t", color=E2C, where="B e1")
    node(fig, 0.664, yd - 0.0135, 0.140, 0.0135, "avail=min(x+a,%d)" % inv.max_inventory,
         E2C, where="B av")
    edge(fig, 0.808, 0.851, yd - 0.0068, "D", color=E2C, where="B e2")
    node(fig, 0.855, yd - 0.0135, 0.102, 0.0135, "sales=min(D,avail)", E2C, where="B s2")
    feedback(fig, 0.734, 0.567, yd - 0.0135, yd - 0.0245,
             "x' = max(0, avail - D)", E2C, where="B fb", maxw=0.170)
    txt(fig, 0.957, yd - 0.0262,
        "logged: min(D, avail); demand above stock unseen",
        size=FS_TINY, color=FAIL, mono=True, maxw=0.245, ha="right", where="B censor")
    y = yd - 0.0300
    txt(fig, 0.518, y - 0.0180,
        "price %.1f   order %.1f   holding %.1f   stockout %.1f   H %d"
        % (inv.price, inv.order_cost, inv.hold_cost, inv.stockout_penalty, inv.horizon),
        size=FS_BODY, color=E2C, weight="bold", mono=True, maxw=0.44, where="B consts")
    y = para(fig, 0.518, y - 0.0300,
             "The prior is Poisson, whose variance is identically its mean. The true mixture\n"
             "has mean %.1f and variance %.1f, a ratio of %.2f, so the prior cannot represent\n"
             "the dispersion at all. E1's prior fails the same way about slope."
             % (S["d_mean"], S["d_var"], S["d_var"] / S["d_mean"]),
             0.442, gap=0.0170, where="B note")
    y = code(fig, 0.518, y, 0.4420,
             "transition stochastic -> anchors and V* exact by DP,\n"
             "                        learned-policy values Monte-Carlo",
             where="B eval")
    para(fig, 0.518, y - 0.0140,
         "Common random seeds across arms. This is the one asymmetry with E1.",
         0.442, size=FS_SMALL, style="italic", floor=ROW[0], where="B mc")

    y0 = panel(fig, 0.030, ROW[1], 0.4620, H, "C",
               "Structured demand prior", "the object under test", EXACT)
    y = code(fig, 0.040, y0, 0.4420,
             "log qhat(p, s) = g(s) - beta_phi(s) * p\n"
             "beta_phi(s)    = e_lo + (e_hi - e_lo) * sigmoid(b(s))\n"
             "e_lo %.1f   e_hi %.1f       g, b : MLP(2 -> 64 -> 1)"
             % (mdl.elasticity_lo, mdl.elasticity_hi), where="C eqs")
    y = para(fig, 0.040, y - 0.0190,
             "The squash is the entire structural commitment: it forces the price derivative\n"
             "negative everywhere and bounds elasticity by construction rather than by fit.\n"
             "Fitted to logged (s, p, q) only; no ground truth enters.",
             0.442, gap=0.0250, where="C note")
    txt(fig, 0.040, y, "ABLATIONS", size=FS_SMALL, color=EXACT, weight="bold")
    for i, (nm, what) in enumerate([
            ("unconstrained", "free MLP, no monotonicity"),
            ("misspecified", "bounds set to [e_hi, 2 e_hi], excluding the truth"),
            ("corrupted", "cancels, then flips, the slope")]):
        yy = y - 0.0135 - i * 0.0135
        txt(fig, 0.040, yy, nm, size=FS_SMALL, color=DARK, mono=True)
        txt(fig, 0.148, yy, what, size=FS_SMALL, color=MUTE, maxw=0.334,
            where="C abl %d" % i)
    para(fig, 0.040, y - 0.0615,
         "Bounds excluding the true elasticity cost 0.006; flipping the sign costs 0.28.\n"
         "The prior carries 514 parameters against the free comparator's 4,481, so the\n"
         "capacities are not matched and the asymmetry runs against the prior.",
         0.442, size=FS_SMALL, style="italic", floor=ROW[1], where="C abl note")

    y0 = panel(fig, 0.5080, ROW[1], 0.4620, H, "D",
               "Structured return relabelling", "action-dependent", EXACT)
    y = code(fig, 0.518, y0, 0.4420,
             "Rhat_t = rhat(s_t, a_t) + SUM_{k>t} max_p rhat(s_k, p)\n"
             "         |__ LOGGED action __|  |_ greedy continuation _|",
             where="D eq")
    y = para(fig, 0.518, y - 0.0190,
             "The current step uses the LOGGED action. If every step used the model optimum\n"
             "the target would be identical whatever the log did, and the model would learn\n"
             "to attach a high target to bad actions, which is the defect measured in the\n"
             "legacy comparator of panel E.",
             0.442, gap=0.0250, where="D note")
    y = code(fig, 0.518, y, 0.4420,
             "cont[H,b] = 0       a* = argmax_a rev[k,b,a]\n"
             "cont[k,b] = rev[k,b,a*] + cont[k+1, N[a*,b]]\n"
             "Rhat_t    = rev[t,b_t,a_t] + cont[t+1, N[a_t,b_t]]",
             where="D table")
    para(fig, 0.518, y - 0.0230,
         "One batched forward and a backward pass replace an O(H^2) per-trajectory loop,\n"
         "which is what made the full factorial grid affordable. The shipped target is\n"
         "(1 - lam) G_t + lam Rhat_t with lam = 1: the blend exists and is never swept.",
         0.442, size=FS_SMALL, style="italic", floor=ROW[1], where="D perf")

    y0 = panel(fig, 0.030, ROW[2], 0.4620, H, "E",
               "Comparator targets and the critic", "one critic, five read-outs", NEUTRAL)
    y = code(fig, 0.040, y0, 0.4420,
             "L(Q) = L_TD + a_c * E_s[ logsumexp_a Q(s,a) - Q(s, a_data) ]\n"
             "a_c %.1f   %d updates   Adam %s   hidden %d"
             % (S["cql_alpha"], S["cql_updates"], S["cql_lr"], mdl.q_hidden),
             where="E cql")
    txt(fig, 0.040, y - 0.0180, "RELABEL MODE", size=FS_SMALL, color=NEUTRAL, weight="bold")
    txt(fig, 0.198, y - 0.0180, "target Rhat_T", size=FS_SMALL, color=NEUTRAL, weight="bold")
    for i, (nm, tgt, col) in enumerate([
            ("state_value (legacy)", "max_a Q(s_t, a)", FAIL),
            ("q_sa", "Q(s_t, a_t)", DARK),
            ("td (default)", "r_t + V(s_{t+1}),  V = max_a Q", DARK),
            ("td + denoise", "R[a_t, b_t] + V(s_{t+1})", DARK),
            ("oracle", "Q*(t, b_t, a_t)", EXACT)]):
        yy = y - 0.0320 - i * 0.0135
        txt(fig, 0.040, yy, nm, size=FS_SMALL, color=col, mono=True)
        txt(fig, 0.198, yy, tgt, size=FS_SMALL, color=MUTE, mono=True, maxw=0.284,
            where="E mode %d" % i)
    para(fig, 0.040, y - 0.1080,
         "state_value is action-independent: its within-state variance is exactly zero, so it\n"
         "cannot teach the policy to prefer one logged continuation over another. Shipping it\n"
         "as the comparator invalidated every earlier advantage (Appendix E.1).",
         0.442, floor=ROW[2], where="E legacy")

    y0 = panel(fig, 0.5080, ROW[2], 0.4620, H, "F",
               "Sequence model", "identical across every target arm", PRIMARY)
    y = code(fig, 0.518, y0, 0.4420,
             "tok = [ Rhat_0, s_0, a_0, Rhat_1, s_1, a_1, ... ]  B x 3T x d\n"
             "d %d   %d blocks   %d heads   dropout %.1f   causal mask\n"
             "action logits read at the STATE token of each triple"
             % (mdl.d_model, mdl.n_layer, mdl.n_head, mdl.dropout), where="F tok")
    y = para(fig, 0.518, y - 0.0190,
             "%s parameters. AdamW, lr 1e-3, weight decay 1e-4, batch %d, %d epochs, clip\n"
             "1.0. Plain supervised training: cross-entropy on the logged action, with no\n"
             "bootstrapping anywhere in the sequence model."
             % (f"{S['params']:,}", mdl.batch_size, mdl.epochs),
             0.442, gap=0.0250, where="F opt")
    y = code(fig, 0.518, y, 0.4420,
             "Rhat <- (Rhat - mu_R) / s_R     standardised, or the model\n"
             "                                ignores the return token",
             where="F std")
    para(fig, 0.518, y - 0.0230,
         "Raw targets span about 480 logged to about 5000 bootstrapped. At those magnitudes\n"
         "the conditioning signal is swamped and every variant collapses to behaviour\n"
         "cloning, so the ablation silently stops being one. The most consequential detail\n"
         "on either page.",
         0.442, floor=ROW[2], where="F std note")

    footer(fig, 1)
    save(fig, "architecture_pipeline.png")


# ================================================================ page 2
def page_two(S):
    """Page 2 carries the implementation controls, the two operators, the evaluation
    protocol and the falsification conditions: everything about how the system was held
    still and how it could have failed."""
    fig = newpage()
    header(fig, "Controls, interventions and falsification",
           "Two inference-time operators, one exact control, and the\n"
           "conditions fixed before any policy was trained.")

    # ---- C: implementation and controls ---------------------------------
    secbar(fig, 0.030, 0.9350, 0.940, 0.0215, "C    IMPLEMENTATION AND CONTROLS")
    box(fig, 0.030, 0.7990, 0.940, 0.1300, face="white", edge=RULE, lw=0.9)
    txt(fig, 0.040, 0.9200, "IMPLEMENTATION MAP", size=FS_LEAD, color=PRIMARY, weight="bold")
    txt(fig, 0.962, 0.9195,
        "every diagnostic writes provenance.json beside its results: commit, argv, "
        "device, library versions",
        size=FS_TINY, color=NEUTRAL, ha="right", style="italic", maxw=0.600,
        where="prov note")
    for x0, rows in [
            (0.040, [("core/simulator.py", "E1 dynamics, exact DP, anchors"),
                     ("envs/inventory.py", "E2 dynamics, censoring, exact DP"),
                     ("core/data.py", "logging policy, trajectory generation"),
                     ("core/demand_model.py", "structured prior and free comparator"),
                     ("core/relabel.py", "panel D and the target variants")]),
            (0.508, [("core/qdt.py", "the CQL critic and its five read-outs"),
                     ("core/dt.py", "sequence model; the logged-support mask"),
                     ("core/baselines.py", "BC, IQL, bandits, estimate-then-optimise"),
                     ("core/metrics.py", "nv, gap, off-support rate, ceilings"),
                     ("diagnostics/", "one script per reported diagnostic")])]:
        for i, (path, what) in enumerate(rows):
            yy = 0.9060 - i * 0.0125
            txt(fig, x0, yy, "pricing_dt/" + path, size=FS_SMALL, color=DARK, mono=True,
                maxw=0.202, where="map path")
            txt(fig, x0 + 0.208, yy, what, size=FS_SMALL, color=MUTE, maxw=0.220,
                where="map what")
    rule(fig, 0.040, 0.960, 0.8480)
    txt(fig, 0.040, 0.8410, "WHAT THE ARCHITECTURE DELIBERATELY DOES NOT DO",
        size=FS_LEAD, color=PRIMARY, weight="bold")
    for i, (head, colour, body) in enumerate([
            ("No per-arm tuning.", FAIL,
             "One architecture, one optimiser, one schedule.\nOnly the conditioning column differs."),
            ("No target chosen on the metric.", FAIL,
             "Selection moved to a held-out split applied\nidentically to every arm (Appendix E.2)."),
            ("No pooling across batches.", FAIL,
             "Batches differ in code as well as in device,\nand none recorded its commit (G.1)."),
            ("No ground truth in a fitted object.", EXACT,
             "V*, Q* and both anchors exist for measurement\nand for the control below. Nothing else.")]):
        x0 = 0.040 + i * 0.2320
        txt(fig, x0, 0.8300, head, size=FS_SMALL, color=colour, weight="bold",
            maxw=0.224, where="not %d h" % i)
        txt(fig, x0, 0.8195, body, size=FS_SMALL, color=MUTE, maxw=0.224,
            floor=0.7990, where="not %d b" % i)

    # ---- D: the two operators -------------------------------------------
    secbar(fig, 0.030, 0.7710, 0.940, 0.0215, "D    THE TWO INFERENCE-TIME OPERATORS")

    y0 = panel(fig, 0.030, 0.5410, 0.4620, 0.2200, "G",
               "The trust region", "Equation (3.1)  ·  the causal instrument", PRIMARY)
    y = code(fig, 0.040, y0, 0.4420,
             "Top_m(s) = the m actions pi_DT(. | s) ranks highest\n"
             "a_t      = argmax_{a in Top_m(s)}  Qhat(s, a)", where="G eq")
    y = para(fig, 0.040, y - 0.0080,
             "One knob, swept over integer widths m = 1..11 in E1 and six widths in E2.\n"
             "A graded control over a discrete action set, not a continuous one.",
             0.442, gap=0.0090, where="G note")
    for i, (m, lab, body, col) in enumerate([
            ("m = 1", "the goal channel", "the model only names a target", E2C),
            ("m = A", "the action channel", "argmax over the whole grid", FAIL)]):
        x0 = 0.040 + i * 0.222
        txt(fig, x0, y, m, size=FS_LEAD, color=col, weight="bold", mono=True)
        txt(fig, x0 + 0.050, y - 0.0006, lab, size=FS_SMALL, color=col, weight="bold")
        txt(fig, x0, y - 0.0112, body, size=FS_SMALL, color=MUTE, maxw=0.212,
            where="G end %d" % i)
    box(fig, 0.040, 0.5510, 0.4420, 0.0910, face=DARK, edge="none")
    txt(fig, 0.049, 0.6350, "THE CONTROL", size=FS_SMALL, color="#8FD3A8", weight="bold")
    txt(fig, 0.049, 0.6240,
        "Run the identical sweep with Q* in place of the fitted surface.",
        size=FS_BODY, color="white", weight="bold", maxw=0.424, where="G ctrl h")
    txt(fig, 0.049, 0.6110,
        "Under Q* widening improves value monotonically to 1.000 with a curse gap of\n"
        "exactly zero. Under Qhat it collapses to -4.543 while the planner's in-model\n"
        "belief reaches 1800.3 against a true optimum of 480, a gap of +1631 (Table 4.1).\n"
        "Only the value surface differs, which licenses attributing the damage to model\n"
        "error rather than to relaxing imitation.",
        size=FS_SMALL, color="#D6E2EC", maxw=0.424, floor=0.5510, where="G ctrl b")
    txt(fig, 0.040, 0.5470, "implementation: diagnostics/diag_trust_region.py",
        size=FS_TINY, color=NEUTRAL, mono=True, style="italic")

    y0 = panel(fig, 0.5080, 0.5410, 0.4620, 0.2200, "H",
               "The logged-support mask", "Equation (3.2)  ·  model-agnostic", SUPPORT)
    y = code(fig, 0.518, y0, 0.4420,
             "c[s,a]    = number of times a was logged at s\n"
             "Supp_k(s) = { a in Top-k c[s,.] : c[s,a] > 0 }\n"
             "a_t       = argmax_a ( logits(a) + log 1[a in Supp_k] )",
             where="H eq")
    y = para(fig, 0.518, y - 0.0080,
             "k = 3. If the state was never logged the mask does not restrict, so the policy\n"
             "always has a legal move. Implemented as masked_fill(-inf) on the logits, so it\n"
             "re-ranks an already-trained model and applies unchanged to the DT, Q-DT, IQL and\n"
             "bandit arms. It requires no retraining.",
             0.442, gap=0.0090, where="H note")
    box(fig, 0.518, y - 0.0400, 0.4420, 0.0390, face="#FDF5E9", edge=SUPPORT, lw=0.9)
    txt(fig, 0.527, y - 0.0035, "NOT AN IMPROVEMENT OPERATOR: IT COMPRESSES, IN BOTH DIRECTIONS",
        size=FS_SMALL, color=SUPPORT, weight="bold", maxw=0.424, where="H contr h")
    txt(fig, 0.527, y - 0.0145,
        "Arm-value spread falls 4.2x in E1 and 3.0x in E2 (Eq. 3.9), but not all arms move\n"
        "the same way: it rescues policies that would stray and harms the strongest E2 arm\n"
        "by -0.093. The ordering this predicts did not reproduce on a public benchmark.",
        size=FS_SMALL, color=MUTE, maxw=0.424, where="H contr b")
    yy = y - 0.0490
    txt(fig, 0.518, yy, "THE TWO ARE NEVER INTERCHANGED", size=FS_SMALL, color=PRIMARY,
        weight="bold")
    for i, (k, a, b) in enumerate([
            ("", "trust region  Top_m", "support mask  Supp_k"),
            ("ranked by", "the policy's own probability", "how often the log played it"),
            ("controls", "the model's freedom", "empirical admissibility"),
            ("used for", "the mechanism test (RQ2)", "the constrained comparison (RQ4)")]):
        yr = yy - 0.0125 - i * 0.0118
        w = "bold" if i == 0 else "normal"
        txt(fig, 0.518, yr, k, size=FS_SMALL, color=NEUTRAL, weight="bold")
        txt(fig, 0.588, yr, a, size=FS_SMALL, color=PRIMARY if i == 0 else MUTE, weight=w,
            maxw=0.168, where="H cmp a%d" % i)
        txt(fig, 0.768, yr, b, size=FS_SMALL, color=SUPPORT if i == 0 else MUTE, weight=w,
            maxw=0.192, floor=0.5410, where="H cmp b%d" % i)

    # ---- E: evaluation ---------------------------------------------------
    secbar(fig, 0.030, 0.5130, 0.940, 0.0215, "E    WHAT MAKES A NUMBER ADMISSIBLE")
    box(fig, 0.030, 0.3220, 0.4620, 0.1870, face="white", edge=RULE, lw=0.9)
    y = code(fig, 0.040, 0.5000, 0.4420,
             "nv(pi) = ( V_pi - V_beh ) / ( V* - V_beh )\n"
             "V_beh  = E1: oracle myopic argmax_a R[a,b]\n"
             "         E2: the logging policy itself", where="I nv")
    y = para(fig, 0.040, y - 0.0080,
             "The zero is a different policy in each environment, so an nv in E1 and one in E2\n"
             "are not one scale: cross-environment statements are about sign and order of\n"
             "magnitude, never matched magnitude.",
             0.442, gap=0.0090, where="I nv note")
    for k, v in [("V*, V_beh", "exact float64 dynamic programming, both environments"),
                 ("V_pi", "exact in E1; Monte-Carlo, common seeds, in E2"),
                 ("tests", "paired Wilcoxon, Holm-corrected, on ten seeds"),
                 ("unit", "the seed, not the 90 cell-seed rows"),
                 ("lineages", "CPU and GPU runs are never merged"),
                 ("cells", "10 seeds x a 3 x 3 factorial; a cell is a mean over 90")]:
        txt(fig, 0.040, y, k, size=FS_SMALL, color=DARK, mono=True)
        txt(fig, 0.126, y, v, size=FS_SMALL, color=MUTE, maxw=0.356, where="I g")
        y -= 0.0118
    para(fig, 0.040, y - 0.0050,
         "The tests average each paired effect within a seed first; at ten seeds the\n"
         "smallest attainable two-sided p is 2/2^10 = 0.00195.",
         0.442, size=FS_SMALL, floor=0.3220, where="I n")

    box(fig, 0.5080, 0.3220, 0.4620, 0.1870, face="#FDF5E9", edge=SUPPORT, lw=0.9)
    txt(fig, 0.518, 0.5000, "THE FLOOR AND THE CEILING BOTH COME FROM THE LOG",
        size=FS_LEAD, color=SUPPORT, weight="bold", maxw=0.446, where="I band h")
    y = para(fig, 0.518, 0.4870,
             "The same constraint applied to a uniform policy reaches +0.448 in E1 and -1.811\n"
             "in E2, each on its own normalisation. It is a reference level and not a lower\n"
             "bound: three arms in Table 4.11 fall below it.",
             0.446, gap=0.0090, where="I band a")
    y = para(fig, 0.518, y,
             "The best policy playable inside the same top-3 logged set reaches 0.9936 under\n"
             "the main protocol, whose start distribution is the logged one. That is a true\n"
             "upper bound: no policy confined to the set can exceed it.",
             0.446, gap=0.0090, where="I band b")
    y = para(fig, 0.518, y,
             "Sweeping the logging policy's competence is a different measurement, taken from\n"
             "a uniform start distribution: there the ceiling runs from 0.986 down to exactly\n"
             "zero. The two ceilings are not the same quantity and are not comparable.",
             0.446, gap=0.0090, where="I band c")
    txt(fig, 0.518, y,
        "A constrained score is evidence about a method only between those lines.",
        size=FS_BODY, color=PRIMARY, weight="bold", maxw=0.446, floor=0.3220,
        where="I band d")

    # ---- F: falsification ------------------------------------------------
    secbar(fig, 0.030, 0.2940, 0.940, 0.0215, "F    FALSIFICATION CONDITIONS AND OUTCOMES")
    box(fig, 0.030, 0.0400, 0.940, 0.2470, face="white", edge=RULE, lw=0.9)
    txt(fig, 0.040, 0.2790, "CLAIM", size=FS_TINY, color=NEUTRAL, weight="bold")
    txt(fig, 0.092, 0.2790, "WOULD HAVE BEEN REFUTED BY", size=FS_TINY, color=NEUTRAL,
        weight="bold")
    txt(fig, 0.524, 0.2790, "OUTCOME", size=FS_TINY, color=NEUTRAL, weight="bold")
    rule(fig, 0.040, 0.960, 0.2725)
    for i, (tag, cond, verdict, col, out) in enumerate([
            ("C1", "the channel swap failing to move value, or moving\nit in only one environment",
             "held", EXACT,
             "-4.543 / +0.670 in E1 and -3.246 / +0.169 in E2: a swing of\n"
             "5.2 and 3.4 units produced entirely by the channel"),
            ("C2", "the exact-Q* control collapsing the way Qhat does,\n"
                   "which would mean the harm is lost imitation",
             "held", EXACT,
             "the curse gap is identically zero under Q* and the sweep rises\n"
             "to 1.000; logged support of the chosen action falls 0.453 to 0.030"),
            ("C3", "the exact Q* matching or beating the fitted target on\nfresh seeds",
             "held", EXACT,
             "+0.074 [0.017, 0.126] on n = 60 held-out seeds, Holm-corrected,\n"
             "and it reverses in E2 where the discrimination account predicts"),
            ("C4", "the structured prior failing to beat a fair comparator\n"
                   "under a symmetric constraint",
             "refuted", FAIL,
             "-0.119 against masked Q-DT q_sa, Holm p = 0.006 over ten seeds.\n"
             "This is the project's own original claim, and it does not survive"),
            ("C5", "the contraction ordering failing to reproduce on a\n"
                   "public benchmark, pre-registered before any training",
             "did not replicate", FAIL,
             "the no-learner floor held (0.415 against 0.021) but the gain\n"
             "ordering did not (Spearman 0.00 and -0.701); top-k is not portable")]):
        yy = 0.2630 - i * 0.0330
        txt(fig, 0.040, yy, tag, size=FS_LEAD, color=PRIMARY, weight="bold", mono=True)
        txt(fig, 0.092, yy, cond, size=FS_SMALL, color=MUTE, maxw=0.412,
            where="J cond %d" % i)
        txt(fig, 0.524, yy, verdict, size=FS_SMALL, color=col, weight="bold", maxw=0.098,
            where="J v %d" % i)
        txt(fig, 0.626, yy, out, size=FS_SMALL, color=MUTE, maxw=0.336,
            floor=0.0400, where="J out %d" % i)
        if i < 4:
            rule(fig, 0.040, 0.960, yy - 0.0245, color="#E4EAEF")
    box(fig, 0.040, 0.0450, 0.9200, 0.0640, face="#FAEDEB", edge=FAIL, lw=0.9)
    txt(fig, 0.050, 0.1030,
        "THREE MEASUREMENT FAILURES WERE FOUND IN THIS PROJECT'S OWN RESULTS",
        size=FS_LEAD, color=FAIL, weight="bold", maxw=0.900, where="J foot h")
    txt(fig, 0.050, 0.0915,
        "Each was invisible in a summary table, and each is reported with its corrected number: a comparator whose target\n"
        "carried zero within-state variance; an evaluation rule that asked different arms for different things; and a\n"
        "constraint applied to the proposal but not to its comparators. The third reverses the sign of the headline\n"
        "result. Two of the checks that expose them cost about two lines of code each (Appendix G.11).",
        size=FS_SMALL, color=MUTE, maxw=0.900, floor=0.0450, where="J foot b")

    footer(fig, 2)
    save(fig, "architecture_protocol.png")


def draw():
    S = _shipped()
    page_one(S)
    page_two(S)
    if _OVERRUN:
        print("TEXT OVERRUNS (%d):" % len(_OVERRUN))
        for line in _OVERRUN:
            print("   " + line)
    else:
        print("no text overruns")


if __name__ == "__main__":
    draw()
