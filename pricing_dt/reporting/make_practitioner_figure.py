"""Draw Appendix D as one portrait page: the four measurements the guide rests on.

Section 5.6 already carries the ordering as Figure 5.1 and the checklist as Table 5.1, so a
second flowchart here would be the same device a third time. What neither of those carries is
the division this page is built on: two of the four checks run on the log alone, before
anything is trained, and two of them cannot be run at all without a rollout or a simulator
someone is willing to trust. That distinction is the practitioner's binding constraint and it
is drawn nowhere else in the dissertation.

Each check therefore carries four things a reader can act on: the quantity as a formula, why it
is worth computing, what it looked like when this project ran it, and the file and line that
compute it. The last is what separates a diagnostic from a recommendation.

Three claims on the page are stated deliberately weakly:

  * The action channel's effect is given as the chosen action's logged probability falling
    from 0.453 to 0.030 as the trust region opens, which is what the results support.
  * Discrimination is computed retrospectively off logged returns; a low ratio indicates a weak
    conditioning signal, but a high one is not sufficient, since realised noise also creates
    within-state spread (Section 4.4).
  * Restricting to logged support is a contraction, not an improvement operator. The page says
    so next to the floor rather than leaving the reader to infer it.

Visual hierarchy, palette and the width/floor checks follow Figures B.1 and C.1-C.2.

Run from anywhere:  python pricing_dt/reporting/make_practitioner_figure.py
"""
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PRIMARY = "#0F3557"
E1C = "#1F6FB2"
E2C = "#6B4E9B"
EXACT = "#2E7D4F"
FAIL = "#B4342A"
SUPPORT = "#B5741C"
NEUTRAL = "#5A6B78"
CODEBG = "#EDF1F5"
RULE = "#C3D0DB"
MUTE = "#3E4C57"
DARK = "#16202A"
LOGBG = "#EAF1F8"          # what the log alone can answer
DEPBG = "#F8F2E8"          # what needs a rollout
SCOPEBG = "#F3F6F9"

FS_TITLE = 16.0
FS_SEC, FS_CHECK = 8.4, 9.0
FS_LEAD, FS_BODY = 6.4, 6.0
FS_CODE, FS_SMALL, FS_TINY = 5.6, 5.4, 5.0

PAGE_W_IN, PAGE_H_IN = 8.27, 11.69
CHAR_EM, LINE_SP = 0.52, 1.42
MONO_EM = 0.60

ARMS = [                       # Table 4.11, in the order the table prints them
    ("Estimate-then-optimise unconstrained support top3", "estimate-then-optimise"),
    ("IQL expectile0.7 beta3 support top3", "IQL"),
    ("Q-DT fixed q_sa support top3", "Q-DT q_sa"),
    ("Q-DT fixed td denoised support top3", "Q-DT td"),
    ("Estimate-then-optimise structured support top3", "EtO structured"),
    ("Structured DT support top3", "structured DT"),
    ("Vanilla DT support top3", "vanilla DT"),
    ("Behaviour cloning support top3", "behaviour cloning"),
    ("Bandit IPS support top3", "CRM bandit"),
    ("Bandit DM support top3", "direct-method bandit"),
]
# The floor and the ceiling are the quantities of Equations (3.6) and (3.10) as reported
# in Section 4.6.2. Unlike the arm values below they are not in this file, because both
# are backward inductions under the true reward rather than an evaluated policy.
FLOOR, CEILING, LOGGER = 0.448, 0.9936, 0.4235

HERE = os.path.dirname(os.path.abspath(__file__))
# Two levels up: this script sits in pricing_dt/reporting/, not beside the results tree.
ROOT = os.path.dirname(os.path.dirname(HERE))


def at(path, anchor):
    """`path:line` for the line where `anchor` is defined, read at draw time.

    The page cites four places in the code. Hard-coded line numbers go stale the first
    time anything above them is edited, and nothing would report it, so they are resolved
    from the source instead and a missing anchor stops the build.
    """
    full = os.path.join(ROOT, path)
    with io.open(full, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if line.lstrip().startswith(anchor):
                return "%s:%d" % (path.split("/")[-1], i)
    raise SystemExit("anchor %r not found in %s" % (anchor, path))
OUT = os.path.join(ROOT, "results", "figures", "practitioner_checks.png")

_OVERRUN = []


def masked_arms():
    """The ten masked arms of Table 4.11, averaged over the full grid, read not typed."""
    import collections, csv, io, statistics
    path = os.path.join(ROOT, "results_family_table_20260825", "four_family_raw.csv")
    agg = collections.defaultdict(list)
    with io.open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            agg[row["method"]].append(float(row["nv"]))
    return [(label, statistics.mean(agg[key])) for key, label in ARMS]


def fit(s, size, maxw, where, mono=False):
    per = (size * (MONO_EM if mono else CHAR_EM) / 72.0) / PAGE_W_IN
    for line in s.split("\n"):
        if len(line) * per > maxw:
            _OVERRUN.append("%-20s wide %5.3f > %5.3f  %r"
                            % (where, len(line) * per, maxw, line[:40]))


def drop(s, size, y, floor, where):
    bottom = y - (s.count("\n") + 1) * (size * LINE_SP / 72.0) / PAGE_H_IN
    if bottom < floor:
        _OVERRUN.append("%-20s deep %5.3f < %5.3f  %r"
                        % (where, bottom, floor, s.split("\n")[0][:40]))


def lh(size):
    return (size * LINE_SP / 72.0) / PAGE_H_IN


def box(fig, x, y, w, h, face=SCOPEBG, edge=RULE, lw=0.8, z=1):
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


def rule(fig, x0, x1, y, color=RULE, lw=0.7):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], transform=fig.transFigure,
                              color=color, linewidth=lw, zorder=3))


def secbar(fig, x, y, w, h, text, note="", face=PRIMARY):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0", transform=fig.transFigure,
        facecolor=face, edgecolor="none", zorder=2))
    fig.text(x + 0.010, y + h / 2.0, text, ha="left", va="center", fontsize=FS_SEC,
             color="white", weight="bold", zorder=3)
    if note:
        fig.text(x + w - 0.010, y + h / 2.0, note, ha="right", va="center",
                 fontsize=FS_TINY, color="#C6D5E2", style="italic", zorder=3)


def code(fig, x, y, w, lines, size=FS_CODE, pad=0.0040, where=""):
    n = lines.count("\n") + 1
    h = n * lh(size) + 2 * pad
    box(fig, x, y - h, w, h, face=CODEBG, edge="none", lw=0, z=2)
    txt(fig, x + pad, y - pad, lines, size=size, color=DARK, mono=True,
        maxw=w - 2 * pad, where=where)
    return y - h


def value_axis(fig, x, y, w, h):
    """Every masked arm on one nv axis, against the floor and the ceiling.

    Drawn because the alternative is asking the reader to believe a sentence. Three of the
    ten arms sit below the no-learner floor, which is what makes it a benchmark and not a
    lower bound, and that is easier to see than to accept on assertion.
    """
    ax = fig.add_axes([x, y, w, h], zorder=5)
    ax.set_facecolor("white")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_yticks([])
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.9, 1.7)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=4.4, colors=MUTE, length=2, width=0.6, pad=1.0)

    ax.axvline(CEILING, color=SUPPORT, lw=1.0, ls=(0, (2, 1.5)))
    ax.axvline(FLOOR, color=SUPPORT, lw=1.2)
    ax.axvline(LOGGER, color=RULE, lw=0.8, ls=":")
    for val, lab, col, dy in ((FLOOR, "floor 0.448", SUPPORT, 1.42),
                              (CEILING, "ceiling 0.9936", SUPPORT, 0.62)):
        ax.annotate(lab, xy=(val, dy), fontsize=4.4, color=col, weight="bold",
                    ha="right" if val > 0.9 else "left",
                    xytext=(-2 if val > 0.9 else 2, 0), textcoords="offset points",
                    va="center")

    below = 0
    for _, v in masked_arms():
        low = v < FLOOR
        below += low
        ax.plot([v], [-0.30], marker="o", ms=3.4, mew=0.0,
                color=FAIL if low else PRIMARY, zorder=6)
    ax.annotate("%d of 10 masked arms fall below the floor" % below, xy=(0.0, 0.62),
                fontsize=4.4, color=FAIL, weight="bold", va="center")
    return ax


def check(fig, y, h, n, title, ask, accent, bg, formula, why, read, looked, impl, floor,
          chart=None):
    """One diagnostic.

    Five things, because four of them are not enough to act on: the quantity, why it is worth
    computing, WHAT TO CONCLUDE from a high or a low value, what it showed when this project
    ran it, and the file and line that compute it. The fourth is evidence and the third is the
    guide; keeping them in separate columns stops the second being read as the first.
    """
    box(fig, 0.030, y, 0.940, h, face="white", edge=RULE, lw=0.9)
    box(fig, 0.030, y, 0.0090, h, face=accent, edge="none", lw=0, z=2)
    fig.patches.append(plt.Circle((0.0330, y + h - 0.0125), 0.0098,
                                  transform=fig.transFigure, facecolor=accent,
                                  edgecolor="none", zorder=5))
    fig.text(0.0330, y + h - 0.0125, n, ha="center", va="center", fontsize=7.6,
             color="white", weight="bold", zorder=6)
    txt(fig, 0.052, y + h - 0.0055, title, size=FS_CHECK, color=PRIMARY, weight="bold",
        maxw=0.300, where=n + " title")
    txt(fig, 0.052, y + h - 0.0200, ask, size=FS_SMALL, color=NEUTRAL, style="italic",
        maxw=0.300, where=n + " ask")

    yy = code(fig, 0.052, y + h - 0.0330, 0.4180, formula, where=n + " eq")
    yy = yy - 0.0090
    txt(fig, 0.052, yy, why, size=FS_BODY, color=MUTE, maxw=0.426, where=n + " why")
    yy -= (why.count("\n") + 1) * lh(FS_BODY) + 0.0075

    # the reading strip: what a high or a low value licenses you to say
    box(fig, 0.052, yy - (read.count("\n") + 1) * lh(FS_SMALL) - 0.0090, 0.4180,
        (read.count("\n") + 1) * lh(FS_SMALL) + 0.0130, face=bg, edge="none", lw=0)
    txt(fig, 0.060, yy - 0.0020, "READ IT", size=FS_TINY, color=accent, weight="bold")
    txt(fig, 0.060, yy - 0.0120, read, size=FS_SMALL, color=MUTE, maxw=0.402,
        floor=floor, where=n + " read")

    txt(fig, 0.500, y + h - 0.0330, "WHAT IT LOOKED LIKE HERE", size=FS_TINY, color=NEUTRAL,
        weight="bold")
    ytxt = y + h - 0.0450
    if chart is not None:
        chart(fig, 0.508, y + h - 0.0900, 0.268, 0.0390)
        ytxt = y + h - 0.0990
    txt(fig, 0.500, ytxt, looked, size=FS_SMALL, color=MUTE, maxw=0.285,
        floor=floor, where=n + " looked")

    txt(fig, 0.812, y + h - 0.0330, "IMPLEMENTATION", size=FS_TINY, color=NEUTRAL,
        weight="bold")
    txt(fig, 0.812, y + h - 0.0450, impl, size=FS_TINY, color=DARK, mono=True,
        maxw=0.150, floor=floor, where=n + " impl")


def draw():
    """One page for one reader: whoever has to decide whether a reported score is evidence.

    Four equal bands, equal gutters. The advice sits inside the check it belongs to, because a reader
    acting on check 4 should not have to remember a line printed under check 1.
    """
    fig = plt.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
    fig.patch.set_facecolor("white")

    RULE_Y = 0.9280
    BAR_H, GAP, H = 0.0215, 0.0165, 0.1480
    BAR_A = 0.9020
    C1 = BAR_A - GAP - H
    C2 = C1 - GAP - H
    BAR_B = C2 - GAP - BAR_H
    C3 = BAR_B - GAP - H
    C4 = C3 - GAP - H

    # ---- header ------------------------------------------------------
    txt(fig, 0.030, 0.9945, "Four diagnostics and the cost of each",
        size=14.0, color=PRIMARY, weight="bold", maxw=0.470, where="title")
    txt(fig, 0.030, 0.9700,
        "For anyone evaluating an offline-learned policy: the four numbers that\n"
        "decide whether a reported score is evidence about the method, what each\n"
        "one costs to obtain, and what it does not license.",
        size=FS_LEAD, color=NEUTRAL, maxw=0.560, floor=RULE_Y, where="subtitle")

    txt(fig, 0.606, 0.9930, "WHAT TRANSFERS", size=FS_TINY, color=EXACT, weight="bold")
    txt(fig, 0.606, 0.9820,
        "The mechanism. A model whose error is\n"
        "not bounded on the actions a policy\n"
        "will select has that error sought out\n"
        "by the argmax, not averaged over.\n"
        "Isolated by control: under an exact\n"
        "Q* the sweep is monotone, gap zero.",
        size=FS_TINY, color=MUTE, maxw=0.182, floor=RULE_Y, where="transfers")
    txt(fig, 0.800, 0.9930, "WHAT DOES NOT", size=FS_TINY, color=FAIL, weight="bold")
    txt(fig, 0.800, 0.9820,
        "The magnitudes, and the ranking of\n"
        "methods. A pre-registered replication\n"
        "got the floor back, not the gain\n"
        "ordering (Spearman 0.00 and -0.701),\n"
        "and no estimator family ranked seven\n"
        "policies on this study's own real log.",
        size=FS_TINY, color=MUTE, maxw=0.170, floor=RULE_Y, where="not")
    rule(fig, 0.030, 0.970, RULE_Y, color=PRIMARY, lw=1.3)

    # ---- A: what the log alone answers -------------------------------
    secbar(fig, 0.030, BAR_A, 0.940, BAR_H, "A    FROM THE LOG ALONE",
           "both run before you train anything: no simulator, no ground truth, no rollout")

    check(fig, C1, H, "1", "Coverage",
          "How much of the intended policy is extrapolation, not evidence?",
          E1C, LOGBG,
          "off(pi)   = P( a_chosen not in supp(log, state) )\n"
          "Supp_k(s) = the k most frequently logged actions at s",
          "Every value estimate at a state-action pair the log does not cover is an\n"
          "extrapolation, and nothing in the fitting procedure marks it as one. This is the\n"
          "cheapest check on the page and the one most often skipped.",
          "High: the score describes extrapolation, not the log. Report the rate beside it.\n"
          "Before choosing any method, check whether the actions you want were already being\n"
          "played: here the myopic optimum was in the log 96.6% of the time and unused.",
          "Opening the admissible set from one action\n"
          "to eleven dropped the chosen action's logged\n"
          "probability from 0.453 to 0.030 and took\n"
          "true policy value from +0.670 to -4.543. The\n"
          "identical sweep under an exact Q* rose to\n"
          "+1.000 instead: coverage, not permissiveness,\n"
          "is what got spent.",
          "pricing_dt/core/\n  " + at("pricing_dt/core/dt.py", "def _supported_actions") + "\n\n_supported_actions(\n  counts, t, b,\n"
          "  n_actions,\n  topk=...)",
          C1)

    check(fig, C2, H, "2", "Discrimination",
          "Can a return target separate actions taken in the same state?",
          E1C, LOGBG,
          "rho = E_b[ sd(g_0 | b) ]  /  sd_b( E[g_0 | b] )        Eq (3.7)",
          "A return-conditioned method needs targets that differ WITHIN a state, not merely\n"
          "across states. Near zero, the conditioning signal carries no action information\n"
          "at all, however accurate the target is.",
          "Near zero: no return-conditioned method can use the target, and its accuracy is\n"
          "beside the point. High is necessary but not sufficient, because realised noise\n"
          "also creates within-state spread. As computed here the ratio is retrospective.",
          "The legacy comparator's within-state spread\n"
          "is exactly zero, which left every arm\n"
          "indistinguishable and is what sent the\n"
          "earlier reading of this experiment wrong.\n"
          "Discrimination tracked value where accuracy\n"
          "did not, which is the finding that was\n"
          "pre-registered and confirmed.",
          "pricing_dt/\n  diagnostics/\n  " + at("pricing_dt/diagnostics/diag_target_stats.py", "def stats_for") + "\n\nstats_for(\n"
          "  rtg_list, trajs,\n  mdp)",
          C2)

    # ---- B: what a rollout is needed for -----------------------------
    secbar(fig, 0.030, BAR_B, 0.940, BAR_H, "B    ONLY ONCE SOMETHING IS DEPLOYED",
           "these cost a rollout, or a simulator you are willing to trust. "
           "There is no log-only substitute.", SUPPORT)

    check(fig, C3, H, "3", "Belief drift",
          "Is the planner's expectation separating from its takings?",
          SUPPORT, DEPBG,
          "gap(pi) = V_model(pi) - V_true(pi)                      Eq (3.4)",
          "This is the only quantity here that makes the failure visible. No check on the\n"
          "model's own predictive fit will show it: a model can be accurate on average and\n"
          "still be optimistic exactly where the argmax lands.",
          "Large and positive: the planner is optimistic where it acts, and predictive fit\n"
          "will not reveal it. Without this number you cannot claim the model's error is\n"
          "bounded on the actions the policy selects, which is the claim the argmax needs.",
          "The action channel expected 1800.3 against a\n"
          "true optimum of 480, a gap of +1631, while\n"
          "the action it chose carried 0.030 logged\n"
          "probability. Its predictive accuracy raised\n"
          "no flag at any point, which is why the\n"
          "quantity has to be measured against realised\n"
          "takings rather than against fit.",
          "pricing_dt/\n  diagnostics/\n  diag_demand_curve\n  _amplification\n  " + at("pricing_dt/diagnostics/diag_demand_curve_amplification.py", "init_inmodel_gap=").split("amplification")[-1],
          C3)

    check(fig, C4, H, "4", "The no-learner floor",
          "Report it beside every constrained score you publish.",
          SUPPORT, DEPBG,
          "floor(k) = nv( Unif( Supp_k ) )                          Eq (3.6)",
          "A constrained policy's score is evidence about the LEARNER only above this line.\n"
          "Beneath it you have measured the constraint. This is the most portable item on\n"
          "the page: it depends on none of the claims that failed to replicate.",
          "At or below the floor you measured the restriction, not the learner; above it,\n"
          "the increment is what the learner contributed. Report floor, ceiling and score\n"
          "together. Restriction is not free: it cost the strongest E2 arm -0.093.",
          "Every masked arm of Table 4.11 on one scale.\n"
          "Uniform choice reached +0.448 here and\n"
          "-1.811 in inventory. Arms had been read as\n"
          "improvements before that line was drawn.",
          "pricing_dt/\n  diagnostics/\n  " + at("pricing_dt/diagnostics/diag_env2_support.py", "def mc_eval_random") + "\n\nuniform policy,\n"
          "  masked",
          C4, chart=value_axis)

    # ---- what to report ----------------------------------------------
    RP_Y, RP_H = C4 - GAP - 0.0560, 0.0560
    box(fig, 0.030, RP_Y, 0.940, RP_H, face="#EEF3EE", edge=EXACT, lw=1.0)
    txt(fig, 0.040, RP_Y + RP_H - 0.0060, "WHAT TO REPORT BESIDE ANY CONSTRAINED SCORE",
        size=FS_LEAD, color=EXACT, weight="bold", maxw=0.500, where="report h")
    for i, (n, item) in enumerate([
            ("1", "the off-support rate of the actions the policy actually selects"),
            ("2", "the no-learner floor under the identical constraint"),
            ("3", "the ceiling that constraint fixes, where ground truth allows it"),
            ("4", "the number of independent seeds the test used, not the number of runs")]):
        x0 = 0.040 + (i % 2) * 0.468
        yy = RP_Y + RP_H - 0.0195 - (i // 2) * 0.0110
        txt(fig, x0, yy, n, size=FS_SMALL, color=EXACT, weight="bold", mono=True)
        txt(fig, x0 + 0.014, yy, item, size=FS_SMALL, color=MUTE, maxw=0.440,
            where="report %d" % i)
    txt(fig, 0.040, RP_Y + RP_H - 0.0430,
        "A score without the first two overstates what the method contributed; without the "
        "third it cannot be placed; without the fourth it cannot be checked.",
        size=FS_SMALL, color=PRIMARY, weight="bold", maxw=0.920, floor=RP_Y,
        where="report kicker")

    # ---- scope and the audit that is not this one ---------------------
    SC_H = 0.0900
    SC_Y = RP_Y - GAP - SC_H
    box(fig, 0.030, SC_Y, 0.6180, SC_H, face=SCOPEBG, edge=RULE, lw=0.9)
    txt(fig, 0.040, SC_Y + SC_H - 0.0060, "SCOPE", size=FS_LEAD, color=PRIMARY,
        weight="bold")
    txt(fig, 0.040, SC_Y + SC_H - 0.0180,
        "This is a diagnostic sequence, not a deployment rule.\n"
        "·  The mechanism evidence comes from two controlled finite-horizon environments.\n"
        "·  Support effects depend on logged action coverage and on how good the logging\n"
        "    policy was; both are properties of the log, not of the method.\n"
        "·  The support-contraction account failed a pre-registered external replication.\n"
        "·  Section 5.6 states the same checklist compactly as Table 5.1, and Figure 5.1\n"
        "    gives the order in which the questions have to be asked.",
        size=FS_SMALL, color=MUTE, maxw=0.598, floor=SC_Y, where="scope")

    box(fig, 0.6600, SC_Y, 0.3100, SC_H, face="#FAEDEB", edge=FAIL, lw=0.9)
    txt(fig, 0.670, SC_Y + SC_H - 0.0060, "A SEPARATE AUDIT IS REQUIRED", size=FS_LEAD,
        color=FAIL, weight="bold", maxw=0.290, where="ethics h")
    txt(fig, 0.670, SC_Y + SC_H - 0.0180,
        "A policy confined to logged support inherits the historical\n"
        "action set, including any option never offered at a state or\n"
        "to a group, though not the logging policy's action\n"
        "frequencies. \"Safe\" here means reduced exposure to\n"
        "unsupported model extrapolation, not fairness: personalised\n"
        "pricing raises questions of discrimination and repeated\n"
        "algorithmic pricing can collude, and this work resolves\n"
        "neither (Sections 5.6 and 2.7).",
        size=FS_TINY, color=MUTE, maxw=0.290, floor=SC_Y, where="ethics b")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, facecolor="white")
    plt.close(fig)
    if _OVERRUN:
        print("TEXT OVERRUNS (%d):" % len(_OVERRUN))
        for line in _OVERRUN:
            print("   " + line)
    else:
        print("no text overruns")
    print("bottom margin %.4f" % SC_Y)
    print("saved %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    draw()
