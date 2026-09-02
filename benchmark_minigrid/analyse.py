"""Apply the decision rule of PREREGISTRATION_SUPPORT_D4RL.md to the run output.

Computes only the four statistics declared in advance. Nothing is added here that
was not pre-specified; anything else would be descriptive by that document's own
terms.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LEARNERS = ["BC", "CQL", "IQL", "DT", "Q-DT"]
FLOOR = "random (no learner)"


def block(d, mask, off_col):
    """Per-seed mask gain, compression, and the off-support/gain rank correlation.

    Three of the dissertation's quantities, on the benchmark logs:
      gain        Equation (3.8), masked minus bare normalised value, per arm.
      compression Equation (3.9), sd across arms bare / sd across arms masked. The
                  MEDIAN over seeds is reported: sd_masked can approach zero, and the
                  mean of such a ratio is not a summary of anything.
      off         Equation (3.5), off-support rate, always read from the BARE arm, so
                  the correlation asks whether the arms the mask had most to correct
                  are the arms it helped.
    """
    out = []
    for seed, g in d.groupby("seed"):
        bare = g[g["mask"] == "bare"].set_index("arm")
        msk = g[g["mask"] == mask].set_index("arm")
        b = bare.loc[LEARNERS, "ret_mean"].to_numpy()
        m = msk.loc[LEARNERS, "ret_mean"].to_numpy()
        gain = m - b
        off = bare.loc[LEARNERS, off_col].to_numpy()
        rho = np.nan
        if np.ptp(off) > 0 and np.ptp(gain) > 0:
            rho = spearmanr(off, gain).statistic
        out.append(dict(seed=seed, sd_bare=b.std(ddof=1), sd_masked=m.std(ddof=1),
                        # guarded: sd_masked can be ~0, and the mean of such a
                        # ratio is meaningless, so the median is reported
                        compression=(b.std(ddof=1) / m.std(ddof=1)
                                     if m.std(ddof=1) > 1e-4 else np.nan),
                        spearman=rho,
                        floor_bare=bare.loc[FLOOR, "ret_mean"],
                        floor_masked=msk.loc[FLOOR, "ret_mean"],
                        best_learner_masked=m.max(),
                        admissible=msk.loc[LEARNERS[0],
                                           "size_top3" if mask == "top3" else "size_tau"]))
    return pd.DataFrame(out)


def main(path):
    d = pd.read_csv(path)
    lines = []
    P = {}
    for mask, off_col in (("top3", "off_bare_top3"), ("bcq_tau", "off_bare_bcq_tau")):
        lines.append(f"\n{'=' * 78}\nMASK = {mask}\n{'=' * 78}")
        for tag in ("expert", "random"):
            sub = d[d["dataset"] == tag]
            if sub.empty:
                continue
            r = block(sub, mask, off_col)
            P[(mask, tag)] = r
            lines.append(f"\n--- logger: {tag}")
            lines.append(f"  mean admissible-set size      {r['admissible'].mean():.2f} of 7")
            lines.append(f"  sd across learners, bare      {r['sd_bare'].mean():.4f}")
            lines.append(f"  sd across learners, masked    {r['sd_masked'].mean():.4f}")
            comp = np.nanmedian(r["compression"]) if r["compression"].notna().any() else np.nan
            lines.append(f"  COMPRESSION (bare/masked)     "
                         f"{'undefined (sd_masked~0)' if np.isnan(comp) else f'{comp:.2f}x (median)'}"
                         f"   per-seed {np.round(r['compression'].to_numpy(), 2)}")
            rho = np.nanmean(r["spearman"]) if r["spearman"].notna().any() else np.nan
            lines.append(f"  Spearman(off-support, gain)   "
                         f"{'undefined (no variation in gain)' if np.isnan(rho) else f'{rho:+.3f}'}"
                         f"   per-seed {np.round(r['spearman'].to_numpy(), 2)}")
            lines.append(f"  no-learner floor  bare {r['floor_bare'].mean():.4f}"
                         f" -> masked {r['floor_masked'].mean():.4f}")
            lines.append(f"  best learner masked           {r['best_learner_masked'].mean():.4f}")
            per = (d[(d.dataset == tag) & (d["mask"].isin(["bare", mask]))]
                   .pivot_table(index="arm", columns="mask", values="ret_mean"))
            per["gain"] = per[mask] - per["bare"]
            off = (d[(d.dataset == tag) & (d["mask"] == "bare")]
                   .groupby("arm")[off_col].mean())
            per["off_support"] = off
            lines.append("  per-arm:")
            for arm in LEARNERS + [FLOOR]:
                if arm in per.index:
                    q = per.loc[arm]
                    lines.append(f"    {arm:20s} bare {q['bare']:+.4f} -> masked "
                                 f"{q[mask]:+.4f}   gain {q['gain']:+.4f}   "
                                 f"off-support {q['off_support']:.3f}")

    lines.append(f"\n{'=' * 78}\nPRE-REGISTERED DECISION RULE\n{'=' * 78}")
    verdict = {}
    for mask in ("top3", "bcq_tau"):
        e, r = P.get((mask, "expert")), P.get((mask, "random"))
        if e is None or r is None:
            continue
        ce = np.nanmedian(e["compression"]) if e["compression"].notna().any() else np.nan
        cr = np.nanmedian(r["compression"]) if r["compression"].notna().any() else np.nan
        rho_e = np.nanmean(e["spearman"]) if e["spearman"].notna().any() else np.nan
        # an undefined statistic cannot support a prediction, so it counts as failed
        p1 = bool(ce > 1)
        p2 = bool(rho_e > 0)
        p3 = bool(cr < ce)
        p4 = e["floor_masked"].mean() > r["floor_masked"].mean()
        verdict[mask] = (p1, p2, p3, p4)
        lines.append(f"\n  mask = {mask}")
        def fmt(v, unit="x"):
            return "undefined" if np.isnan(v) else f"{v:.2f}{unit}"

        lines.append(f"    P1 compression > 1 (expert)            "
                     f"{'HOLDS' if p1 else 'REFUTED'}   ({fmt(ce)})")
        lines.append(f"    P2 Spearman(off-support, gain) > 0     "
                     f"{'HOLDS' if p2 else 'REFUTED'}   ({fmt(rho_e, '')})")
        lines.append(f"    P3 compression smaller under random    "
                     f"{'HOLDS' if p3 else 'REFUTED'}   ({fmt(cr)} vs {fmt(ce)})")
        lines.append(f"    P4 floor higher under expert logger    {'HOLDS' if p4 else 'REFUTED'}"
                     f"   ({e['floor_masked'].mean():.4f} vs {r['floor_masked'].mean():.4f})")
        if p1 and p2 and p3:
            lines.append("    => contraction account externally validated on this benchmark")
        elif p1 and p2:
            lines.append("    => phenomenon replicates, explanation does not; account "
                         "downgraded to a description")
        else:
            lines.append("    => contraction result does not generalise beyond E1/E2; "
                         "restrict the claim by name")
    lines.append("\n  n = 3 training seeds: these are directional outcomes against "
                 "pre-declared predictions, not significance tests.")
    print("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    txt = main(sys.argv[1])
    if len(sys.argv) > 2:
        open(sys.argv[2], "w", encoding="utf-8").write(txt)
