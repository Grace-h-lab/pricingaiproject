"""Where does the LOGGING policy itself sit on the normalised scale?

The zero of `nv` is the oracle myopic policy, not the logging policy (Chapter 3,
"Value normalisation"). That makes the logging policy's own position an ordinary
measured quantity rather than a definition -- and it is the quantity a reader needs
in order to interpret "worse than the log" statements, because nothing in the study
reports it.

The E1 log is a 50/50 mixture of two complementary region specialists, so its
expected value is the mean of the two specialists' exact values over the same start
states. Start bins follow the same convention as the Gate-2 grid
(`_traj_start_bins`), so the numbers are directly comparable with the published
per-cell results.
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: main().
#
# Exact backward induction for the logging policy itself, giving the value a constrained
# score has to be read against.
#
# Implements or follows:
#   - Fujimoto, S., Conti, E., Ghavamzadeh, M. and Pineau, J. (2019) 'Benchmarking Batch
#     Deep Reinforcement Learning Algorithms', eq. 17. arXiv:1910.01708.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import argparse
import csv
import json
import os

import numpy as np

from pricing_dt.core import config as C
from pricing_dt.core import provenance
from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.data import RegionSpecialised
from pricing_dt.experiments.experiments import _setup, _traj_start_bins


def _specialist_value_exact(mdp, low_region, init_bins, expert_q, eps=0.05):
    """Exact expected value of one region specialist.

    The specialist is stochastic -- with probability `eps` it falls back to the
    myopic action even inside its own region -- so a deterministic evaluator would
    measure the wrong policy. E1's transition is deterministic, so the expected
    value is obtained exactly by backward induction over the two-point action
    distribution rather than by Monte-Carlo.
    """
    n_bins = mdp.cfg.n_ref_bins
    mid = n_bins // 2
    V = np.zeros((mdp.H + 1, n_bins), dtype=float)
    for t in range(mdp.H - 1, -1, -1):
        for b in range(n_bins):
            a_myopic = int(mdp.R[:, b].argmax())
            in_region = (b < mid) if low_region else (b >= mid)
            q_my = mdp.R[a_myopic, b] + V[t + 1, mdp.N[a_myopic, b]]
            if not in_region:
                V[t, b] = q_my
                continue
            a_opt = int(mdp.pistar[t, b])
            a_mix = int(round((1 - expert_q) * a_myopic + expert_q * a_opt))
            q_mix = mdp.R[a_mix, b] + V[t + 1, mdp.N[a_mix, b]]
            V[t, b] = (1 - eps) * q_mix + eps * q_my
    return float(V[0, list(init_bins)].mean())


def _specialist_values(mdp, init_bins, seed, expert_q):
    return {"low_region": _specialist_value_exact(mdp, True, init_bins, expert_q),
            "high_region": _specialist_value_exact(mdp, False, init_bins, expert_q)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results_logger_value")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = C.smoke() if a.smoke else C.full()
    sizes = [100] if a.smoke else [100, 400, 1600]
    noises = [0.5] if a.smoke else [0.05, 0.20, 0.50]
    seeds = range(1 if a.smoke else a.seeds)

    os.makedirs(a.outdir, exist_ok=True)
    # Record the commit, device and library versions this run actually used; the
    # parameters alone do not pin a result. See REPRODUCE.md.
    provenance.stamp(a.outdir, replace=True)
    rows = []
    for n in sizes:
        for noise in noises:
            for seed in seeds:
                cfg.data.n_train_traj = int(n)
                mdp, _, _, _ = _setup(cfg, {"demand_noise": float(noise)}, seed=seed)
                trajs = D.make_stitching_necessary(mdp, int(n), float(noise), seed,
                                                   cfg.data.expert_q)
                init_bins = _traj_start_bins(trajs)
                v_opt = float(mdp.Vstar[0, init_bins].mean())

                def oracle_myopic(obs):
                    b, _ = mdp.decode_obs(obs)
                    return int(mdp.R[:, b].argmax())

                v_anchor, _ = mdp.evaluate_policy_fn(oracle_myopic, init_bins)
                spec = _specialist_values(mdp, init_bins, seed, cfg.data.expert_q)
                # the log is half trajectories from each specialist
                v_logger = 0.5 * (spec["low_region"] + spec["high_region"])
                rows.append(dict(
                    N=int(n), noise=float(noise), seed=int(seed),
                    v_logger=v_logger, v_anchor=float(v_anchor), v_opt=v_opt,
                    v_low=spec["low_region"], v_high=spec["high_region"],
                    nv_logger=float(M.normalised_value(v_logger, v_anchor, v_opt)),
                    nv_low=float(M.normalised_value(spec["low_region"], v_anchor, v_opt)),
                    nv_high=float(M.normalised_value(spec["high_region"], v_anchor, v_opt)),
                ))
                print(f"  N={n:5d} noise={noise:.2f} seed={seed}  "
                      f"nv_logger={rows[-1]['nv_logger']:+.4f}  "
                      f"(low {rows[-1]['nv_low']:+.4f}, high {rows[-1]['nv_high']:+.4f})")

    path = os.path.join(a.outdir, "logger_value.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    nv = np.array([r["nv_logger"] for r in rows])
    summary = dict(n_rows=len(rows), nv_logger_mean=float(nv.mean()),
                   nv_logger_sd=float(nv.std(ddof=1)) if len(nv) > 1 else 0.0,
                   nv_logger_min=float(nv.min()), nv_logger_max=float(nv.max()))
    json.dump(summary, open(os.path.join(a.outdir, "summary.json"), "w"), indent=1)
    print(f"\nlogging policy nv: mean {summary['nv_logger_mean']:+.4f} "
          f"(sd {summary['nv_logger_sd']:.4f}, range {summary['nv_logger_min']:+.4f} "
          f"to {summary['nv_logger_max']:+.4f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
