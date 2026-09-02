"""Gate 3A real-data OPE sensitivity diagnostics.

The dissertation's real-data track is deliberately descriptive because Online
Retail II has no ground-truth counterfactual policy value. This script strengthens
that descriptive check with standard OPE diagnostics:

  * DM / IPS / SNIPS / DR estimates for logged, vanilla-DT, and BC policies.
  * Clipping sensitivity over cumulative importance weights.
  * Effective sample size and action-overlap diagnostics.
  * Fixed-nuisance bootstrap CIs over held-out real episodes.

The bootstrap resamples episodes after fitting pi_b and qhat once on the held-out
set. It is a sensitivity diagnostic, not a full nested nuisance-refit interval.

References:
  - Chen, D. (2012), Online Retail II.
  - Jiang and Li (2016), sequential doubly robust OPE.
  - Uehara, Shi and Kallus (2026), recent OPE review.
  - Shimizu et al. (2025), future/non-stationary OPE.

Run:  python -m pricing_dt.diagnostics.diag_gate3_real_ope --preset full --outdir results_gate3_real_ope
      Needs the Online Retail II workbook, which is not distributed: see Appendix H.3 for
      the DOI, and pass its path with --real-data. Without it the run stops at the loader.
"""
import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.core import ope
from pricing_dt.core import realdata as RD
from pricing_dt.core.baselines import qnet_action_probs, train_bc, train_iql
from pricing_dt.core.dt import _supported_actions, dt_action_probs, train_dt
from pricing_dt.core.relabel import logged_rtg


class _RealBins:
    """Reference-price binning for the real log, mirroring `simulator.obs_to_bins`.

    Real trajectories carry the same observation as the simulator — scaled
    reference price, scaled time — but their `ref_bins` field is unused, so the
    bin is recovered from the observation against the same price-bin edges the
    actions were discretised on.
    """

    def __init__(self, edges, horizon, n_prices):
        self.edges = np.asarray(edges, dtype=float)
        self.H = int(horizon)
        self.cfg = SimpleNamespace(n_prices=int(n_prices), n_ref_bins=int(n_prices))

    def obs_to_bins(self, obs):
        obs = np.asarray(obs, dtype=float)
        idx = np.digitize(obs[:, 0], self.edges) - 1
        return np.clip(idx, 0, self.cfg.n_ref_bins - 1).astype(int)

    def step_index(self, obs):
        t = int(round(float(np.asarray(obs, dtype=float)[1]) * max(1, self.H - 1)))
        return min(max(t, 0), self.H - 1)


def _real_support_counts(trajs, binner):
    """Logged action counts per (t, reference bin), from the TRAINING split."""
    counts = np.zeros((binner.H, binner.cfg.n_ref_bins, binner.cfg.n_prices))
    for tr in trajs:
        bins = binner.obs_to_bins(tr.obs)
        for t, a in enumerate(tr.actions):
            if t < binner.H:
                counts[t, int(bins[t]), int(a)] += 1.0
    return counts


def _masked_probs(prob_fn, binner, counts, topk):
    """Restrict a policy's action distribution to the logged support.

    Off-policy evaluation needs a distribution rather than an argmax, so the
    simulator's masked re-ranking becomes a renormalisation over the same
    admissible set. The set itself is `dt._supported_actions`, unchanged.
    """

    def fn(obs):
        p = np.asarray(prob_fn(obs), dtype=float)
        b = int(binner.obs_to_bins(np.asarray(obs, dtype=float)[None])[0])
        sup = _supported_actions(counts, binner.step_index(obs), b,
                                 binner.cfg.n_prices, topk=topk)
        q = np.where(sup, p, 0.0)
        s = q.sum()
        return q / s if s > 0 else p

    return fn


def _safe_probs(fn, obs, n_actions, clip=1e-8):
    p = np.asarray(fn(obs), dtype=float)
    if p.shape[0] != n_actions:
        raise ValueError(f"Policy returned {p.shape[0]} actions, expected {n_actions}")
    p = np.maximum(p, clip)
    return p / p.sum()


def _vhat(pi_e, qhat, obs, n_actions):
    pe = _safe_probs(pi_e, obs, n_actions)
    return float(sum(pe[a] * qhat(obs, a) for a in range(n_actions)))


def _trajectory_terms(trajs, pi_e, pi_b, qhat, n_actions, w_clip=10.0):
    dm_vals, ips_vals, dr_vals = [], [], []
    snips_num, snips_den = [], []
    step_weights, terminal_weights = [], []
    pb_at_target_argmax, pe_at_logged = [], []

    for tr in trajs:
        H = len(tr.actions)
        dm_vals.append(_vhat(pi_e, qhat, tr.obs[0], n_actions))
        w = 1.0
        ips_total, sn_num, sn_den = 0.0, 0.0, 0.0
        dr_acc = dm_vals[-1]
        for t in range(H):
            pe = _safe_probs(pi_e, tr.obs[t], n_actions)
            pb = _safe_probs(pi_b, tr.obs[t], n_actions)
            a = int(tr.actions[t])
            rho = pe[a] / max(pb[a], 1e-8)
            w = min(w * rho, w_clip)
            step_weights.append(w)
            pe_at_logged.append(pe[a])
            pb_at_target_argmax.append(pb[int(np.argmax(pe))])

            ips_total += w * float(tr.rewards[t])
            sn_num += w * float(tr.rewards[t])
            sn_den += w

            q_sa = qhat(tr.obs[t], a)
            dr_acc += w * (float(tr.rewards[t]) - q_sa)
            if t + 1 < H:
                dr_acc += w * _vhat(pi_e, qhat, tr.obs[t + 1], n_actions)
        terminal_weights.append(w)
        ips_vals.append(ips_total)
        dr_vals.append(dr_acc)
        snips_num.append(sn_num)
        snips_den.append(sn_den)

    return {
        "dm": np.asarray(dm_vals, dtype=float),
        "ips": np.asarray(ips_vals, dtype=float),
        "dr": np.asarray(dr_vals, dtype=float),
        "snips_num": np.asarray(snips_num, dtype=float),
        "snips_den": np.asarray(snips_den, dtype=float),
        "step_weights": np.asarray(step_weights, dtype=float),
        "terminal_weights": np.asarray(terminal_weights, dtype=float),
        "pb_at_target_argmax": np.asarray(pb_at_target_argmax, dtype=float),
        "pe_at_logged": np.asarray(pe_at_logged, dtype=float),
    }


def _ess(weights):
    """Kish effective sample size, (sum w)^2 / sum w^2.

    Divided by the number of weights by the caller, this is the effective sample
    fraction of Equation (3.11). `step_ess_frac` weights steps and is the figure
    Section 5.5 quotes; `terminal_ess_frac` weights whole episodes.
    """
    weights = np.asarray(weights, dtype=float)
    denom = np.square(weights).sum()
    if denom <= 0:
        return float("nan")
    return float(weights.sum() ** 2 / denom)


def _mean_ci(vals, rng, n_boot):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean()
        for _ in range(n_boot)
    ])
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _snips_ci(num, den, horizon, rng, n_boot):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    idx = np.arange(len(num))
    point = horizon * num.sum() / max(den.sum(), 1e-8)
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(horizon * num[b].sum() / max(den[b].sum(), 1e-8))
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _overlap_rows(test_trajs, policies, pi_b, n_actions):
    rows = []
    for name, pi_e in policies.items():
        pb_argmax, pe_logged, ent_e, ent_b = [], [], [], []
        for tr in test_trajs:
            for obs, a in zip(tr.obs, tr.actions):
                pe = _safe_probs(pi_e, obs, n_actions)
                pb = _safe_probs(pi_b, obs, n_actions)
                pb_argmax.append(pb[int(np.argmax(pe))])
                pe_logged.append(pe[int(a)])
                ent_e.append(-float(np.sum(pe * np.log(np.maximum(pe, 1e-12)))))
                ent_b.append(-float(np.sum(pb * np.log(np.maximum(pb, 1e-12)))))
        pb_argmax = np.asarray(pb_argmax)
        pe_logged = np.asarray(pe_logged)
        rows.append({
            "policy": name,
            "n_decisions": int(len(pb_argmax)),
            "mean_pb_at_target_argmax": float(pb_argmax.mean()),
            "p05_pb_at_target_argmax": float(np.percentile(pb_argmax, 5)),
            "target_argmax_pb_lt_0.01": float((pb_argmax < 0.01).mean()),
            "target_argmax_pb_lt_0.05": float((pb_argmax < 0.05).mean()),
            "mean_target_prob_on_logged_action": float(pe_logged.mean()),
            "mean_target_entropy": float(np.mean(ent_e)),
            "mean_logger_entropy": float(np.mean(ent_b)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "full"], default="full")
    ap.add_argument("--outdir", default="results_gate3_real_ope")
    ap.add_argument("--real-data", default=None,
                    help="Path to Online Retail II CSV/XLSX. If omitted, tries ucimlrepo as a best-effort fallback.")
    ap.add_argument("--clips", default="1,3,10,30")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3,
                    help="logged-support mask width, matching the simulator arms")
    ap.add_argument("--device", default=None,
                    help="torch device for the learned arms (default: auto)")
    args = ap.parse_args()

    cfg = C.smoke() if args.preset == "smoke" else C.full()
    obs_dim, n_actions = 2, cfg.sim.n_prices
    clips = [float(x) for x in args.clips.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    try:
        df = RD.load_online_retail_ii(args.real_data)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(2)
    train, test, edges = RD.build_real_trajectories(
        df, n_prices=cfg.sim.n_prices, horizon=cfg.sim.horizon, seed=args.seed)
    train_trajs = [t for t, _ in train]
    test_trajs = [t for t, _ in test]
    if len(train_trajs) < 5 or len(test_trajs) < 3:
        raise RuntimeError("Too few real episodes after filtering; relax filters.")

    dt = train_dt(D.pack_dt(train_trajs, logged_rtg(train_trajs)), obs_dim,
                  n_actions, cfg.model, seed=args.seed)
    bc = train_bc(train_trajs, obs_dim, n_actions, cfg.model, seed=args.seed)
    shim = SimpleNamespace(H=cfg.sim.horizon)
    target = float(np.quantile(np.concatenate(logged_rtg(train_trajs)), 0.95))
    pi_b = ope.estimate_behaviour_policy(test_trajs, n_actions)
    policies = {
        "logged_policy_estimated": pi_b,
        "vanilla_dt": lambda o: dt_action_probs(dt, shim, target, o),
        "bc": lambda o: qnet_action_probs(bc, o),
    }

    # The leading simulator arm is a support-masked IQL, so the real-log back-test
    # carries it too. The mask is applied to EVERY learned arm rather than only the
    # new one: applying it asymmetrically is the measurement failure this project
    # documents in §4.6.1.
    binner = _RealBins(edges, cfg.sim.horizon, n_actions)
    counts = _real_support_counts(train_trajs, binner)
    iql = train_iql(train_trajs, None, obs_dim, n_actions, cfg.model,
                    device=args.device, seed=args.seed)
    policies["iql"] = lambda o: qnet_action_probs(iql, o)
    for name, base in (("iql", policies["iql"]),
                       ("vanilla_dt", policies["vanilla_dt"]),
                       ("bc", policies["bc"])):
        policies[f"{name}_support_top{args.topk}"] = _masked_probs(
            base, binner, counts, args.topk)
    qhats = {
        "weak_action_only": ope.fit_qhat(test_trajs, n_actions, state_dependent=False),
        "strong_state_action": ope.fit_qhat(test_trajs, n_actions, state_dependent=True),
    }

    rng = np.random.default_rng(args.seed)
    rows = []
    for qname, qhat in qhats.items():
        for pname, pi_e in policies.items():
            for clip in clips:
                terms = _trajectory_terms(test_trajs, pi_e, pi_b, qhat,
                                          n_actions, w_clip=clip)
                dm, dm_lo, dm_hi = _mean_ci(terms["dm"], rng, args.n_boot)
                ips, ips_lo, ips_hi = _mean_ci(terms["ips"], rng, args.n_boot)
                dr, dr_lo, dr_hi = _mean_ci(terms["dr"], rng, args.n_boot)
                sn, sn_lo, sn_hi = _snips_ci(terms["snips_num"],
                                             terms["snips_den"], cfg.sim.horizon,
                                             rng, args.n_boot)
                rows += [
                    {"policy": pname, "qhat": qname, "clip": clip, "estimator": "DM",
                     "value": dm, "ci95_low": dm_lo, "ci95_high": dm_hi},
                    {"policy": pname, "qhat": qname, "clip": clip, "estimator": "IPS",
                     "value": ips, "ci95_low": ips_lo, "ci95_high": ips_hi},
                    {"policy": pname, "qhat": qname, "clip": clip, "estimator": "SNIPS",
                     "value": sn, "ci95_low": sn_lo, "ci95_high": sn_hi},
                    {"policy": pname, "qhat": qname, "clip": clip, "estimator": "DR",
                     "value": dr, "ci95_low": dr_lo, "ci95_high": dr_hi},
                ]
                for row in rows[-4:]:
                    row.update({
                        "n_train_episodes": int(len(train_trajs)),
                        "n_test_episodes": int(len(test_trajs)),
                        "target_q95": target if pname == "vanilla_dt" else np.nan,
                        "step_ess": _ess(terms["step_weights"]),
                        "step_ess_frac": _ess(terms["step_weights"]) / max(len(terms["step_weights"]), 1),
                        "terminal_ess": _ess(terms["terminal_weights"]),
                        "terminal_ess_frac": _ess(terms["terminal_weights"]) / max(len(terms["terminal_weights"]), 1),
                        "max_step_weight": float(np.max(terms["step_weights"])),
                        "p95_step_weight": float(np.percentile(terms["step_weights"], 95)),
                        "mean_pb_at_target_argmax": float(terms["pb_at_target_argmax"].mean()),
                        "target_argmax_pb_lt_0.01": float((terms["pb_at_target_argmax"] < 0.01).mean()),
                        "mean_target_prob_on_logged_action": float(terms["pe_at_logged"].mean()),
                    })

    sens = pd.DataFrame(rows)
    sens_path = os.path.join(args.outdir, "gate3_real_ope_sensitivity.csv")
    sens.to_csv(sens_path, index=False)
    overlap = _overlap_rows(test_trajs, policies, pi_b, n_actions)
    overlap_path = os.path.join(args.outdir, "gate3_real_ope_overlap.csv")
    overlap.to_csv(overlap_path, index=False)

    protocol = {
        "date": "2026-07-27",
        "stage": "Gate 3A real-data OPE sensitivity",
        "preset": args.preset,
        "real_data": args.real_data,
        "n_rows_after_cleaning": int(len(df)),
        "n_train_episodes": int(len(train_trajs)),
        "n_test_episodes": int(len(test_trajs)),
        "n_prices": int(n_actions),
        "horizon": int(cfg.sim.horizon),
        "clips": clips,
        "n_boot": int(args.n_boot),
        "bootstrap_note": "Fixed-nuisance bootstrap over held-out episodes.",
        "source_note": (
            "Online Retail II has no ground-truth counterfactual policy value; "
            "these estimates are sensitivity diagnostics, not headline evidence."
        ),
    }
    with open(os.path.join(args.outdir, "gate3_real_ope_protocol.json"),
              "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    print(f"Wrote {sens_path}")
    print(f"Wrote {overlap_path}")
    show = sens[(sens["qhat"].eq("strong_state_action")) & (sens["clip"].eq(10.0))]
    cols = ["policy", "estimator", "value", "ci95_low", "ci95_high",
            "step_ess_frac", "mean_pb_at_target_argmax"]
    print(show[cols].to_string(index=False))


if __name__ == "__main__":
    main()
