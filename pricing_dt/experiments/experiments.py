"""Experiment runners E0-E3.

E0  testbed validation
E1  sequential-stitching diagnostic for vanilla DT
E2  structured relabelling vs corrected Q-DT across a factorial scan
E2AB prior-isolation ablation (A/B/C/D, with D using corrected Q-DT)
E3  RQ3 / C3: pooled vs segmented DR under non-stationary logging
"""

# ---------------------------------------------------------------- REFERENCES
#
# Core code here: _setup(), e1_vanilla_failure(), e2_core(), e3_ope().
#
# The experiment runners, and the place the evaluation anchors are constructed: the
# oracle myopic policy that defines 0 and the dynamic-programming optimum that defines
# 1.
#
# Implements or follows:
#   - Fu, J., Kumar, A., Nachum, O., Tucker, G. and Levine, S. (2020) 'D4RL: Datasets for
#     Deep Data-Driven Reinforcement Learning'. arXiv:2004.07219.
#
# Full entries: the reference list of the dissertation this code accompanies.
# ----------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import torch

from pricing_dt.core import data as D
from pricing_dt.core import metrics as M
from pricing_dt.core.simulator import PricingMDP
from pricing_dt.core.demand_model import (StructuredDemandModel, UnconstrainedDemandModel,
                          MisspecifiedStructuredDemandModel, fit_demand_model)
from pricing_dt.core.relabel import relabel_dataset, logged_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.dt import train_dt, make_dt_policy, dt_action_probs
from pricing_dt.core.baselines import train_bc, policy_from_qnet
from pricing_dt.core import ope


def _seed(s):
    np.random.seed(s); torch.manual_seed(s)


def _setup(cfg, sim_overrides=None, seed=0):
    # IMPORTANT: work on a COPY of the sim config. Mutating cfg.sim in place leaks
    # state across experiments in a full `--exp all` run — e.g. e_seqnec sweeps
    # `delta` and would leave cfg.sim.delta at its last value (4.0), silently
    # changing the reference-price strength for every later experiment (E1/E2/E2-AB/
    # mis/E3). Copying makes each call start from the intended SimConfig defaults.
    import copy
    sc = copy.deepcopy(cfg.sim)
    if sim_overrides:
        for k, v in sim_overrides.items():
            setattr(sc, k, v)
    sc.seed = seed
    mdp = PricingMDP(sc)
    mdp.solve_optimal()
    init = mdp.initial_bins(cfg.data.n_eval_episodes)
    # Exact anchors for normalisation.
    #
    # NOTE ON THE NAME. The zero anchor is the ORACLE MYOPIC policy -- argmax over
    # the TRUE reward matrix mdp.R, i.e. the contextual-bandit solution under
    # perfect one-step information. It is NOT the logging policy, despite the
    # legacy names `v_beh` here and `v_behaviour_expected` in the released CSVs,
    # which are kept so previously published result files stay readable. A fitted
    # one-step pipeline is a competitor on this scale, not the origin (it scores
    # about -1.45). See Chapter 3, "Value normalisation".
    v_opt, _ = mdp.evaluate_policy_fn(lambda o: int(mdp.pistar[int(round(o[1]*(mdp.H-1))),
                                                                 mdp.ref_to_bin(mdp.cfg.p_min+o[0]*(mdp.cfg.p_max-mdp.cfg.p_min))]), init)
    oracle_myopic = lambda o: int(mdp.R[:, mdp.ref_to_bin(mdp.cfg.p_min+o[0]*(mdp.cfg.p_max-mdp.cfg.p_min))].argmax())
    v_myopic_oracle, _ = mdp.evaluate_policy_fn(oracle_myopic, init)
    return mdp, init, v_opt, v_myopic_oracle


def _eval_dt(model, mdp, init, train_rtg):
    target = float(np.quantile(np.concatenate(train_rtg), 0.95))
    pol = make_dt_policy(model, mdp, target)
    v, _ = mdp.evaluate_policy_fn(pol, init)
    return v, target


def _policy_value_by_bin(policy_fn, mdp, trajs):
    """Exact value of a policy evaluated FROM each unique trajectory start bin.
    Returns {bin: V_from_bin}, for the well-posed per-start stitching comparison."""
    bins = sorted(set(int(tr.ref_bins[0]) for tr in trajs))
    # One rollout over all start bins at once. Episodes are independent given the
    # policy, so this is identical to the old per-bin loop but lets a batched
    # policy do a single forward per timestep instead of one per (bin, timestep).
    _, per_start = mdp.evaluate_policy_fn(policy_fn, np.array(bins))
    return {b: float(v) for b, v in zip(bins, per_start)}


def _traj_start_bins(trajs):
    """Initial reference bins of the logged trajectories. Evaluating learned
    policies from THIS distribution (not a uniform one) keeps them comparable to
    the de-noised best-logged-return ceiling, which is itself computed over these
    same trajectories (avoids the start-distribution mismatch that would otherwise
    make the ceiling exceed the start-averaged optimum)."""
    return np.array([int(tr.ref_bins[0]) for tr in trajs])


# ----------------------------- E0 -----------------------------
def e0_testbed(cfg, outdir):
    """E0. Validate the testbed before any learner is run on it.

    Checks that the dynamic-programming optimum agrees with exact evaluation of the same
    policy, and that demand falls monotonically in price at a mid reference. Nothing
    downstream is interpretable if either fails, so this runs first.
    """
    mdp, init, v_opt, v_beh = _setup(cfg, seed=0)
    # sanity: DP optimal value equals exact evaluation of pistar
    Vdp = mdp.Vstar[0].mean()
    # demand monotonicity sanity at mid reference
    ref = mdp.ref_grid[mdp.cfg.n_ref_bins // 2]
    dem = [mdp.expected_demand(p, ref) for p in mdp.prices]
    monotone = bool(np.all(np.diff(dem) <= 1e-9))
    rows = [dict(check="optimal_value_DP", value=round(float(Vdp), 3)),
            dict(check="optimal_value_exact_rollout", value=round(float(v_opt), 3)),
            dict(check="behaviour(myopic)_value", value=round(float(v_beh), 3)),
            dict(check="demand_monotone_in_price", value=monotone),
            dict(check="optimality_gap(opt-beh)", value=round(float(v_opt - v_beh), 3))]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "e0_testbed.csv"), index=False)
    return df


# ----------------------------- E1 -----------------------------
def e1_vanilla_failure(cfg, outdir):
    """E1. Establish that the unaided Decision Transformer fails on these logs.

    Runs the low and the high noise level only, on the stitching-necessary logs where no
    single trajectory is globally good. Establishes the gap that the conditioning targets
    of `e2_core` are then asked to close.
    """
    rows = []
    obs_dim, A = 2, cfg.sim.n_prices
    for noise in cfg.exp.noise_levels[::len(cfg.exp.noise_levels) - 1 or 1]:  # low & high
        for seed in cfg.exp.seeds:
            _seed(seed)
            mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, cfg.data.n_train_traj, noise, seed, cfg.data.expert_q)
            init_traj = _traj_start_bins(trajs)
            v_opt_sm = float(mdp.Vstar[0, init_traj].mean())   # optimum from the SAME starts
            bc = train_bc(trajs, obs_dim, A, cfg.model, seed=seed)
            v_bc, _ = mdp.evaluate_policy_fn(policy_from_qnet(bc), init_traj)
            dt = train_dt(D.pack_dt(trajs, logged_rtg(trajs)), obs_dim, A, cfg.model, seed=seed)
            v_dt, tgt = _eval_dt(dt, mdp, init_traj, logged_rtg(trajs))
            # well-posed per-start stitching comparison (C1): policy value from each
            # logged start vs the best de-noised logged trajectory from the SAME start
            pv = _policy_value_by_bin(make_dt_policy(dt, mdp, tgt), mdp, trajs)
            ss = M.stitching_score(pv, trajs, mdp)
            rows.append(dict(noise=noise, seed=seed,
                             v_behaviour=round(v_beh, 2), v_optimal=round(v_opt_sm, 2),
                             v_BC=round(v_bc, 2), v_vanillaDT=round(v_dt, 2),
                             stitch_avg_margin_vanillaDT=round(ss["avg_margin"], 2),
                             stitch_beat_fraction_vanillaDT=round(ss["beat_fraction"], 3),
                             vanilla_stitches=bool(ss["beat_fraction"] > 0.5)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "e1_vanilla_failure.csv"), index=False)
    return df


# ----------------------------- E2 -----------------------------
def _train_variants(trajs, mdp, cfg, obs_dim, A, seed):
    """Fit demand model, build the three RTG columns, and train the three DT variants."""
    dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
    fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
    rtg_vanilla = logged_rtg(trajs)
    rtg_struct = relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)
    rtg_value, _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
    out = {}
    out["vanillaDT"] = (train_dt(D.pack_dt(trajs, rtg_vanilla), obs_dim, A, cfg.model, seed=seed), rtg_vanilla)
    out["Q-DT"] = (train_dt(D.pack_dt(trajs, rtg_value), obs_dim, A, cfg.model, seed=seed), rtg_value)
    out["structuredDT"] = (train_dt(D.pack_dt(trajs, rtg_struct), obs_dim, A, cfg.model, seed=seed), rtg_struct)
    return out, dm


def e2_core(cfg, outdir):
    """The main experiment: the three conditioning targets over the data-size and
    noise grid, each on the same logs, the same seeds and the same evaluation.

    The arms differ only
    in the relabelling applied before `train_dt`, which is what lets a difference in
    policy value be read as a difference in the target.
    """
    rows = []
    obs_dim, A = 2, cfg.sim.n_prices
    for N in cfg.exp.data_sizes:
        for noise in cfg.exp.noise_levels:
            for seed in cfg.exp.seeds:
                _seed(seed)
                mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
                trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
                init_traj = _traj_start_bins(trajs)
                v_opt_sm = float(mdp.Vstar[0, init_traj].mean())
                v_beh_data = float(np.mean([tr.total_return for tr in trajs]))
                variants, _ = _train_variants(trajs, mdp, cfg, obs_dim, A, seed)
                rec = dict(N=N, noise=noise, seed=seed,
                           v_optimal=round(v_opt_sm, 2),
                           v_behaviour_expected=round(v_beh, 2),
                           v_behaviour_realised=round(v_beh_data, 2))
                for name, (model, rtg) in variants.items():
                    target = float(np.quantile(np.concatenate(rtg), 0.95))
                    v, _ = mdp.evaluate_policy_fn(make_dt_policy(model, mdp, target), init_traj)
                    rec[f"nv_{name}"] = round(M.normalised_value(v, v_beh, v_opt_sm), 3)
                    rec[f"uplift_{name}"] = round(v - v_beh_data, 2)  # over logged behaviour
                    # well-posed per-start stitching score (C1)
                    pv = _policy_value_by_bin(make_dt_policy(model, mdp, target), mdp, trajs)
                    ss = M.stitching_score(pv, trajs, mdp)
                    rec[f"stitch_margin_{name}"] = round(ss["avg_margin"], 2)
                    rec[f"stitches_{name}"] = bool(ss["beat_fraction"] > 0.5)
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "e2_core_raw.csv"), index=False)

    # Broad surface: structured - corrected Q-DT by (N, noise), with paired test across seeds.
    # Claim-critical fixed-QDT/held-out diagnostics live in diag_c2_fixed.py and companions.
    summ = []
    for N in cfg.exp.data_sizes:
        for noise in cfg.exp.noise_levels:
            cell = df[(df.N == N) & (df.noise == noise)]
            med, p = M.paired_test(cell["nv_structuredDT"].values, cell["nv_Q-DT"].values)
            summ.append(dict(N=N, noise=noise,
                             adv_structured_minus_QDT=round(float(
                                 cell["nv_structuredDT"].mean() - cell["nv_Q-DT"].mean()), 3),
                             paired_median=round(med, 3), p_value=round(p, 4) if p == p else float("nan")))
    sdf = pd.DataFrame(summ)
    sdf["p_holm"] = M.holm(np.nan_to_num(sdf["p_value"].values, nan=1.0))
    sdf.to_csv(os.path.join(outdir, "e2_C2_surface.csv"), index=False)
    return df, sdf


# ----------------------------- E2-AB -----------------------------
def e2ab_ablation(cfg, outdir):
    """E2-AB. Isolate the structural prior at the hardest cell only.

    Fixes the smallest data size and the highest noise level -- the cell where the prior
    has the most to contribute -- and varies the prior alone, everything else held to
    `e2_core`'s settings. Being one cell rather than the grid, its n is the seed count,
    not the grid count.
    """
    rows = []
    obs_dim, A = 2, cfg.sim.n_prices
    # hardest cell: smallest data, highest noise
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    for seed in cfg.exp.seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        v_beh_data = float(np.mean([tr.total_return for tr in trajs]))

        def run_struct(model_dm, lam=cfg.model.relabel_lambda):
            rtg = relabel_dataset(trajs, model_dm, mdp, lam)
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            v, _ = _eval_dt(m, mdp, init, rtg)
            return M.normalised_value(v, v_beh, v_opt)

        # A: full structured prior
        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        nvA = run_struct(dmA)
        # B: unconstrained demand model (no prior)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)
        nvB = run_struct(dmB)
        # C: mis-specified prior (wrong elasticity bounds)
        dmC = StructuredDemandModel(obs_dim, cfg.model.elasticity_hi, cfg.model.elasticity_hi * 2)
        fit_demand_model(dmC, trajs, mdp, cfg.model.demand_epochs)
        nvC = run_struct(dmC)
        # D: bootstrapped value (Q-DT)
        rtgD, _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
        mD = train_dt(D.pack_dt(trajs, rtgD), obs_dim, A, cfg.model, seed=seed)
        vD, _ = _eval_dt(mD, mdp, init, rtgD)
        nvD = M.normalised_value(vD, v_beh, v_opt)

        rows.append(dict(seed=seed, N=N, noise=noise,
                         A_full_prior=round(nvA, 3), B_no_constraints=round(nvB, 3),
                         C_misspecified=round(nvC, 3), D_bootstrapped_value=round(nvD, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "e2ab_ablation.csv"), index=False)
    return df


# ----------------------------- E3 -----------------------------
def _mean_pairwise_tv(loggers, mdp):
    """Mean total-variation distance between segment loggers' action distributions
    over the (ref-bin, t) grid -- the measured logging-drift level for C3."""
    B, H = mdp.cfg.n_ref_bins, mdp.H
    tvs = []
    for i in range(len(loggers)):
        for j in range(i + 1, len(loggers)):
            d = [0.5 * np.abs(loggers[i].probs(b, t) - loggers[j].probs(b, t)).sum()
                 for b in range(B) for t in range(H)]
            tvs.append(float(np.mean(d)))
    return float(np.mean(tvs)) if tvs else 0.0


def e3_ope(cfg, outdir):
    """RQ3 / C3. Pooled vs segmented doubly robust OPE under a non-stationary logger.

    Three design points make this a fair test. Each is load-bearing: set any of them the
    other way and the effect the test is looking for disappears.
      1. PREFERENCE drift: segment loggers interpolate myopic->optimal (see
         data.SoftmaxQSnapshot), so the drift knob produces a real total-variation
         sweep, not just a softmax-temperature change.
      2. WEAK q_hat: the headline uses a state-INDEPENDENT outcome model so DR leans
         on its importance weights and a mis-specified pooled propensity surfaces as
         bias. A state-dependent q_hat is reported alongside (suffix _strong) to show
         the effect is genuinely MASKED by a capable direct method: the
         estimator-visibility caveat, demonstrated rather than asserted.
      3. Matched target: the policy whose value is estimated (a memoryless argmax DT
         controller) is exactly the policy whose true value v_true is computed, so the
         reported bias is not contaminated by a target-policy mismatch.
    The non-stationarity-specific quantity is |bias_pooled| - |bias_segmented|, which
    is 0 at zero drift by construction and grows with drift if segmentation helps.
    """
    rows = []
    obs_dim, A = 2, cfg.sim.n_prices
    noise = cfg.exp.noise_levels[len(cfg.exp.noise_levels) // 2]
    for drift in cfg.exp.drift_levels:
        for seed in cfg.exp.seeds:
            _seed(seed)
            mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
            # drift controls how different the segment loggers are (spread)
            trajs, loggers = D.make_nonstationary(mdp, cfg.data.n_segments,
                                                  cfg.data.traj_per_segment, noise,
                                                  temp=0.5, seed=seed, spread=drift)
            tv = _mean_pairwise_tv(loggers, mdp)
            # target policy: structured DT trained on this log
            dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            rtg = relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)
            dt = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            target = float(np.quantile(np.concatenate(rtg), 0.95))
            # MEMORYLESS deterministic DT controller: the OPE target == the policy
            # whose exact value we compute, so bias is not a target mismatch artefact.
            _pcache = {}
            def pe_dist(o):
                k = (round(float(o[0]), 4), round(float(o[1]), 4))
                if k not in _pcache:
                    _pcache[k] = dt_action_probs(dt, mdp, target, o)
                return _pcache[k]
            pol = lambda o: int(np.argmax(pe_dist(o)))
            v_true, _ = mdp.evaluate_policy_fn(pol, init)
            def pe(o):                       # deterministic one-hot target for OPE
                p = np.zeros(A); p[pol(o)] = 1.0; return p

            # True per-segment loggers (simulator ground truth) isolate the
            # non-stationarity bias without behaviour-policy estimation noise.
            def logger_probs(lg):
                def f(o):
                    b, t = mdp.decode_obs(o)
                    return lg.probs(b, t)
                return f
            seg_probs = [logger_probs(lg) for lg in loggers]
            pooled_probs = lambda o: np.mean([f(o) for f in seg_probs], axis=0)

            rec = dict(drift=drift, seed=seed, logger_tv=round(tv, 3),
                       v_true=round(v_true, 2),
                       true_uplift_over_beh=round(v_true - v_beh, 2))
            # headline = weak (state-independent) q_hat; _strong = capable q_hat (masks)
            for suffix, qh in [("", ope.fit_qhat(trajs, A, state_dependent=False)),
                               ("_strong", ope.fit_qhat(trajs, A, state_dependent=True))]:
                vals, ws = [], []
                for s in range(cfg.data.n_segments):
                    sub = [tr for tr in trajs if tr.seg == s]
                    vals.append(ope.dr_value(sub, pe, seg_probs[s], qh, A)); ws.append(len(sub))
                ws = np.array(ws, float); ws /= ws.sum()
                v_seg = float(np.dot(ws, vals))
                v_pooled = ope.dr_value(trajs, pe, pooled_probs, qh, A)
                rec[f"bias_pooled{suffix}"] = round(v_pooled - v_true, 2)
                rec[f"bias_segmented{suffix}"] = round(v_seg - v_true, 2)

            # ---- Fully DATA-DRIVEN pipeline (no oracle), the deployable setting ----
            # pi_b is ESTIMATED from the logged (state, action) pairs and change-points
            # are DETECTED from data, rather than using the simulator's true loggers and
            # true segment ids. Weak (state-independent) q_hat so the bias still surfaces.
            qh_weak = ope.fit_qhat(trajs, A, state_dependent=False)
            pb_pool_est = ope.estimate_behaviour_policy(trajs, A)
            v_pool_est = ope.dr_value(trajs, pe, pb_pool_est, qh_weak, A)
            seg_ids = ope.detect_segments(trajs, max_segments=cfg.data.n_segments + 2)
            vals_e, ws_e = [], []
            for s in np.unique(seg_ids):
                sub = [tr for tr, sid in zip(trajs, seg_ids) if sid == s]
                if len(sub) < 3:
                    continue
                pb_e = ope.estimate_behaviour_policy(sub, A)
                vals_e.append(ope.dr_value(sub, pe, pb_e, qh_weak, A)); ws_e.append(len(sub))
            ws_e = np.array(ws_e, float); ws_e /= ws_e.sum()
            v_seg_est = float(np.dot(ws_e, vals_e))
            rec["bias_pooled_estpi"] = round(v_pool_est - v_true, 2)
            rec["bias_segmented_estpi"] = round(v_seg_est - v_true, 2)
            rec["n_detected_segments"] = int(len(np.unique(seg_ids)))
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "e3_ope.csv"), index=False)
    # summary: mean |bias| by drift, for weak (headline) and strong (masked) q_hat
    summ = df.groupby("drift").agg(
        logger_tv=("logger_tv", lambda x: round(x.mean(), 3)),
        mean_abs_bias_pooled=("bias_pooled", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_segmented=("bias_segmented", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_pooled_strong=("bias_pooled_strong", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_segmented_strong=("bias_segmented_strong", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_pooled_estpi=("bias_pooled_estpi", lambda x: round(np.abs(x).mean(), 2)),
        mean_abs_bias_segmented_estpi=("bias_segmented_estpi", lambda x: round(np.abs(x).mean(), 2)),
        n_detected_segments=("n_detected_segments", lambda x: round(x.mean(), 1)),
    ).reset_index()
    summ["segmentation_benefit"] = (summ["mean_abs_bias_pooled"]
                                    - summ["mean_abs_bias_segmented"]).round(2)
    summ["segmentation_benefit_estpi"] = (summ["mean_abs_bias_pooled_estpi"]
                                          - summ["mean_abs_bias_segmented_estpi"]).round(2)
    summ.to_csv(os.path.join(outdir, "e3_ope_summary.csv"), index=False)
    return df, summ


# ----------------------------- Real-data: calibration -----------------------------
def calibrate(cfg, outdir, real_path=None):
    """Estimate price elasticity from Online Retail II and suggest simulator
    parameters so the ground-truth experiments run at a realistic elasticity."""
    from pricing_dt.core import realdata as RD
    df = RD.load_online_retail_ii(real_path)
    cal = RD.calibrate_elasticity(df)
    sug = RD.suggest_sim_params(cal["median_elasticity"])
    out = {**cal, **{f"suggested_{k}": v for k, v in sug.items()}}
    odf = pd.DataFrame([out])
    odf.to_csv(os.path.join(outdir, "calibration.csv"), index=False)
    return odf


# ----------------------------- Real-data: realism check -----------------------------
def e_realism(cfg, outdir, real_path=None):
    """Run vanilla-DT / BC + OPE on real held-out logs (no ground truth), and fit
    the interpretable structured demand curve. OPE estimates only."""
    import torch
    from types import SimpleNamespace
    from pricing_dt.core import realdata as RD
    from pricing_dt.core.demand_model import StructuredDemandModel
    from pricing_dt.core.baselines import qnet_action_probs

    obs_dim, A = 2, cfg.sim.n_prices
    df = RD.load_online_retail_ii(real_path)
    train, test, edges = RD.build_real_trajectories(
        df, n_prices=cfg.sim.n_prices, horizon=cfg.sim.horizon)
    train_trajs = [t for t, _ in train]
    test_trajs = [t for t, _ in test]
    if len(train_trajs) < 5 or len(test_trajs) < 3:
        raise RuntimeError("Too few real episodes after filtering; relax build_real_trajectories filters.")

    # interpretable demand curve (structured prior) fit on real (state, price, demand)
    S = np.concatenate([t.obs for t, _ in train])
    P = np.concatenate([p for _, p in train])
    Dm = np.concatenate([t.rewards / np.maximum(p, 1e-6) for t, p in train])
    dm = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
    opt = torch.optim.Adam(dm.parameters(), lr=1e-2)
    St, Pt, Dt = torch.tensor(S), torch.tensor(P), torch.tensor(Dm)
    for _ in range(cfg.model.demand_epochs * 3):
        opt.zero_grad(); ((dm(Pt, St) - Dt) ** 2).mean().backward(); opt.step()
    with torch.no_grad():
        s0 = torch.tensor(np.array([[0.5, 0.5]], np.float32)).repeat(cfg.sim.n_prices, 1)
        dem = dm(torch.tensor(edges, dtype=torch.float32), s0).cpu().numpy()
    monotone = bool(np.all(np.diff(dem) <= 1e-6))

    # train policies on real train split; evaluate on held-out test by OPE
    dt = train_dt(D.pack_dt(train_trajs, logged_rtg(train_trajs)), obs_dim, A, cfg.model)
    bc = train_bc(train_trajs, obs_dim, A, cfg.model)
    shim = SimpleNamespace(H=cfg.sim.horizon)
    target = float(np.quantile(np.concatenate(logged_rtg(train_trajs)), 0.95))
    pe_dt = lambda o: dt_action_probs(dt, shim, target, o)
    pe_bc = lambda o: qnet_action_probs(bc, o)
    pb = ope.estimate_behaviour_policy(test_trajs, A)
    qh = ope.fit_qhat(test_trajs, A)
    v_dt = ope.dr_value(test_trajs, pe_dt, pb, qh, A)
    v_bc = ope.dr_value(test_trajs, pe_bc, pb, qh, A)
    v_logged = float(np.mean([t.total_return for t in test_trajs]))

    rows = [dict(metric="n_train_episodes", value=round(len(train_trajs), 3)),
            dict(metric="n_test_episodes", value=round(len(test_trajs), 3)),
            dict(metric="demand_curve_monotone_in_price", value=monotone),
            dict(metric="OPE_value_logged_policy(test)", value=round(v_logged, 3)),
            dict(metric="OPE_value_vanillaDT", value=round(v_dt, 3)),
            dict(metric="OPE_value_BC", value=round(v_bc, 3)),
            dict(metric="OPE_uplift_DT_over_logged", value=round(v_dt - v_logged, 3))]
    odf = pd.DataFrame(rows)
    odf.to_csv(os.path.join(outdir, "realism.csv"), index=False)
    return odf


# ----------------------------- C0: sequential-necessity test -----------------------------
def e_seqnec(cfg, outdir):
    """RQ1 / C0. Is the testbed genuinely sequential? Scan the reference-price
    strength (delta). For each, compare the EXACT intertemporal-optimal value
    against the EXACT best myopic per-period-greedy value (both by dynamic
    programming on the known MDP -- no training, no estimation). A materially
    positive, growing gap confirms the reference-price coupling creates real
    sequential structure; gap ~ 0 at delta=0 (a disguised contextual bandit)
    would refute the RL framing. This is an exact, simulator-only result.
    """
    rows = []
    for delta in cfg.exp.ref_strengths:
        mdp, init, v_opt, v_myopic = _setup(cfg, {"delta": delta}, seed=0)
        gap = v_opt - v_myopic
        rel = gap / max(v_opt, 1e-8)
        rows.append(dict(ref_strength_delta=delta,
                         v_intertemporal_optimal=round(v_opt, 3),
                         v_myopic_greedy=round(v_myopic, 3),
                         sequential_gap=round(gap, 3),
                         relative_gap=round(rel, 4),
                         sequential_structure_material=bool(rel > 0.02)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "c0_sequential_necessity.csv"), index=False)
    return df


# ----------------------------- Misspecification scan (formal result) -----------------------------
def e2_mis(cfg, outdir):
    """Formal misspecification result (promoted from the E2-AB cell). Scan a
    single 'prior wrongness' severity axis: at 0 the structured prior is correct;
    growing severity pushes the elasticity bounds wrong and adds a price-increasing
    term until the raw log-demand prior is invalid. Quantifies whether the
    structured DT's advantage erodes or reverses as the prior is made
    progressively wrong, in the hardest regime (smallest data, highest noise).
    Q-DT (no demand prior) is the reference floor.
    """
    rows = []
    obs_dim, A = 2, cfg.sim.n_prices
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    for sev in cfg.exp.misspec_levels:
        for seed in cfg.exp.seeds:
            _seed(seed)
            mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
            trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
            # structured prior at this severity
            dm = MisspecifiedStructuredDemandModel(obs_dim, cfg.model.elasticity_lo,
                                                   cfg.model.elasticity_hi, severity=sev)
            fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
            # Is the corrupted prior still monotone in price?  Use the raw
            # log-demand, not forward(), because forward() clamps large values
            # and can turn an invalid increasing curve into a constant one.
            with torch.no_grad():
                import numpy as _np
                ps = torch.tensor(mdp.prices, dtype=torch.float32)
                s_mid = torch.tensor(mdp.obs(mdp.ref_grid[mdp.cfg.n_ref_bins // 2], 0),
                                     dtype=torch.float32).repeat(len(ps), 1)
                log_dem = dm.log_demand(ps, s_mid).cpu().numpy()
                dem = dm(ps, s_mid).cpu().numpy()
                monotone = bool(_np.all(_np.diff(log_dem) <= 1e-6))
                degenerate_after_clamp = bool(_np.ptp(dem) <= 1e-6)
            rtg = relabel_dataset(trajs, dm, mdp, cfg.model.relabel_lambda)
            m = train_dt(D.pack_dt(trajs, rtg), obs_dim, A, cfg.model, seed=seed)
            v, _ = _eval_dt(m, mdp, init, rtg)
            nv_struct = M.normalised_value(v, v_beh, v_opt)
            # Q-DT floor (no demand prior at all)
            rtgQ, _ = value_relabel(trajs, mdp, obs_dim, A, cfg.model, seed=seed)
            mQ = train_dt(D.pack_dt(trajs, rtgQ), obs_dim, A, cfg.model, seed=seed)
            vQ, _ = _eval_dt(mQ, mdp, init, rtgQ)
            nv_q = M.normalised_value(vQ, v_beh, v_opt)
            rows.append(dict(severity=sev, seed=seed, prior_monotone=monotone,
                             prior_degenerate_after_clamp=degenerate_after_clamp,
                             nv_structured=round(nv_struct, 3),
                             nv_QDT_floor=round(nv_q, 3),
                             advantage_over_QDT=round(nv_struct - nv_q, 3)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "mis_scan_raw.csv"), index=False)
    summ = df.groupby("severity").agg(
        prior_monotone=("prior_monotone", "all"),
        prior_degenerate_after_clamp=("prior_degenerate_after_clamp", "all"),
        mean_nv_structured=("nv_structured", lambda x: round(np.nanmean(x), 3)),
        mean_nv_QDT=("nv_QDT_floor", lambda x: round(np.nanmean(x), 3)),
        mean_advantage=("advantage_over_QDT", lambda x: round(np.nanmean(x), 3)),
    ).reset_index()
    summ.to_csv(os.path.join(outdir, "mis_scan_summary.csv"), index=False)
    return summ
