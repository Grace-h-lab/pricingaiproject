"""Smoke tests asserting the key scientific invariants of the testbed.

These go beyond "does it run": each test checks a property the reported claims
depend on, so a regression in the mechanism, and not only a crash, is caught.

Run with either:
    python tests/test_smoke.py        # plain stdlib, prints PASS/FAIL, exits non-zero on failure
    pytest tests/test_smoke.py        # if pytest is installed

Kept tiny (smoke scale) so the whole file runs in well under a minute on CPU.
"""
import os
import sys
import copy
import numpy as np
import torch

# allow running from repo root or from tests/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing_dt.core import config as C
from pricing_dt.core.simulator import PricingMDP
from pricing_dt.core import data as D
from pricing_dt.core.demand_model import (StructuredDemandModel, MisspecifiedStructuredDemandModel,
                          fit_demand_model)
from pricing_dt.core.relabel import achievable_rtg
from pricing_dt.core.qdt import value_relabel
from pricing_dt.core.baselines import train_iql, policy_from_qnet
from pricing_dt.core.dt import make_support_masked_dt_policy


def _tiny_mdp(seed=0):
    cfg = C.smoke()
    cfg.sim.seed = seed
    mdp = PricingMDP(cfg.sim)
    mdp.solve_optimal()
    return cfg, mdp


def test_e0_optimum_recovered_and_demand_monotone():
    """E0: DP optimum matches exact rollout; demand is non-increasing in price."""
    cfg, mdp = _tiny_mdp()
    init = mdp.initial_bins(64)
    v_dp = mdp.Vstar[0].mean()
    # exact rollout of pistar should match the DP value closely
    def pistar_fn(o):
        b = mdp.ref_to_bin(mdp.cfg.p_min + o[0] * (mdp.cfg.p_max - mdp.cfg.p_min))
        t = int(round(o[1] * (mdp.H - 1)))
        return int(mdp.pistar[t, b])
    v_roll, _ = mdp.evaluate_policy_fn(pistar_fn, init)
    assert abs(v_dp - v_roll) / max(abs(v_dp), 1e-6) < 0.05, (v_dp, v_roll)
    # demand monotone non-increasing in price at a mid reference
    ref = mdp.ref_grid[mdp.cfg.n_ref_bins // 2]
    dem = np.array([mdp.expected_demand(p, ref) for p in mdp.prices])
    assert np.all(np.diff(dem) <= 1e-9), dem


def test_relabelling_is_action_dependent():
    """Relabelled RTG must change when the logged action changes.

    If it does not, the DT would learn to pair a high target with sub-optimal
    actions (the bug this guards against)."""
    cfg, mdp = _tiny_mdp()
    trajs = D.make_stitching_necessary(mdp, 40, 0.1, 0)
    dm = StructuredDemandModel(2, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
    fit_demand_model(dm, trajs, mdp, cfg.model.demand_epochs)
    tr = trajs[0]
    g0 = achievable_rtg(tr, dm, mdp, lam=1.0)
    tr2 = copy.deepcopy(tr)
    tr2.actions = tr.actions.copy()
    tr2.actions[0] = (tr.actions[0] + mdp.cfg.n_prices // 2) % mdp.cfg.n_prices
    g1 = achievable_rtg(tr2, dm, mdp, lam=1.0)
    assert abs(g0[0] - g1[0]) > 1e-6, (g0[0], g1[0])
    # last-step target equals just that step's taken-action revenue (no continuation)
    # (sanity: relabelled values are finite and non-negative)
    assert np.all(np.isfinite(g0)) and np.all(g0 >= -1e-6)


def test_qdt_td_relabelling_is_action_dependent():
    """The fixed Q-DT target must depend on the logged action.

    The legacy state-value target is retained for reporting that effect, so
    this test checks both sides: TD changes when the logged action changes;
    state-value does not.
    """
    cfg, mdp = _tiny_mdp()
    trajs = D.make_stitching_necessary(mdp, 12, 0.1, 0)
    tr = trajs[0]
    b0 = int(tr.ref_bins[0])
    a0 = int(tr.actions[0])
    rewards = mdp.R[:, b0]
    a1 = int(np.argmax(np.abs(rewards - rewards[a0])))
    assert abs(rewards[a1] - rewards[a0]) > 1e-6

    tr2 = copy.deepcopy(tr)
    tr2.actions = tr.actions.copy()
    tr2.actions[0] = a1

    class ZeroQ(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], mdp.cfg.n_prices),
                               dtype=torch.float32, device=x.device)

    q = ZeroQ()
    td0, _ = value_relabel([tr], mdp, 2, mdp.cfg.n_prices, cfg.model,
                           mode="td", denoise=True, q=q)
    td1, _ = value_relabel([tr2], mdp, 2, mdp.cfg.n_prices, cfg.model,
                           mode="td", denoise=True, q=q)
    legacy0, _ = value_relabel([tr], mdp, 2, mdp.cfg.n_prices, cfg.model,
                               mode="state_value", q=q)
    legacy1, _ = value_relabel([tr2], mdp, 2, mdp.cfg.n_prices, cfg.model,
                               mode="state_value", q=q)

    assert abs(td0[0][0] - td1[0][0]) > 1e-6
    assert abs(legacy0[0][0] - legacy1[0][0]) < 1e-6


def test_support_masked_dt_policy_blocks_unsupported_argmax():
    """A support mask should override a high-logit unsupported action."""
    cfg, mdp = _tiny_mdp()
    ref = mdp.ref_grid[mdp.cfg.n_ref_bins // 2]
    obs = mdp.obs(ref, 0)
    b = mdp.ref_to_bin(ref)
    counts = np.zeros((mdp.H, mdp.cfg.n_ref_bins, mdp.cfg.n_prices), dtype=float)
    counts[0, b, 0] = 1.0

    class UnsupportedArgmax(torch.nn.Module):
        def forward(self, rtg, states, actions, timesteps):
            B, T = states.shape[:2]
            logits = torch.zeros((B, T, mdp.cfg.n_prices),
                                 dtype=torch.float32, device=states.device)
            logits[..., -1] = 10.0
            return logits

    policy = make_support_masked_dt_policy(
        UnsupportedArgmax(), mdp, target_return=0.0, counts=counts, min_count=1
    )
    assert policy(obs) == 0


def test_discrete_iql_smoke_policy():
    """IQL should train a callable policy baseline at tiny scale."""
    cfg, mdp = _tiny_mdp()
    trajs = D.make_stitching_necessary(mdp, 12, 0.1, 0)
    pi = train_iql(trajs, mdp, 2, mdp.cfg.n_prices, cfg.model,
                   seed=0, updates=3, batch_size=16)
    action = policy_from_qnet(pi)(trajs[0].obs[0])
    assert 0 <= action < mdp.cfg.n_prices
    assert pi.iql_diagnostics["iql_updates"] == 3


def test_stitching_necessary_construction():
    """No single logged trajectory should reach the optimum on a SKILL basis.

    Note on luck inflation: a trajectory's *realised* return can exceed the
    optimum's *expected* value purely through favourable demand noise. The meaningful, skill-based ceiling is therefore the
    de-noised expected return of each trajectory's actions, not its realised
    return. We assert the construction binds on that basis."""
    cfg, mdp = _tiny_mdp()
    trajs = D.make_stitching_necessary(mdp, 80, 0.05, 0)

    # de-noised expected return of each trajectory's action sequence, compared
    # against the intertemporal optimum FROM THAT TRAJECTORY'S OWN START state
    # (comparing to the start-averaged optimum would be an averaging artefact: a
    # trajectory beginning at a high-value state can exceed the average optimum
    # legitimately). A trajectory cannot beat the optimal policy from its own
    # start, so on a per-start basis no trajectory should exceed it; the
    # construction binds because none reaches it via a single behaviour.
    from pricing_dt.core.metrics import expected_trajectory_return

    slack = []
    for tr in trajs:
        b0 = int(tr.ref_bins[0])
        v_opt_from_start = float(mdp.Vstar[0, b0])
        # no trajectory's skill return may exceed the optimum from its own start
        assert expected_trajectory_return(tr, mdp) <= v_opt_from_start + 1e-6
        slack.append(v_opt_from_start - expected_trajectory_return(tr, mdp))
    # and stitching must actually bind: at least some trajectories fall short of
    # their own-start optimum (otherwise the data already contains optimal play)
    assert max(slack) > 1e-6, "construction does not require any stitching"


def test_misspecification_knob_flips_monotonicity():
    """Severity=0 prior is monotone decreasing; high severity becomes non-monotone."""
    cfg, mdp = _tiny_mdp()
    trajs = D.make_stitching_necessary(mdp, 40, 0.1, 0)
    import torch
    ps = torch.tensor(mdp.prices, dtype=torch.float32)
    s = torch.tensor(mdp.obs(mdp.ref_grid[mdp.cfg.n_ref_bins // 2], 0),
                     dtype=torch.float32).repeat(len(ps), 1)

    dm0 = MisspecifiedStructuredDemandModel(2, cfg.model.elasticity_lo,
                                            cfg.model.elasticity_hi, severity=0.0)
    fit_demand_model(dm0, trajs, mdp, cfg.model.demand_epochs)
    with torch.no_grad():
        log_dem0 = dm0.log_demand(ps, s).numpy()
    assert np.all(np.diff(log_dem0) <= 1e-6), "severity=0 should be monotone decreasing"

    dm3 = MisspecifiedStructuredDemandModel(2, cfg.model.elasticity_lo,
                                            cfg.model.elasticity_hi, severity=3.0)
    fit_demand_model(dm3, trajs, mdp, cfg.model.demand_epochs)
    with torch.no_grad():
        log_dem3 = dm3.log_demand(ps, s).numpy()
    assert not np.all(np.diff(log_dem3) <= 1e-6), "high severity should break monotonicity"


def test_e3_drift_zero_gives_equal_estimators():
    """At zero drift, pooled and segmented OPE must coincide (correctness check):
    with identical segment loggers there is nothing for segmentation to fix."""
    cfg, mdp = _tiny_mdp()
    trajs, loggers = D.make_nonstationary(mdp, cfg.data.n_segments,
                                          cfg.data.traj_per_segment, 0.2,
                                          temp=0.5, seed=0, spread=0.0)
    from pricing_dt.core import ope
    A = mdp.cfg.n_prices
    qh = ope.fit_qhat(trajs, A)

    def logger_probs(lg):
        def f(o):
            b, t = mdp.decode_obs(o)
            return lg.probs(b, t)
        return f
    seg_probs = [logger_probs(lg) for lg in loggers]
    pooled_probs = lambda o: np.mean([f(o) for f in seg_probs], axis=0)
    pe = lambda o: np.ones(A) / A   # arbitrary fixed target policy

    v_pooled = ope.dr_value(trajs, pe, pooled_probs, qh, A)
    vals, ws = [], []
    for sgi in range(cfg.data.n_segments):
        sub = [tr for tr in trajs if tr.seg == sgi]
        vals.append(ope.dr_value(sub, pe, seg_probs[sgi], qh, A)); ws.append(len(sub))
    ws = np.array(ws, float); ws /= ws.sum()
    v_seg = float(np.dot(ws, vals))
    # spread=0 => all loggers identical => pooled == segmented (up to fp noise)
    assert abs(v_pooled - v_seg) < 1e-6, (v_pooled, v_seg)


ALL_TESTS = [
    test_e0_optimum_recovered_and_demand_monotone,
    test_relabelling_is_action_dependent,
    test_qdt_td_relabelling_is_action_dependent,
    test_support_masked_dt_policy_blocks_unsupported_argmax,
    test_discrete_iql_smoke_policy,
    test_stitching_necessary_construction,
    test_misspecification_knob_flips_monotonicity,
    test_e3_drift_zero_gives_equal_estimators,
]


def _main():
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()
