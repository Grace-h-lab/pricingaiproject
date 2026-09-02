"""Second-environment FEASIBILITY PROBE — does condition (b) hold?

Run this BEFORE building the full channel suite on the inventory environment. It asks
one question only: **is planning against the structural prior actually toxic here?**

The channel contrast needs both halves. The goal-channel half ("relabelling helps") is
the easy one; if the action-channel half ("planning is destroyed by the same model")
does not reproduce, the environment cannot support the comparison at all and no amount
of relabelling work rescues it. The pricing testbed's signature is specific and must be
matched qualitatively:

    estimate-then-optimize true value  -4.543 (far BELOW the logging policy)
    planner's in-model value           ~1800 vs true optimum ~480  (3.75x)
    optimizer's-curse gap              +1631

so the probe reports the same three quantities. Passing is not "the planner is a bit
worse"; it is "the planner believes something wildly untrue and is destroyed by it".

ARMS
  logging policy         conservative order-up-to (the data source, the 0 anchor)
  optimal policy         exact DP on the true kernel (the 1 anchor)
  EtO_poisson            fit a Poisson demand (structural prior: variance == mean, so
                         overdispersion is UNREPRESENTABLE), plan on it, evaluate truly
  EtO_empirical          fit a free histogram over observed sales (no structural prior),
                         plan on it, evaluate truly -- the unconstrained comparator

Both fitted models see only CENSORED observations (`sales = min(demand, available)`),
which is what the logs contain. The probe also reports the fitted vs true demand
mean/variance so the mechanism is visible rather than inferred.
"""
import argparse
import numpy as np
import pandas as pd
import os

from pricing_dt.envs.inventory import InvConfig, InventoryMDP, OrderUpTo, _poisson_pmf


def generate_logs(mdp, logger, n_ep, rng):
    """Roll the logging policy, recording what a real log would contain."""
    rows = []
    for _ in range(n_ep):
        x = int(rng.integers(0, mdp.cfg.max_inventory // 2 + 1))
        for t in range(mdp.H):
            a = logger.action(x, t)
            rew, nxt, sales, censored = mdp.step(x, a, rng)
            rows.append(dict(t=t, x=x, a=a, reward=rew, sales=sales,
                             avail=min(x + a, mdp.cfg.max_inventory),
                             censored=censored, next_x=nxt))
            x = nxt
    return pd.DataFrame(rows)


def _uncensored(log):
    """Only uncensored observations reveal true demand; using all `sales` values
    biases the MEAN downward as well as the variance, which confounds the structural
    effect under test with ordinary censoring bias. Standard practice, and here it
    isolates the one error we care about: Poisson cannot express overdispersion."""
    u = log[~log["censored"].astype(bool)]
    return u if len(u) > 10 else log


def fit_poisson(log, kmax):
    """Structural prior: Poisson fitted to uncensored demand. Variance is then forced
    equal to the mean, so overdispersion is structurally unrepresentable."""
    return _poisson_pmf(max(_uncensored(log)["sales"].mean(), 1e-6), kmax)


def fit_empirical(log, kmax):
    """No structural prior: free histogram over uncensored demand."""
    p = np.zeros(kmax + 1)
    for s in _uncensored(log)["sales"].values:
        p[int(min(s, kmax))] += 1
    return p / max(p.sum(), 1)


def pmf_stats(p):
    k = np.arange(len(p))
    m = float(np.dot(k, p))
    return m, float(np.dot((k - m) ** 2, p))


def plan_and_evaluate(mdp, pmf):
    """Backward induction on the FITTED demand, then exact evaluation on the TRUE one.
    Returns (true value, in-model value, tabular policy)."""
    Rh, Ph = mdp._step_arrays(pmf)
    _Q, Vh, pih = mdp.solve_optimal(R=Rh, P=Ph)          # plan in the fitted model
    mdp.solve_optimal()                                   # restore true Q*/V*/pi*
    V_true = mdp.evaluate_tabular_policy(pih)             # evaluate on the truth
    return V_true, Vh, pih


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--level", type=int, default=14,
                    help="order-up-to level of the logger. Swept during the probe: lower means "
                         "more censoring but also a WORSE logger, which collapses the "
                         "normalisation anchor and makes 'beats the logger' a trivial bar. "
                         "14 keeps the logger competent (70.5 of an 80.9 optimum) while "
                         "still censoring the spike tail (13.6%).")
    args = ap.parse_args()

    cfg = InvConfig()
    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        mdp = InventoryMDP(InvConfig(seed=seed))
        mdp.solve_optimal()

        # initial state distribution: uniform over the lower half of the grid
        init = np.zeros(mdp.n_states)
        init[: mdp.cfg.max_inventory // 2 + 1] = 1.0
        init /= init.sum()

        logger = OrderUpTo(mdp, args.level, rng)
        log = generate_logs(mdp, logger, args.episodes, rng)

        v_beh = mdp.evaluate_policy_fn(
            lambda o: logger.action(*mdp.decode_obs(o)), init)
        v_opt = float(init @ mdp.Vstar[0])

        out = {"seed": seed, "v_beh": round(v_beh, 1), "v_opt": round(v_opt, 1),
               "censor_rate": round(float(log["censored"].mean()), 3),
               "true_mean": round(mdp.demand_mean(), 2),
               "true_var": round(mdp.demand_var(), 2)}

        for tag, fitter in (("poisson", fit_poisson), ("empirical", fit_empirical)):
            pmf = fitter(log, cfg.max_demand)
            m, v = pmf_stats(pmf)
            V_true, V_model, _pi = plan_and_evaluate(mdp, pmf)
            v_true = float(init @ V_true[0])
            v_model = float(init @ V_model[0])
            denom = v_opt - v_beh
            out[f"fit_mean_{tag}"] = round(m, 2)
            out[f"fit_var_{tag}"] = round(v, 2)
            out[f"EtO_{tag}_true"] = round(v_true, 1)
            out[f"EtO_{tag}_inmodel"] = round(v_model, 1)
            out[f"EtO_{tag}_gap"] = round(v_model - v_true, 1)
            out[f"EtO_{tag}_nv"] = round((v_true - v_beh) / denom, 3) if denom > 1e-8 else float("nan")
        rows.append(out)
        print(f"seed {seed}: censor={out['censor_rate']:.2f} | "
              f"poisson nv={out['EtO_poisson_nv']:+.2f} "
              f"(in-model {out['EtO_poisson_inmodel']:.0f} vs true {out['EtO_poisson_true']:.0f}) | "
              f"empirical nv={out['EtO_empirical_nv']:+.2f}")

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "env2_probe.csv"), index=False)

    print("\n=== environment sanity ===")
    print(f"  true demand mean {df.true_mean.mean():.2f}  variance {df.true_var.mean():.2f} "
          f"(overdispersed: variance/mean = {df.true_var.mean()/df.true_mean.mean():.2f})")
    print(f"  censoring rate under the logger: {df.censor_rate.mean():.1%}")
    print(f"  logging value {df.v_beh.mean():.1f}   optimal value {df.v_opt.mean():.1f}")

    print("\n=== condition (b): is planning against the prior TOXIC? ===")
    print(f"  {'model':<12} {'fit mean':>9} {'fit var':>9} {'in-model':>9} "
          f"{'true':>8} {'curse gap':>10} {'nv':>7}")
    for tag in ("poisson", "empirical"):
        print(f"  {tag:<12} {df['fit_mean_'+tag].mean():>9.2f} {df['fit_var_'+tag].mean():>9.2f} "
              f"{df['EtO_'+tag+'_inmodel'].mean():>9.1f} {df['EtO_'+tag+'_true'].mean():>8.1f} "
              f"{df['EtO_'+tag+'_gap'].mean():>+10.1f} {df['EtO_'+tag+'_nv'].mean():>+7.3f}")

    nv = df["EtO_poisson_nv"].mean()
    gap = df["EtO_poisson_gap"].mean()
    var_ratio = df["fit_var_poisson"].mean() / df.true_var.mean()
    print("\n=== verdict ===")
    print(f"  structural prior under-states demand variance by "
          f"{100*(1-var_ratio):.0f}% ({df['fit_var_poisson'].mean():.1f} vs "
          f"{df.true_var.mean():.1f}) -- the mechanism is present."
          if var_ratio < 0.8 else
          f"  WARNING: the prior does NOT under-state variance (ratio {var_ratio:.2f}); "
          f"the intended mechanism is absent.")
    if nv < 0 and gap > 0:
        print(f"  >>> (b) HOLDS. Planning on the structural prior lands BELOW the logging "
              f"policy (nv {nv:+.3f}) while the model believes it is worth "
              f"{df['EtO_poisson_inmodel'].mean():.0f} (curse gap {gap:+.1f}). "
              f"The environment can support the channel contrast; proceed to the goal-channel half.")
    elif gap > 0:
        print(f"  >>> (b) PARTIAL. The curse gap is positive ({gap:+.1f}) but planning still "
              f"beats the logger (nv {nv:+.3f}). The optimism exists but is not destructive; "
              f"strengthen it (lower the order-up-to level, raise p_spike/mu_spike) before "
              f"committing to the full suite.")
    else:
        print(f"  >>> (b) FAILS (nv {nv:+.3f}, gap {gap:+.1f}). Planning is not toxic here, so "
              f"this environment cannot reproduce the action-channel half. Do not build the "
              f"full suite on it.")


if __name__ == "__main__":
    main()
