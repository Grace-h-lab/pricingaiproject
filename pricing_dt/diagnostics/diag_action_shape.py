"""Action-level shape mechanism probe — closes the optimism-verdict loop.

The optimism verdict (diag_optimism_verdict.py) showed A_structured beats a
magnitude-MATCHED unstructured optimism (B_matchA): the
advantage is the SHAPE of the target, not its level. But the start-level
shape-corr did NOT explain it (B ranks START states marginally better). The
mechanism must live at ACTION granularity: from each state, does the relabel
target rank the ACTIONS like the true Q*(s,a)? The return-conditioned DT is
pushed toward whichever action receives the higher achievable target FROM A GIVEN
STATE, so the per-state action ranking is what drives which action it stitches.

For each demand model we build its implied achievable-return Q-surface
  Qhat_dm(s, a) = r_hat(s, a) + sum_{k>t} max_p r_hat(s_k, p)   (Algorithm 1, but
for EVERY candidate first action a, rolled forward under the known transition),
and compare its per-state action ranking against the exact Qstar[t, b, :].

KEY: B_matchA = B scaled by a per-seed constant. Scaling is monotone, so it
leaves every per-state action RANKING identical to B. Hence if B ranks actions
worse than A, NO global inflation can fix it -> this is precisely why B_matchA
cannot reach A. The probe makes that mechanism explicit.

Metrics, per seed (hardest cell), averaged over states:
  spearman_A/B   : mean Spearman(Qhat_dm(s,.), Qstar(s,.)) over actions.
  argmax_A/B     : fraction of states where argmax_a Qhat == argmax_a Qstar
                   (the optimal action the DT would be steered toward).
Reported for ALL states and split by logged COVERAGE (visited vs unvisited
(ref_bin, t)), since the prior should help most where B has no data.
"""
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr

from pricing_dt.core import config as C
from pricing_dt.core import data as D
from pricing_dt.experiments.experiments import _setup, _seed
from pricing_dt.core.demand_model import StructuredDemandModel, UnconstrainedDemandModel, fit_demand_model


def demand_q(dm, mdp, b0, t0, device="cpu"):
    """Achievable-return surface over first actions from state (ref bin b0, t0):
    Qhat[a] = de-noised revenue of action a at (ref,t0) + greedy demand-model
    continuation under the KNOWN reference-price transition. Mirrors relabel.achievable_rtg."""
    prices = torch.tensor(mdp.prices, dtype=torch.float32, device=device)
    A, H = len(mdp.prices), mdp.H
    out = np.zeros(A, np.float32)
    dm.eval()
    with torch.no_grad():
        for a in range(A):
            ref = mdp.ref_grid[b0]
            s_t = torch.tensor(mdp.obs(ref, t0), dtype=torch.float32, device=device).unsqueeze(0)
            total = float(prices[a] * dm(prices[a].unsqueeze(0), s_t).squeeze())
            ref = mdp.ref_grid[mdp.N[a, mdp.ref_to_bin(ref)]]
            for k in range(t0 + 1, H):
                s = torch.tensor(mdp.obs(ref, k), dtype=torch.float32, device=device)
                s_rep = s.unsqueeze(0).repeat(A, 1)
                rev = (prices * dm(prices, s_rep)).cpu().numpy()
                a_best = int(rev.argmax())
                total += float(rev[a_best])
                ref = mdp.ref_grid[mdp.N[a_best, mdp.ref_to_bin(ref)]]
            out[a] = total
    return out


def _ranking_stats(dm, mdp, covered):
    """Mean Spearman(Qhat, Qstar) and argmax-agreement over states, split by coverage."""
    H, B = mdp.H, mdp.cfg.n_ref_bins
    rec = {k: [] for k in ("sp_all", "am_all", "sp_in", "am_in", "sp_out", "am_out")}
    for t in range(H):
        for b in range(B):
            qhat = demand_q(dm, mdp, b, t)
            qstar = mdp.Qstar[t, b, :]
            if qhat.std() < 1e-9 or qstar.std() < 1e-9:
                continue
            sp = spearmanr(qhat, qstar).correlation
            am = float(int(qhat.argmax()) == int(qstar.argmax()))
            rec["sp_all"].append(sp); rec["am_all"].append(am)
            tag = "in" if (b, t) in covered else "out"
            rec[f"sp_{tag}"].append(sp); rec[f"am_{tag}"].append(am)
    return {k: (float(np.mean(v)) if v else float("nan")) for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg = C.smoke() if args.smoke else C.full()
    obs_dim = 2
    N = min(cfg.exp.data_sizes); noise = max(cfg.exp.noise_levels)
    seeds = cfg.exp.seeds
    print(f"delta={cfg.sim.delta}  cell: N={N} noise={noise}  seeds={seeds}\n")

    rows = []
    for seed in seeds:
        _seed(seed)
        mdp, init, v_opt, v_beh = _setup(cfg, {"demand_noise": noise}, seed=seed)
        trajs = D.make_stitching_necessary(mdp, N, noise, seed, cfg.data.expert_q)
        covered = set((int(tr.ref_bins[t]), t) for tr in trajs for t in range(mdp.H))

        dmA = StructuredDemandModel(obs_dim, cfg.model.elasticity_lo, cfg.model.elasticity_hi)
        fit_demand_model(dmA, trajs, mdp, cfg.model.demand_epochs)
        dmB = UnconstrainedDemandModel(obs_dim)
        fit_demand_model(dmB, trajs, mdp, cfg.model.demand_epochs)

        sA = _ranking_stats(dmA, mdp, covered)
        sB = _ranking_stats(dmB, mdp, covered)
        cov_frac = len(covered) / (mdp.H * mdp.cfg.n_ref_bins)
        rows.append(dict(seed=seed, coverage=round(cov_frac, 2),
                         spearman_A=round(sA["sp_all"], 3), spearman_B=round(sB["sp_all"], 3),
                         argmax_A=round(sA["am_all"], 3), argmax_B=round(sB["am_all"], 3),
                         spearman_A_incov=round(sA["sp_in"], 3), spearman_B_incov=round(sB["sp_in"], 3),
                         spearman_A_oocov=round(sA["sp_out"], 3), spearman_B_oocov=round(sB["sp_out"], 3),
                         argmax_A_oocov=round(sA["am_out"], 3), argmax_B_oocov=round(sB["am_out"], 3)))
        print(f"seed {seed}: spearman A={sA['sp_all']:.3f} B={sB['sp_all']:.3f} | "
              f"argmax A={sA['am_all']:.3f} B={sB['am_all']:.3f} | "
              f"OOcov spearman A={sA['sp_out']:.3f} B={sB['sp_out']:.3f}")

    import pandas as pd, os
    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "action_shape_scan.csv"), index=False)

    print("\n=== mechanism summary (mean over seeds) ===")
    print(f"  per-state action-rank Spearman vs Qstar : A={df.spearman_A.mean():.3f}  B={df.spearman_B.mean():.3f}")
    print(f"  optimal-action argmax agreement          : A={df.argmax_A.mean():.3f}  B={df.argmax_B.mean():.3f}")
    print(f"  Spearman, IN-coverage states             : A={df.spearman_A_incov.mean():.3f}  B={df.spearman_B_incov.mean():.3f}")
    print(f"  Spearman, OUT-of-coverage states         : A={df.spearman_A_oocov.mean():.3f}  B={df.spearman_B_oocov.mean():.3f}")
    print(f"  argmax,   OUT-of-coverage states         : A={df.argmax_A_oocov.mean():.3f}  B={df.argmax_B_oocov.mean():.3f}")
    dsp = df.spearman_A.mean() - df.spearman_B.mean()
    verdict = ("A ranks actions vs Qstar BETTER -> action-level shape is the mechanism"
               if dsp > 0.02 else "no action-level ranking edge for A -> mechanism unexplained")
    print(f"\n  >>> {verdict}  (A-B spearman = {dsp:+.3f})")


if __name__ == "__main__":
    main()
