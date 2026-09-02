"""CLI entry point for the broad E0-E3 reproduction pipeline.

Simulator experiments (exact ground truth; no download needed):
  python run.py --exp all --preset smoke           # fast pipeline check
  python run.py --exp all --preset full            # broad simulator results
  python run.py --exp e2  --preset full

Real-data track (Online Retail II, UCI id=502):
  python run.py --exp calib   --real-data online_retail_II.csv
  python run.py --exp realism --preset full --real-data online_retail_II.csv

`calib` estimates real price elasticity and prints simulator parameters to use;
`realism` runs vanilla-DT/BC + OPE on real held-out logs (OPE estimates only,
since real data has no ground-truth policy value). The smoke preset is for
pipeline checks. Current Q-DT-relative claims should be read through the
fixed-QDT diagnostics in README/RUN_GUIDE, not only this broad runner.

Reference anchors: Chen (2012) for Online Retail II; Chen et al. (2021) for
Decision Transformer; Yamagata et al. (2023) for Q-DT; Jiang and Li (2016) and
Uehara, Shi and Kallus (2026) for OPE.
"""
import argparse
import os
from pricing_dt.core import config as C
from pricing_dt.core import provenance
from pricing_dt.core.torch_utils import device_report
from pricing_dt.experiments import experiments as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["e0", "seqnec", "e1", "e2", "e2ab", "mis", "e3", "all",
                             "calib", "realism"])
    ap.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--real-data", dest="real_data", default=None,
                    help="Path to Online Retail II CSV/XLSX. If omitted, tries ucimlrepo as a best-effort fallback.")
    args = ap.parse_args()

    cfg = C.smoke() if args.preset == "smoke" else C.full()
    outdir = args.outdir or cfg.exp.outdir
    os.makedirs(outdir, exist_ok=True)
    # This is the entry point RUN_GUIDE tells a reader to use, so it records the same
    # commit, dirty-tree state, device and library versions as every other run.
    provenance.stamp(outdir, replace=True,
                     extra={"preset": args.preset, "exp": args.exp})
    print(f"[pricing_dt] preset={args.preset}  {device_report()}")

    def show(name, obj):
        print(f"\n===== {name} =====")
        print(obj if not isinstance(obj, tuple) else obj[-1])

    if args.exp in ("e0", "all"):
        show("E0 testbed", E.e0_testbed(cfg, outdir))
    if args.exp in ("seqnec", "all"):
        show("C0 sequential-necessity (RQ1) - gap vs reference strength", E.e_seqnec(cfg, outdir))
    if args.exp in ("e1", "all"):
        show("E1 vanilla-DT failure (RQ1/C1)", E.e1_vanilla_failure(cfg, outdir))
    if args.exp in ("e2", "all"):
        show("E2 broad factorial diagnostic", E.e2_core(cfg, outdir))
    if args.exp in ("e2ab", "all"):
        show("E2-AB prior-isolation diagnostic", E.e2ab_ablation(cfg, outdir))
    if args.exp in ("mis", "all"):
        show("Misspecification scan (formal) - advantage vs prior wrongness", E.e2_mis(cfg, outdir))
    if args.exp in ("e3", "all"):
        show("E3 non-stationary OPE (RQ3/C3) - |bias| by drift", E.e3_ope(cfg, outdir))
    if args.exp == "calib":
        try:
            show("Real-data elasticity calibration", E.calibrate(cfg, outdir, args.real_data))
        except RuntimeError as exc:
            print(f"\nERROR: {exc}")
            raise SystemExit(2)
    if args.exp == "realism":
        try:
            show("Real-data realism check (OPE only)", E.e_realism(cfg, outdir, args.real_data))
        except RuntimeError as exc:
            print(f"\nERROR: {exc}")
            raise SystemExit(2)

    print(f"\nAll CSVs written to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
