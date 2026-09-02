"""Go / no-go check for the appendix LLM pilot run.

Run the pilot (one seed, log_only) first, then point this at its output directory.
It answers one question: is this model producing a *policy*, or is it producing
noise that the harness will silently convert into a myopic fallback?

    python check_llm_pilot.py <pilot-outdir>

The pilot directory it was written for is not part of this package; the two language-model
batches that ship are results_appendix_llm_controls_20260821/ and
results_appendix_llm_deepseek_20260824/.

Three gates, in the order that matters. Failing gate 1 or 2 means the normalised
value is not a measurement of the model and the full run would waste money.
"""
import sys
import os
import pandas as pd

FLOOR = 0.448          # random + mask, 10 seeds (results_appendix_llm_controls_20260821)
CHANCE_MYOPIC = 1 / 11  # uniform choice over the price grid


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]
    raw = pd.read_csv(os.path.join(d, "appendix_llm_raw.csv"))

    # The 0.448 floor was measured on the FULL preset (11 prices, H=8, N=100). A
    # smoke run is a different environment (9 prices, H=6, N=60) with its own
    # floor, so comparing the two is meaningless. Detect and withhold rather than
    # print a number that looks authoritative.
    preset = "unknown"
    ppath = os.path.join(d, "appendix_llm_protocol.json")
    if os.path.exists(ppath):
        import json
        preset = json.load(open(ppath, encoding="utf-8")).get("preset", "unknown")
    comparable = preset == "full"
    llm = raw[raw.method.str.startswith("LLM ")]
    if llm.empty:
        sys.exit(f"No LLM arm in {d}. Was --provider set to something other than heuristic?")

    bare = llm[llm.support_mask == "none"]
    masked = llm[llm.support_mask != "none"]
    name = bare.method.iloc[0]
    print(f"\nmodel arm : {name}")
    print(f"seeds     : {sorted(bare.seed.unique())}")
    print(f"states    : {len(pd.read_csv(os.path.join(d, 'appendix_llm_actions.csv')))} action rows\n")

    ok = True

    # --- gate 1: did the responses parse? ---
    v = bare.valid_output_rate.mean()
    g1 = v >= 0.90
    ok &= g1
    print(f"[{'PASS' if g1 else 'FAIL'}] gate 1  parse rate        {v:6.1%}   (need >= 90%)")
    if not g1:
        print("        -> responses are not parseable. Unparsed states fall back to the")
        print("           myopic action, so nv is not a measurement. Try raising")
        print("           --max-tokens, or inspect raw_response in appendix_llm_actions.csv.")

    # --- gate 2: is the output a policy or a constant? ---
    m = bare.modal_action_share.mean()
    dsn = bare.distinct_actions_used.mean()
    ent = bare.action_entropy_normalised.mean()
    mm = bare.match_myopic_rate.mean()
    g2 = m <= 0.90 and dsn >= 3
    ok &= g2
    print(f"[{'PASS' if g2 else 'FAIL'}] gate 2  modal share       {m:6.1%}   (need <= 90%)")
    print(f"                 distinct actions   {dsn:6.1f}   (need >= 3)")
    print(f"                 entropy (norm.)    {ent:6.3f}")
    print(f"                 myopic-match       {mm:6.1%}   (chance = {CHANCE_MYOPIC:.1%})")
    if not g2:
        print("        -> near-constant output: this is a degenerate emission, not a")
        print("           policy. STOP. Report the degeneracy; do not run the full grid.")
    elif mm < CHANCE_MYOPIC * 0.5:
        print("        -> note: myopic-match is well below chance, which is itself a")
        print("           red flag even though the modal-share gate passed.")

    # --- gate 3: is it an interesting point on the dose-response axis? ---
    off = bare.off_support_rate.mean()
    print(f"[INFO] gate 3  off-support rate    {off:6.1%}   (higher = more interesting)")

    print(f"\n{'-' * 62}")
    if masked.empty:
        print("No masked arm found.")
    else:
        b, mk = bare.nv.mean(), masked.nv.mean()
        print(f"bare nv                {b:+.4f}")
        print(f"masked nv              {mk:+.4f}   (mask gain {mk - b:+.4f})")
        if comparable:
            print(f"no-learner floor       {FLOOR:+.4f}   (random + mask, 10 seeds)")
            print(f"attributable to model  {mk - FLOOR:+.4f}   <- the number that matters")
        else:
            print(f"no-learner floor       n/a      (preset is '{preset}', not 'full')")
            print("        -> the 0.448 floor was measured on the full preset; a smoke")
            print("           run is a different environment with its own floor, so this")
            print("           comparison is withheld rather than printed misleadingly.")
        if not comparable:
            pass
        elif not ok:
            print("\n*** The figures above are NOT interpretable: a gate failed. ***")
        elif mk - FLOOR > 0.05:
            print("\nReading: clears the floor -> a positive result about prompt-only")
            print("pricing under a support constraint.")
        else:
            print("\nReading: at or below the floor -> the LLM behaves like every other")
            print("unconstrained action channel and contributes little once contained.")
            print("This is a publishable outcome, not a failed run.")

    if bare.off_support_rate.mean() < 0.02:
        print("\nNOTE: bare off-support is ~0, so the mask has nothing to do and the")
        print("      mask gain is structurally 0. In log_only that is expected of ANY")
        print("      ranker, because unobserved actions carry a null revenue and are")
        print("      avoided without being forbidden -- the heuristic proxy does the")
        print("      same. To test whether the constraint BINDS, use")
        print("      --info-modes oracle_info, where every action carries a number and")
        print("      going off-support is a live choice.")

    print(f"\n{'GO -- proceed' if ok else 'NO-GO -- do not spend on the full grid'}")
    if not comparable:
        print("Next: re-run the pilot on the FULL preset (drop --smoke, keep --seeds 0)")
        print("      before spending on the whole grid.\n")
    else:
        print("Full run: drop --seeds; pick --info-modes per the NOTE above.\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
