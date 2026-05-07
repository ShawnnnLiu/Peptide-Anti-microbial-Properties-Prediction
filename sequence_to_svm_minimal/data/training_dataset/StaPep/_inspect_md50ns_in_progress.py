#!/usr/bin/env python3
"""
READ-ONLY snapshot of the in-progress 50 ns MD run.

Compares the rows already saved in stapled_amps_features_training_XZ_md50ns.csv
against the same DRAMP_IDs in the 4 ns reference, and answers:

  * Are partial rows clean (no NaN MD, no extraction_error)?
  * Did the canary DRAMP21542 escape the 0% helix trap?
  * Did the cohort mean helix shift up vs the 4 ns run?
  * Did per-peptide noise drop vs the 4 ns single-trajectory + replicate spread?

Does NOT write to any file the running job uses. Safe to invoke at any time.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

HERE   = os.path.dirname(os.path.abspath(__file__))
F50    = os.path.join(HERE, "stapled_amps_features_training_XZ_md50ns.csv")
F04    = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns.csv")
F04R2  = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns_REP2.csv")

MD = ["helix_percent", "sheet_percent", "loop_percent",
      "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]


def main() -> None:
    if not os.path.exists(F50):
        print(f"[ERROR] {F50} not found yet.", file=sys.stderr)
        sys.exit(1)
    df50 = pd.read_csv(F50, encoding="utf-8")
    df04 = pd.read_csv(F04, encoding="utf-8")

    print("=" * 72)
    print(f"  Snapshot of {os.path.basename(F50)}")
    print("=" * 72)
    print(f"  Rows in partial 50 ns CSV : {len(df50)}")
    print(f"  Rows in reference 4 ns CSV: {len(df04)}")

    err_mask = df50["extraction_error"].notna() & (df50["extraction_error"].astype(str).str.strip() != "")
    print(f"  Rows with extraction_error: {int(err_mask.sum())}")
    if err_mask.any():
        print(df50.loc[err_mask, ["DRAMP_ID", "extraction_error"]].to_string(index=False))

    nan_mask = df50[MD].isna().any(axis=1)
    print(f"  Rows with any NaN MD col  : {int(nan_mask.sum())}")
    if nan_mask.any():
        print(df50.loc[nan_mask, ["DRAMP_ID"] + MD].to_string(index=False))

    # ── Cohort stats overlap (only peptides present in both) ──
    common = sorted(set(df50["DRAMP_ID"]) & set(df04["DRAMP_ID"]))
    a50 = df50[df50["DRAMP_ID"].isin(common)].set_index("DRAMP_ID").sort_index()
    a04 = df04[df04["DRAMP_ID"].isin(common)].set_index("DRAMP_ID").sort_index()
    print()
    print(f"  Common DRAMP_IDs (50 ns ∩ 4 ns): {len(common)}")

    print()
    print("  ── Cohort mean per MD feature (overlap subset) ──")
    print(f"  {'feature':<18}{'4 ns':>12}{'50 ns':>12}{'shift':>12}{'rel%':>10}")
    for f in MD:
        m04 = a04[f].mean()
        m50 = a50[f].mean()
        shift = m50 - m04
        rel = (100.0 * shift / abs(m04)) if m04 else float("nan")
        print(f"  {f:<18}{m04:>12.3f}{m50:>12.3f}{shift:>+12.3f}{rel:>+10.1f}%")

    # ── Canary check ──
    canary = "DRAMP21542"
    print()
    print(f"  ── Canary peptide: {canary} (Mag(i+4)1,15(A9K) — was 0% helix at 4 ns) ──")
    if canary in df50["DRAMP_ID"].values:
        r50 = df50[df50["DRAMP_ID"] == canary].iloc[0]
        r04 = df04[df04["DRAMP_ID"] == canary].iloc[0]
        for f in MD:
            print(f"     {f:<18} 4 ns = {r04[f]:>9.3f}    50 ns = {r50[f]:>9.3f}    Δ = {r50[f]-r04[f]:+.3f}")
        if r50["helix_percent"] > 0.20:
            print(f"     >>> ESCAPED the 0% trap. helix_percent climbed from {r04['helix_percent']:.3f} to {r50['helix_percent']:.3f}")
        elif r50["helix_percent"] > r04["helix_percent"] + 0.05:
            print(f"     >>> partial escape: helix climbed but still below 0.20")
        else:
            print(f"     >>> still trapped — 50 ns did not help this peptide")
    else:
        print(f"     {canary} not yet in the partial CSV. Estimated remaining queue position: "
              f"~{len(df04) - len(df50)} peptides.")

    # ── Per-peptide head-to-head ──
    print()
    print("  ── Per-peptide deltas (50 ns − 4 ns), abs values ──")
    delta = (a50[MD] - a04[MD]).abs()
    rows = []
    for f in MD:
        rows.append({
            "feature":      f,
            "mean_abs_d":   delta[f].mean(),
            "med_abs_d":    delta[f].median(),
            "max_abs_d":    delta[f].max(),
            "ref_mean":     a04[f].mean(),
            "shift/ref":    (a50[f].mean() - a04[f].mean()) / abs(a04[f].mean()) if a04[f].mean() else float("nan"),
        })
    summary = pd.DataFrame(rows)
    print(summary.round(3).to_string(index=False))

    # ── Replicate-MD noise floor reference ──
    if os.path.exists(F04R2):
        rep2 = pd.read_csv(F04R2, encoding="utf-8")
        rep_common = sorted(set(rep2["DRAMP_ID"]) & set(df50["DRAMP_ID"]) & set(df04["DRAMP_ID"]))
        if rep_common:
            print()
            print(f"  ── Replicate-noise floor reference (peptides: {', '.join(rep_common)}) ──")
            print(f"     Compares 4 ns rep1 vs 4 ns rep2 (same protocol, different RNG seed).")
            r1 = df04[df04["DRAMP_ID"].isin(rep_common)].set_index("DRAMP_ID")
            r2 = rep2[rep2["DRAMP_ID"].isin(rep_common)].set_index("DRAMP_ID")
            d50 = a50[a50.index.isin(rep_common)]
            print(f"     {'feature':<18}{'rep-noise':>12}{'50 ns shift':>14}{'snr':>10}")
            for f in MD:
                noise = (r1[f] - r2[f]).abs().mean()
                shift = abs(d50[f].mean() - r1[f].mean())
                snr = shift / noise if noise else float("nan")
                tag = "  ← signal beats noise" if snr > 1 else ""
                print(f"     {f:<18}{noise:>12.4f}{shift:>+14.4f}{snr:>10.2f}{tag}")

    print()
    print("  (snapshot — the run is still progressing; numbers will update as more rows land.)")


if __name__ == "__main__":
    main()
