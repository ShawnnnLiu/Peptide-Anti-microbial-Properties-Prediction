#!/usr/bin/env python3
"""
Compare MD features for the 3 replicate peptides against the original 4 ns run.

OpenMM's LangevinIntegrator uses a clock-based RNG seed when none is supplied
(StaPep does not call setRandomNumberSeed), so re-launching the same script
already produces an independent trajectory — no patch required to obtain
replicate behaviour.

Inputs:
    stapled_amps_features_training_XZ_md4ns.csv               (replicate 1)
    stapled_amps_features_training_XZ_md4ns_REP2.csv          (replicate 2)

Output:
    md_replicate_comparison.csv  (per-peptide, per-feature delta)
    md_replicate_comparison.txt  (human-readable summary)
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REP1 = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns.csv")
REP2 = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns_REP2.csv")
OUT_CSV = os.path.join(HERE, "md_replicate_comparison.csv")
OUT_TXT = os.path.join(HERE, "md_replicate_comparison.txt")

MD_FEATS = ["helix_percent", "sheet_percent", "loop_percent",
            "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]
SEQ_FEATS = ["length", "weight", "hydrophobic_index", "charge",
             "isoelectric_point", "lyticity_index"]


def main() -> None:
    if not os.path.exists(REP2):
        print(f"[ERROR] Replicate file not found: {REP2}", file=sys.stderr)
        print("Run the replicate MD job first (see WSL command).", file=sys.stderr)
        sys.exit(1)

    a = pd.read_csv(REP1, encoding="utf-8").set_index("DRAMP_ID")
    b = pd.read_csv(REP2, encoding="utf-8").set_index("DRAMP_ID")

    common = sorted(set(a.index) & set(b.index))
    if not common:
        print("[ERROR] No common DRAMP_IDs between rep1 and rep2.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for did in common:
        for f in MD_FEATS:
            v1, v2 = a.loc[did, f], b.loc[did, f]
            rows.append({
                "DRAMP_ID": did,
                "feature":  f,
                "rep1":     v1,
                "rep2":     v2,
                "delta":    (v2 - v1) if pd.notna(v1) and pd.notna(v2) else np.nan,
                "abs_delta": abs(v2 - v1) if pd.notna(v1) and pd.notna(v2) else np.nan,
                "rel_pct":  (100.0 * (v2 - v1) / v1) if pd.notna(v1) and v1 != 0 else np.nan,
            })
    long = pd.DataFrame(rows)
    long.to_csv(OUT_CSV, index=False, encoding="utf-8")

    buf: list[str] = []
    buf.append("=" * 78)
    buf.append("  MD reproducibility — replicate 2 vs replicate 1")
    buf.append("=" * 78)
    buf.append(f"  Replicates compared: rep1 = {os.path.basename(REP1)}")
    buf.append(f"                       rep2 = {os.path.basename(REP2)}")
    buf.append(f"  Common DRAMP_IDs   : {len(common)}  ({', '.join(common)})")
    buf.append(f"  Per-peptide delta CSV -> {OUT_CSV}")
    buf.append("")
    buf.append("Per-feature noise summary across the {} peptides:".format(len(common)))
    buf.append(f"  {'feature':<18}{'mean(|delta|)':>15}{'max(|delta|)':>15}"
               f"{'mean(rep1)':>14}{'noise/signal':>15}")
    summary_rows = []
    for f in MD_FEATS:
        sub = long[long["feature"] == f]
        mean_abs = sub["abs_delta"].mean()
        max_abs  = sub["abs_delta"].max()
        ref_mu   = a.loc[common, f].mean()
        ratio    = mean_abs / abs(ref_mu) if ref_mu else float("nan")
        summary_rows.append(dict(feature=f, mean_abs=mean_abs, max_abs=max_abs,
                                 ref_mean=ref_mu, ratio=ratio))
        buf.append(f"  {f:<18}{mean_abs:>15.4f}{max_abs:>15.4f}"
                   f"{ref_mu:>14.3f}{ratio:>15.3f}")

    buf.append("")
    buf.append("Per-peptide MD-feature deltas (rep2 - rep1):")
    pivot = long.pivot(index="DRAMP_ID", columns="feature", values="delta")[MD_FEATS]
    pivot = pivot.loc[common]
    for line in pivot.round(4).to_string().split("\n"):
        buf.append(f"  {line}")

    buf.append("")
    buf.append("Sequence-only features (should be identical across replicates):")
    for f in SEQ_FEATS:
        if f in a.columns and f in b.columns:
            d = (b.loc[common, f] - a.loc[common, f]).abs().max()
            buf.append(f"  {f:<22}  max |delta| across {len(common)} peptides = {d}")

    buf.append("")
    buf.append("Interpretation guide:")
    buf.append("  * mean |delta| / mean(rep1) is the rough fractional noise floor.")
    buf.append("    Values < 0.05 mean the feature is reproducible at the per-peptide level.")
    buf.append("    Values 0.05-0.20 mean cohort-level statistics are reliable but per-peptide")
    buf.append("    labels are noisy; use as ranked features, not absolute values.")
    buf.append("    Values > 0.20 mean a single trajectory does not pin the feature down;")
    buf.append("    consider triplicate-and-average or longer MD before using as regression label.")

    text = "\n".join(buf)
    print(text)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nReport saved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
