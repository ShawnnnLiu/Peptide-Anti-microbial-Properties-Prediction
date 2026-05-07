#!/usr/bin/env python3
"""
Sanity-check report for the XZ-only 4 ns MD training feature CSV.

Three layers of checks:

  1. Internal consistency  (DSSP partition closes to 1, scaling laws,
                            non-negative ranges) — any failure is a bug.
  2. Distributional sanity (feature ranges, AMP vs decoy separation,
                            outlier flags at ±3 sigma).
  3. MD-protocol drift     (compare overlapping DRAMP_IDs against the
                            legacy stapled_amps_features.csv to detect
                            systematic shifts caused by the 4 ns vs
                            previous protocol).

Outputs:
  sanity_report_md4ns.txt          — human-readable report
  sanity_outliers_md4ns.csv        — per-peptide outlier table (>3 sigma in any feature)
  sanity_legacy_vs_md4ns.csv       — per-peptide deltas vs legacy run (overlapping IDs only)
"""
from __future__ import annotations
import os
import sys
from typing import Iterable

import numpy as np
import pandas as pd

HERE     = os.path.dirname(os.path.abspath(__file__))
TRAIN    = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns.csv")
LEGACY   = os.path.join(HERE, "stapled_amps_features.csv")
DECOYS   = os.path.join(HERE, "stapled_decoys.csv")

REPORT   = os.path.join(HERE, "sanity_report_md4ns.txt")
OUTLIERS = os.path.join(HERE, "sanity_outliers_md4ns.csv")
DELTA    = os.path.join(HERE, "sanity_legacy_vs_md4ns.csv")

SEQ_FEATS = ["length", "weight", "hydrophobic_index", "charge", "aromaticity",
             "isoelectric_point", "fraction_arginine", "fraction_lysine",
             "lyticity_index"]
MD_FEATS  = ["helix_percent", "sheet_percent", "loop_percent",
             "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]


def section(buf: list[str], title: str) -> None:
    bar = "=" * 72
    buf.append("")
    buf.append(bar)
    buf.append(f"  {title}")
    buf.append(bar)


def warn(buf: list[str], msg: str) -> None:
    buf.append(f"  [WARN] {msg}")


def info(buf: list[str], msg: str) -> None:
    buf.append(f"  {msg}")


def check_internal(df: pd.DataFrame, buf: list[str]) -> None:
    section(buf, "1. Internal consistency")

    n = len(df)
    info(buf, f"Rows: {n}")

    # DSSP partition closure
    dssp = df["helix_percent"] + df["sheet_percent"] + df["loop_percent"]
    bad = ((dssp - 1.0).abs() > 0.01)
    info(buf, f"helix+sheet+loop within 0.01 of 1.0: {n - int(bad.sum())}/{n}")
    if bad.any():
        warn(buf, f"DSSP partition does NOT close for {int(bad.sum())} rows:")
        for _, r in df[bad].iterrows():
            buf.append(f"      {r['DRAMP_ID']}  sum={dssp.loc[r.name]:.4f}")

    # Non-negative ranges
    for col, must in [
        ("helix_percent", ">=0,<=1"), ("sheet_percent", ">=0,<=1"),
        ("loop_percent", ">=0,<=1"), ("mean_bfactor", ">=0"),
        ("mean_gyrate", ">0"),       ("num_hbonds", ">=0"),
        ("psa", ">0"),               ("sasa", ">0"),
    ]:
        if must == ">=0,<=1":
            bad = ~df[col].between(0.0, 1.0)
        elif must == ">=0":
            bad = ~(df[col] >= 0)
        elif must == ">0":
            bad = ~(df[col] > 0)
        if bad.any():
            warn(buf, f"{col} {must} violated by {int(bad.sum())} rows: "
                       f"{df.loc[bad, 'DRAMP_ID'].tolist()[:5]}...")
        else:
            info(buf, f"{col} {must} OK ({n}/{n})")

    # PSA <= SASA
    bad = df["psa"] > df["sasa"]
    if bad.any():
        warn(buf, f"psa > sasa for {int(bad.sum())} rows (expected psa <= sasa).")

    # mean_gyrate scaling vs length
    rg_pred = 2.2 * np.power(df["length"], 0.5)   # crude alpha-helical heuristic, AA
    rel = (df["mean_gyrate"] / rg_pred).clip(0.1, 10)
    info(buf, f"mean_gyrate / (2.2 * sqrt(length))  median={rel.median():.2f} "
              f"5%={rel.quantile(0.05):.2f}  95%={rel.quantile(0.95):.2f}  "
              f"(target ~1.0; <0.7 collapsed, >1.4 extended)")


def check_distribution(df: pd.DataFrame, buf: list[str],
                       outliers_path: str) -> None:
    section(buf, "2. Distributional sanity")

    info(buf, "Per-feature stats (4 ns MD, n={}):".format(len(df)))
    stats = df[SEQ_FEATS + MD_FEATS].describe().round(3).T[["mean", "std", "min", "max"]]
    for line in stats.to_string().split("\n"):
        info(buf, line)

    # Helix expectation flag
    helix_mean = df["helix_percent"].mean()
    if helix_mean < 0.20:
        warn(buf, f"Mean helix_percent = {helix_mean:.3f}. Stapled AMPs in literature "
                   f"typically show 0.30-0.80. Consider checking whether 4 ns is "
                   f"long enough to refold from the linear starting structure, or "
                   f"whether DSSP cutoff in StaPep is strict.")
    else:
        info(buf, f"Mean helix_percent = {helix_mean:.3f} (within literature range).")

    # Decoy comparison
    if os.path.exists(DECOYS):
        dec = pd.read_csv(DECOYS, encoding="utf-8")
        info(buf, "")
        info(buf, "AMP vs Decoy mean (separation should be visible for "
                   "charge / fraction_lysine / helix_percent / lyticity_index):")
        info(buf, f"  {'feature':<22}{'AMP':>10}{'decoy':>10}{'|delta|/sigma':>16}")
        for f in MD_FEATS + ["charge", "fraction_lysine", "lyticity_index"]:
            if f in dec.columns and f in df.columns:
                amu, asd = df[f].mean(), df[f].std()
                dmu, dsd = dec[f].mean(), dec[f].std()
                pooled = (asd + dsd) / 2 if (asd + dsd) > 0 else float("nan")
                z = abs(amu - dmu) / pooled if pooled else float("nan")
                info(buf, f"  {f:<22}{amu:>10.3f}{dmu:>10.3f}{z:>16.2f}")
    else:
        warn(buf, f"No decoy file at {DECOYS}; skipping AMP-vs-decoy contrast.")

    # Outlier flag at +/-3 sigma
    z = (df[MD_FEATS] - df[MD_FEATS].mean()) / df[MD_FEATS].std()
    outlier_mask = (z.abs() > 3).any(axis=1)
    out_df = df.loc[outlier_mask, ["DRAMP_ID", "Sequence"] + MD_FEATS].copy()
    if len(out_df):
        info(buf, "")
        info(buf, f"+/-3 sigma outliers in any MD feature: {len(out_df)}")
        info(buf, f"  -> {outliers_path}")
    else:
        info(buf, "No +/-3 sigma outliers in MD features.")
    out_df.to_csv(outliers_path, index=False, encoding="utf-8")


def check_legacy_drift(df: pd.DataFrame, buf: list[str], delta_path: str) -> None:
    section(buf, "3. Drift vs legacy stapled_amps_features.csv")

    if not os.path.exists(LEGACY):
        warn(buf, f"No legacy file at {LEGACY}; skipping drift comparison.")
        return

    legacy = pd.read_csv(LEGACY, encoding="utf-8")
    common = sorted(set(df["DRAMP_ID"]) & set(legacy["DRAMP_ID"]))
    info(buf, f"Overlapping DRAMP_IDs (training vs legacy): {len(common)}")
    if not common:
        return

    a = df[df["DRAMP_ID"].isin(common)].set_index("DRAMP_ID").sort_index()
    b = legacy[legacy["DRAMP_ID"].isin(common)].set_index("DRAMP_ID").sort_index()

    rows = []
    for f in MD_FEATS:
        if f in a.columns and f in b.columns:
            d = (a[f] - b[f]).dropna()
            if len(d):
                rows.append({
                    "feature": f,
                    "n_overlap": len(d),
                    "mean_delta_4ns_minus_legacy": round(d.mean(), 4),
                    "median_delta": round(d.median(), 4),
                    "std_delta":   round(d.std(), 4),
                    "max_abs_delta": round(d.abs().max(), 4),
                })

    info(buf, "")
    info(buf, "Per-feature drift (positive = 4 ns larger than legacy):")
    drift = pd.DataFrame(rows)
    for line in drift.to_string(index=False).split("\n"):
        info(buf, line)

    # Per-peptide table
    per_pep = (a[MD_FEATS] - b[MD_FEATS]).reset_index()
    per_pep.to_csv(delta_path, index=False, encoding="utf-8")
    info(buf, f"  per-peptide delta table -> {delta_path}")


def main() -> None:
    if not os.path.exists(TRAIN):
        print(f"Missing {TRAIN}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(TRAIN, encoding="utf-8")

    buf: list[str] = []
    section(buf, f"StaPep MD sanity report  ({os.path.basename(TRAIN)})")

    check_internal(df, buf)
    check_distribution(df, buf, OUTLIERS)
    check_legacy_drift(df, buf, DELTA)

    section(buf, "Recommended next manual checks")
    info(buf, "* CD-helicity comparison: cross-check predicted helix_percent against")
    info(buf, "  experimental %helicity for Mag(i+4)1,15(A9K), Pleu(i+4)1,15(A9K),")
    info(buf, "  CAP(i+4)1,23(L17K), Esc(i+4)1,14(A7K) from PMID 31427820 Table 1.")
    info(buf, "* Replicate-MD: re-run 3-5 random peptides with a different OpenMM seed;")
    info(buf, "  if helix_percent drifts > 0.10 between replicates the single-trajectory")
    info(buf, "  features are too noisy to use as regression labels.")
    info(buf, "* Trajectory eyeball: load /tmp/stapep_md/<DRAMP_ID>/{pep_vac.prmtop,traj.dcd}")
    info(buf, "  in pytraj/VMD for any +/-3 sigma outlier and confirm no obvious chain break.")

    text = "\n".join(buf)
    print(text)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nReport saved -> {REPORT}")


if __name__ == "__main__":
    main()
