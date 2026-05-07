#!/usr/bin/env python3
"""Quick completeness check on the XZ-only 4 ns training feature CSV."""
from __future__ import annotations
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ID_CSV = os.path.join(HERE, "stapled_amps_combined_paper_dataset_XZ_only.csv")
FEAT   = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns.csv")

SEQ_FEATS = ["length", "weight", "hydrophobic_index", "charge", "aromaticity",
             "isoelectric_point", "fraction_arginine", "fraction_lysine",
             "lyticity_index"]
MD_FEATS  = ["helix_percent", "sheet_percent", "loop_percent",
             "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]

ids = pd.read_csv(ID_CSV, encoding="utf-8")["DRAMP_ID"].astype(str)
df  = pd.read_csv(FEAT, encoding="utf-8")

print(f"Identity CSV rows : {len(ids)}")
print(f"Feature CSV  rows : {len(df)}")
print(f"Unique DRAMP_IDs  : {df['DRAMP_ID'].nunique()}")

missing = set(ids) - set(df["DRAMP_ID"].astype(str))
extra   = set(df["DRAMP_ID"].astype(str)) - set(ids)
print(f"Missing from features (in ID file but not features): {len(missing)} {sorted(missing) if missing else ''}")
print(f"Extra in features (not in ID file)                 : {len(extra)} {sorted(extra) if extra else ''}")

err_mask = df["extraction_error"].notna() & (df["extraction_error"].astype(str).str.strip() != "")
print(f"\nRows with extraction_error: {int(err_mask.sum())}")
if err_mask.any():
    print(df.loc[err_mask, ["DRAMP_ID", "extraction_error"]].to_string(index=False))

print("\nNaN counts per feature column:")
for col in SEQ_FEATS + MD_FEATS:
    n = int(df[col].isna().sum())
    flag = "" if n == 0 else "  <-- gap"
    print(f"  {col:<22} {n:>4}{flag}")

bad_md = df[df[MD_FEATS].isna().any(axis=1)]
print(f"\nRows with at least one NaN MD column: {len(bad_md)}")
if len(bad_md):
    print(bad_md[["DRAMP_ID"] + MD_FEATS].to_string(index=False))

ok = (~err_mask) & df[MD_FEATS].notna().all(axis=1) & df[SEQ_FEATS].notna().all(axis=1)
print(f"\nFully-complete rows (no error, no NaN): {int(ok.sum())} / {len(df)}")
