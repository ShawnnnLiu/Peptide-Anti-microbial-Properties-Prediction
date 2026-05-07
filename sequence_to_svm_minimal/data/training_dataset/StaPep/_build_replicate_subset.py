#!/usr/bin/env python3
"""
Build a 3-peptide identity CSV for a replicate-MD reproducibility check.

Choice rationale:
  DRAMP21558  short (7-mer, KⓍWKJⓍK)         - shortest XZ peptide, smallest cohort variance
  DRAMP21542  Mag(i+4)1,15(A9K) (23-mer)       - paper reference, central case
  DRAMP21541  CAP(i+4)1,23(L17K) (32-mer, CAP) - longest peptide, slowest convergence

Output: replicate_subset.csv  (preserves full schema of the source identity CSV
                               so run_amp_md_features.py can read it as-is)
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "stapled_amps_combined_paper_dataset_XZ_only.csv")
OUT  = os.path.join(HERE, "replicate_subset.csv")
PICKS = ["DRAMP21558", "DRAMP21542", "DRAMP21541"]

df = pd.read_csv(SRC, encoding="utf-8")
sub = df[df["DRAMP_ID"].isin(PICKS)].copy()
# Sort to canonical order
sub["__order"] = sub["DRAMP_ID"].apply(lambda x: PICKS.index(x))
sub = sub.sort_values("__order").drop(columns="__order").reset_index(drop=True)
sub.to_csv(OUT, index=False, encoding="utf-8")
print(f"Wrote {len(sub)} rows -> {OUT}")
print(sub[["DRAMP_ID", "Sequence", "Sequence_Length",
           "N_terminal_Modification", "C_terminal_Modification"]].to_string(index=False))
