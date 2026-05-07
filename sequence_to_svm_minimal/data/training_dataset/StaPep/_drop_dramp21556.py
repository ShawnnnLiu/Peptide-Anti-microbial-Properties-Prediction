#!/usr/bin/env python3
"""Drop DRAMP21556 (Mag(i+7)13) from the XZ-only training dataset.

Removed because MD failed with "Energy is NaN" after MD_MAX_ATTEMPTS=2 retries
during the 4 ns implicit-GB run. Kept in audit file for traceability.
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ID   = os.path.join(HERE, "stapled_amps_combined_paper_dataset_XZ_only.csv")
FT   = os.path.join(HERE, "stapled_amps_features_training_XZ_md4ns.csv")
AUDIT = os.path.join(HERE, "dropped_md_failures.csv")
DROP = "DRAMP21556"

ids  = pd.read_csv(ID, encoding="utf-8")
feat = pd.read_csv(FT, encoding="utf-8")
print(f"Before -> identity: {len(ids)}  features: {len(feat)}")

dropped_id   = ids[ids["DRAMP_ID"] == DROP].copy()
dropped_feat = feat[feat["DRAMP_ID"] == DROP].copy()

ids  = ids[ids["DRAMP_ID"]  != DROP].reset_index(drop=True)
feat = feat[feat["DRAMP_ID"] != DROP].reset_index(drop=True)

ids.to_csv(ID, index=False, encoding="utf-8")
feat.to_csv(FT, index=False, encoding="utf-8")

seq = dropped_id["Sequence"].iloc[0] if len(dropped_id) else ""
err = dropped_feat["extraction_error"].iloc[0] if len(dropped_feat) else "unknown"

audit_row = pd.DataFrame([{"DRAMP_ID": DROP, "Sequence": seq, "reason": err}])
if os.path.exists(AUDIT):
    existing = pd.read_csv(AUDIT, encoding="utf-8")
    audit_row = pd.concat([existing, audit_row], ignore_index=True).drop_duplicates(
        subset="DRAMP_ID", keep="last"
    )
audit_row.to_csv(AUDIT, index=False, encoding="utf-8")

print(f"After  -> identity: {len(ids)}  features: {len(feat)}")
print(f"Audit  -> {AUDIT}")
print(audit_row.tail(5).to_string(index=False))
