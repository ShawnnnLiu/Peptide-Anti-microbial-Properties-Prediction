#!/usr/bin/env python3
"""
Drop K-stapled (lysine-tethered) peptides from the combined training dataset.

Advisor rule (31 Mar 2026):
    "take out data set with K stapling (like KFFⓀKLKKAVⓀKGFKKFAKV) because they
     are cyclic and different. Just use the ones that have X and Z"

We keep only hydrocarbon-stapled peptides (X=S5 pentenyl, Z=R8 octenyl) and
regular natural-AA peptides. Any sequence containing a circled K (Ⓚ U+24C0 or
ⓚ U+24DA) is dropped.

Inputs:
    stapled_amps_combined_paper_dataset.csv
    stapled_amps_features_combined_paper_dataset.csv

Outputs:
    stapled_amps_combined_paper_dataset_XZ_only.csv
    stapled_amps_features_combined_paper_dataset_XZ_only.csv
    dropped_k_stapled.csv   (audit log of what was removed)
"""
from __future__ import annotations

import os
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas required", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
IN_AMP = os.path.join(HERE, "stapled_amps_combined_paper_dataset.csv")
IN_FEAT = os.path.join(HERE, "stapled_amps_features_combined_paper_dataset.csv")
OUT_AMP = os.path.join(HERE, "stapled_amps_combined_paper_dataset_XZ_only.csv")
OUT_FEAT = os.path.join(HERE, "stapled_amps_features_combined_paper_dataset_XZ_only.csv")
AUDIT = os.path.join(HERE, "dropped_k_stapled.csv")

K_MARKERS = ("\u24c0", "\u24da")  # Ⓚ, ⓚ


def is_k_stapled(seq: str) -> bool:
    if not isinstance(seq, str):
        return False
    return any(m in seq for m in K_MARKERS)


def main() -> None:
    amps = pd.read_csv(IN_AMP, encoding="utf-8")
    feat = pd.read_csv(IN_FEAT, encoding="utf-8")

    mask_k = amps["Sequence"].apply(is_k_stapled)
    dropped = amps[mask_k].copy()
    dropped_ids = set(dropped["DRAMP_ID"])

    amps_keep = amps[~mask_k].copy()
    feat_keep = feat[~feat["DRAMP_ID"].isin(dropped_ids)].copy()

    amps_keep.to_csv(OUT_AMP, index=False, encoding="utf-8")
    feat_keep.to_csv(OUT_FEAT, index=False, encoding="utf-8")
    dropped[["DRAMP_ID", "Sequence"]].to_csv(AUDIT, index=False, encoding="utf-8")

    print(f"Input rows:         {len(amps)}  (features: {len(feat)})")
    print(f"K-stapled dropped:  {len(dropped)}")
    print(f"Kept (X/Z + plain): {len(amps_keep)}  (features: {len(feat_keep)})")
    print()
    print(f"Wrote AMPs     -> {OUT_AMP}")
    print(f"Wrote Features -> {OUT_FEAT}")
    print(f"Wrote audit    -> {AUDIT}")


if __name__ == "__main__":
    main()
