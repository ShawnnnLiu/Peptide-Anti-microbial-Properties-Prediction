"""Verify that all claimed-covered DRAMP IDs are fully populated in stapled_amps_features.csv."""
from pathlib import Path

import pandas as pd

# This script lives in the StaPep dataset directory; resolve inputs relative to
# it rather than a machine-specific absolute path.
HERE = Path(__file__).resolve().parent
FEAT = HERE / "stapled_amps_features.csv"

PAPER_DRAMP_IDS = [
    # 37 paper-sheet peptides that mapped to DRAMP
    *(f"DRAMP{n}" for n in range(21504, 21540)),  # 21504..21539
    "DRAMP21542",
]
FIG_DRAMP_IDS = [
    "DRAMP21540", "DRAMP21541", "DRAMP21542", "DRAMP21543",
    *(f"DRAMP{n}" for n in range(21544, 21558)),  # 21544..21557 Mag(i+7)1..14
]

MD_COLS = [
    "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]
SEQ_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine", "lyticity_index",
]


def report(name: str, ids):
    print(f"=== {name} ({len(ids)} ids) ===")
    df = pd.read_csv(FEAT)
    df = df.set_index("DRAMP_ID")
    missing_ids = [i for i in ids if i not in df.index]
    if missing_ids:
        print(f"MISSING ROWS ({len(missing_ids)}): {missing_ids}")
    present = [i for i in ids if i in df.index]
    sub = df.loc[present]
    # sequence descriptors
    seq_missing = sub[SEQ_COLS].isna().any(axis=1)
    if seq_missing.any():
        print(f"rows with any seq-only NaN: {int(seq_missing.sum())}")
        print(sub[seq_missing][SEQ_COLS].head(10))
    else:
        print("seq-only descriptors: all filled")
    md_missing = sub[MD_COLS].isna().any(axis=1)
    if md_missing.any():
        print(f"rows with any MD NaN: {int(md_missing.sum())}")
        print(sub[md_missing][MD_COLS].head(10))
    else:
        print("MD descriptors: all filled")
    # extraction_error column (if exists)
    if "extraction_error" in sub.columns:
        errs = sub["extraction_error"].dropna()
        errs = errs[errs.astype(str).str.len() > 0]
        if len(errs):
            print(f"non-empty extraction_error: {len(errs)}")
            print(errs.head(10))
        else:
            print("extraction_error: all empty")
    print()


def main():
    report("Paper extraction sheet -> DRAMP (37)", sorted(set(PAPER_DRAMP_IDS)))
    report("Figures: 4-species + Mag(i+7)1..14 (18)", sorted(set(FIG_DRAMP_IDS)))


if __name__ == "__main__":
    main()
