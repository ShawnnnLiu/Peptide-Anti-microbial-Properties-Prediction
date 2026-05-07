#!/usr/bin/env python3
"""
Merge DRAMP stapled AMP tables + paper supplement into official training-ready CSVs.

Reads:
  stapled_amps.csv
  stapled_amps_paper_supplement.csv   (run build_paper_supplement_mag115.py first)
  stapled_amps_features.csv
  stapled_amps_paper_supplement_features.csv  (from run_amp_md_features.py on supplement)

Writes:
  stapled_amps_combined_paper_dataset.csv
  stapled_amps_features_combined_paper_dataset.csv

Duplicate DRAMP_ID: supplement row wins (keep='last') so you can refresh paper rows.
"""
from __future__ import annotations

import os
import sys

try:
    import pandas as pd
except ImportError:
    print("pandas required: pip install pandas", file=sys.stderr)
    sys.exit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = {
    "amps_base": os.path.join(HERE, "stapled_amps.csv"),
    "amps_paper": os.path.join(HERE, "stapled_amps_paper_supplement.csv"),
    "feat_base": os.path.join(HERE, "stapled_amps_features.csv"),
    "feat_paper": os.path.join(HERE, "stapled_amps_paper_supplement_features.csv"),
}
OUT_AMP = os.path.join(HERE, "stapled_amps_combined_paper_dataset.csv")
OUT_FEAT = os.path.join(HERE, "stapled_amps_features_combined_paper_dataset.csv")


def main() -> None:
    if not os.path.isfile(FILES["amps_paper"]):
        print(f"Missing {FILES['amps_paper']} — run build_paper_supplement_mag115.py", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(FILES["feat_paper"]):
        print(
            f"Missing {FILES['feat_paper']} — run StaPep on supplement first, e.g.\n"
            "  WSL: conda activate stap && python run_amp_md_features.py "
            "--amp-csv stapled_amps_paper_supplement.csv "
            "--out stapled_amps_paper_supplement_features.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    amps = pd.read_csv(FILES["amps_base"], encoding="utf-8-sig")
    ap = pd.read_csv(FILES["amps_paper"], encoding="utf-8-sig")
    amps_c = pd.concat([amps, ap], ignore_index=True)
    amps_c = amps_c.drop_duplicates(subset=["DRAMP_ID"], keep="last")
    amps_c.to_csv(OUT_AMP, index=False, encoding="utf-8")

    feat = pd.read_csv(FILES["feat_base"], encoding="utf-8-sig")
    fp = pd.read_csv(FILES["feat_paper"], encoding="utf-8-sig")
    # align columns (paper features must match base schema)
    for c in feat.columns:
        if c not in fp.columns:
            fp[c] = pd.NA
    fp = fp[[c for c in feat.columns]]
    feat_c = pd.concat([feat, fp], ignore_index=True)
    feat_c = feat_c.drop_duplicates(subset=["DRAMP_ID"], keep="last")
    feat_c.to_csv(OUT_FEAT, index=False, encoding="utf-8")

    print(f"Wrote {len(amps_c)} AMP rows -> {OUT_AMP}")
    print(f"Wrote {len(feat_c)} feature rows -> {OUT_FEAT}")


if __name__ == "__main__":
    main()
