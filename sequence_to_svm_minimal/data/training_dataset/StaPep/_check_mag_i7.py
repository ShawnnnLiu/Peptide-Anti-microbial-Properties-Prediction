"""Report MD-completeness of the Mag(i+7) series and locate the Magainin II parent row."""
from pathlib import Path

import pandas as pd

# This script lives in the StaPep dataset directory; resolve inputs relative to
# it rather than a machine-specific absolute path.
HERE = Path(__file__).resolve().parent
AMP = HERE / "stapled_amps.csv"
FEAT = HERE / "stapled_amps_features.csv"

md_cols = ["helix_percent", "sheet_percent", "loop_percent",
           "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]


def main():
    amps = pd.read_csv(AMP)
    feat = pd.read_csv(FEAT).set_index("DRAMP_ID")

    print("=== Mag(i+7) series in stapled_amps.csv ===")
    sub = amps[amps["Name"].astype(str).str.contains(r"Mag\(i\+7\)", regex=True, na=False)][["DRAMP_ID", "Name", "Hiden_Sequence"]]
    for _, r in sub.iterrows():
        did = r["DRAMP_ID"]
        if did not in feat.index:
            status = "NOT IN FEATURES FILE"
        else:
            f = feat.loc[did]
            any_missing = f[md_cols].isna().any()
            status = "MD incomplete (NaN)" if any_missing else "MD complete"
            err = f.get("extraction_error")
            if isinstance(err, str) and len(err) > 0:
                last = err.strip().splitlines()[-1][:100]
                status += " | ERROR: " + last
        print(f"  {did:<12} {r['Name']:<14} {status}")

    print()
    print("=== Magainin II linear parent row (Hiden_Sequence == GIGKFLHSAKKFGKAFVGEIMNS) ===")
    m = amps[amps["Hiden_Sequence"].astype(str) == "GIGKFLHSAKKFGKAFVGEIMNS"]
    print("rows with that exact Hiden_Sequence:", len(m))
    if len(m):
        print(m[["DRAMP_ID", "Name"]].to_string(index=False))
    else:
        print("  (none — Magainin II is not a standalone row)")


if __name__ == "__main__":
    main()
