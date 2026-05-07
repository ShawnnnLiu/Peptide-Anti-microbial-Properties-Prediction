import pandas as pd

AMP = r"c:/Users/bioin/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep/stapled_amps.csv"
FEAT = r"c:/Users/bioin/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep/stapled_amps_features.csv"

amps = pd.read_csv(AMP)
feat = pd.read_csv(FEAT)

md_cols = ["helix_percent", "sheet_percent", "loop_percent",
           "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]
seq_cols = ["length", "weight", "hydrophobic_index", "charge", "aromaticity",
            "isoelectric_point", "fraction_arginine", "fraction_lysine", "lyticity_index"]

name_map = amps.set_index("DRAMP_ID")["Name"].to_dict()
pub_map  = amps.set_index("DRAMP_ID")["Pubmed_ID"].astype(str).to_dict()

df = feat.copy()
df["any_md_nan"] = df[md_cols].isna().any(axis=1)
df["any_seq_nan"] = df[seq_cols].isna().any(axis=1)
df["Name"] = df["DRAMP_ID"].map(name_map)
df["Pubmed_ID"] = df["DRAMP_ID"].map(pub_map)

print("Total rows in features file:", len(df))
print("Rows with ANY MD NaN:", int(df["any_md_nan"].sum()))
print("Rows with ANY seq-only NaN:", int(df["any_seq_nan"].sum()))
print()

bad = df[df["any_md_nan"]]
print("=== Rows with missing MD features (any column) ===")
cols_show = ["DRAMP_ID", "Name", "Pubmed_ID", "extraction_error"]
for _, r in bad[cols_show].iterrows():
    err = r["extraction_error"]
    if isinstance(err, str) and len(err) > 0:
        last = err.strip().splitlines()[-1][:90]
    else:
        last = "(no error message)"
    print(f"  {r['DRAMP_ID']:<12} {str(r['Name'])[:35]:<35} Pub {r['Pubmed_ID']:<12} -> {last}")
