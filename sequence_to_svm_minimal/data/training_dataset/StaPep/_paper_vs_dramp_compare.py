"""One-off: paper extraction CSV vs stapled_amps DRAMP rows (PubMed 31427820)."""
import re
import pandas as pd

PAPER = r"c:/Users/bioin/Downloads/Stapled Peptide Data Extraction - Stapled Peptide Data Extraction.csv"
DRAMP = r"c:/Users/bioin/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep/stapled_amps.csv"
OUT = r"c:/Users/bioin/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep/paper_vs_dramp_side_by_side.csv"


def norm_aa(s: str) -> str:
    s = s.upper()
    return s.replace("B", "J")  # norleucine: paper/DRAMP use B or J


def norm_paper_seq(s: str) -> str:
    s = re.sub(r"\[S5\]", "X", str(s))
    return norm_aa(s)


def norm_dramp_name(n: str) -> str:
    return re.sub(r"\s+", "", n.replace("(", "(").lower())


def norm_paper_peptide_name(n: str) -> str:
    return re.sub(r"\s+", "", n.strip().lower())


def main():
    paper = pd.read_csv(PAPER, encoding="utf-8-sig")
    paper["norm_seq"] = paper["Sequence"].astype(str).map(norm_paper_seq)

    dr = pd.read_csv(DRAMP, encoding="utf-8-sig")
    dr = dr[dr["Pubmed_ID"].astype(str).str.contains("31427820", na=False)].copy()
    dr["norm_seq"] = dr["Hiden_Sequence"].astype(str).map(norm_aa)
    dr["norm_name"] = dr["Name"].astype(str).map(norm_dramp_name)

    # Index DRAMP by normalized sequence (first wins for duplicates)
    seq_to_row = {}
    for _, r in dr.iterrows():
        k = r["norm_seq"]
        if k not in seq_to_row:
            seq_to_row[k] = r

    rows = []
    for _, p in paper.iterrows():
        pname = norm_paper_peptide_name(str(p["Peptide"]))
        match = None
        how = ""
        # 1) sequence
        r = seq_to_row.get(p["norm_seq"])
        if r is not None:
            match = r
            how = "sequence"
        else:
            # 2) name (fuzzy: paper vs DRAMP capitalization)
            cand = dr[dr["norm_name"] == pname]
            if len(cand) == 1:
                match = cand.iloc[0]
                how = "name"
            elif len(cand) > 1:
                match = cand.iloc[0]
                how = "name_ambiguous_first"

        if match is None:
            rows.append(
                {
                    "paper_Peptide": p["Peptide"],
                    "paper_Sequence_norm": p["norm_seq"],
                    "match_how": "NOT_IN_DRAMP",
                    "DRAMP_ID": "",
                    "DRAMP_Name": "",
                    "DRAMP_Hiden_Sequence": "",
                }
            )
        else:
            rows.append(
                {
                    "paper_Peptide": p["Peptide"],
                    "paper_Sequence_norm": p["norm_seq"],
                    "match_how": how,
                    "DRAMP_ID": match["DRAMP_ID"],
                    "DRAMP_Name": match["Name"],
                    "DRAMP_Hiden_Sequence": match["Hiden_Sequence"],
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    n_miss = (out["match_how"] == "NOT_IN_DRAMP").sum()
    print("paper rows:", len(paper))
    print("DRAMP 31427820 rows:", len(dr))
    print("NOT_IN_DRAMP:", int(n_miss))
    print("wrote", OUT)
    if n_miss:
        print("\nMissing:")
        print(out[out["match_how"] == "NOT_IN_DRAMP"][["paper_Peptide", "paper_Sequence_norm"]].to_string(index=False))


if __name__ == "__main__":
    main()
