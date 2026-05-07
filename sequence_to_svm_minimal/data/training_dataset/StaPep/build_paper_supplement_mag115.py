#!/usr/bin/env python3
"""
Build stapled_amps_paper_supplement.csv — Mag(i+4)1,15 multi-mutation variants
(Ref. 31427820) that are tabulated in the paper figures but not present as
separate rows in stapled_amps.csv (DRAMP21542 is only Mag(i+4)1,15(A9K)).

Run from anywhere:
  python build_paper_supplement_mag115.py

Output: stapled_amps_paper_supplement.csv (same columns as stapled_amps.csv)
"""
from __future__ import annotations

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AMP_MASTER = os.path.join(HERE, "stapled_amps.csv")
OUT_CSV = os.path.join(HERE, "stapled_amps_paper_supplement.csv")

# Hiden_Sequence: X = S5 staple (StaPep); B = norleucine at 21 when present;
# mutations vs Mag2 + double staple scaffold 1,15 (four X at 2,6,16,20).
ROWS = [
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_K4R_A9K",
        "Name": "Mag(i+4)1,15(K4R,A9K)",
        "Hiden_Sequence": "GXGRFXHSKKKFGKAXVGEXBNS",
        "mic_ec": 3.12,
        "mic_bc": 12.50,
        "mic_pa": 6.25,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 90.5,
    },
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_K4R_A9K_K10R",
        "Name": "Mag(i+4)1,15(K4R,A9K,K10R)",
        "Hiden_Sequence": "GXGRFXHSKRKFGKAXVGEXBNS",
        "mic_ec": 3.12,
        "mic_bc": 12.50,
        "mic_pa": 6.25,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 76.2,
    },
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_K4H_A9K",
        "Name": "Mag(i+4)1,15(K4H,A9K)",
        "Hiden_Sequence": "GXGHFXHSKKKFGKAXVGEXBNS",
        "mic_ec": 3.12,
        "mic_bc": 25.0,
        "mic_pa": 6.25,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 45.6,
    },
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_K4H_A9K_K10H",
        "Name": "Mag(i+4)1,15(K4H,A9K,K10H)",
        "Hiden_Sequence": "GXGHFXHSKHKFGKAXVGEXBNS",
        "mic_ec": 3.12,
        "mic_bc": 25.0,
        "mic_pa": 50.0,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 13.3,
    },
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_K4H_A9K_K10H_K11H",
        "Name": "Mag(i+4)1,15(K4H,A9K,K10H,K11H)",
        "Hiden_Sequence": "GXGHFXHSKHHFGKAXVGEXBNS",
        "mic_ec": 17.68,
        "mic_bc": 50.0,
        "mic_pa": 50.0,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 0.11,
    },
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_A9K_B21A",
        "Name": "Mag(i+4)1,15(A9K,B21A)",
        "Hiden_Sequence": "GXGKFXHSKKKFGKAXVGEXANS",
        "mic_ec": 6.25,
        "mic_bc": ">50",
        "mic_pa": 8.80,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 0.66,
    },
    {
        "DRAMP_ID": "PAPER_31427820_MAG1_15_A9K_B21A_N22K_S23K",
        "Name": "Mag(i+4)1,15(A9K,B21A,N22K,S23K)",
        "Hiden_Sequence": "GXGKFXHSKKKFGKAXVGEAKKK",
        "mic_ec": 1.56,
        "mic_bc": ">50",
        "mic_pa": 3.12,
        "mic_sa": ">50",
        "hrptec_lysis_pct_100ug_ml": 1.67,
    },
]


def _mic_block(r: dict) -> str:
    return (
        f"[Ref.31427820] Gram-positive bacteria: Staphylococcus aureus ATCC 25923 "
        f"(MIC = {r['mic_sa']} μg/mL), Bacillus cereus ATCC 14579 (MIC = {r['mic_bc']} μg/mL);"
        f"##Gram-negative bacteria: Escherichia coli ATCC 25922 (MIC = {r['mic_ec']} μg/mL), "
        f"Pseudomonas aeruginosa ATCC 27853 (MIC = {r['mic_pa']} μg/mL)"
    )


def _cyto(r: dict) -> str:
    return (
        f"[Ref.31427820] HRPTECs: {r['hrptec_lysis_pct_100ug_ml']}% lysis at 100 μg/mL "
        f"(figure; renal epithelial cytotoxicity, not RBC hemolysis)."
    )


def main() -> None:
    with open(AMP_MASTER, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

    if not fieldnames:
        raise SystemExit("Could not read stapled_amps.csv header")

    out_rows = []
    for r in ROWS:
        row = {k: "" for k in fieldnames}
        row["DRAMP_ID"] = r["DRAMP_ID"]
        row["Sequence"] = r["Hiden_Sequence"].replace("X", "Ⓧ")
        row["Hiden_Sequence"] = r["Hiden_Sequence"]
        row["Original_Sequence"] = "GIGKFLHSAKKFGKAFVGEIMNS"
        row["Sequence_Length"] = "23"
        row["Name"] = r["Name"]
        row["Source"] = "Synthetic construct"
        row["Activity"] = "Antimicrobial, Antibacterial, Anti-Gram+, Anti-Gram-"
        row["Target_Organism"] = _mic_block(r)
        row["Hemolytic_Activity"] = (
            "No separate single-concentration RBC hemolysis %% quoted for this variant "
            "in the extracted figure; see Cytotoxicity for HRPTEC %% lysis."
        )
        row["Cytotoxicity"] = _cyto(r)
        row["Pubmed_ID"] = "31427820"
        row["Linear/Cyclic"] = "Cyclic (Stapled)"
        row["N_terminal_Modification"] = "Free"
        row["C_terminal_Modification"] = "Amidation"
        row["Special_Amino_Acid_and_Stapling_Position"] = (
            "①The X (positions: 2, 6, 16 and 20) in Hiden_Sequence indicates S5 stapling "
            "(pentenyl alanine crosslink). ②B at position 21 is norleucine when present."
        )
        out_rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
