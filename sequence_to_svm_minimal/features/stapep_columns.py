"""Shared feature-column definitions for stapled-peptide models.

Centralised replacement for STAPEP_COLS / QSAR_COLS / TEST_NAMES /
QSAR_TEST_NAME_MAP / SVM_PARAM_GRID duplications that appeared verbatim
in run_stapep_svm.py, run_combined_svm.py, run_mic_svm.py,
run_stapep_svm_no_loop.py, predict_mic_svm.py and others.
"""

# 17 StaPep features (sequence-only 9 + MD-derived 8).
# Order matches the canonical training CSVs.
STAPEP_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]

# Derived 18th feature used by predict_mic_svm.py — runtime: sasa - psa.
STAPEP_COLS_WITH_HSASA = STAPEP_COLS + ["hydrophobic_sasa"]

# 14-feature subset used by the StaPep paper's Fig. 6 RF regression — also
# used by regression/predict_pmic_regression.py, predict_pmic_all_organisms.py,
# and predict_hemolysis_regression.py. Strict subset of STAPEP_COLS minus
# (lyticity_index, sheet_percent, sasa).
STAPEP_COLS_PAPER_14 = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent", "mean_bfactor", "mean_gyrate",
    "num_hbonds", "psa",
]

# 12 QSAR descriptors from the 2016 PNAS SVM paper.
QSAR_COLS = [
    "netCharge", "FC", "LW", "DP", "NK", "AE", "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104", "tau4_GRAR740104",
    "QSO50_GRAR740104", "QSO29_GRAR740104",
]

# Canonical test-peptide names. Order matches the rows in
# test_stapled_features.csv.
TEST_NAMES = [
    "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    "Mag20",  "Mag25", "Mag31",     "Mag36",
]

# QSAR test CSV uses a different naming convention; map back to TEST_NAMES.
QSAR_TEST_NAME_MAP = {
    "Buf(i+4)_12"          : "Buf12",
    "Buf(i+4)_13"          : "Buf13",
    "Buf(i+4)_13_Q9K"      : "Buf13_Q9K",
    "Buf(i+4)_12_V15K_L19K": "Buf12_V15K_L19K",
    "Mag_20"               : "Mag20",
    "Mag_25"               : "Mag25",
    "Mag_31"               : "Mag31",
    "Mag_36"               : "Mag36",
}

# Standard RBF-SVM hyperparameter grid (paper-style C × γ tuning).
SVM_PARAM_GRID = {
    "svc__C"    : [0.1, 1, 10, 100, 1000],
    "svc__gamma": ["scale", 1e-3, 1e-2, 0.1],
}
