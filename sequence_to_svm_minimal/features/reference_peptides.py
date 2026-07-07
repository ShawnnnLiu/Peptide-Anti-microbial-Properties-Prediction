"""Reference peptide constants shared across model scripts.

Only includes data that has been verified byte-identical across all current
callers. Native peptide feature dicts (BUF_WT / MAG_NATIVE) are intentionally
NOT centralized here yet — they exist in two precision tiers (3-5 digit vs
full numpy precision) and unifying them would silently change ML inputs.
That cleanup belongs in a separate slice with experimental cross-validation.
"""

# Literature E. coli MIC values for Buforin test variants (μg/mL).
# The advisor's source table header says "mg/mL" but is a known typo — values
# are μg/mL. None = censored entries (e.g. ">100", "low (unclear)", not reported).
#
# Used identically by: regression/predict_pmic_stapled_variants.py,
# svm/predict_mic_svm.py, mlp/predict_mic_mlp.py (all verified byte-identical
# during the SVM and MLP slices).
LITERATURE_MIC_ECOLI: dict[str, dict[str, float | None]] = {
    # F10W variants
    "Buf_i4_16_F10W":  {"mic_ugml": 5.2,   "mw": 2429.9},
    "Buf_i4_14_F10W":  {"mic_ugml": 29.2,  "mw": 2453.8},
    "Buf_i4_4_F10W":   {"mic_ugml": 100.0, "mw": 2523.0},
    "Buf_i4_3_F10W":   {"mic_ugml": 6.3,   "mw": 2579.1},
    "Buf_i7_9_F10W":   {"mic_ugml": 3.1,   "mw": 2500.0},
    "Buf_i7_6_F10W":   {"mic_ugml": 22.9,  "mw": 2637.2},
    "Buf_i7_1_F10W":   {"mic_ugml": None,  "mw": 2551.0},  # >100 (censored)
    # Original variants
    "Buf12":           {"mic_ugml": 6.25,  "mw": 2491.93},
    "Buf13":           {"mic_ugml": 100.0, "mw": 2514.96},
    "Buf13_Q9K":       {"mic_ugml": None,  "mw": 2515.01},  # "low" (unclear)
    "Buf12_V15K_L19K": {"mic_ugml": None,  "mw": 2536.0},   # not reported
}
