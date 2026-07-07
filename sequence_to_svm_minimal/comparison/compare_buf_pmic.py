#!/usr/bin/env python3
"""
compare_buf_pmic.py
===================
Side-by-side comparison of:
  LEFT  — Literature pMIC values for the 7 stapled Buforin variants
           from the advisor's table (MIC in μg/mL, E. coli).
  RIGHT — RF-predicted pMIC for the SAME 7 variants,
           using StaPep MD features from buf_advisor_variants_features.csv.

The comparison is now 1-to-1: same variant appears in both panels.
"""

import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy import stats
import re

warnings.filterwarnings("ignore")

# Make utils/ and features/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import PROJECT_ROOT, STAPEP_DIR
from utils.mic_units import mic_to_pmic_ugml as mic_ugml_to_pmic, pmic_to_mic_uM
from features.stapep_columns import STAPEP_COLS_PAPER_14 as FEATURE_COLS

# ─── Paths ────────────────────────────────────────────────────────────────────
AMPS_META      = STAPEP_DIR / "stapled_amps.csv"
AMPS_FEAT      = STAPEP_DIR / "stapled_amps_features.csv"
# Features from the new batch run (run_buf_variants_stapep.py at the top level)
ADVISOR_FEAT   = PROJECT_ROOT.parent.parent / "buf_advisor_variants_features.csv"
# Fall back to old test file if new one not yet generated
TEST_FEAT_OLD  = STAPEP_DIR / "test_stapled_features.csv"

# ─── Advisor's table (7 variants, MIC in μg/mL — corrected from mg/mL typo) ──
# pMIC = 6 − log10(MIC_μM)  where  MIC_μM = MIC_μg/mL × 1000 / MW_Da
#
# NOTE: Buf(i+7)1 has MIC >100 μg/mL → skip pMIC (shown as hatched bar)
LITERATURE = [
    # (label,                   MW_Da,  MIC_μg/mL,  hemo_pct, stapep_id)
    ("Buf(i+4)16\n(F10W)",  2429.9,   5.2,  12.6, "Buf_i4_16_F10W"),
    ("Buf(i+4)14\n(F10W)",  2453.8,  29.2,   2.9, "Buf_i4_14_F10W"),
    ("Buf(i+4)4\n(F10W)",   2523.0, 100.0,   2.4, "Buf_i4_4_F10W"),
    ("Buf(i+4)3\n(F10W)",   2579.1,   6.3,   3.1, "Buf_i4_3_F10W"),
    ("Buf(i+7)9\n(F10W)",   2500.0,   3.1,  57.0, "Buf_i7_9_F10W"),
    ("Buf(i+7)6\n(F10W)",   2637.2,  22.9,   3.0, "Buf_i7_6_F10W"),
    ("Buf(i+7)1\n(F10W)",   2551.0,  None,   2.3, "Buf_i7_1_F10W"),
]

# ─── Helper: unit conversions ─────────────────────────────────────────────────
def _lit_pmic(mic_ugml, mw_da):
    """μg/mL + MW → pMIC, with None-safe handling for censored MIC values."""
    if mic_ugml is None:
        return None
    return mic_ugml_to_pmic(mic_ugml, mw_da)


def tier(pmic):
    mic = pmic_to_mic_uM(pmic)
    if mic <  2: return "Very Strong"
    if mic <  5: return "Strong"
    if mic < 10: return "Moderate"
    return "Weak"

TIER_COLORS = {
    "Very Strong": "#1a9641",
    "Strong":      "#a6d96a",
    "Moderate":    "#fdae61",
    "Weak":        "#d7191c",
}

def bar_color(pmic):
    return TIER_COLORS.get(tier(pmic), "#bbbbbb")

# ─── Train RF on stapled AMP dataset ─────────────────────────────────────────
def load_and_train():
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    df   = pd.merge(meta[["DRAMP_ID","Target_Organism"]],
                    feat[["DRAMP_ID"] + FEATURE_COLS],
                    on="DRAMP_ID", how="inner").dropna(subset=FEATURE_COLS)

    _ECOLI_RE = re.compile(
        r"(?:E\.?\s?coli|Escherichia\s+coli)(?:\s+\w+)*\s*"
        r"\(MIC(?:99\.9)?(?:[\d.]*)?\s*[=≥>]?\s*([\d.]+)\s*([μu]g/mL|[μu]M)",
        re.IGNORECASE,
    )

    def get_mic_uM(row):
        m = _ECOLI_RE.search(str(row["Target_Organism"]))
        if not m: return None
        v, u = float(m.group(1)), m.group(2).lower()
        if "um" in u or "μm" in u: return v
        mw = row.get("weight", 0)
        return (v / mw * 1000.0) if mw > 0 else None

    df["mic_uM"] = df.apply(get_mic_uM, axis=1)
    df = df.dropna(subset=["mic_uM"])
    df["pMIC"]   = 6 - np.log10(df["mic_uM"].clip(lower=1e-6))

    X = df[FEATURE_COLS].values.astype(float)
    y = df["pMIC"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    rf_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv    = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv  = cross_val_predict(rf_cv, X, y, cv=cv)
    r_cv, _ = stats.pearsonr(y, y_cv)

    print(f"  Training set: n={len(df)}  |  5-fold CV Pearson R = {r_cv:.3f}")
    return rf, r_cv, len(df)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Buforin Stapled Variants — Literature vs. RF Predicted pMIC")
    print("=" * 70)

    # ── Load RF model ─────────────────────────────────────────────────────────
    print("\n  Training RF model on stapled AMP dataset …")
    rf, r_cv, n_train = load_and_train()

    # ── Load advisor variant features ─────────────────────────────────────────
    if os.path.exists(ADVISOR_FEAT):
        feat_df = pd.read_csv(ADVISOR_FEAT)
        print(f"\n  Loaded advisor variant features: {ADVISOR_FEAT}")
        print(f"  Variants found: {feat_df['peptide_id'].tolist()}")
        source_note = "(StaPep MD features — same variants as literature)"
    else:
        print(f"\n  ⚠  {ADVISOR_FEAT} not found.")
        print(f"     Run run_buf_variants_stapep.py first in WSL.")
        print(f"     Falling back to old Buf12/Buf13 test variants …")
        feat_df = pd.read_csv(TEST_FEAT_OLD)
        source_note = "(old test variants — different staple positions)"

    # ── Compute predictions for each literature variant ───────────────────────
    lit_label, lit_mw, lit_mic, lit_hemo, lit_ids = zip(*[
        (r[0], r[1], r[2], r[3], r[4]) for r in LITERATURE
    ])

    lit_pmics = [_lit_pmic(m, mw) for m, mw in zip(lit_mic, lit_mw)]

    pred_pmics = []
    for pid in lit_ids:
        row = feat_df[feat_df["peptide_id"] == pid]
        if row.empty or row[FEATURE_COLS].isnull().any(axis=1).all():
            pred_pmics.append(None)
        else:
            x   = np.array([[row.iloc[0][f] for f in FEATURE_COLS]])
            pred_pmics.append(float(rf.predict(x)[0]))

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"  {'Peptide':<22}  {'Lit MIC μg/mL':>13}  {'Lit μM':>7}  {'Lit pMIC':>8}  "
          f"{'Pred pMIC':>9}  {'Pred MIC μg/mL':>14}")
    print("  " + "─" * 76)

    for label, mw, mic_ugml, hemo, pid, lpm, ppm in zip(
            lit_label, lit_mw, lit_mic, lit_hemo,
            lit_ids, lit_pmics, pred_pmics):
        lname  = label.replace("\n", " ")
        mic_s  = f"{mic_ugml:.1f}" if mic_ugml else ">100"
        mic_um = f"{mic_ugml*1000/mw:.2f}" if mic_ugml else "—"
        lpm_s  = f"{lpm:.3f}" if lpm else "—"
        if ppm is not None:
            pmic_uM   = pmic_to_mic_uM(ppm)
            pmic_ugml = pmic_uM * mw / 1000.0
            ppm_s     = f"{ppm:.3f}"
            p_ugml_s  = f"{pmic_ugml:.1f} μg/mL"
        else:
            ppm_s    = "N/A (no MD)"
            p_ugml_s = "—"
        print(f"  {lname:<22}  {mic_s:>13}  {mic_um:>7}  {lpm_s:>8}  "
              f"{ppm_s:>9}  {p_ugml_s:>14}")

    print("  " + "─" * 76)
    print("\n  ⚠  All units: MIC in μg/mL (corrected from mg/mL typo in original table)")

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows = len(LITERATURE)
    fig, (ax_lit, ax_pred) = plt.subplots(1, 2, figsize=(16, 6),
                                           gridspec_kw={"wspace": 0.38})

    y_pos  = np.arange(n_rows)
    x_lo, x_hi = 3.8, 6.5

    # Tier reference lines helper
    def add_tier_lines(ax):
        for mic_thresh, lbl in [(2, "2 μM"), (5, "5 μM"), (10, "10 μM")]:
            pm_line = 6 - math.log10(mic_thresh)
            ax.axvline(pm_line, color="gray", lw=0.8, ls="--", alpha=0.6)
            ax.text(pm_line, -0.65, lbl, ha="center", fontsize=7, color="gray")

    # ── Panel A — Literature ──────────────────────────────────────────────────
    colors_lit  = [bar_color(pm) if pm else "#cccccc" for pm in lit_pmics]
    vals_lit    = [pm if pm else 4.2 for pm in lit_pmics]
    bars_a = ax_lit.barh(y_pos, vals_lit, color=colors_lit,
                          edgecolor="black", linewidth=0.7, height=0.55)

    for i, (pm, mw, mic_ugml, hemo) in enumerate(zip(lit_pmics, lit_mw, lit_mic, lit_hemo)):
        if pm is None:
            bars_a[i].set_hatch("///")
            ax_lit.text(4.25, i, "  >100 μg/mL", va="center", fontsize=7.5, color="#555")
        else:
            mic_uM = mic_ugml * 1000 / mw
            ax_lit.text(pm + 0.02, i,
                        f"  {pm:.2f}  ({mic_ugml:.1f} μg/mL / {mic_uM:.2f} μM)"
                        f"  |  {hemo:.0f}% hem.",
                        va="center", fontsize=7)

    ax_lit.set_yticks(y_pos)
    ax_lit.set_yticklabels(list(lit_label), fontsize=8.5)
    ax_lit.invert_yaxis()
    ax_lit.set_xlim(x_lo, x_hi)
    ax_lit.set_xlabel("pMIC  (= 6 − log₁₀[MIC in μM])", fontsize=10)
    ax_lit.set_title("Panel A — Literature Values\n(Advisor's table, μg/mL)",
                     fontsize=10, fontweight="bold")
    add_tier_lines(ax_lit)

    # i+4 / i+7 group divider
    ax_lit.axhline(3.5, color="#555", lw=0.8, ls=":")
    ax_lit.text(x_lo + 0.05, 1.5, "i+4 staples", fontsize=7.5, color="#333",
                style="italic", va="center")
    ax_lit.text(x_lo + 0.05, 5.0, "i+7 staples", fontsize=7.5, color="#333",
                style="italic", va="center")

    # ── Panel B — RF Predictions ──────────────────────────────────────────────
    colors_pred = [bar_color(pm) if pm else "#cccccc" for pm in pred_pmics]
    vals_pred   = [pm if pm else 4.2 for pm in pred_pmics]
    bars_b = ax_pred.barh(y_pos, vals_pred, color=colors_pred,
                           edgecolor="black", linewidth=0.7, height=0.55)

    for i, (pm, mw) in enumerate(zip(pred_pmics, lit_mw)):
        if pm is None:
            bars_b[i].set_hatch("///")
            ax_pred.text(4.25, i, "  MD features not yet run",
                         va="center", fontsize=7.5, color="#999")
        else:
            mic_uM   = pmic_to_mic_uM(pm)
            mic_ugml = mic_uM * mw / 1000.0
            ax_pred.text(pm + 0.02, i,
                         f"  {pm:.2f}  ({mic_uM:.2f} μM / {mic_ugml:.1f} μg/mL)",
                         va="center", fontsize=7)

    ax_pred.set_yticks(y_pos)
    ax_pred.set_yticklabels(list(lit_label), fontsize=8.5)
    ax_pred.invert_yaxis()
    ax_pred.set_xlim(x_lo, x_hi)
    ax_pred.set_xlabel("Predicted pMIC  (= 6 − log₁₀[MIC in μM])", fontsize=10)
    ax_pred.set_title(f"Panel B — RF Model Predictions {source_note}\n"
                      f"(Trained on {n_train} stapled AMPs,  5-fold CV R = {r_cv:.2f})",
                      fontsize=9, fontweight="bold")
    add_tier_lines(ax_pred)
    ax_pred.axhline(3.5, color="#555", lw=0.8, ls=":")
    ax_pred.text(x_lo + 0.05, 1.5, "i+4 staples", fontsize=7.5, color="#333",
                 style="italic", va="center")
    ax_pred.text(x_lo + 0.05, 5.0, "i+7 staples", fontsize=7.5, color="#333",
                 style="italic", va="center")

    # ── Shared legend ─────────────────────────────────────────────────────────
    tier_patches = [
        mpatches.Patch(color="#1a9641", label="Very Strong  (<2 μM)"),
        mpatches.Patch(color="#a6d96a", label="Strong  (2–5 μM)"),
        mpatches.Patch(color="#fdae61", label="Moderate  (5–10 μM)"),
        mpatches.Patch(color="#d7191c", label="Weak  (>10 μM)"),
        mpatches.Patch(facecolor="white", edgecolor="#333",
                       hatch="///", label=">100 μg/mL / no MD run"),
    ]
    fig.legend(handles=tier_patches, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
               title="MIC Tier Color Code", title_fontsize=8.5)

    fig.suptitle(
        "Buforin Stapled Variants (F10W) — Literature MIC vs. RF-Predicted pMIC\n"
        "Same variants in both panels  |  MIC units: μg/mL  |  E. coli",
        fontsize=10, fontweight="bold",
    )

    out = "buforin_pmic_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out}")

    # ── Pearson R between lit and predicted (for variants with both) ──────────
    paired = [(l, p) for l, p in zip(lit_pmics, pred_pmics) if l and p]
    if len(paired) >= 3:
        l_arr = np.array([x[0] for x in paired])
        p_arr = np.array([x[1] for x in paired])
        r_lit_pred, _ = stats.pearsonr(l_arr, p_arr)
        print(f"\n  Pearson R (literature pMIC vs. predicted pMIC) = {r_lit_pred:.3f}")
        print(f"  Based on {len(paired)} variants with both literature MIC and MD features")

    print("  Done.\n")


if __name__ == "__main__":
    main()
