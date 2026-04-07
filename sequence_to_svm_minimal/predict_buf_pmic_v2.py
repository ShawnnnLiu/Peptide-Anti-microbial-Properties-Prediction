#!/usr/bin/env python3
"""
predict_buf_pmic_v2.py
=======================
Improved pMIC RF regression (E. coli) — mirrors hemolysis v2 structure.

Improvements over original predict_pmic_regression.py:
  1. Expanded feature set: 18 features
     (added lyticity_index, sasa, sheet_percent, hydrophobic_sasa)
  2. Improved MIC parsing:
     - μM direct                                  (primary)
     - μg/mL → μM conversion via MW               (secondary)
     - IC50 / MBC entries (converted same as MIC)  (tertiary)
     - Handles ">" and "<" bounds (excluded)
  3. Tests on all 12 Buforin variants with literature MIC comparison
  4. 3-panel figure matching hemolysis v2 layout:
     Panel A — 5-fold CV scatter (training)
     Panel B — Predicted vs Literature pMIC (Buforin variants)
     Panel C — Bar chart of all 12 Buforin variant predictions
"""

import re, math, warnings, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "training_dataset" / "StaPep"

AMPS_META = DATA_DIR / "stapled_amps.csv"
AMPS_FEAT = DATA_DIR / "stapled_amps_features.csv"
F10W_FEAT = DATA_DIR / "test_buf_specific_stapep_features.csv"
TEST_FEAT = DATA_DIR / "test_stapled_features.csv"

# ── Feature columns (expanded from 14 -> 18) ──────────────────────────────────
FEATURE_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index",           # NEW: amphipathic patterning score
    "helix_percent",
    "sheet_percent",            # NEW: beta-sheet content
    "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds",
    "psa", "sasa",              # NEW: total SASA
    "hydrophobic_sasa",         # NEW: derived = sasa - psa
]

# ── Buf WT features (from 5 ns StaPep MD + derived) ───────────────────────────
BUF_WT_FEATURES = {
    "length": 21, "weight": 2473.829,
    "hydrophobic_index": -0.8142857142857142, "charge": 6.094,
    "aromaticity": 0.047619, "isoelectric_point": 11.9999,
    "fraction_arginine": 0.23810, "fraction_lysine": 0.04762,
    "lyticity_index": 300.106,
    "helix_percent": 0.17819,
    "sheet_percent": 0.0007619,
    "loop_percent": 0.82105,
    "mean_bfactor": 573.434, "mean_gyrate": 12.001,
    "num_hbonds": 0, "psa": 1064.217, "sasa": 2038.292,
    "hydrophobic_sasa": 2038.292 - 1064.217,  # = 974.075
}

# ── Literature MIC values for F10W Buforin variants (from advisor table) ──────
# (MIC_ugml, MW_Da)  — E. coli
# Buf(i+7)1 has MIC >100 ug/mL -> None (excluded from pMIC correlation)
LITERATURE_MIC = {
    "Buf_i4_16_F10W":  (5.2,   2429.9),
    "Buf_i4_14_F10W":  (29.2,  2453.8),
    "Buf_i4_4_F10W":   (100.0, 2523.0),
    "Buf_i4_3_F10W":   (6.3,   2579.1),
    "Buf_i7_9_F10W":   (3.1,   2500.0),
    "Buf_i7_6_F10W":   (22.9,  2637.2),
    "Buf_i7_1_F10W":   (None,  2551.0),   # >100 ug/mL
}

DISPLAY_LABELS = {
    "Buf_WT":             "Buf WT\n(native)",
    "Buf_i4_16_F10W":     "Buf(i+4)16\n(F10W)",
    "Buf_i4_14_F10W":     "Buf(i+4)14\n(F10W)",
    "Buf_i4_4_F10W":      "Buf(i+4)4\n(F10W)",
    "Buf_i4_3_F10W":      "Buf(i+4)3\n(F10W)",
    "Buf_i7_9_F10W":      "Buf(i+7)9\n(F10W)",
    "Buf_i7_6_F10W":      "Buf(i+7)6\n(F10W)",
    "Buf_i7_1_F10W":      "Buf(i+7)1\n(F10W)",
    "Buf12":              "Buf(i+4)12",
    "Buf13":              "Buf(i+4)13",
    "Buf13_Q9K":          "Buf(i+4)13\nQ9K",
    "Buf12_V15K_L19K":    "Buf(i+4)12\nV15K,L19K",
}

# ══════════════════════════════════════════════════════════════════════════════
# MIC EXTRACTION  (E. coli — comprehensive)
# ══════════════════════════════════════════════════════════════════════════════

# Organism pattern: E. coli / Escherichia coli + optional strain
_COLI_ORG = r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+[\w\-]+)*\s*"

# MIC (various suffixes: MIC, MIC99.9, MIC50, etc.)
_MIC_UM_RE   = re.compile(
    _COLI_ORG + r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u03bcuU]M",
    re.IGNORECASE,
)
_MIC_UGML_RE = re.compile(
    _COLI_ORG + r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u03bcuU]g/mL",
    re.IGNORECASE,
)

# IC50 as fallback (some entries only have IC50)
_IC50_UM_RE  = re.compile(
    _COLI_ORG + r"\([^)]*?IC50\s*[=:]\s*([><=\u2265]?)\s*([\d.]+)\s*[\u03bcuU]M",
    re.IGNORECASE,
)
_IC50_UGML_RE = re.compile(
    _COLI_ORG + r"\([^)]*?IC50\s*[=:]\s*([><=\u2265]?)\s*([\d.]+)\s*[\u03bcuU]g/mL",
    re.IGNORECASE,
)


def _parse_mic_match(m):
    """Return float value from regex match, or None if censored (> or <)."""
    if m is None:
        return None
    sign = m.group(1).strip()
    if sign in (">", "<", "\u2265", "\u2264"):   # ≥ ≤ — skip bounds
        return None
    try:
        return float(m.group(2))
    except ValueError:
        return None


def get_ecoli_mic_uM(text, mw_da):
    """Return E. coli MIC in uM from any supported format, or None."""
    if not isinstance(text, str):
        return None

    # 1. Try MIC in uM directly
    v = _parse_mic_match(_MIC_UM_RE.search(text))
    if v is not None:
        return v

    # 2. Try MIC in ug/mL -> convert to uM
    v_ugml = _parse_mic_match(_MIC_UGML_RE.search(text))
    if v_ugml is not None and mw_da > 0:
        return v_ugml / mw_da * 1000.0

    # 3. Try IC50 in uM
    v = _parse_mic_match(_IC50_UM_RE.search(text))
    if v is not None:
        return v

    # 4. Try IC50 in ug/mL
    v_ugml = _parse_mic_match(_IC50_UGML_RE.search(text))
    if v_ugml is not None and mw_da > 0:
        return v_ugml / mw_da * 1000.0

    return None


# ── Unit helpers ──────────────────────────────────────────────────────────────
def pmic_to_mic_uM(pmic):
    """pMIC -> MIC in uM."""
    return 10 ** (6 - pmic)

def mic_uM_to_pmic(mic_uM):
    """MIC in uM -> pMIC."""
    if mic_uM is None or mic_uM <= 0:
        return None
    return 6.0 - math.log10(mic_uM)

def mic_ugml_to_pmic(mic_ugml, mw_da):
    """MIC in ug/mL -> pMIC via MW."""
    if mic_ugml is None or mic_ugml <= 0 or mw_da <= 0:
        return None
    mic_uM = mic_ugml * 1000.0 / mw_da
    return 6.0 - math.log10(mic_uM)

def tier(mic_uM):
    if mic_uM < 2:   return "Very Strong"
    if mic_uM < 5:   return "Strong"
    if mic_uM < 10:  return "Moderate"
    return "Weak"

TIER_COLORS = {
    "Very Strong": "#1a9641",
    "Strong":      "#a6d96a",
    "Moderate":    "#fdae61",
    "Weak":        "#d7191c",
}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  pMIC RF Regression v2 -- Expanded Features + Buf Variant Testing")
    print("=" * 72)

    # ── 1. Load training data ────────────────────────────────────────────────
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)

    # Add hydrophobic_sasa to training features
    feat["hydrophobic_sasa"] = feat["sasa"] - feat["psa"]

    base_cols = [c for c in FEATURE_COLS if c != "hydrophobic_sasa"]
    df = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat[["DRAMP_ID"] + base_cols + ["hydrophobic_sasa"]],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=FEATURE_COLS)
    print(f"\n  Stapled AMPs with complete features: {len(df)}")

    # ── 2. Extract E. coli MIC ───────────────────────────────────────────────
    df["mic_uM"] = df.apply(
        lambda r: get_ecoli_mic_uM(r["Target_Organism"], r["weight"]), axis=1
    )
    df_mic = df[df["mic_uM"].notna() & (df["mic_uM"] > 0)].copy()
    df_mic["pMIC"] = df_mic["mic_uM"].apply(lambda x: 6.0 - math.log10(x))
    n_train = len(df_mic)

    print(f"  Peptides with E. coli MIC: {n_train}")
    print(f"  pMIC range: {df_mic['pMIC'].min():.2f} - {df_mic['pMIC'].max():.2f}")
    print(f"  pMIC mean:  {df_mic['pMIC'].mean():.2f}  "
          f"(~{pmic_to_mic_uM(df_mic['pMIC'].mean()):.1f} uM)")

    # ── 3. Train RF ──────────────────────────────────────────────────────────
    X_train = df_mic[FEATURE_COLS].values.astype(float)
    y_train = df_mic["pMIC"].values

    rf = RandomForestRegressor(n_estimators=300, random_state=42,
                                max_depth=None, min_samples_leaf=3,
                                n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(rf, X_train, y_train, cv=cv)
    r_cv, _   = pearsonr(y_train, y_cv)
    r2_cv     = r2_score(y_train, y_cv)
    mae_cv    = mean_absolute_error(y_train, y_cv)
    rf.fit(X_train, y_train)

    print(f"\n  RF (18 features, n={n_train}):")
    print(f"    5-fold CV:  Pearson R = {r_cv:.3f}  |  R2 = {r2_cv:.3f}  "
          f"|  MAE = {mae_cv:.3f}")

    # ── Feature importance ───────────────────────────────────────────────────
    importances = rf.feature_importances_
    imp_order   = np.argsort(importances)[::-1]
    print(f"\n  Top 10 feature importances:")
    for rank, idx in enumerate(imp_order[:10]):
        print(f"    {rank+1:>2}. {FEATURE_COLS[idx]:<22}  {importances[idx]:.4f}")

    # ── 4. Load ALL Buforin variant features ─────────────────────────────────
    test_entries = []

    # 4a. Buf WT
    test_entries.append({
        "pid": "Buf_WT",
        "features": np.array([[BUF_WT_FEATURES[f] for f in FEATURE_COLS]]),
        "stapled": False, "group": "native",
    })

    # 4b. 7 F10W variants
    f10w_df = pd.read_csv(F10W_FEAT)
    f10w_df["hydrophobic_sasa"] = f10w_df["sasa"] - f10w_df["psa"]
    for _, row in f10w_df.iterrows():
        test_entries.append({
            "pid": row["peptide_id"],
            "features": np.array([[row[f] for f in FEATURE_COLS]]),
            "stapled": True, "group": "F10W",
        })

    # 4c. 4 Buf variants from test_stapled_features.csv
    test_df = pd.read_csv(TEST_FEAT)
    test_df["hydrophobic_sasa"] = test_df["sasa"] - test_df["psa"]
    buf_test = test_df[test_df["peptide_id"].str.lower().str.startswith("buf")]
    for _, row in buf_test.iterrows():
        test_entries.append({
            "pid": row["peptide_id"],
            "features": np.array([[row[f] for f in FEATURE_COLS]]),
            "stapled": True, "group": "NGC_variants",
        })

    print(f"\n  Total Buforin test variants: {len(test_entries)}")

    # ── 5. Predict pMIC ──────────────────────────────────────────────────────
    for e in test_entries:
        pred_pmic = float(rf.predict(e["features"])[0])
        pred_mic  = pmic_to_mic_uM(pred_pmic)
        e["pred_pmic"]   = pred_pmic
        e["pred_mic_uM"] = pred_mic

        # Get MW for ug/mL conversion
        if e["pid"] == "Buf_WT":
            mw = BUF_WT_FEATURES["weight"]
        else:
            lit = LITERATURE_MIC.get(e["pid"])
            if lit:
                mw = lit[1]
            else:
                mw = float(e["features"][0][1])  # weight is 2nd feature
        e["mw"] = mw
        e["pred_mic_ugml"] = pred_mic * mw / 1000.0

        # Literature values
        lit = LITERATURE_MIC.get(e["pid"])
        if lit and lit[0] is not None:
            mic_ugml, mw_lit = lit
            mic_uM   = mic_ugml * 1000.0 / mw_lit
            lit_pmic  = 6.0 - math.log10(mic_uM)
            e["lit_mic_ugml"] = mic_ugml
            e["lit_mic_uM"]   = mic_uM
            e["lit_pmic"]     = lit_pmic
        elif lit and lit[0] is None:
            # >100 ug/mL — show in bar chart but not in correlation
            e["lit_mic_ugml"] = None
            e["lit_mic_uM"]   = None
            e["lit_pmic"]     = None
            e["lit_censored"] = True
        else:
            e["lit_mic_ugml"] = None
            e["lit_mic_uM"]   = None
            e["lit_pmic"]     = None

    # ── 6. Results table ─────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  {'Variant':<22}  {'Pred pMIC':>9}  {'MIC(uM)':>8}  {'MIC(ug/mL)':>10}  "
          f"{'Tier':>12}  {'Lit pMIC':>8}  {'Lit MIC(ug/mL)':>14}  {'Delta pMIC':>10}")
    print(f"{'='*100}")
    for e in test_entries:
        dname = DISPLAY_LABELS.get(e["pid"], e["pid"]).replace("\n", " ")
        pred_s = f"{e['pred_pmic']:.3f}"
        mic_s  = f"{e['pred_mic_uM']:.2f}"
        ugml_s = f"{e['pred_mic_ugml']:.1f}"
        tier_s = tier(e["pred_mic_uM"])

        if e["lit_pmic"] is not None:
            lpm_s  = f"{e['lit_pmic']:.3f}"
            lugml  = f"{e['lit_mic_ugml']:.1f}"
            delta  = e['pred_pmic'] - e['lit_pmic']
            delta_s = f"{delta:+.3f}"
        elif e.get("lit_censored"):
            lpm_s  = ">100"
            lugml  = ">100"
            delta_s = "  --"
        else:
            lpm_s = lugml = delta_s = "  --"

        tag = " *" if not e["stapled"] else ""
        print(f"  {dname:<22}  {pred_s:>9}  {mic_s:>8}  {ugml_s:>10}  "
              f"{tier_s:>12}  {lpm_s:>8}  {lugml:>14}  {delta_s:>10}{tag}")
    print(f"{'='*100}")
    print(f"  * = non-stapled (native)")

    # ── 7. Correlation with literature ───────────────────────────────────────
    paired = [(e["pred_pmic"], e["lit_pmic"])
              for e in test_entries if e.get("lit_pmic") is not None]
    pred_arr = np.array([p[0] for p in paired])
    lit_arr  = np.array([p[1] for p in paired])
    n_paired = len(paired)

    if n_paired >= 3:
        r_pl, p_pl   = pearsonr(pred_arr, lit_arr)
        rho_pl, p_rho = spearmanr(pred_arr, lit_arr)
        mae_test = mean_absolute_error(lit_arr, pred_arr)
        print(f"\n  Predicted vs Literature pMIC (N={n_paired}):")
        print(f"    Pearson R  = {r_pl:.3f}  (p={p_pl:.4f})")
        print(f"    Spearman r = {rho_pl:.3f}  (p={p_rho:.4f})")
        print(f"    MAE        = {mae_test:.3f}")
    else:
        r_pl = rho_pl = mae_test = None

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 7.5))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3],
                           left=0.04, right=0.97,
                           top=0.83, bottom=0.12, wspace=0.32)
    ax_cv  = fig.add_subplot(gs[0])
    ax_sc  = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    # ── Panel A: Training 5-fold CV scatter ──────────────────────────────────
    ax_cv.scatter(y_train, y_cv, alpha=0.55, s=50, color="#2c7bb6",
                  edgecolors="white", linewidths=0.4, zorder=3,
                  label=f"Peptides (n={n_train})")

    lo = min(y_train.min(), y_cv.min()) - 0.15
    hi = max(y_train.max(), y_cv.max()) + 0.15
    ax_cv.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    slope, intercept, *_ = stats.linregress(y_train, y_cv)
    x_grid = np.linspace(lo, hi, 200)
    ax_cv.plot(x_grid, slope * x_grid + intercept,
               "-", color="#d7191c", lw=1.5, label="Linear fit")

    ax_cv.text(0.05, 0.95,
               f"Pearson R = {r_cv:.2f}\nR\u00b2 = {r2_cv:.2f}\n"
               f"MAE = {mae_cv:.3f}\nn = {n_train}\n"
               f"features = {len(FEATURE_COLS)}",
               transform=ax_cv.transAxes, fontsize=9, va="top",
               fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="lightyellow", alpha=0.85))

    ax_cv.set_xlabel("Actual pMIC", fontsize=9)
    ax_cv.set_ylabel("Predicted pMIC (5-fold CV)", fontsize=9)
    ax_cv.set_title("Panel A -- Training Set (5-fold CV)\n"
                    "E. coli MIC, RF Regression",
                    fontsize=10, fontweight="bold")
    ax_cv.set_xlim(lo, hi); ax_cv.set_ylim(lo, hi)
    ax_cv.legend(fontsize=7, loc="lower right")
    ax_cv.set_aspect("equal", adjustable="box")
    ax_cv.grid(alpha=0.2, lw=0.5)

    # ── Panel B: Predicted vs Literature scatter ─────────────────────────────
    if n_paired >= 3:
        group_colors = {"F10W": "#4363d8", "NGC_variants": "#e6194b"}
        for e in test_entries:
            if e.get("lit_pmic") is None:
                continue
            c = group_colors.get(e["group"], "#333")
            marker = "o" if e["group"] == "F10W" else "s"
            ax_sc.scatter(e["lit_pmic"], e["pred_pmic"],
                          s=110, marker=marker, color=c,
                          edgecolors="k", linewidths=0.7, zorder=5)
            dname = DISPLAY_LABELS.get(e["pid"], e["pid"]).replace("\n", " ")
            # Smart label offset
            x_off, y_off = 8, 5
            ax_sc.annotate(dname, (e["lit_pmic"], e["pred_pmic"]),
                           textcoords="offset points", xytext=(x_off, y_off),
                           fontsize=6.5, color=c, fontweight="bold")

        lo2 = min(min(pred_arr), min(lit_arr)) - 0.2
        hi2 = max(max(pred_arr), max(lit_arr)) + 0.3
        ax_sc.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.8, alpha=0.5, label="y = x")

        sl2, ic2, *_ = stats.linregress(lit_arr, pred_arr)
        x_fit = np.linspace(lo2, hi2, 100)
        ax_sc.plot(x_fit, sl2 * x_fit + ic2,
                   color="#d7191c", lw=1.3, ls="--", alpha=0.7)

        ax_sc.text(0.05, 0.95,
                   f"Pearson R  = {r_pl:.2f}\n"
                   f"Spearman \u03c1 = {rho_pl:.2f}\n"
                   f"MAE = {mae_test:.3f}\n"
                   f"N = {n_paired}",
                   transform=ax_sc.transAxes, fontsize=9, va="top",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="lightyellow", alpha=0.85))

        ax_sc.set_xlabel("Literature pMIC", fontsize=9)
        ax_sc.set_ylabel("RF Predicted pMIC", fontsize=9)
        ax_sc.set_title("Panel B -- Predicted vs Literature\n"
                        "F10W Buforin Variants (E. coli)",
                        fontsize=10, fontweight="bold")
        ax_sc.set_xlim(lo2, hi2); ax_sc.set_ylim(lo2, hi2)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.grid(alpha=0.2, lw=0.5)

        leg_h = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#4363d8", markeredgecolor="k",
                   markersize=8, label="F10W variants"),
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="#e6194b", markeredgecolor="k",
                   markersize=8, label="Buf12/13 variants"),
        ]
        ax_sc.legend(handles=leg_h, fontsize=7, loc="lower right")

    # ── Panel C: Bar chart ───────────────────────────────────────────────────
    order = [
        "Buf_WT",
        "Buf_i4_16_F10W", "Buf_i4_14_F10W", "Buf_i4_4_F10W", "Buf_i4_3_F10W",
        "Buf_i7_9_F10W", "Buf_i7_6_F10W", "Buf_i7_1_F10W",
        "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    ]
    entry_map = {e["pid"]: e for e in test_entries}
    ordered = [entry_map[pid] for pid in order if pid in entry_map]

    bar_labels    = [DISPLAY_LABELS.get(e["pid"], e["pid"]) for e in ordered]
    bar_pred_pmic = [e["pred_pmic"] for e in ordered]
    bar_pred_mic  = [e["pred_mic_uM"] for e in ordered]
    bar_pred_ugml = [e["pred_mic_ugml"] for e in ordered]
    bar_lit_pmic  = [e.get("lit_pmic") for e in ordered]
    bar_lit_ugml  = [e.get("lit_mic_ugml") for e in ordered]
    bar_stapled   = [e["stapled"] for e in ordered]
    bar_censored  = [e.get("lit_censored", False) for e in ordered]

    y_pos = np.arange(len(ordered))
    bar_h = 0.35

    # Predicted bars
    colors_pred = [TIER_COLORS[tier(m)] for m in bar_pred_mic]
    bars_p = ax_bar.barh(y_pos + bar_h/2, bar_pred_pmic, bar_h,
                         color=colors_pred, edgecolor="k", linewidth=0.6,
                         label="RF Predicted")

    # Literature bars (where available)
    lit_vals_plot = [v if v is not None else 0 for v in bar_lit_pmic]
    lit_colors = [TIER_COLORS[tier(pmic_to_mic_uM(v))] if v is not None else "#eeeeee"
                  for v in bar_lit_pmic]
    bars_l = ax_bar.barh(y_pos - bar_h/2, lit_vals_plot, bar_h,
                         color=lit_colors, edgecolor="k", linewidth=0.6,
                         alpha=0.6, hatch="//",
                         label="Literature")

    # Hide bars with no literature data
    for i, (v, cens) in enumerate(zip(bar_lit_pmic, bar_censored)):
        if v is None and not cens:
            bars_l[i].set_visible(False)
        elif v is None and cens:
            # Mark as >100 ug/mL
            bars_l[i].set_visible(False)  # hide the 0-length bar

    # Hatch native (non-stapled)
    for i, is_st in enumerate(bar_stapled):
        if not is_st:
            bars_p[i].set_hatch("xxx")

    # Annotations
    for i, (pmic, mic, ugml, lit_pm, lit_ugml, is_st, cens) in enumerate(
            zip(bar_pred_pmic, bar_pred_mic, bar_pred_ugml,
                bar_lit_pmic, bar_lit_ugml, bar_stapled, bar_censored)):
        tag = " *" if not is_st else ""
        ax_bar.text(pmic + 0.02, y_pos[i] + bar_h/2,
                    f" {pmic:.2f}  ({mic:.1f} uM / {ugml:.1f} ug/mL){tag}",
                    va="center", fontsize=6.5, fontweight="bold")
        if lit_pm is not None:
            ax_bar.text(lit_pm + 0.02, y_pos[i] - bar_h/2,
                        f" {lit_pm:.2f}  ({lit_ugml:.1f} ug/mL)",
                        va="center", fontsize=6, color="#555")
        elif cens:
            ax_bar.text(0.02 + min(bar_pred_pmic) - 0.3, y_pos[i] - bar_h/2,
                        " Lit: >100 ug/mL",
                        va="center", fontsize=6, color="#999", style="italic")

    # Tier boundary lines
    for mic_thresh, lbl, lc in [(2, "2 uM (VS)", "#1a9641"),
                                  (5, "5 uM (S)",  "#a6d96a"),
                                  (10, "10 uM (M)", "#fdae61")]:
        pm_line = 6 - math.log10(mic_thresh)
        ax_bar.axvline(pm_line, color=lc, linestyle="--", lw=0.9, alpha=0.7)
        ax_bar.text(pm_line, -0.65, lbl, ha="center", fontsize=6, color=lc)

    # Group dividers
    ax_bar.axhline(0.5, color="#555", lw=0.8, ls=":")   # WT | i+4 F10W
    ax_bar.axhline(4.5, color="#555", lw=0.8, ls=":")   # i+4 F10W | i+7 F10W
    ax_bar.axhline(7.5, color="#555", lw=0.8, ls=":")   # i+7 F10W | Buf12/13

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("pMIC  (= 6 - log10[MIC in uM])", fontsize=9)
    x_lo = min(bar_pred_pmic) - 0.4
    x_hi = max(bar_pred_pmic) + 1.8
    ax_bar.set_xlim(x_lo, x_hi)
    ax_bar.set_title("Panel C -- All Buforin Variants\n"
                     "Predicted (solid) vs Literature (hatched)",
                     fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7.5, loc="lower right")
    ax_bar.grid(axis="x", alpha=0.2, lw=0.5)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "pMIC RF Regression v2 -- Expanded Features + Buf Variant Testing\n"
        f"{len(FEATURE_COLS)} StaPep features  |  n_train = {n_train}  |  "
        f"E. coli MIC  |  pMIC = 6 - log10(MIC in uM)",
        fontsize=12, fontweight="bold",
    )

    tier_patches = [mpatches.Patch(color=c, label=f"{t}") for t, c in TIER_COLORS.items()]
    tier_patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                       hatch="xxx", label="Non-stapled (native)"))
    tier_patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                       hatch="//", label="Literature value"))
    fig.legend(handles=tier_patches, loc="lower center", ncol=6,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, title="MIC Tier", title_fontsize=8)

    out = BASE / "buf_pmic_regression_v2.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # ── Comparison with v1 ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Original (14-feature) vs v2 (18-feature) comparison:")
    print(f"{'='*72}")
    print(f"  {'Metric':<30}  {'Original':>10}  {'v2':>10}  {'Change':>10}")
    print(f"  {'='*62}")
    print(f"  {'Features':.<30}  {'14':>10}  {f'{len(FEATURE_COLS)}':>10}  {'+4':>10}")
    print(f"  {'Training samples':.<30}  {'~147':>10}  {f'{n_train}':>10}  {'--':>10}")
    print(f"  {'CV Pearson R':.<30}  {'~0.83':>10}  {f'{r_cv:.3f}':>10}  {'--':>10}")
    if n_paired >= 3:
        print(f"  {'Test Pearson R (lit)':.<30}  {'N/A':>10}  {f'{r_pl:.3f}':>10}  {'NEW':>10}")
        print(f"  {'Test Spearman rho':.<30}  {'N/A':>10}  {f'{rho_pl:.3f}':>10}  {'NEW':>10}")
    print()


if __name__ == "__main__":
    main()
