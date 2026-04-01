#!/usr/bin/env python3
"""
predict_pmic_regression.py

Replicates StaPep paper Section 3.3 / Figure 6:
  - Random Forest regression to predict pMIC (E. coli) from 14 MD features
  - Scatter plot: actual vs. predicted pMIC with Pearson R (mimics paper Fig. 6)
  - Also predicts pMIC for Buforin II and Magainin II (non-stapled)

Usage:
    python predict_pmic_regression.py
    python predict_pmic_regression.py --save my_figure.png
"""

import re
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = "data/training_dataset/StaPep"
AMPS_META = f"{DATA_DIR}/stapled_amps.csv"
AMPS_FEAT = f"{DATA_DIR}/stapled_amps_features.csv"

# 14 features from the StaPep paper
FEATURE_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent", "mean_bfactor", "mean_gyrate",
    "num_hbonds", "psa",
]

# ─── Query peptides (from 5 ns StaPep MD runs) ────────────────────────────────
QUERY_PEPTIDES = {
    "Buforin II": {
        "sequence":          "TRSSRAGLQWPVGRVHRLLRK",
        "length":            21,
        "weight":            2473.829,
        "hydrophobic_index": -0.8142857142857142,
        "charge":            6.094,
        "aromaticity":       0.047619047619047616,
        "isoelectric_point": 11.999967765808105,
        "fraction_arginine": 0.23809523809523808,
        "fraction_lysine":   0.047619047619047616,
        "helix_percent":     0.17819047619047618,
        "loop_percent":      0.821047619047619,
        "mean_bfactor":      573.4344313305434,
        "mean_gyrate":       12.001021374843388,
        "num_hbonds":        0,
        "psa":               1064.2171630859375,
    },
    "Magainin II": {
        "sequence":          "GIGKFLHSAKKFGKAFVGEIMNS",
        "length":            23,
        "weight":            2466.832,
        "hydrophobic_index": 0.08260869565217388,
        "charge":            3.095,
        "aromaticity":       0.13043478260869565,
        "isoelectric_point": 10.00138339996338,
        "fraction_arginine": 0.0,
        "fraction_lysine":   0.17391304347826086,
        "helix_percent":     0.16786956521739127,
        "loop_percent":      0.8080869565217391,
        "mean_bfactor":      895.0195772326506,
        "mean_gyrate":       12.213103738248654,
        "num_hbonds":        1,
        "psa":               918.0007934570312,
    },
}

# ─── MIC parsing (comprehensive) ──────────────────────────────────────────────
# Handles all formats found in the dataset:
#   E.coli (MIC99.9= 16 μg/mL)
#   E. coli (MIC= 1.0 μM)
#   E.coli (MIC = 4.4 μM), ...
#   Escherichia coli (MIC = 25 μg/mL), ...
#   Escherichia coli ATCC 700926 (IC50 = 14.8 μM, MIC > 20 μM)
#
# Captures: group(1) = inequality sign (> < =), group(2) = numeric value
_COLI_ORG = r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+\w+)*\s*"
_MIC_UM    = r"\([^)]*?MIC[\w.]*\s*([><=≥]?)\s*([\d.]+)\s*[μu]M"
_MIC_UGML  = r"\([^)]*?MIC[\w.]*\s*([><=≥]?)\s*([\d.]+)\s*[μu]g/mL"

_ECOLI_UM_RE   = re.compile(_COLI_ORG + _MIC_UM,   re.IGNORECASE)
_ECOLI_UGML_RE = re.compile(_COLI_ORG + _MIC_UGML, re.IGNORECASE)


def _parse_match(m) -> float | None:
    """Return float value from regex match, or None if censored (> or <)."""
    if m is None:
        return None
    sign = m.group(1).strip()
    if sign in (">", "<"):       # censored — skip
        return None
    try:
        return float(m.group(2))
    except ValueError:
        return None


def get_ecoli_mic_uM(text: str, mw_da: float) -> float | None:
    """Return E. coli MIC in μM from any supported format, or None."""
    if not isinstance(text, str):
        return None

    # 1. Try μM directly
    v = _parse_match(_ECOLI_UM_RE.search(text))
    if v is not None:
        return v

    # 2. Try μg/mL → convert to μM using MW
    v_ugml = _parse_match(_ECOLI_UGML_RE.search(text))
    if v_ugml is not None and mw_da > 0:
        return v_ugml / mw_da * 1000.0   # (μg/mL) / (g/mol) * 1000 = μM
    return None


def pmic_to_mic_uM(pmic: float) -> float:
    """pMIC = -log10(MIC in mol/L)  →  MIC in μM."""
    return 10 ** (6 - pmic)


def tier(mic_uM: float) -> str:
    if mic_uM < 2:
        return "Very Strong (<2 μM)"
    elif mic_uM < 5:
        return "Strong (2–5 μM)"
    elif mic_uM < 10:
        return "Moderate (5–10 μM)"
    else:
        return "Weak (>10 μM)"


# ─── Load & merge data ────────────────────────────────────────────────────────
def load_training_data() -> pd.DataFrame:
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)

    # Merge on DRAMP_ID so MW is available for μg/mL → μM conversion
    df = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat[["DRAMP_ID"] + FEATURE_COLS],
        on="DRAMP_ID", how="inner",
    )
    df = df.dropna(subset=FEATURE_COLS)

    df["mic_uM"] = df.apply(
        lambda r: get_ecoli_mic_uM(r["Target_Organism"], r["weight"]), axis=1
    )
    df = df[df["mic_uM"].notna() & (df["mic_uM"] > 0)].copy()

    # pMIC = -log10(MIC in mol/L) = 6 - log10(MIC in μM)
    df["pMIC"] = df["mic_uM"].apply(lambda x: 6.0 - math.log10(x))
    return df.reset_index(drop=True)


# ─── Model ────────────────────────────────────────────────────────────────────
def train_rf_split(df: pd.DataFrame, test_size: float = 0.20):
    """
    80/20 train/test split → scatter plot on held-out test set (mirrors paper Fig.6).
    Also compute 5-fold CV Pearson R on the training portion.
    Returns: rf_full, X_test, y_test, y_pred_test, r_test, r_cv5
    """
    X = df[FEATURE_COLS].values.astype(float)
    y = df["pMIC"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_test = rf.predict(X_test)

    r_test, _ = stats.pearsonr(y_test, y_pred_test)
    r2_test   = r2_score(y_test, y_pred_test)

    # 5-fold CV on full dataset for a robust secondary metric
    rf_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv5   = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(rf_cv, X, y, cv=cv5)
    r_cv5, _  = stats.pearsonr(y, y_pred_cv)

    # Final model trained on ALL data (for Buforin / Magainin predictions)
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_full.fit(X, y)

    return rf_full, y_test, y_pred_test, r_test, r2_test, r_cv5


# ─── Figure (mimic StaPep Fig. 6) ────────────────────────────────────────────
def make_figure(y_actual:   np.ndarray,
                y_pred:     np.ndarray,
                pearson_r:  float,
                n_train:    int,
                n_test:     int,
                r_cv5:      float,
                query_preds: dict,
                save_path:  str = "pmic_regression.png"):
    """
    Left panel  : scatter of test-set actual vs. predicted pMIC — mimics Fig. 6.
    Right panel : predicted pMIC for Buforin II and Magainin II.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # ── Panel A: Fig. 6 replica ──────────────────────────────────────────────
    ax = axes[0]

    ax.scatter(y_actual, y_pred,
               color="#E8866F", edgecolors="white", linewidths=0.4,
               s=65, zorder=3, alpha=0.92,
               label=f"Test set (n={n_test})")

    # Linear regression fit + 95 % CI on test points
    slope, intercept, _, _, stderr = stats.linregress(y_actual, y_pred)
    x_fit  = np.linspace(y_actual.min() - 0.1, y_actual.max() + 0.1, 300)
    y_fit  = slope * x_fit + intercept

    n_pts  = len(y_actual)
    x_mean = y_actual.mean()
    se_fit = stderr * np.sqrt(
        1 / n_pts + (x_fit - x_mean) ** 2 / np.sum((y_actual - x_mean) ** 2)
    )
    t_crit = t_dist.ppf(0.975, df=n_pts - 2)

    ax.fill_between(x_fit, y_fit - t_crit * se_fit, y_fit + t_crit * se_fit,
                    color="#4878CF", alpha=0.18, zorder=2)
    ax.plot(x_fit, y_fit, color="#4878CF", linewidth=2.0, zorder=4)

    # Pearson R — top-left, exactly as in paper
    ax.text(0.05, 0.91, f"Pearson R: {pearson_r:.2f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="none", alpha=0.85))
    ax.text(0.05, 0.80, f"5-fold CV R: {r_cv5:.2f}  (n={n_train+n_test})",
            transform=ax.transAxes, fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="none", alpha=0.75))

    ax.set_xlabel("Actual pMIC",    fontsize=12)
    ax.set_ylabel("Predicted pMIC", fontsize=12)
    ax.set_title("Fig. 6 Replica — RF Regression: Actual vs. Predicted pMIC\n"
                 f"(E. coli, stapled AMPs — 80/20 hold-out test set)", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right")

    # ── Panel B: query peptide predictions ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_title("Buforin II & Magainin II\nPredicted pMIC (E. coli)", fontsize=10)

    names  = list(query_preds.keys())
    preds  = [query_preds[n]["pred_pmic"]   for n in names]
    mics   = [query_preds[n]["pred_mic_uM"]  for n in names]
    tiers  = [query_preds[n]["tier"]         for n in names]
    colors = ["#2CA02C", "#9467BD"]
    y_pos  = [0.65, 0.32]

    x_lo, x_hi = 4.0, 6.5

    # Tier background shading
    ax2.axvspan(x_lo, 5.0, alpha=0.06, color="#d62728")   # Weak
    ax2.axvspan(5.0,  5.3, alpha=0.08, color="#ff7f0e")   # Moderate
    ax2.axvspan(5.3,  5.7, alpha=0.08, color="#2ca02c")   # Strong
    ax2.axvspan(5.7,  x_hi, alpha=0.08, color="#1f77b4")  # Very Strong

    for xv, lbl, lc in [(5.0, "Mod\n(10μM)", "#d62728"),
                         (5.3, "Str\n(5μM)",  "#ff7f0e"),
                         (5.7, "VS\n(2μM)",   "#1f77b4")]:
        ax2.axvline(xv, color=lc, linestyle="--", linewidth=1.1, alpha=0.8)
        ax2.text(xv, 0.97, lbl, transform=ax2.get_xaxis_transform(),
                 ha="center", va="top", fontsize=6.5, color=lc)

    for yp, pred, mic_uM, name, color in zip(y_pos, preds, mics, names, colors):
        ax2.scatter(pred, yp, color=color, s=170, zorder=5,
                    edgecolors="black", linewidths=0.8)
        t_short = tier(mic_uM).split("(")[0].strip()
        ax2.annotate(
            f"{name}\npMIC = {pred:.2f}\nMIC ≈ {mic_uM:.1f} μM\n({t_short})",
            xy=(pred, yp), xytext=(0, -40), textcoords="offset points",
            ha="center", fontsize=8.5, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.9),
        )

    # Training mean reference line
    ax2.axvline(np.mean(y_actual), color="gray", linestyle=":",
                linewidth=1.3, alpha=0.8)
    ax2.text(np.mean(y_actual) + 0.05, 0.87,
             f"Dataset\nmean\n{np.mean(y_actual):.2f}",
             fontsize=7, color="gray", va="top",
             transform=ax2.get_xaxis_transform() if False else ax2.transData)

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(0.08, 0.88)
    ax2.set_xlabel("Predicted pMIC", fontsize=11)
    ax2.set_yticks([])

    plt.tight_layout(pad=1.2)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved  →  {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="pMIC regression — StaPep Example 3 replica"
    )
    parser.add_argument("--save", default="pmic_regression.png",
                        help="Output figure path (default: pmic_regression.png)")
    args = parser.parse_args()

    print("=" * 65)
    print("  pMIC Regression  ─  StaPep Section 3.3 Replica")
    print("=" * 65)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\n  Loading training data …")
    df = load_training_data()
    n  = len(df)
    print(f"  Peptides with E. coli MIC : {n}")
    print(f"  pMIC range : {df['pMIC'].min():.2f} – {df['pMIC'].max():.2f}")
    print(f"  pMIC mean  : {df['pMIC'].mean():.2f}  "
          f"≈  MIC {pmic_to_mic_uM(df['pMIC'].mean()):.1f} μM")

    print(f"\n  MIC distribution (μM):")
    vc = df["mic_uM"].value_counts().sort_index()
    for v, cnt in list(vc.items())[:25]:
        bar = "█" * cnt
        print(f"    {v:>8.2f} μM  {bar} ×{cnt}")

    # ── Training / evaluation ────────────────────────────────────────────────
    n_test  = max(int(round(n * 0.20)), 5)
    n_train = n - n_test
    print(f"\n  Strategy: 80/20 train/test split "
          f"(train={n_train}, test={n_test})")
    print("  Training Random Forest …")

    rf, y_test, y_pred_test, r_test, r2_test, r_cv5 = train_rf_split(df)

    print(f"\n  Test-set   Pearson R = {r_test:.3f}   R² = {r2_test:.3f}")
    print(f"  5-fold CV  Pearson R = {r_cv5:.3f}   (all {n} peptides)")

    # ── Query peptide predictions (model trained on ALL data) ────────────────
    print("\n" + "─" * 65)
    print(f"  {'Peptide':<14}  {'Pred pMIC':>9}  {'MIC (μM)':>10}  Tier")
    print("─" * 65)

    query_results = {}
    for name, feats in QUERY_PEPTIDES.items():
        x_q      = np.array([[feats[f] for f in FEATURE_COLS]])
        pred_p   = float(rf.predict(x_q)[0])
        pred_mic = pmic_to_mic_uM(pred_p)
        t_str    = tier(pred_mic)

        print(f"  {name:<14}  {pred_p:>9.3f}  {pred_mic:>10.2f}  {t_str}")
        query_results[name] = {
            "pred_pmic":   pred_p,
            "pred_mic_uM": pred_mic,
            "tier":        t_str,
            "sequence":    feats["sequence"],
        }

    print("─" * 65)
    print("\n  ⚠  Buforin II and Magainin II are NON-STAPLED peptides fed into")
    print("     a model trained on STAPLED AMPs → out-of-distribution estimates.")

    # ── Figure ──────────────────────────────────────────────────────────────
    print("\n  Generating figure …")
    make_figure(
        y_actual    = y_test,
        y_pred      = y_pred_test,
        pearson_r   = r_test,
        n_train     = n_train,
        n_test      = n_test,
        r_cv5       = r_cv5,
        query_preds = query_results,
        save_path   = args.save,
    )

    print("  Done.\n")


if __name__ == "__main__":
    main()
