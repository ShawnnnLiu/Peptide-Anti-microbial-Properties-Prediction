#!/usr/bin/env python3
"""
predict_pmic_stapled_variants.py
================================
RF regression for MIC (E. coli) of stapled peptides.
  • Train on all stapled AMPs with parseable E. coli MIC
  • 5-fold CV on training set
  • Test on ALL Buforin variants (original + F10W)
  • 3-panel figure matching hemolysis regression format

Units: μg/mL throughout (the literature table header "mg/mL" is a typo).
Internally the model works in pMIC = -log10(MIC_M) for linearity;
all display is converted back to μg/mL.

Usage:
    python predict_pmic_stapled_variants.py
"""

import io, re, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold

# Make utils/ and features/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import PROJECT_ROOT, STAPEP_DIR
from utils.mic_units import mic_to_pmic_ugml as mic_to_pmic, pmic_to_mic_ugml
from features.stapep_columns import STAPEP_COLS_WITH_HSASA as FEATURE_COLS
from features.reference_peptides import LITERATURE_MIC_ECOLI as LITERATURE_MIC

warnings.filterwarnings("ignore")

# Path aliases kept for any in-file legacy references.
BASE = PROJECT_ROOT
DATA = STAPEP_DIR

# ── Buf WT features (from run_buforin_stapep.py, averaged two runs) ─────────
# Kept inline pending native-peptide-feature-dict centralization (deferred —
# 14-col and 18-col copies exist at different float precisions; unifying
# silently changes ML inputs and needs experimental validation).
BUF_WT_FEATURES = {
    "length": 21, "weight": 2473.829,
    "hydrophobic_index": -0.8142857142857142, "charge": 6.094,
    "aromaticity": 0.047619047619047616,
    "isoelectric_point": 11.999967765808105,
    "fraction_arginine": 0.23809523809523808,
    "fraction_lysine": 0.047619047619047616,
    "lyticity_index": 300.106,
    "helix_percent": 0.17819, "sheet_percent": 0.00076,
    "loop_percent": 0.82105,
    "mean_bfactor": 573.434, "mean_gyrate": 12.001,
    "num_hbonds": 0,
    "psa": 1064.217, "sasa": 2038.292,
    "hydrophobic_sasa": 2038.292 - 1064.217,
}

# ── MIC parsing helpers ─────────────────────────────────────────────────────
_COLI = r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+\w+)*\s*"
_MIC_UM   = r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M"
_MIC_UGML = r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL"
_RE_UM    = re.compile(_COLI + _MIC_UM,   re.IGNORECASE)
_RE_UGML  = re.compile(_COLI + _MIC_UGML, re.IGNORECASE)

# Fallback: any organism with E. coli MIC
_GENERIC_UM   = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M", re.I)
_GENERIC_UGML = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL", re.I)


def _parse(m):
    if m is None:
        return None
    mod = m.group(1).strip()
    if mod in (">", "<", "≥"):
        return None
    try:
        return float(m.group(2))
    except Exception:
        return None


def get_mic_ugml(text, mw):
    """Extract E. coli MIC in μg/mL from Target_Organism text."""
    if not isinstance(text, str):
        return None
    # Try direct μg/mL match first
    v = _parse(_RE_UGML.search(text))
    if v is not None:
        return v
    # Try μM and convert
    v = _parse(_RE_UM.search(text))
    if v is not None and mw > 0:
        return v * mw / 1000.0
    # Fallback patterns
    v = _parse(_GENERIC_UGML.search(text))
    if v is not None:
        return v
    v = _parse(_GENERIC_UM.search(text))
    if v is not None and mw > 0:
        return v * mw / 1000.0
    return None


def metrics(y_true, y_pred, label=""):
    r, p = stats.pearsonr(y_true, y_pred)
    rho  = stats.spearmanr(y_true, y_pred).statistic
    mae  = np.mean(np.abs(y_true - y_pred))
    if label:
        print(f"\n  {label}:")
        print(f"    Pearson R  = {r:.3f}  (p = {p:.2e})")
        print(f"    Spearman ρ = {rho:.3f}")
        print(f"    MAE        = {mae:.2f} μg/mL")
    return r, rho, mae


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    # Unicode-safe stdout for Windows console. Done inside main() so the
    # module remains importable without side effects on sys.stdout.
    if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  MIC Regression (RF-18): Train on Stapled AMPs, Test on Buforin")
    print("=" * 70)

    # ── 1. Load & parse training data ────────────────────────────────────────
    meta = pd.read_csv(DATA / "stapled_amps.csv")
    feat = pd.read_csv(DATA / "stapled_amps_features_training_XZ_md50ns.csv")

    # Merge on DRAMP_ID
    df = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat,
        on="DRAMP_ID", how="inner",
    )

    # Compute hydrophobic_sasa
    df["hydrophobic_sasa"] = df["sasa"] - df["psa"]

    # Parse E. coli MIC
    df["mic_ugml"] = df.apply(
        lambda r: get_mic_ugml(r["Target_Organism"], r["weight"]), axis=1
    )
    df = df[df["mic_ugml"].notna() & (df["mic_ugml"] > 0)].copy()
    df["pMIC"] = df.apply(lambda r: mic_to_pmic(r["mic_ugml"], r["weight"]), axis=1)

    # Drop rows with NaN features
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    X_train = df[FEATURE_COLS].values.astype(float)
    y_train_pmic = df["pMIC"].values
    y_train_ugml = df["mic_ugml"].values
    mw_train = df["weight"].values
    n_train = len(df)

    print(f"\n  Training set: n = {n_train}")
    print(f"  MIC range: {y_train_ugml.min():.1f} – {y_train_ugml.max():.1f} μg/mL")
    print(f"  pMIC range: {y_train_pmic.min():.2f} – {y_train_pmic.max():.2f}")

    # ── 2. Train RF & 5-fold CV ──────────────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_pred_pmic = cross_val_predict(rf, X_train, y_train_pmic, cv=kf)
    # Convert CV pMIC predictions back to μg/mL
    cv_pred_ugml = np.array([
        pmic_to_mic_ugml(p, mw) for p, mw in zip(cv_pred_pmic, mw_train)
    ])

    # Final model trained on all data
    rf.fit(X_train, y_train_pmic)

    print("\n" + "=" * 70)
    print("  TRAINING 5-FOLD CV")
    print("=" * 70)

    r_cv, rho_cv, mae_cv = metrics(y_train_ugml, cv_pred_ugml, "RF-18 (μg/mL)")

    # Also report pMIC-space CV metrics
    r_cv_p, _ = stats.pearsonr(y_train_pmic, cv_pred_pmic)
    mae_cv_p  = np.mean(np.abs(y_train_pmic - cv_pred_pmic))
    print(f"\n  pMIC space: R={r_cv_p:.3f}, MAE={mae_cv_p:.3f}")

    # ── 3. Load ALL Buf test variants ────────────────────────────────────────
    test_f10w = pd.read_csv(DATA / "test_buf_specific_stapep_features.csv")
    test_orig = pd.read_csv(DATA / "test_stapled_features.csv")

    # Keep only Buf variants from test_orig (not Magainin)
    test_orig = test_orig[test_orig["peptide_id"].str.startswith("Buf")].copy()

    # Compute hydrophobic_sasa
    test_f10w["hydrophobic_sasa"] = test_f10w["sasa"] - test_f10w["psa"]
    test_orig["hydrophobic_sasa"] = test_orig["sasa"] - test_orig["psa"]

    # ── 4. Build test variant list ───────────────────────────────────────────
    test_variants = []

    # Buf WT (native, non-stapled)
    x_wt = np.array([[BUF_WT_FEATURES[f] for f in FEATURE_COLS]])
    pred_wt_pmic = float(rf.predict(x_wt)[0])
    pred_wt_ugml = pmic_to_mic_ugml(pred_wt_pmic, BUF_WT_FEATURES["weight"])
    test_variants.append({
        "name": "Buf WT\n(native)", "pred_pmic": pred_wt_pmic,
        "pred_ugml": pred_wt_ugml, "lit_ugml": None,
        "mw": BUF_WT_FEATURES["weight"],
    })

    # F10W variants
    for _, row in test_f10w.iterrows():
        pid = row["peptide_id"]
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        pred_pmic = float(rf.predict(x)[0])
        mw = row["weight"]
        pred_ugml = pmic_to_mic_ugml(pred_pmic, mw)
        lit = LITERATURE_MIC.get(pid, {})
        test_variants.append({
            "name": pid.replace("_F10W", "\n(F10W)"),
            "pred_pmic": pred_pmic,
            "pred_ugml": pred_ugml,
            "lit_ugml": lit.get("mic_ugml"),
            "mw": mw,
        })

    # Original Buf variants
    for _, row in test_orig.iterrows():
        pid = row["peptide_id"]
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        pred_pmic = float(rf.predict(x)[0])
        mw = row["weight"]
        pred_ugml = pmic_to_mic_ugml(pred_pmic, mw)
        lit = LITERATURE_MIC.get(pid, {})
        test_variants.append({
            "name": pid,
            "pred_pmic": pred_pmic,
            "pred_ugml": pred_ugml,
            "lit_ugml": lit.get("mic_ugml"),
            "mw": mw,
        })

    # ── 5. Buforin test metrics (variants with literature MIC) ───────────────
    with_lit = [v for v in test_variants if v["lit_ugml"] is not None]

    print("\n" + "=" * 70)
    print(f"  BUFORIN TEST SET (n = {len(with_lit)} variants with literature MIC)")
    print("=" * 70)

    if len(with_lit) >= 3:
        lit_arr  = np.array([v["lit_ugml"]  for v in with_lit])
        pred_arr = np.array([v["pred_ugml"] for v in with_lit])
        r_buf, rho_buf, mae_buf = metrics(lit_arr, pred_arr, "RF-18 on Buforin (μg/mL)")

        # pMIC space
        lit_pmic  = np.array([mic_to_pmic(v["lit_ugml"], v["mw"]) for v in with_lit])
        pred_pmic = np.array([v["pred_pmic"] for v in with_lit])
        r_bp, _ = stats.pearsonr(lit_pmic, pred_pmic)
        mae_bp  = np.mean(np.abs(lit_pmic - pred_pmic))
        print(f"\n  pMIC space: R={r_bp:.3f}, MAE={mae_bp:.3f}")
    else:
        r_buf, rho_buf, mae_buf = np.nan, np.nan, np.nan
        r_bp, mae_bp = np.nan, np.nan

    # ── 6. Per-variant table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PER-VARIANT PREDICTIONS (μg/mL)")
    print(f"{'='*80}")
    print(f"\n  {'Variant':<22} {'Pred pMIC':>10} {'Pred μg/mL':>11} {'Lit μg/mL':>10} {'Error':>10}")
    print(f"  {'-'*65}")

    for v in test_variants:
        name = v["name"].replace("\n", " ")
        pred_p = f"{v['pred_pmic']:.3f}"
        pred_u = f"{v['pred_ugml']:.1f}"
        if v["lit_ugml"] is not None:
            lit_u = f"{v['lit_ugml']:.1f}"
            err = f"{v['pred_ugml'] - v['lit_ugml']:+.1f}"
        else:
            lit_u, err = "---", "---"
        print(f"  {name:<22} {pred_p:>10} {pred_u:>11} {lit_u:>10} {err:>10}")

    # ── 7. Feature importance ────────────────────────────────────────────────
    importances = rf.feature_importances_
    fi_order = np.argsort(importances)[::-1]
    print(f"\n  Top-5 feature importances:")
    for rank, idx in enumerate(fi_order[:5]):
        print(f"    {rank+1}. {FEATURE_COLS[idx]:<22} {importances[idx]:.4f}")

    # ── 8. Figure (3 panels) ─────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        "MIC Prediction (RF-18): Trained on Stapled AMPs, Tested on Buforin Variants\n"
        f"Training n={n_train} | Buforin test n={len(with_lit)}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── Panel A: Buforin Actual vs Predicted (μg/mL, log scale) ──────────────
    if len(with_lit) >= 3:
        lit_vals  = np.array([v["lit_ugml"]  for v in with_lit])
        pred_vals_lit = np.array([v["pred_ugml"] for v in with_lit])

        for v in with_lit:
            marker = "^" if "i7" in v["name"] else "o"
            color  = "#e6194b" if "i7" in v["name"] else "#4363d8"
            ax1.scatter(v["lit_ugml"], v["pred_ugml"], s=100, marker=marker,
                        color=color, edgecolors="k", linewidths=0.8, zorder=5)
            # Label each point
            short = v["name"].replace("\n", " ").replace("(F10W)", "").strip()
            ax1.annotate(short, (v["lit_ugml"], v["pred_ugml"]),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=6.5, alpha=0.85)

        blim_lo = max(0.5, min(lit_vals.min(), pred_vals_lit.min()) * 0.4)
        blim_hi = max(lit_vals.max(), pred_vals_lit.max()) * 3
        ax1.plot([blim_lo, blim_hi], [blim_lo, blim_hi], "--", color="#333",
                 lw=1, alpha=0.5, label="Perfect prediction")

        # 2× error band
        bxs = np.logspace(np.log10(blim_lo), np.log10(blim_hi), 100)
        ax1.fill_between(bxs, bxs / 2, bxs * 2, alpha=0.08, color="#2ca02c",
                         label="2× range")

        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlim(blim_lo, blim_hi); ax1.set_ylim(blim_lo, blim_hi)

        from matplotlib.lines import Line2D
        buf_legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#4363d8",
                   markeredgecolor="k", markersize=8, label="Buf i+4"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6194b",
                   markeredgecolor="k", markersize=8, label="Buf i+7"),
        ]
        ax1.legend(handles=buf_legend, fontsize=8, loc="upper left")

    ax1.set_xlabel("Literature MIC (μg/mL)", fontsize=11)
    ax1.set_ylabel("Predicted MIC (μg/mL)", fontsize=11)
    ax1.set_title(f"Panel A — Buforin: Actual vs Predicted (n={len(with_lit)})\n"
                  f"R={r_buf:.3f}  |  Spearman ρ={rho_buf:.3f}  |  MAE={mae_buf:.1f} μg/mL",
                  fontsize=10, fontweight="bold")
    ax1.grid(alpha=0.2, which="both")
    ax1.set_aspect("equal")

    # ── Panel B: Training CV (μg/mL, log scale) ─────────────────────────────
    ax2.scatter(y_train_ugml, cv_pred_ugml, alpha=0.5, s=40, color="#2c7bb6",
                edgecolors="white", linewidths=0.3, zorder=3)

    # Perfect prediction line
    lim_lo = max(0.5, min(y_train_ugml.min(), cv_pred_ugml.min()) * 0.5)
    lim_hi = max(y_train_ugml.max(), cv_pred_ugml.max()) * 2
    ax2.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="#333", lw=1, alpha=0.5)

    # 2× and 4× error bands
    xs = np.logspace(np.log10(lim_lo), np.log10(lim_hi), 100)
    ax2.fill_between(xs, xs / 2, xs * 2, alpha=0.06, color="#2ca02c", label="2× range")
    ax2.fill_between(xs, xs / 4, xs * 4, alpha=0.04, color="#ff7f0e", label="4× range")

    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lim_lo, lim_hi); ax2.set_ylim(lim_lo, lim_hi)
    ax2.set_xlabel("Actual MIC (μg/mL)", fontsize=11)
    ax2.set_ylabel("Predicted MIC — 5-fold CV (μg/mL)", fontsize=11)
    ax2.set_title(f"Panel B — Training 5-fold CV (n={n_train})\n"
                  f"R={r_cv:.2f} (μg/mL)  |  R={r_cv_p:.2f} (pMIC)  |  MAE={mae_cv:.1f} μg/mL",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(alpha=0.2, which="both")
    ax2.set_aspect("equal")

    # ── Panel C: Buforin predictions bar chart ───────────────────────────────
    n_vars = len(test_variants)
    y_pos = np.arange(n_vars)
    bar_h = 0.35

    labels = [v["name"] for v in test_variants]

    # Predicted bars
    pred_bar_vals = [v["pred_ugml"] for v in test_variants]
    ax3.barh(y_pos, pred_bar_vals, height=bar_h, color="#2c7bb6", alpha=0.8,
             edgecolor="white", linewidth=0.5, label="RF-18 predicted", zorder=3)

    # Literature markers
    for i, v in enumerate(test_variants):
        if v["lit_ugml"] is not None:
            ax3.plot(v["lit_ugml"], i, "D", color="#2ca02c", markersize=8,
                     markeredgecolor="k", markeredgewidth=0.8, zorder=5)

    ax3.plot([], [], "D", color="#2ca02c", markeredgecolor="k",
             markersize=8, label="Literature MIC")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("MIC (μg/mL, E. coli)", fontsize=10)
    ax3.set_title(f"Panel C — Buforin Predictions (n={len(with_lit)} with lit)",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(axis="x", alpha=0.2)
    ax3.set_xscale("log")

    # Accuracy box
    if len(with_lit) >= 3:
        acc_text = (
            f"Buforin Test Accuracy (n={len(with_lit)})\n"
            f"{'Metric':<12} {'ug/mL':>8}  {'pMIC':>8}\n"
            f"{'-'*30}\n"
            f"{'Pearson R':<12} {r_buf:>8.3f}  {r_bp:>8.3f}\n"
            f"{'Spearman':<12} {rho_buf:>8.3f}\n"
            f"{'MAE':<12} {mae_buf:>7.1f}  {mae_bp:>8.3f}"
        )
        ax3.text(0.98, 0.02, acc_text, transform=ax3.transAxes,
                 fontsize=7.5, fontfamily="monospace",
                 verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                           edgecolor="#999", alpha=0.9),
                 zorder=10)

    plt.tight_layout()
    out = BASE / "buf_mic_regression.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Training CV:   R={r_cv:.3f},  MAE={mae_cv:.1f} μg/mL  (pMIC: R={r_cv_p:.3f}, MAE={mae_cv_p:.3f})")
    if len(with_lit) >= 3:
        print(f"  Buforin test:  R={r_buf:.3f},  MAE={mae_buf:.1f} μg/mL  (pMIC: R={r_bp:.3f}, MAE={mae_bp:.3f})")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
