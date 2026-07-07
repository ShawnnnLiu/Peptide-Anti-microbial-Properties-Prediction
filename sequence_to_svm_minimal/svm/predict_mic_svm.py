#!/usr/bin/env python3
"""
predict_mic_svm.py
==================
SVM Classifier (AMP vs Decoy) — distance-to-margin analysis.

  • Train RBF-SVC on ALL stapled AMPs + Decoys (binary classification)
  • Compute `decision_function` (signed distance to hyperplane) for:
      – Training AMPs that have E. coli MIC data
      – All Buforin test variants
  • Create figures showing correlation between MIC and distance-to-margin
    for BOTH training and test sets

Usage:
    conda run -n esm_env python predict_mic_svm.py
"""

import io, re, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, cross_val_predict)

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

# ── Buf WT features ─────────────────────────────────────────────────────────
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
_GENERIC_UM   = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M", re.I)
_GENERIC_UGML = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL", re.I)


def _parse(m):
    if m is None: return None
    mod = m.group(1).strip()
    if mod in (">", "<", "\u2265"): return None
    try: return float(m.group(2))
    except: return None


def get_mic_ugml(text, mw):
    if not isinstance(text, str): return None
    v = _parse(_RE_UGML.search(text))
    if v is not None: return v
    v = _parse(_RE_UM.search(text))
    if v is not None and mw > 0: return v * mw / 1000.0
    v = _parse(_GENERIC_UGML.search(text))
    if v is not None: return v
    v = _parse(_GENERIC_UM.search(text))
    if v is not None and mw > 0: return v * mw / 1000.0
    return None


def pearson_safe(x, y):
    if len(x) < 3: return np.nan, np.nan
    return stats.pearsonr(x, y)


def spearman_safe(x, y):
    if len(x) < 3: return np.nan, np.nan
    return stats.spearmanr(x, y)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    # Unicode-safe stdout for Windows console. Done inside main() so the
    # module remains importable without side effects on sys.stdout.
    if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  SVM Classifier Margin vs MIC")
    print("  Train AMP/Decoy on ALL stapled peptides | Test on Buforin variants")
    print("=" * 72)

    # ── 1. Load ALL training data (AMPs + Decoys) ────────────────────────────
    meta = pd.read_csv(DATA / "stapled_amps.csv")
    feat = pd.read_csv(DATA / "stapled_amps_features.csv")
    dec  = pd.read_csv(DATA / "stapled_decoys.csv")

    # AMPs
    amp_feat = feat.copy()
    amp_feat["hydrophobic_sasa"] = amp_feat["sasa"] - amp_feat["psa"]
    amp_feat["label"] = 1

    # Decoys
    dec["hydrophobic_sasa"] = dec["sasa"] - dec["psa"]
    dec["label"] = -1

    # Combined classifier training set
    clf_df = pd.concat([
        amp_feat[FEATURE_COLS + ["label"]],
        dec[FEATURE_COLS + ["label"]],
    ], ignore_index=True).dropna(subset=FEATURE_COLS)

    X_clf = clf_df[FEATURE_COLS].values.astype(float)
    y_clf = clf_df["label"].values
    n_amp = (y_clf == 1).sum()
    n_dec = (y_clf == -1).sum()

    print(f"\n  Classifier training: {len(clf_df)} total  "
          f"(AMPs={n_amp}, Decoys={n_dec})")

    # ── 2. Identify training AMPs that have E. coli MIC ──────────────────────
    df_amp = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat, on="DRAMP_ID", how="inner",
    )
    df_amp["hydrophobic_sasa"] = df_amp["sasa"] - df_amp["psa"]
    df_amp["mic_ugml"] = df_amp.apply(
        lambda r: get_mic_ugml(r["Target_Organism"], r["weight"]), axis=1
    )
    df_mic = df_amp[df_amp["mic_ugml"].notna() & (df_amp["mic_ugml"] > 0)].copy()
    df_mic["pMIC"] = df_mic.apply(
        lambda r: mic_to_pmic(r["mic_ugml"], r["weight"]), axis=1
    )
    df_mic = df_mic.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    X_mic_train = df_mic[FEATURE_COLS].values.astype(float)
    y_mic_ugml  = df_mic["mic_ugml"].values
    y_mic_pmic  = df_mic["pMIC"].values
    mw_mic      = df_mic["weight"].values
    n_mic = len(df_mic)

    print(f"  Training AMPs with E. coli MIC: {n_mic}")
    print(f"  MIC range: {y_mic_ugml.min():.1f} - {y_mic_ugml.max():.1f} ug/mL")

    # ── 3. Train SVM Classifier with GridSearch ──────────────────────────────
    print(f"\n{'='*72}")
    print("  Training SVM Classifier (RBF-SVC, AMP vs Decoy)")
    print(f"{'='*72}")

    svc_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42)),
    ])
    svc_grid = {
        "svc__C":     [0.1, 1, 10, 100, 1000],
        "svc__gamma": ["scale", 1e-3, 1e-2, 0.1],
    }
    gs = GridSearchCV(
        svc_pipe, svc_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_clf, y_clf)
    best_svc = gs.best_estimator_

    print(f"  Best params: C={gs.best_params_['svc__C']}, "
          f"gamma={gs.best_params_['svc__gamma']}")
    print(f"  Best CV AUC: {gs.best_score_:.3f}")

    # CV accuracy
    cv_acc = cross_val_score(
        best_svc, X_clf, y_clf,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    print(f"  CV Accuracy: {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")

    # ── 4. Decision margins for training AMPs with MIC ───────────────────────
    train_margins = best_svc.decision_function(X_mic_train)
    train_probs   = best_svc.predict_proba(X_mic_train)[:, 1]

    print(f"\n{'='*72}")
    print(f"  TRAINING AMPs WITH MIC — Margin vs MIC correlation")
    print(f"{'='*72}")

    r_train_ugml, p_train_ugml = pearson_safe(train_margins, y_mic_ugml)
    rho_train_ugml, _ = spearman_safe(train_margins, y_mic_ugml)
    r_train_pmic, _   = pearson_safe(train_margins, y_mic_pmic)
    rho_train_pmic, _ = spearman_safe(train_margins, y_mic_pmic)

    # Also: margin vs log10(MIC)
    log_mic = np.log10(y_mic_ugml)
    r_train_logmic, _ = pearson_safe(train_margins, log_mic)
    rho_train_logmic, _ = spearman_safe(train_margins, log_mic)

    print(f"\n  {'Correlation':<35} {'Pearson R':>10} {'Spearman rho':>13}")
    print(f"  {'-'*35} {'-'*10} {'-'*13}")
    print(f"  {'Margin vs MIC (ug/mL)':<35} {r_train_ugml:>10.3f} {rho_train_ugml:>13.3f}")
    print(f"  {'Margin vs log10(MIC)':<35} {r_train_logmic:>10.3f} {rho_train_logmic:>13.3f}")
    print(f"  {'Margin vs pMIC':<35} {r_train_pmic:>10.3f} {rho_train_pmic:>13.3f}")
    print(f"\n  Margin range (train AMPs): {train_margins.min():.3f} to {train_margins.max():.3f}")
    print(f"  Mean margin: {train_margins.mean():.3f}")

    # ── 5. Load ALL Buf test variants ────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BUFORIN TEST SET")
    print(f"{'='*72}")

    test_f10w = pd.read_csv(DATA / "test_buf_specific_stapep_features.csv")
    test_orig = pd.read_csv(DATA / "test_stapled_features.csv")
    test_orig = test_orig[test_orig["peptide_id"].str.startswith("Buf")].copy()

    test_f10w["hydrophobic_sasa"] = test_f10w["sasa"] - test_f10w["psa"]
    test_orig["hydrophobic_sasa"] = test_orig["sasa"] - test_orig["psa"]

    test_variants = []

    # Buf WT
    x_wt = np.array([[BUF_WT_FEATURES[f] for f in FEATURE_COLS]])
    margin_wt = float(best_svc.decision_function(x_wt)[0])
    prob_wt   = float(best_svc.predict_proba(x_wt)[0, 1])
    test_variants.append({
        "name": "Buf WT", "short": "Buf WT",
        "margin": margin_wt, "prob_amp": prob_wt,
        "lit_ugml": None, "mw": BUF_WT_FEATURES["weight"],
        "group": "WT",
    })

    # F10W variants
    for _, row in test_f10w.iterrows():
        pid = row["peptide_id"]
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        margin = float(best_svc.decision_function(x)[0])
        prob   = float(best_svc.predict_proba(x)[0, 1])
        lit = LITERATURE_MIC.get(pid, {})
        short = pid.replace("Buf_", "").replace("_F10W", "")
        group = "i+7" if "i7" in pid else "i+4"
        test_variants.append({
            "name": pid, "short": short,
            "margin": margin, "prob_amp": prob,
            "lit_ugml": lit.get("mic_ugml"), "mw": row["weight"],
            "group": group,
        })

    # Original variants
    for _, row in test_orig.iterrows():
        pid = row["peptide_id"]
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        margin = float(best_svc.decision_function(x)[0])
        prob   = float(best_svc.predict_proba(x)[0, 1])
        lit = LITERATURE_MIC.get(pid, {})
        test_variants.append({
            "name": pid, "short": pid,
            "margin": margin, "prob_amp": prob,
            "lit_ugml": lit.get("mic_ugml"), "mw": row["weight"],
            "group": "Original",
        })

    # ── 6. Test set correlations ─────────────────────────────────────────────
    with_lit = [v for v in test_variants if v["lit_ugml"] is not None]
    all_test = test_variants

    if len(with_lit) >= 3:
        buf_margins = np.array([v["margin"] for v in with_lit])
        buf_mic     = np.array([v["lit_ugml"] for v in with_lit])
        buf_logmic  = np.log10(buf_mic)
        buf_pmic    = np.array([mic_to_pmic(v["lit_ugml"], v["mw"]) for v in with_lit])

        r_buf_ugml, p_buf = pearson_safe(buf_margins, buf_mic)
        rho_buf_ugml, _   = spearman_safe(buf_margins, buf_mic)
        r_buf_logmic, _   = pearson_safe(buf_margins, buf_logmic)
        rho_buf_logmic, _ = spearman_safe(buf_margins, buf_logmic)
        r_buf_pmic, _     = pearson_safe(buf_margins, buf_pmic)
        rho_buf_pmic, _   = spearman_safe(buf_margins, buf_pmic)

        print(f"\n  Buforin test (n={len(with_lit)} with literature MIC):")
        print(f"  {'Correlation':<35} {'Pearson R':>10} {'Spearman rho':>13}")
        print(f"  {'-'*35} {'-'*10} {'-'*13}")
        print(f"  {'Margin vs MIC (ug/mL)':<35} {r_buf_ugml:>10.3f} {rho_buf_ugml:>13.3f}")
        print(f"  {'Margin vs log10(MIC)':<35} {r_buf_logmic:>10.3f} {rho_buf_logmic:>13.3f}")
        print(f"  {'Margin vs pMIC':<35} {r_buf_pmic:>10.3f} {rho_buf_pmic:>13.3f}")
    else:
        r_buf_ugml = rho_buf_ugml = np.nan
        r_buf_logmic = rho_buf_logmic = np.nan
        r_buf_pmic = rho_buf_pmic = np.nan

    # ── 7. Per-variant table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PER-VARIANT RESULTS")
    print(f"{'='*80}")
    print(f"\n  {'Variant':<22} {'Margin':>8} {'P(AMP)':>7} {'Lit MIC':>10} {'Group':>8}")
    print(f"  {'-'*60}")

    for v in test_variants:
        lit_s = f"{v['lit_ugml']:.1f}" if v["lit_ugml"] is not None else "---"
        print(f"  {v['name']:<22} {v['margin']:>8.3f} {v['prob_amp']:>7.3f} "
              f"{lit_s:>10} {v['group']:>8}")

    # ── 8. Figures ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        "SVM Classifier Margin vs MIC (E. coli)\n"
        f"Training: {n_amp} AMPs + {n_dec} Decoys | "
        f"Test: {len(all_test)} Buforin variants ({len(with_lit)} with MIC)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Panel A: Training AMPs — Margin vs MIC (log scale) ──────────────────
    ax = axes[0, 0]
    sc = ax.scatter(train_margins, y_mic_ugml, c=y_mic_pmic,
                    cmap="RdYlGn", s=50, alpha=0.6,
                    edgecolors="white", linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=ax, label="pMIC (higher = more potent)")

    # Trend line (log-linear)
    slope, intercept, _, _, _ = stats.linregress(train_margins, log_mic)
    x_fit = np.linspace(train_margins.min() - 0.3, train_margins.max() + 0.3, 100)
    y_fit = 10 ** (slope * x_fit + intercept)
    ax.plot(x_fit, y_fit, "--", color="#d62728", lw=2, alpha=0.7,
            label=f"log-linear fit")

    ax.set_yscale("log")
    ax.set_xlabel("SVM Decision Margin (distance to hyperplane)", fontsize=11)
    ax.set_ylabel("MIC (ug/mL) [log scale]", fontsize=11)
    ax.set_title(f"Panel A: Training AMPs (n={n_mic})\n"
                 f"Pearson R={r_train_logmic:.3f} | Spearman rho={rho_train_logmic:.3f}  "
                 f"(margin vs log10 MIC)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    # ── Panel B: Buforin Test — Margin vs MIC (log scale) ────────────────────
    ax = axes[0, 1]
    colors_map = {"i+4": "#4363d8", "i+7": "#e6194b", "Original": "#3cb44b", "WT": "#f58231"}
    markers_map = {"i+4": "o", "i+7": "^", "Original": "s", "WT": "D"}

    # Plot training AMPs as background
    ax.scatter(train_margins, y_mic_ugml, alpha=0.15, s=20,
               color="#bbbbbb", edgecolors="none", zorder=1,
               label=f"Training AMPs (n={n_mic})")

    # Plot Buf variants (those with literature MIC)
    for v in with_lit:
        ax.scatter(v["margin"], v["lit_ugml"], s=120,
                   marker=markers_map.get(v["group"], "o"),
                   color=colors_map.get(v["group"], "#333"),
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.annotate(v["short"], (v["margin"], v["lit_ugml"]),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=7.5, fontweight="bold", alpha=0.9)

    # Also plot variants WITHOUT MIC at a dashed y position
    no_lit = [v for v in all_test if v["lit_ugml"] is None]
    for v in no_lit:
        ax.scatter(v["margin"], 1.0, s=80, alpha=0.5,
                   marker=markers_map.get(v["group"], "o"),
                   color=colors_map.get(v["group"], "#333"),
                   edgecolors="k", linewidths=0.5, zorder=4)
        ax.annotate(v["short"] + "\n(no MIC)", (v["margin"], 1.0),
                    textcoords="offset points", xytext=(7, -12),
                    fontsize=6.5, alpha=0.7, style="italic")

    # Trend line through Buf points
    if len(with_lit) >= 3:
        sl2, int2, _, _, _ = stats.linregress(buf_margins, buf_logmic)
        x_f2 = np.linspace(
            min(buf_margins.min(), *[v["margin"] for v in no_lit]) - 0.3,
            max(buf_margins.max(), *[v["margin"] for v in no_lit]) + 0.3, 100)
        y_f2 = 10 ** (sl2 * x_f2 + int2)
        ax.plot(x_f2, y_f2, "--", color="#d62728", lw=1.5, alpha=0.6,
                label="Buf trend (log-linear)")

    buf_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4363d8",
               markeredgecolor="k", markersize=9, label="i+4 (F10W)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6194b",
               markeredgecolor="k", markersize=9, label="i+7 (F10W)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3cb44b",
               markeredgecolor="k", markersize=9, label="Original Buf"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#f58231",
               markeredgecolor="k", markersize=9, label="Buf WT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbb",
               markeredgecolor="none", markersize=7, label=f"Training AMPs"),
    ]
    ax.legend(handles=buf_legend, fontsize=7.5, loc="upper left")

    ax.set_yscale("log")
    ax.set_xlabel("SVM Decision Margin (distance to hyperplane)", fontsize=11)
    ax.set_ylabel("Literature MIC (ug/mL) [log scale]", fontsize=11)
    ax.set_title(f"Panel B: Buforin Test Variants (n={len(with_lit)} with MIC)\n"
                 f"Pearson R={r_buf_logmic:.3f} | Spearman rho={rho_buf_logmic:.3f}  "
                 f"(margin vs log10 MIC)",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2, which="both")

    # ── Panel C: Both sets overlaid — Margin vs pMIC ─────────────────────────
    ax = axes[1, 0]
    ax.scatter(train_margins, y_mic_pmic, alpha=0.4, s=40,
               color="#2c7bb6", edgecolors="white", linewidths=0.2,
               zorder=2, label=f"Training AMPs (n={n_mic})")

    if len(with_lit) >= 3:
        for v in with_lit:
            pmic_v = mic_to_pmic(v["lit_ugml"], v["mw"])
            ax.scatter(v["margin"], pmic_v, s=120,
                       marker=markers_map.get(v["group"], "o"),
                       color=colors_map.get(v["group"], "#333"),
                       edgecolors="k", linewidths=0.8, zorder=5)
            ax.annotate(v["short"], (v["margin"], pmic_v),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, fontweight="bold", alpha=0.9)

    # Combined trend (training only, for reference)
    sl3, int3, _, _, _ = stats.linregress(train_margins, y_mic_pmic)
    x_f3 = np.linspace(
        min(train_margins.min(), *[v["margin"] for v in all_test]) - 0.5,
        max(train_margins.max(), *[v["margin"] for v in all_test]) + 0.5, 100)
    y_f3 = sl3 * x_f3 + int3
    ax.plot(x_f3, y_f3, "--", color="#d62728", lw=1.5, alpha=0.6,
            label=f"Training trend (slope={sl3:.3f})")

    ax.set_xlabel("SVM Decision Margin", fontsize=11)
    ax.set_ylabel("pMIC = -log10(MIC_M)  [higher = more potent]", fontsize=11)
    ax.set_title(f"Panel C: Margin vs pMIC (combined view)\n"
                 f"Train: R={r_train_pmic:.3f} | "
                 f"Buf: R={r_buf_pmic:.3f}",
                 fontsize=10, fontweight="bold")
    ax.legend(handles=buf_legend, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.2)

    # ── Panel D: Bar chart — Margin + P(AMP) for all Buf variants ────────────
    ax = axes[1, 1]
    names = [v["short"] for v in all_test]
    margins = [v["margin"] for v in all_test]
    probs = [v["prob_amp"] for v in all_test]

    x_pos = np.arange(len(all_test))
    bar_colors = [colors_map.get(v["group"], "#333") for v in all_test]

    bars = ax.bar(x_pos, margins, color=bar_colors, alpha=0.75,
                  edgecolor="k", linewidth=0.5)

    # Add P(AMP) annotation on bars
    for i, (b, p) in enumerate(zip(bars, probs)):
        y_pos = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, y_pos + 0.03,
                f"P={p:.2f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold")

    # Add MIC label below bars for those with lit MIC
    for i, v in enumerate(all_test):
        if v["lit_ugml"] is not None:
            ax.text(i, -0.15, f"{v['lit_ugml']:.1f}",
                    ha="center", va="top", fontsize=6.5, color="#d62728",
                    fontweight="bold")

    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("SVM Decision Margin", fontsize=11)
    ax.set_xlabel("Buforin Variant", fontsize=11)
    ax.set_title("Panel D: SVM Margin per Buforin Variant\n"
                 "(red values = literature MIC in ug/mL)",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.15, axis="y")

    # Annotation box for test correlations
    if not np.isnan(r_buf_logmic):
        box_text = (f"Buf Margin-MIC Correlation\n"
                    f"R = {r_buf_logmic:.3f} (vs log10 MIC)\n"
                    f"rho = {rho_buf_logmic:.3f}\n"
                    f"n = {len(with_lit)} variants")
        ax.text(0.98, 0.98, box_text, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#ffffcc", edgecolor="#999", alpha=0.9))

    plt.tight_layout()
    out = BASE / "buf_mic_svm_margin.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"\n  SVM Classifier:  AUC={gs.best_score_:.3f}, "
          f"Accuracy={cv_acc.mean():.3f}")
    print(f"  Best C={gs.best_params_['svc__C']}, "
          f"gamma={gs.best_params_['svc__gamma']}")
    print(f"\n  Training AMPs with MIC (n={n_mic}):")
    print(f"    Margin vs log10(MIC):  R={r_train_logmic:.3f}, "
          f"rho={rho_train_logmic:.3f}")
    print(f"    Margin vs pMIC:        R={r_train_pmic:.3f}, "
          f"rho={rho_train_pmic:.3f}")
    if not np.isnan(r_buf_logmic):
        print(f"\n  Buforin test (n={len(with_lit)}):")
        print(f"    Margin vs log10(MIC):  R={r_buf_logmic:.3f}, "
              f"rho={rho_buf_logmic:.3f}")
        print(f"    Margin vs pMIC:        R={r_buf_pmic:.3f}, "
              f"rho={rho_buf_pmic:.3f}")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
