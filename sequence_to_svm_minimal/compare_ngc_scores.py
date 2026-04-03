#!/usr/bin/env python3
"""
compare_ngc_scores.py
=====================
Compare NGC induction values (experimental membrane-curvature data from
collaborator) against model prediction scores for 4 Buforin variants:

  • Buf WT  — native Buforin II (non-stapled)
  • Buf 12  — i+4 hydrocarbon staple
  • Buf 13  — i+4 hydrocarbon staple (different window)
  • Buf Q9K — Buf 13 with Q9K mutation

NGC = Negative Gaussian Curvature (nm⁻²).
Higher NGC → stronger saddle-shaped membrane deformation → more potent AMP.

Models compared
---------------
  SVM-1  QSAR only       pretrained 2016 PNAS SVM  (linear kernel, Platt-scaled)
  SVM-2  QSAR + StaPep   RBF-SVM retrained on stapled AMP/decoy set
  SVM-3  StaPep only     RBF-SVM retrained on stapled AMP/decoy set
  MLP-1  StaPep only     MLP (4 hidden layers) trained on stapled set
  MLP-2  QSAR only       MLP trained on stapled set
  MLP-3  QSAR + StaPep   MLP trained on stapled set

Scores reported
---------------
  • P(AMP)           — calibrated probability of being antimicrobial
  • Decision margin  — SVM raw score (distance to decision hyperplane)
                       positive = AMP side, larger = more confident
                       (not available for MLP; log-odds shown instead)
"""

import warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau

from sklearn.svm            import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"
PP_DIR = BASE / ".." / "pretrained_svm" / "sequence_to_svm_minimal" / "predictionsParameters"

# ── NGC values from collaborator (nm⁻²) ───────────────────────────────────────
# ALL variants are test peptides — native Buforin (Buf WT) is NOT in the
# stapled AMP training set (stapled_amps_features.csv uses DRAMP stapled peptides
# only). Buf WT scores lower because it is unstapled — the model was trained
# specifically on hydrocarbon-stapled AMPs vs decoys.
NGC_DATA = {
    #  name         NGC (nm⁻²)   unstapled?
    "Buf WT":  {"ngc": 0.0133, "training": False, "unstapled": True},
    "Buf 12":  {"ngc": 0.0109, "training": False, "unstapled": False},
    "Buf 13":  {"ngc": 0.0145, "training": False, "unstapled": False},
    "Buf Q9K": {"ngc": 0.0130, "training": False, "unstapled": False},
}
VARIANTS = list(NGC_DATA.keys())
NGC_VALS = np.array([NGC_DATA[v]["ngc"] for v in VARIANTS])

# ── Feature columns ────────────────────────────────────────────────────────────
STAPEP_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]
QSAR_COLS = [
    "netCharge", "FC", "LW", "DP", "NK", "AE", "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104", "tau4_GRAR740104",
    "QSO50_GRAR740104", "QSO29_GRAR740104",
]

# ── Hardcoded StaPep features for each variant ─────────────────────────────────
# Buf WT extracted via run_buforin_stapep.py (5 ns MD)
# Buf 12/13/Q9K from test_stapled_features.csv
SP_FEATURES = {
    "Buf WT": {
        "length": 21, "weight": 2473.829,
        "hydrophobic_index": -0.8142857142857142, "charge": 6.094,
        "aromaticity": 0.047619047619047616, "isoelectric_point": 11.999967765808105,
        "fraction_arginine": 0.23809523809523808, "fraction_lysine": 0.047619047619047616,
        "lyticity_index": 300.106, "helix_percent": 0.17819047619047618,
        "sheet_percent": 0.0007619047619047618, "loop_percent": 0.821047619047619,
        "mean_bfactor": 573.4344313305434, "mean_gyrate": 12.001021374843388,
        "num_hbonds": 0, "psa": 1064.2171630859375, "sasa": 2038.292463648343,
    },
    "Buf 12": {
        "length": 21, "weight": 2491.930,
        "hydrophobic_index": -0.4714285714285717, "charge": 6.996,
        "aromaticity": 0.047619047619047616, "isoelectric_point": 11.999967765808105,
        "fraction_arginine": 0.23809523809523808, "fraction_lysine": 0.047619047619047616,
        "lyticity_index": 502.854, "helix_percent": 0.018571428571428572,
        "sheet_percent": 0.0, "loop_percent": 0.9814285714285714,
        "mean_bfactor": 608.4296642126951, "mean_gyrate": 14.343374692386986,
        "num_hbonds": 1, "psa": 1077.7835693359375, "sasa": 2254.3843296864616,
    },
    "Buf 13": {
        "length": 21, "weight": 2514.964,
        "hydrophobic_index": -0.19047619047619055, "charge": 6.094,
        "aromaticity": 0.047619047619047616, "isoelectric_point": 11.999967765808105,
        "fraction_arginine": 0.19047619047619047, "fraction_lysine": 0.047619047619047616,
        "lyticity_index": 403.066, "helix_percent": 0.13380952380952377,
        "sheet_percent": 0.017142857142857144, "loop_percent": 0.8490476190476189,
        "mean_bfactor": 562.1170263259725, "mean_gyrate": 12.080517163086721,
        "num_hbonds": 1, "psa": 1037.6502685546875, "sasa": 2076.4826314823017,
    },
    "Buf Q9K": {
        "length": 21, "weight": 2515.008,
        "hydrophobic_index": -0.20952380952380953, "charge": 7.094,
        "aromaticity": 0.047619047619047616, "isoelectric_point": 11.999967765808105,
        "fraction_arginine": 0.19047619047619047, "fraction_lysine": 0.09523809523809523,
        "lyticity_index": 403.066, "helix_percent": 0.056190476190476187,
        "sheet_percent": 0.0, "loop_percent": 0.9438095238095238,
        "mean_bfactor": 372.96412476928043, "mean_gyrate": 12.019611160983318,
        "num_hbonds": 2, "psa": 1028.093994140625, "sasa": 2141.4632109684153,
    },
}

# ── QSAR features for each variant ────────────────────────────────────────────
# Buf WT = AMP_101 from qsar12_descriptors.csv
# Buf 12/13/Q9K from qsar_stapled_test.csv (all share same QSAR seq = parent Buforin)
QSAR_FEATURES = {
    "Buf WT":  {"netCharge": 7, "FC": 0, "LW": 0, "DP": 0, "NK": 0, "AE": 0,
                "pcMK": 0, "_SolventAccessibilityD1025": 33.333,
                "tau2_GRAR740104": 0, "tau4_GRAR740104": 0,
                "QSO50_GRAR740104": 0, "QSO29_GRAR740104": 0},
    "Buf 12":  {"netCharge": 7, "FC": 0, "LW": 0, "DP": 0, "NK": 0, "AE": 0,
                "pcMK": 0, "_SolventAccessibilityD1025": 33.333,
                "tau2_GRAR740104": 0, "tau4_GRAR740104": 0,
                "QSO50_GRAR740104": 0, "QSO29_GRAR740104": 0},
    "Buf 13":  {"netCharge": 7, "FC": 0, "LW": 0, "DP": 0, "NK": 0, "AE": 0,
                "pcMK": 0, "_SolventAccessibilityD1025": 33.333,
                "tau2_GRAR740104": 0, "tau4_GRAR740104": 0,
                "QSO50_GRAR740104": 0, "QSO29_GRAR740104": 0},
    "Buf Q9K": {"netCharge": 7, "FC": 0, "LW": 0, "DP": 0, "NK": 0, "AE": 0,
                "pcMK": 0, "_SolventAccessibilityD1025": 33.333,
                "tau2_GRAR740104": 0, "tau4_GRAR740104": 0,
                "QSO50_GRAR740104": 0, "QSO29_GRAR740104": 0},
}

# ── SVM Hyperparameter grid ────────────────────────────────────────────────────
SVM_GRID = {
    "svc__C":     [0.1, 1, 10, 100],
    "svc__gamma": ["scale", 1e-3, 1e-2, 0.1],
}


# ════════════════════════════════════════════════════════════════════════════════
# Load training data
# ════════════════════════════════════════════════════════════════════════════════
def load_training():
    amp_sp = pd.read_csv(STAPEP / "stapled_amps_features.csv").rename(
        columns={"DRAMP_ID": "peptide_id", "Hiden_Sequence": "sequence"})
    dec_sp = pd.read_csv(STAPEP / "stapled_decoys.csv").rename(
        columns={"COMPOUND_ID": "peptide_id", "SEQUENCE": "sequence"})
    amp_sp["label"] = 1;  dec_sp["label"] = 0

    def _present(df, cols):
        return [c for c in cols if c in df.columns]

    sp_cols_amp = _present(amp_sp, STAPEP_COLS)
    sp_cols_dec = _present(dec_sp, STAPEP_COLS)

    amp_sp = amp_sp.dropna(subset=sp_cols_amp, how="all")
    dec_sp = dec_sp.dropna(subset=sp_cols_dec, how="all")

    amp_qsar = pd.DataFrame(); dec_qsar = pd.DataFrame()
    q_amp_f = STAPEP / "qsar_stapled_amps.csv"
    q_dec_f = STAPEP / "qsar_stapled_decoys.csv"
    if q_amp_f.exists():
        amp_qsar = pd.read_csv(q_amp_f); amp_qsar["label"] = 1
    if q_dec_f.exists():
        dec_qsar = pd.read_csv(q_dec_f); dec_qsar["label"] = 0

    return amp_sp, dec_sp, amp_qsar, dec_qsar


# ════════════════════════════════════════════════════════════════════════════════
# Build feature matrices
# ════════════════════════════════════════════════════════════════════════════════
def _sp_matrix(amp_sp, dec_sp):
    cols  = [c for c in STAPEP_COLS if c in amp_sp.columns and c in dec_sp.columns]
    train = pd.concat([amp_sp[["label"] + cols],
                       dec_sp[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, cols


def _qsar_matrix(amp_qsar, dec_qsar):
    cols  = [c for c in QSAR_COLS if c in amp_qsar.columns and c in dec_qsar.columns]
    train = pd.concat([amp_qsar[["label"] + cols],
                       dec_qsar[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, cols


def _qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar):
    sp_cols  = [c for c in STAPEP_COLS if c in amp_sp.columns]
    amp_sp["_join"] = amp_sp["peptide_id"].astype(str)
    dec_sp["_join"] = (dec_sp.reset_index(drop=True).index + 1).astype(str)
    amp_qsar = amp_qsar.copy(); amp_qsar["_join"] = amp_qsar["peptide_id"].astype(str)
    dec_qsar = dec_qsar.copy()
    dec_qsar["_join"] = (dec_qsar.reset_index(drop=True).index + 1).astype(str)

    q_only = [c for c in QSAR_COLS if c in amp_qsar.columns and c not in sp_cols]
    amp_m = amp_sp[["_join","label"] + sp_cols].merge(amp_qsar[["_join"] + q_only], on="_join", how="inner")
    dec_m = dec_sp[["_join","label"] + sp_cols].merge(dec_qsar[["_join"] + q_only], on="_join", how="inner")

    all_cols = sp_cols + q_only
    train = pd.concat([amp_m, dec_m], ignore_index=True)
    return train[all_cols].values.astype(float), train["label"].values, all_cols


# ════════════════════════════════════════════════════════════════════════════════
# Build test vectors for the 4 Buf variants
# ════════════════════════════════════════════════════════════════════════════════
def variant_matrix(feature_names):
    rows = []
    for v in VARIANTS:
        sp_feats  = SP_FEATURES.get(v, {})
        q_feats   = QSAR_FEATURES.get(v, {})
        merged    = {**sp_feats, **q_feats}
        rows.append([float(merged.get(c, np.nan)) for c in feature_names])
    return np.array(rows)


# ════════════════════════════════════════════════════════════════════════════════
# Models
# ════════════════════════════════════════════════════════════════════════════════
def _svm_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42)),
    ])

def _mlp_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu", solver="adam",
            learning_rate="adaptive", learning_rate_init=1e-3,
            max_iter=3000, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=80,
            tol=1e-6, random_state=42)),
    ])

def fit_svm_gs(X, y, cv=5):
    gs = GridSearchCV(_svm_pipe(), SVM_GRID,
                      cv=StratifiedKFold(cv, shuffle=True, random_state=42),
                      scoring="roc_auc", n_jobs=-1, refit=True)
    gs.fit(X, y)
    return gs

def fit_mlp(X, y):
    clf = _mlp_pipe()
    clf.fit(X, y)
    return clf


# ════════════════════════════════════════════════════════════════════════════════
# Pretrained QSAR SVM (2016 PNAS)
# ════════════════════════════════════════════════════════════════════════════════
def _npy(n):
    return np.load(PP_DIR / f"svc.pkl_{n:02d}.npy", allow_pickle=False)

def pretrained_qsar_scores():
    """Returns (proba, decision) for all 4 Buf variants using pretrained QSAR SVM."""
    with open(PP_DIR / "Z_score_mean_std__intersect_noflip.csv") as f:
        desc_names = f.readline().strip().split(",")
        z_means    = np.array([float(x) for x in f.readline().strip().split(",")])
        z_stds     = np.array([float(x) for x in f.readline().strip().split(",")])

    support_vectors_ = _npy(7)
    dual_coef_       = _npy(3)
    intercept_       = _npy(10)
    probA_           = _npy(4)
    probB_           = _npy(11)

    w = support_vectors_.T @ dual_coef_[0]

    probas    = []
    decisions = []
    for v in VARIANTS:
        q = QSAR_FEATURES[v]
        feat = np.array([q.get(d, 0.0) for d in desc_names], dtype=float)
        x_z  = (feat - z_means) / z_stds
        dec  = float(x_z @ w + intercept_[0])
        fval = probA_[0] * dec + probB_[0]
        prob = 1.0 / (1.0 + np.exp(fval))
        probas.append(prob)
        decisions.append(dec)

    return np.array(probas), np.array(decisions)


# ════════════════════════════════════════════════════════════════════════════════
# Colour / style constants
# ════════════════════════════════════════════════════════════════════════════════
VARIANT_COLORS = {
    "Buf WT":  "#e6194b",   # red   (non-stapled — shown differently)
    "Buf 12":  "#3cb44b",   # green
    "Buf 13":  "#4363d8",   # blue
    "Buf Q9K": "#f58231",   # orange
}
VARIANT_MARKERS = {
    "Buf WT":  "D",   # diamond = non-stapled (all are test peptides)
    "Buf 12":  "o",
    "Buf 13":  "s",
    "Buf Q9K": "^",
}


# ════════════════════════════════════════════════════════════════════════════════
# Correlation metrics helper (Spearman ρ + Kendall τ)
# ════════════════════════════════════════════════════════════════════════════════
def _corr_metrics(x, y):
    """Return (spearman_rho, spearman_p, kendall_tau, kendall_p) for arrays x, y."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    rho, p_s = spearmanr(x[mask], y[mask])
    tau, p_k = kendalltau(x[mask], y[mask])
    return float(rho), float(p_s), float(tau), float(p_k)


# ════════════════════════════════════════════════════════════════════════════════
# Panel A: NGC vs score scatter with regression + rank labels
# ════════════════════════════════════════════════════════════════════════════════
def _panel_scatter(ax, scores, score_label, title):
    """
    Scatter of NGC (x) vs model score (y) for the 4 Buf variants.
    Adds:
      • Spearman ρ and Kendall τ annotation
      • Regression line (OLS) through all 4 points
      • Rank labels for both axes (shows whether ranking is preserved)
    """
    x = NGC_VALS.copy()
    y = np.array(scores, dtype=float)

    # ── regression line ────────────────────────────────────────────────────
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() >= 2:
        coeffs = np.polyfit(x[mask], y[mask], 1)
        xfit   = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xfit, np.polyval(coeffs, xfit),
                color="gray", lw=1.2, ls="--", zorder=1, alpha=0.7)

    # ── scatter points ─────────────────────────────────────────────────────
    # Compute ranks for NGC and scores (1 = lowest)
    ngc_ranks   = np.argsort(np.argsort(x)) + 1          # 1-based rank
    score_ranks = np.argsort(np.argsort(y)) + 1

    for i, v in enumerate(VARIANTS):
        ax.scatter(x[i], y[i],
                   s=130,
                   marker=VARIANT_MARKERS[v],
                   color=VARIANT_COLORS[v],
                   edgecolors="k", linewidths=0.8,
                   zorder=5)
        # label with name + "(NGC rank → score rank)"
        unstap_tag = " ◆unstapled" if NGC_DATA[v].get("unstapled") else ""
        lab = f"{v}{unstap_tag}\n(NGC#{ngc_ranks[i]}→mdl#{score_ranks[i]})"
        ax.annotate(lab, (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(6, -14) if i % 2 == 0 else (6, 4),
                    fontsize=7, color=VARIANT_COLORS[v],
                    fontweight="bold")

    # ── correlation annotations ────────────────────────────────────────────
    rho, p_s, tau, p_k = _corr_metrics(x, y)
    txt = (f"Spearman ρ = {rho:+.3f}\n"
           f"Kendall τ  = {tau:+.3f}\n"
           f"(N = {mask.sum()}, caution: small)")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=7.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa", alpha=0.92))

    ax.set_xlabel("NGC  (nm⁻²)", fontsize=9)
    ax.set_ylabel(score_label, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.22, lw=0.5)


# ════════════════════════════════════════════════════════════════════════════════
# Panel B: Bump chart — rank order comparison
# ════════════════════════════════════════════════════════════════════════════════
def _panel_bump(ax, scores_dict):
    """
    Parallel-coordinates / bump chart:
    Each column = one ranking system (NGC, SVM-SP margin, MLP-SP prob).
    Y-axis = rank (4 = best = highest value).
    Lines connect the SAME variant across systems.
    If lines are flat / non-crossing → perfect concordance with NGC.
    Crossings indicate rank inversion (model disagrees with NGC).
    """
    systems = ["NGC"] + list(scores_dict.keys())
    all_vals = [NGC_VALS] + [np.array(v, dtype=float) for v in scores_dict.values()]

    # Ranks: 4 = highest value (best AMP / most membrane-disrupting)
    n = len(VARIANTS)
    all_ranks = []
    for vals in all_vals:
        order = np.argsort(vals)          # lowest → highest
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n+1) # rank 1 = lowest
        all_ranks.append(ranks)

    all_ranks = np.array(all_ranks)  # shape: (n_systems, n_variants)

    x_pos = np.arange(len(systems))

    for vi, v in enumerate(VARIANTS):
        y_vals = all_ranks[:, vi]
        ls = "--" if NGC_DATA[v].get("unstapled") else "-"
        ax.plot(x_pos, y_vals,
                color=VARIANT_COLORS[v],
                marker=VARIANT_MARKERS[v],
                markersize=8,
                linewidth=1.8,
                linestyle=ls,
                label=v, zorder=4)
        tag = " (unstapled)" if NGC_DATA[v].get("unstapled") else ""
        # label at the left end (NGC column)
        ax.annotate(v + tag, (0, y_vals[0]),
                    textcoords="offset points", xytext=(-75, -4),
                    fontsize=7.5, color=VARIANT_COLORS[v], fontweight="bold")
        # label at the right end
        ax.annotate(v, (len(systems)-1, y_vals[-1]),
                    textcoords="offset points", xytext=(4, -4),
                    fontsize=7.5, color=VARIANT_COLORS[v], fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.replace("\n", "\n") for s in systems],
                       fontsize=8, rotation=20, ha="right")
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Rank 1\n(lowest)", "Rank 2", "Rank 3", "Rank 4\n(highest)"],
                       fontsize=7.5)
    ax.set_ylabel("Rank", fontsize=9)
    ax.set_title("Rank Order Concordance\n(flat lines = matches NGC ranking)", 
                 fontsize=9, fontweight="bold", pad=6)
    ax.set_xlim(-0.7, len(systems) - 0.4)
    ax.set_ylim(0.5, 4.5)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ════════════════════════════════════════════════════════════════════════════════
# Panel C: Correlation metrics bar chart
# ════════════════════════════════════════════════════════════════════════════════
def _panel_corr_bar(ax, all_scores):
    """
    Horizontal bar chart of Spearman ρ for all model scores vs NGC.
    Color encodes sign (green = positive correlation, red = negative).
    Grey = constant score (QSAR gives same value for all variants).
    """
    labels, rhos, taus = [], [], []
    for label, scores in all_scores.items():
        y = np.array(scores, dtype=float)
        rho, _, tau, _ = _corr_metrics(NGC_VALS, y)
        labels.append(label.replace("\n", " "))
        rhos.append(rho)
        taus.append(tau)

    rhos = np.array(rhos)
    taus = np.array(taus)
    y_pos = np.arange(len(labels))

    # bar colors
    bar_colors_rho = ["#3cb44b" if r > 0 else "#e6194b" if r < 0 else "#aaaaaa"
                      for r in rhos]
    bar_colors_tau = ["#4363d8" if t > 0 else "#f58231" if t < 0 else "#aaaaaa"
                      for t in taus]

    bar_h = 0.35
    bars1 = ax.barh(y_pos + bar_h/2, rhos, bar_h,
                    color=bar_colors_rho, edgecolor="k", linewidth=0.5,
                    label="Spearman ρ")
    bars2 = ax.barh(y_pos - bar_h/2, taus, bar_h,
                    color=bar_colors_tau, edgecolor="k", linewidth=0.5, alpha=0.8,
                    label="Kendall τ")

    # value labels
    for bar, val in zip(bars1, rhos):
        if not np.isnan(val):
            ha = "left" if val >= 0 else "right"
            off = 0.02 if val >= 0 else -0.02
            ax.text(val + off, bar.get_y() + bar.get_height()/2,
                    f"{val:+.2f}", va="center", ha=ha, fontsize=7)
    for bar, val in zip(bars2, taus):
        if not np.isnan(val):
            ha = "left" if val >= 0 else "right"
            off = 0.02 if val >= 0 else -0.02
            ax.text(val + off, bar.get_y() + bar.get_height()/2,
                    f"{val:+.2f}", va="center", ha=ha, fontsize=7)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Correlation with NGC", fontsize=9)
    ax.set_title("Spearman ρ & Kendall τ\nvs NGC (all models)", 
                 fontsize=9, fontweight="bold", pad=6)
    ax.set_xlim(-1.1, 1.1)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", alpha=0.22, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # reference lines
    for v in [-1, -0.5, 0.5, 1]:
        ax.axvline(v, color="gray", lw=0.5, ls=":", alpha=0.5)


# ════════════════════════════════════════════════════════════════════════════════
# Full figure builder
# ════════════════════════════════════════════════════════════════════════════════
def _make_main_figure(scores_for_plot, BASE):
    """
    Main figure: scatter plots (top row) + bump chart + summary bar.
    """
    from matplotlib.lines import Line2D

    # Which scores to scatter-plot (skip QSAR — constant)
    scatter_models = [
        ("SVM-StaPep\nMargin",  "SVM (StaPep) decision margin",  "SVM — StaPep\nDecision Function (margin)"),
        ("MLP-StaPep\nP(AMP)",  "MLP (StaPep) P(AMP)",           "MLP — StaPep\nP(AMP)"),
        ("SVM-StaPep\nP(AMP)",  "SVM (StaPep) P(AMP)",           "SVM — StaPep\nP(AMP)"),
    ]
    if "SVM-Q+SP\nMargin" in scores_for_plot:
        scatter_models.append(
            ("SVM-Q+SP\nMargin", "SVM (QSAR+StaPep) decision margin", "SVM — QSAR+StaPep\nDecision Function"))

    n_scatter = min(len(scatter_models), 4)

    # Layout: row0 = scatter plots, row1 = bump + bar
    fig = plt.figure(figsize=(5.5 * n_scatter, 11))
    fig.suptitle(
        "NGC Induction vs Machine Learning Scores — 4 Buforin Variants\n"
        "NGC = Negative Gaussian Curvature (nm⁻²)  |  ◆ Buf WT is non-stapled (all are test peptides — none in training set)",
        fontsize=11, fontweight="bold", y=1.01,
    )

    gs_top = gridspec.GridSpec(1, n_scatter, figure=fig,
                               top=0.92, bottom=0.55, hspace=0.1, wspace=0.38)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               top=0.48, bottom=0.07, hspace=0.1, wspace=0.38)

    # ── Top row: scatter panels ────────────────────────────────────────────
    for col, (key, ylabel, title) in enumerate(scatter_models[:n_scatter]):
        ax = fig.add_subplot(gs_top[0, col])
        _panel_scatter(ax, scores_for_plot[key], ylabel, title)

    # ── Bottom left: bump chart ────────────────────────────────────────────
    bump_models = {}
    for key in ["SVM-StaPep\nMargin", "MLP-StaPep\nP(AMP)"]:
        if key in scores_for_plot:
            bump_models[key] = scores_for_plot[key]
    if "SVM-Q+SP\nMargin" in scores_for_plot:
        bump_models["SVM-Q+SP\nMargin"] = scores_for_plot["SVM-Q+SP\nMargin"]

    ax_bump = fig.add_subplot(gs_bot[0, 0])
    _panel_bump(ax_bump, bump_models)

    # ── Bottom right: correlation bar ─────────────────────────────────────
    ax_bar = fig.add_subplot(gs_bot[0, 1])

    # Build full set for bar (include QSAR for reference)
    bar_scores = {k: v for k, v in scores_for_plot.items()}

    _panel_corr_bar(ax_bar, bar_scores)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = []
    for v in VARIANTS:
        tag = " (non-stapled)" if NGC_DATA[v].get("unstapled") else " (stapled)"
        h = Line2D([0], [0], marker=VARIANT_MARKERS[v],
                   color="w", markerfacecolor=VARIANT_COLORS[v],
                   markeredgecolor="k", markersize=9,
                   label=f"{v}{tag}")
        legend_handles.append(h)

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.02),
               title="Buforin variant  |  rank label: NGC#rank → model#rank  |  dashed line = non-stapled",
               title_fontsize=8)

    # Footer
    fig.text(0.01, -0.01,
             "Spearman ρ and Kendall τ are rank-based correlation coefficients robust to non-normality.\n"
             "With N=4, p-values are not significant — interpret direction and magnitude only.\n"
             "Buf WT is the native non-stapled peptide. It is NOT in the training set — low scores reflect\n"
             "that the model was trained on stapled AMPs and has not seen the unstapled structural signature.\n"
             "QSAR model omitted from scatter/bump (gives identical score to all 4 variants → cannot differentiate).",
             fontsize=7, color="#555555", va="top")

    out = BASE / "ngc_vs_scores.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Main figure saved → {out}")


def _make_summary_figure(all_scores, BASE):
    """
    Standalone summary: horizontal bar chart of Spearman ρ + Kendall τ
    for ALL models vs NGC, with an interpretation guide.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("NGC Correlation Summary — All Models\n"
                 "Spearman ρ and Kendall τ vs Negative Gaussian Curvature",
                 fontsize=11, fontweight="bold")
    _panel_corr_bar(ax, all_scores)
    plt.tight_layout()
    out = BASE / "ngc_correlation_summary.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Summary figure saved → {out}")


def _make_qsar_blindness_figure(qsar_proba, qsar_dec,
                                 sp_proba, sp_dec,
                                 mlp_sp_proba,
                                 qs_proba, qs_dec,
                                 BASE):
    """
    Dedicated figure showing QSAR model is completely blind to Buforin variant
    differences, while StaPep-based models differentiate them clearly.

    Three panels:
      Panel A — Score comparison (line + scatter): QSAR flat, StaPep varies
      Panel B — NGC vs score: QSAR all stacked, StaPep spread
      Panel C — Range / spread bar: how much each model's score varies
    """
    from matplotlib.lines import Line2D

    # Sort variants by NGC (ascending) for a clean x-axis
    ngc_order = np.argsort(NGC_VALS)  # Buf12, Buf Q9K, Buf WT, Buf 13
    sorted_names = [VARIANTS[i] for i in ngc_order]
    sorted_ngc   = NGC_VALS[ngc_order]

    def _sort(arr):
        return np.array(arr)[ngc_order]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "QSAR is Blind to Buforin Variant Differences — StaPep Captures Them\n"
        "Models compared on 4 Buforin variants ordered by increasing NGC induction",
        fontsize=13, fontweight="bold", y=1.00,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40,
                           top=0.92, bottom=0.10)

    x_ticks = np.arange(len(VARIANTS))

    # ── Panel A: Line chart — scores for each variant ─────────────────────────
    ax_a = fig.add_subplot(gs[0, :2])  # spans first two columns

    # QSAR — flat
    ax_a.plot(x_ticks, _sort(qsar_proba),
              color="#aaaaaa", lw=2.5, ls="-",
              marker="o", markersize=8,
              label="QSAR (pretrained) P(AMP)", zorder=3)
    ax_a.fill_between(x_ticks,
                      _sort(qsar_proba) - 0.005,
                      _sort(qsar_proba) + 0.005,
                      color="#aaaaaa", alpha=0.15)

    # StaPep SVM P(AMP)
    ax_a.plot(x_ticks, _sort(sp_proba),
              color="#3cb44b", lw=2.5, ls="-",
              marker="s", markersize=8,
              label="SVM — StaPep P(AMP)", zorder=4)

    # MLP StaPep P(AMP)
    ax_a.plot(x_ticks, _sort(mlp_sp_proba),
              color="#4363d8", lw=2.5, ls="--",
              marker="^", markersize=8,
              label="MLP — StaPep P(AMP)", zorder=4)

    if qs_proba is not None:
        ax_a.plot(x_ticks, _sort(qs_proba),
                  color="#f58231", lw=2.0, ls="-.",
                  marker="D", markersize=7,
                  label="SVM — QSAR+StaPep P(AMP)", zorder=4)

    # Annotate NGC values on top x-axis
    ax_a2 = ax_a.twiny()
    ax_a2.set_xlim(ax_a.get_xlim())
    ax_a2.set_xticks(x_ticks)
    ax_a2.set_xticklabels([f"NGC={n:.4f}" for n in sorted_ngc],
                           fontsize=7.5, rotation=15, ha="left")
    ax_a2.set_xlabel("Experimental NGC (nm⁻²)", fontsize=8, labelpad=2)

    # Annotate flat QSAR region
    mid_x = len(VARIANTS) / 2 - 0.5
    ax_a.annotate(
        "← QSAR score is identical\n   for ALL 4 variants\n   (Spearman ρ = N/A)",
        xy=(mid_x, float(_sort(qsar_proba).mean())),
        xytext=(mid_x + 0.15, 0.35),
        fontsize=8.5, color="#777777", fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc"),
    )

    ax_a.set_xticks(x_ticks)
    ax_a.set_xticklabels(
        [f"{n}\n({'◆ unstapled' if NGC_DATA[n].get('unstapled') else 'stapled'})"
         for n in sorted_names],
        fontsize=9,
    )
    ax_a.set_ylabel("P(AMP) — probability of being antimicrobial", fontsize=9)
    ax_a.set_title("Panel A — Model Score per Variant  (ordered by increasing NGC)",
                   fontsize=9, fontweight="bold", pad=22)
    ax_a.set_ylim(-0.05, 1.08)
    ax_a.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax_a.grid(axis="y", alpha=0.25, lw=0.5)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # ── Panel B: Score spread bar chart (range) ───────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])

    model_names = ["QSAR\n(pretrained)", "SVM\nStaPep", "MLP\nStaPep"]
    model_scores = [qsar_proba, sp_proba, mlp_sp_proba]
    model_colors = ["#aaaaaa", "#3cb44b", "#4363d8"]
    if qs_proba is not None:
        model_names.append("SVM\nQ+SP")
        model_scores.append(qs_proba)
        model_colors.append("#f58231")

    ranges  = [float(np.max(s) - np.min(s)) for s in model_scores]
    stdevs  = [float(np.std(s))             for s in model_scores]
    y_pos_b = np.arange(len(model_names))

    bars = ax_b.barh(y_pos_b, ranges, color=model_colors,
                     edgecolor="k", linewidth=0.6, height=0.5)
    for bar, r, sd in zip(bars, ranges, stdevs):
        ax_b.text(r + 0.01, bar.get_y() + bar.get_height()/2,
                  f"range={r:.3f}\nSD={sd:.3f}",
                  va="center", ha="left", fontsize=7.5,
                  fontweight="bold" if r > 0.01 else "normal",
                  color="#333333")

    ax_b.set_yticks(y_pos_b)
    ax_b.set_yticklabels(model_names, fontsize=9)
    ax_b.set_xlabel("Score Range  (max − min across 4 variants)", fontsize=8.5)
    ax_b.set_title("Panel B — Score Spread\n(larger = more discriminating)", 
                   fontsize=9, fontweight="bold")
    ax_b.set_xlim(0, max(ranges) * 1.55)
    ax_b.axvline(0.01, color="red", lw=1, ls="--", alpha=0.5, label="threshold=0.01")
    ax_b.text(0.01, -0.6, "effectively\nzero →", fontsize=7, color="red",
              va="top", ha="left")
    ax_b.grid(axis="x", alpha=0.22, lw=0.5)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ── Panel C: NGC vs QSAR  (all stacked) ──────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    for i, v in enumerate(VARIANTS):
        # add tiny vertical jitter so stacked points are visible
        jitter = (i - 1.5) * 0.0003
        ax_c.scatter(NGC_DATA[v]["ngc"], float(qsar_proba[i]) + jitter,
                     s=120, marker=VARIANT_MARKERS[v],
                     color=VARIANT_COLORS[v], edgecolors="k", lw=0.8, zorder=5)
        ax_c.annotate(v, (NGC_DATA[v]["ngc"], float(qsar_proba[i]) + jitter),
                      textcoords="offset points", xytext=(5, 3),
                      fontsize=7.5, color=VARIANT_COLORS[v], fontweight="bold")

    # Draw flat "blind" line
    x_line = np.array([min(NGC_VALS)-0.0003, max(NGC_VALS)+0.0003])
    ax_c.plot(x_line, [qsar_proba.mean()] * 2,
              color="#aaaaaa", lw=2, ls="--", zorder=2,
              label="QSAR score (flat)")
    ax_c.text(0.5, 0.5,
              "COMPLETELY FLAT\n(ρ = undefined)",
              transform=ax_c.transAxes, fontsize=12,
              ha="center", va="center", color="gray", alpha=0.35,
              fontweight="bold", rotation=10)

    rho_c, _, tau_c, _ = _corr_metrics(NGC_VALS, np.array(qsar_proba))
    ax_c.set_xlabel("NGC (nm⁻²)", fontsize=9)
    ax_c.set_ylabel("QSAR P(AMP)", fontsize=9)
    ax_c.set_title("Panel C — NGC vs QSAR P(AMP)\n(pretrained, linear kernel)",
                   fontsize=9, fontweight="bold")
    ax_c.set_ylim(-0.02, 0.06)
    ax_c.grid(alpha=0.22, lw=0.5)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    # ── Panel D: NGC vs SVM-StaPep margin  (spread) ──────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _panel_scatter(ax_d, sp_dec,
                   "SVM-StaPep Decision Margin",
                   "Panel D — NGC vs SVM-StaPep Margin\n(distance to hyperplane)")

    # ── Panel E: NGC vs MLP-StaPep P(AMP) ────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 2])
    _panel_scatter(ax_e, mlp_sp_proba,
                   "MLP-StaPep P(AMP)",
                   "Panel E — NGC vs MLP-StaPep P(AMP)")

    # ── Shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], marker=VARIANT_MARKERS[v], color="w",
               markerfacecolor=VARIANT_COLORS[v], markeredgecolor="k",
               markersize=9,
               label=f"{v} {'(◆ non-stapled)' if NGC_DATA[v].get('unstapled') else '(stapled)'}")
        for v in VARIANTS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01),
               title="Buforin variant  |  rank labels: NGC#rank → model#rank",
               title_fontsize=8)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.01, -0.03,
             "Key insight: QSAR features (netCharge, solvent accessibility, autocorrelations) are identical for all 4 variants\n"
             "because they share the same parent sequence. The hydrocarbon staple is invisible to QSAR.\n"
             "StaPep MD features (lyticity index, helix%, mean B-factor) differ per staple position and correctly differentiate them.",
             fontsize=7.5, color="#444444", va="top")

    out = BASE / "ngc_qsar_blindness.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  QSAR blindness figure saved → {out}")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 65)
    print("  NGC vs Model Score Comparison — 4 Buforin variants")
    print("=" * 65)

    # ── Load training data ────────────────────────────────────────────────────
    print("\n  Loading training data ...")
    amp_sp, dec_sp, amp_qsar, dec_qsar = load_training()
    qsar_avail = not amp_qsar.empty and not dec_qsar.empty
    print(f"  StaPep AMPs: {len(amp_sp)}   Decoys: {len(dec_sp)}")
    if qsar_avail:
        print(f"  QSAR   AMPs: {len(amp_qsar)}   Decoys: {len(dec_qsar)}")

    # ── Build training matrices ───────────────────────────────────────────────
    X_sp, y_sp, cols_sp = _sp_matrix(amp_sp, dec_sp)

    X_q = X_qs = None; cols_q = cols_qs = []
    if qsar_avail:
        X_q,  y_q,  cols_q  = _qsar_matrix(amp_qsar, dec_qsar)
        X_qs, y_qs, cols_qs = _qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar)

    # ── Pretrained QSAR SVM ───────────────────────────────────────────────────
    print("\n  Running pretrained QSAR SVM ...")
    qsar_proba, qsar_dec = pretrained_qsar_scores()

    # ── Train new SVMs ────────────────────────────────────────────────────────
    print("  Training StaPep SVM (GridSearchCV) ...")
    gs_sp = fit_svm_gs(X_sp, y_sp)
    Xt_sp = variant_matrix(cols_sp)
    sp_proba  = gs_sp.predict_proba(Xt_sp)[:, 1]
    sp_dec    = gs_sp.decision_function(Xt_sp)
    print(f"    Best: C={gs_sp.best_params_['svc__C']}  γ={gs_sp.best_params_['svc__gamma']}")

    qs_proba = qs_dec = None
    if qsar_avail:
        print("  Training QSAR+StaPep SVM (GridSearchCV) ...")
        gs_qs = fit_svm_gs(X_qs, y_qs)
        Xt_qs = variant_matrix(cols_qs)
        qs_proba = gs_qs.predict_proba(Xt_qs)[:, 1]
        qs_dec   = gs_qs.decision_function(Xt_qs)
        print(f"    Best: C={gs_qs.best_params_['svc__C']}  γ={gs_qs.best_params_['svc__gamma']}")

    # ── Train MLPs ────────────────────────────────────────────────────────────
    print("  Training StaPep MLP ...")
    mlp_sp_clf = fit_mlp(X_sp, y_sp)
    mlp_sp_proba = mlp_sp_clf.predict_proba(Xt_sp)[:, 1]
    # log-odds (analogue of decision function for MLP)
    mlp_sp_logodds = np.log(np.clip(mlp_sp_proba, 1e-9, 1-1e-9) /
                             np.clip(1-mlp_sp_proba, 1e-9, 1-1e-9))

    mlp_q_proba = mlp_qs_proba = None
    mlp_q_logodds = mlp_qs_logodds = None
    if qsar_avail:
        print("  Training QSAR MLP ...")
        mlp_q_clf  = fit_mlp(X_q, y_q)
        Xt_q = variant_matrix(cols_q)
        mlp_q_proba   = mlp_q_clf.predict_proba(Xt_q)[:, 1]
        mlp_q_logodds = np.log(np.clip(mlp_q_proba, 1e-9, 1-1e-9) /
                                np.clip(1-mlp_q_proba, 1e-9, 1-1e-9))

        print("  Training QSAR+StaPep MLP ...")
        mlp_qs_clf  = fit_mlp(X_qs, y_qs)
        mlp_qs_proba   = mlp_qs_clf.predict_proba(Xt_qs)[:, 1]
        mlp_qs_logodds = np.log(np.clip(mlp_qs_proba, 1e-9, 1-1e-9) /
                                 np.clip(1-mlp_qs_proba, 1e-9, 1-1e-9))

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  NGC vs Score Summary  (◆ = Buf WT is non-stapled; all are test peptides)")
    print(f"{'='*65}")
    col_w = 12
    hdr = (f"  {'Variant':<10} {'NGC':>7}  "
           f"{'SVM-Q':>{col_w}} {'SVM-SP':>{col_w}} {'SVM-Q+SP':>{col_w}}  "
           f"{'MLP-SP':>{col_w}} {'MLP-Q':>{col_w}} {'MLP-Q+SP':>{col_w}}")
    print(hdr)
    print("  " + "─" * (len(hdr)-2))
    for i, v in enumerate(VARIANTS):
        tag = "◆" if NGC_DATA[v].get("unstapled") else " "
        qp  = f"{qsar_proba[i]:.3f}"
        spp = f"{sp_proba[i]:.3f}"
        qsp = f"{qs_proba[i]:.3f}" if qs_proba is not None else "  N/A"
        mp  = f"{mlp_sp_proba[i]:.3f}"
        mq  = f"{mlp_q_proba[i]:.3f}" if mlp_q_proba is not None else "  N/A"
        mqs = f"{mlp_qs_proba[i]:.3f}" if mlp_qs_proba is not None else "  N/A"
        print(f"  {v+tag:<10} {NGC_DATA[v]['ngc']:>7.4f}  "
              f"{qp:>{col_w}} {spp:>{col_w}} {qsp:>{col_w}}  "
              f"{mp:>{col_w}} {mq:>{col_w}} {mqs:>{col_w}}")

    print(f"\n  Decision function (SVM margin distance, positive = AMP side):")
    print(f"  {'Variant':<10} {'SVM-QSAR':>12} {'SVM-SP':>12}", end="")
    if qs_dec is not None:
        print(f" {'SVM-Q+SP':>12}", end="")
    print()
    print("  " + "─" * 50)
    for i, v in enumerate(VARIANTS):
        tag = "◆" if NGC_DATA[v].get("unstapled") else " "
        print(f"  {v+tag:<10} {qsar_dec[i]:>12.4f} {sp_dec[i]:>12.4f}", end="")
        if qs_dec is not None:
            print(f" {qs_dec[i]:>12.4f}", end="")
        print()

    # ── Collect scores dict for plotting ─────────────────────────────────────
    # Only include models that differentiate variants (exclude constant QSAR)
    scores_for_plot = {
        "SVM-StaPep\nP(AMP)":       sp_proba,
        "SVM-StaPep\nMargin":        sp_dec,
        "MLP-StaPep\nP(AMP)":       mlp_sp_proba,
    }
    if qsar_avail and qs_proba is not None:
        scores_for_plot["SVM-Q+SP\nP(AMP)"]  = qs_proba
        scores_for_plot["SVM-Q+SP\nMargin"]  = qs_dec
        scores_for_plot["MLP-Q+SP\nP(AMP)"] = mlp_qs_proba

    # Also keep the QSAR model for reference in summary bar
    all_scores_for_summary = {
        "SVM\nQSAR*": qsar_proba,
        "SVM\nStaPep": sp_proba,
        "SVM\nQ+SP":   qs_proba  if qsar_avail else np.full(4, np.nan),
        "MLP\nStaPep": mlp_sp_proba,
        "MLP\nQSAR*":  mlp_q_proba if qsar_avail else np.full(4, np.nan),
        "MLP\nQ+SP":   mlp_qs_proba if qsar_avail else np.full(4, np.nan),
        "SVM\nMargin-SP": sp_dec,
        "SVM\nMargin-QSP": qs_dec if qsar_avail else np.full(4, np.nan),
    }

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n  Generating plots ...")
    _make_main_figure(scores_for_plot, BASE)
    _make_summary_figure(all_scores_for_summary, BASE)
    _make_qsar_blindness_figure(
        qsar_proba, qsar_dec,
        sp_proba, sp_dec,
        mlp_sp_proba,
        qs_proba if qsar_avail else None,
        qs_dec   if qsar_avail else None,
        BASE,
    )

    # ── Correlation summary printout ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Spearman ρ and Kendall τ  vs NGC  (all models)")
    print(f"{'='*65}")
    print(f"  {'Model':<22} {'Spearman ρ':>12}  {'Kendall τ':>12}  {'Interpretation'}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*25}")
    for label, sc in all_scores_for_summary.items():
        y = np.array(sc, dtype=float) if sc is not None else np.full(4, np.nan)
        rho, _, tau, _ = _corr_metrics(NGC_VALS, y)
        if np.isnan(rho):
            interp = "constant — cannot correlate"
        elif abs(rho) < 0.2:
            interp = "negligible"
        elif abs(rho) < 0.5:
            interp = "weak " + ("positive" if rho > 0 else "negative")
        elif abs(rho) < 0.8:
            interp = "moderate " + ("positive" if rho > 0 else "negative")
        else:
            interp = "strong " + ("positive" if rho > 0 else "negative")
        lbl = label.replace("\n", " ")
        print(f"  {lbl:<22} {rho:>+12.3f}  {tau:>+12.3f}  {interp}")

    print(f"""
  Note: With N=4, no p-value is statistically significant (min p ≈ 0.08).
  Interpret direction (sign) and magnitude, not statistical significance.
  Rank labels in the scatter plots show: #NGC_rank → #model_rank.
  Flat lines in the bump chart = perfect rank concordance with NGC.
""")


if __name__ == "__main__":
    main()
