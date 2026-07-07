#!/usr/bin/env python3
"""
plot_mic_distribution.py
========================
Visualise the MIC training data used by predict_mic_single.py.

Panels
------
  1. Log-scale MIC histogram + KDE with tier boundary lines
  2. Tier count bar chart (class balance)
  3. RF feature importance — 4-tier E. coli MIC classification (our task)
  4. RF feature importance — binary AMP vs decoy (mirrors StaPep paper Fig 4B)
  5. RF feature importance — PNAS permeability (Permeable vs Impermeable)

Usage
-----
  conda run -n esm_env python plot_mic_distribution.py
  conda run -n esm_env python plot_mic_distribution.py --save mic_distribution.png
"""

from __future__ import annotations

import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute   import SimpleImputer
from sklearn.pipeline import Pipeline as SKPipeline

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parent.parent
STAPEP    = BASE / "data" / "training_dataset" / "StaPep"
FEAT_CSV  = STAPEP / "stapled_amps_features.csv"
AMP_CSV   = STAPEP / "stapled_amps.csv"
DECOY_CSV = STAPEP / "stapled_decoys.csv"
PNAS_CSV  = STAPEP / "PNAS_paper_datasets" / "Stapled-peptide_permeability_filtered.csv"

FEATURES = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index",
    "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]

# The 14 features shown in the StaPep paper Fig 4B
# (excludes sheet_percent, lyticity_index, sasa)
PAPER_FEATURES = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa",
]

# Human-readable labels matching the StaPep paper's axis labels
FEATURE_LABELS = {
    "length"           : "Peptide Length",
    "weight"           : "Molecular Weight",
    "hydrophobic_index": "Hydrophobicity Index",
    "charge"           : "Net Charge",
    "aromaticity"      : "Aromaticity",
    "isoelectric_point": "Isoelectric Point",
    "fraction_arginine": "Fraction of Arginine",
    "fraction_lysine"  : "Fraction of Lysine",
    "lyticity_index"   : "Lyticity Index",
    "helix_percent"    : "Helical Content",
    "sheet_percent"    : "Sheet Content",
    "loop_percent"     : "Loop Content",
    "mean_bfactor"     : "Mean B-factor",
    "mean_gyrate"      : "Mean Radius of Gyration",
    "num_hbonds"       : "Number of Hydrogen Bonds",
    "psa"              : "Polar Surface Area",
    "sasa"             : "Solvent Accessible Surface Area",
}

BINS        = [0, 2, 5, 10, np.inf]
TIER_LABELS = ["Very strong\n(<2 μM)", "Strong\n(2–5 μM)",
               "Moderate\n(5–10 μM)", "Weak\n(>10 μM)"]
SHORT       = ["VeryStrong", "Strong", "Moderate", "Weak"]
TIER_COLORS = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]


# ── Data loaders ───────────────────────────────────────────────────────────────
def parse_ecoli_mic(amp_csv: Path) -> pd.DataFrame:
    amp = pd.read_csv(amp_csv)
    pat = re.compile(
        r'(?:Escherichia\s+coli|E\.?\s*coli)[^(]*'
        r'\(MIC[^=]*=\s*([\d.]+)\s*([\u03bcμ\xb5]M|[\u03bcμ\xb5]g/mL)', re.I)
    rows = []
    for _, row in amp.iterrows():
        m = pat.search(str(row.get("Target_Organism", "")))
        if m:
            rows.append({"DRAMP_ID": row["DRAMP_ID"],
                         "mic_raw" : float(m.group(1)),
                         "unit"    : m.group(2)})
    return pd.DataFrame(rows)


def build_mic_df() -> pd.DataFrame:
    """MIC-tier dataset (147 peptides with E. coli MIC)."""
    mic  = parse_ecoli_mic(AMP_CSV)
    feat = pd.read_csv(FEAT_CSV)
    mc   = list(dict.fromkeys(["DRAMP_ID"] + FEATURES))
    df   = mic.merge(feat[mc], on="DRAMP_ID", how="inner").reset_index(drop=True)

    df["mic_uM"] = df["mic_raw"].astype(float)
    mask = df["unit"].str.lower().str.contains("g/ml").values
    df.loc[mask, "mic_uM"] = (df["mic_raw"].values[mask] * 1000
                               / df["weight"].values[mask])

    df = df.dropna(subset=["mic_uM"] + FEATURES).reset_index(drop=True)
    df["tier"]     = pd.cut(df["mic_uM"], bins=BINS, labels=[0, 1, 2, 3]).astype(int)
    df["log2_mic"] = np.log2(np.clip(df["mic_uM"].values, 1e-3, None))
    return df


def build_binary_df() -> pd.DataFrame:
    """Binary AMP-vs-decoy dataset (188 AMPs + 355 decoys)."""
    amp   = pd.read_csv(FEAT_CSV)[FEATURES + ["label"]].copy()
    amp["label"] = 1

    dec   = pd.read_csv(DECOY_CSV)[FEATURES].copy()
    dec["label"] = 0

    df = pd.concat([amp, dec], ignore_index=True)
    df[FEATURES] = df[FEATURES].astype(float)
    return df


def build_pnas_df() -> pd.DataFrame:
    """PNAS permeability dataset — Permeable(1) vs Impermeable(0)."""
    df = pd.read_csv(PNAS_CSV)
    avail = [f for f in PAPER_FEATURES if f in df.columns]
    df = df[avail + ["Permeability"]].copy()
    df["label"] = (df["Permeability"].str.strip().str.lower() == "permeable").astype(int)
    df[avail] = df[avail].astype(float)
    return df, avail


# ── Reference peptides ─────────────────────────────────────────────────────────
REFS = {
    "BufII\n(non-stapled)": {"mic_uM": 2.0,  "color": "#8e44ad"},
    "MagII\n(non-stapled)": {"mic_uM": 3.0,  "color": "#16a085"},
}


# ── RF helper ──────────────────────────────────────────────────────────────────
def _train_rf(X: pd.DataFrame, y: pd.Series,
              balanced: bool = True) -> np.ndarray:
    cw = "balanced" if balanced else None
    pipe = SKPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf",      RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=1, class_weight=cw,
            random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X, y)
    return pipe.named_steps["rf"].feature_importances_


def _importance_bars(ax, importances: np.ndarray,
                     feature_list: list,
                     use_paper_labels: bool = False,
                     title: str = ""):
    """Draw a horizontal importance bar chart on ax."""
    imp = pd.Series(importances, index=feature_list)
    order      = imp.sort_values(ascending=True).index
    imp_sorted = imp[order]

    q75 = imp_sorted.quantile(0.75)
    q25 = imp_sorted.quantile(0.25)
    colors = []
    for v in imp_sorted:
        if v >= q75:
            colors.append(TIER_COLORS[0])    # green — top 25%
        elif v <= q25:
            colors.append(TIER_COLORS[3])    # red   — bottom 25%
        else:
            colors.append("#3498db")          # blue  — middle 50%

    ax.barh(range(len(feature_list)), imp_sorted.values,
            color=colors, edgecolor="#1a1a2e", linewidth=0.5, height=0.7)

    for i, val in enumerate(imp_sorted.values):
        ax.text(val + imp_sorted.max() * 0.01, i, f"{val:.4f}",
                va="center", ha="left", color="white", fontsize=7.5)

    ytick_labels = ([FEATURE_LABELS[f] for f in order]
                    if use_paper_labels else list(order))
    ax.set_yticks(range(len(feature_list)))
    ax.set_yticklabels(ytick_labels, fontsize=8.5, color="white")
    ax.set_xlabel("RF Mean Decrease in Impurity  (importance score)",
                  color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=10, pad=8)

    legend_els = [
        Line2D([0], [0], color=TIER_COLORS[0], lw=0, marker="s",
               markersize=9, label="Top 25% most important"),
        Line2D([0], [0], color="#3498db",       lw=0, marker="s",
               markersize=9, label="Middle 50%"),
        Line2D([0], [0], color=TIER_COLORS[3], lw=0, marker="s",
               markersize=9, label="Bottom 25% least important"),
    ]
    ax.legend(handles=legend_els, fontsize=7.5, framealpha=0.3,
              labelcolor="white", facecolor="#2c2c4e", edgecolor="none",
              loc="lower right")


# ── Main figure ────────────────────────────────────────────────────────────────
def make_figure(mic_df: pd.DataFrame, binary_df: pd.DataFrame,
                pnas_df: pd.DataFrame, pnas_feats: list,
                save_path: str | None = None):

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("#1a1a2e")

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.38,
                          left=0.07, right=0.97, top=0.95, bottom=0.03)

    ax1 = fig.add_subplot(gs[0, 0])   # Panel 1: MIC histogram
    ax2 = fig.add_subplot(gs[0, 1])   # Panel 2: tier bar
    ax3 = fig.add_subplot(gs[1, :])   # Panel 3: RF importance — MIC tiers
    ax4 = fig.add_subplot(gs[2, :])   # Panel 4: RF importance — AMP vs decoy
    ax5 = fig.add_subplot(gs[3, :])   # Panel 5: RF importance — PNAS permeability

    for ax in [ax1, ax2, ax3, ax4, ax5]:
        _style_ax(ax)

    # ── Panel 1: log-scale MIC histogram + KDE ────────────────────────────────
    mic_vals = mic_df["mic_uM"].values
    log_vals = np.log2(np.clip(mic_vals, 1e-3, None))

    for i, (lo, hi, col) in enumerate(zip(BINS[:-1], BINS[1:], TIER_COLORS)):
        mask = (mic_vals > lo) & (mic_vals <= (hi if not np.isinf(hi) else 1e6))
        if mask.any():
            ax1.hist(np.log2(np.clip(mic_vals[mask], 1e-3, None)),
                     bins=20, color=col, alpha=0.75, edgecolor="#1a1a2e",
                     linewidth=0.5, label=SHORT[i])

    kde_x = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 400)
    kde   = gaussian_kde(log_vals, bw_method=0.4)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(kde_x, kde(kde_x), color="white", lw=2, alpha=0.85, zorder=5)
    ax1_twin.set_yticks([])
    ax1_twin.set_facecolor("#1a1a2e")

    for boundary in [2, 5, 10]:
        ax1.axvline(np.log2(boundary), color="white", lw=1.2,
                    linestyle="--", alpha=0.6, zorder=4)
        ax1.text(np.log2(boundary) + 0.05,
                 ax1.get_ylim()[1] * 0.97 if ax1.get_ylim()[1] > 0 else 1,
                 f"{boundary} μM", color="white", fontsize=7.5, va="top", alpha=0.8)

    for pep_name, info in REFS.items():
        xpos = np.log2(info["mic_uM"])
        ax1.axvline(xpos, color=info["color"], lw=1.8, linestyle=":", zorder=6)
        ax1.text(xpos + 0.06, 0.92, pep_name, color=info["color"],
                 fontsize=7.5, va="top", transform=ax1.get_xaxis_transform())

    tick_uM  = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100]
    tick_log = [np.log2(v) for v in tick_uM]
    ax1.set_xticks(tick_log)
    ax1.set_xticklabels([f"{v}" for v in tick_uM], fontsize=8, color="white")
    ax1.set_xlabel("MIC  (μM,  log₂ scale)", color="white", fontsize=10)
    ax1.set_ylabel("Count", color="white", fontsize=10)
    ax1.set_title("MIC Distribution  (E. coli, stapled AMP training set)",
                  color="white", fontsize=11, pad=8)
    ax1.legend(fontsize=8, framealpha=0.3,
               labelcolor="white", facecolor="#2c2c4e", edgecolor="none")

    # ── Panel 2: tier class balance bar ───────────────────────────────────────
    counts = mic_df["tier"].value_counts().sort_index()
    bars = ax2.bar(range(len(SHORT)),
                   [counts.get(i, 0) for i in range(len(SHORT))],
                   color=TIER_COLORS, edgecolor="#1a1a2e", linewidth=0.8, width=0.6)
    for bar, cnt in zip(bars, [counts.get(i, 0) for i in range(len(SHORT))]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3, str(cnt),
                 ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

    ax2.set_xticks(range(len(SHORT)))
    ax2.set_xticklabels(TIER_LABELS, fontsize=8.5, color="white")
    ax2.set_ylabel("Number of peptides", color="white", fontsize=10)
    ax2.set_title("Class Balance  (4-tier MIC split)", color="white",
                  fontsize=11, pad=8)
    ax2.set_ylim(0, max(counts.values) * 1.18)
    ax2.text(0.98, 0.97, f"n = {len(mic_df)}", transform=ax2.transAxes,
             ha="right", va="top", color="#aaaaaa", fontsize=9)

    # ── Panel 3: RF importance — 4-tier MIC (paper features only) ────────────
    print("  Panel 3: Training RF on 4-tier MIC (paper features) …", end=" ", flush=True)
    imp_mic = _train_rf(mic_df[PAPER_FEATURES].astype(float), mic_df["tier"].astype(int))
    print("done")
    _importance_bars(ax3, imp_mic, feature_list=PAPER_FEATURES, use_paper_labels=True,
                     title=f"Panel 3 — RF Feature Importance  "
                           f"(4-tier E. coli MIC,  n={len(mic_df)},  paper features only)")

    # ── Panel 4: RF importance — binary AMP vs decoy (replicates paper) ───────
    print("  Panel 4: Training RF on AMP vs decoy (paper features) …", end=" ", flush=True)
    imp_bin = _train_rf(binary_df[PAPER_FEATURES].astype(float),
                        binary_df["label"].astype(int),
                        balanced=False)
    print("done")

    n_amp   = int(binary_df["label"].sum())
    n_decoy = len(binary_df) - n_amp
    _importance_bars(ax4, imp_bin, feature_list=PAPER_FEATURES, use_paper_labels=True,
                     title=f"Panel 4 — RF Feature Importance  "
                           f"(AMP vs Decoy, default weights,  n={n_amp} AMPs + {n_decoy} decoys  "
                           f"— replicates StaPep paper Fig 4B)")

    # ── Panel 5: RF importance — PNAS permeability ────────────────────────────
    print("  Panel 5: Training RF on PNAS permeability …", end=" ", flush=True)
    n_perm   = int(pnas_df["label"].sum())
    n_imperm = len(pnas_df) - n_perm
    imp_pnas = _train_rf(pnas_df[pnas_feats].astype(float),
                         pnas_df["label"].astype(int),
                         balanced=False)
    print("done")
    _importance_bars(ax5, imp_pnas, feature_list=pnas_feats, use_paper_labels=True,
                     title=f"Panel 5 — RF Feature Importance  "
                           f"(PNAS: Permeable vs Impermeable,  "
                           f"n={n_perm} permeable + {n_imperm} impermeable  "
                           f"— matches our decoy feature space)")

    # ── Overall title ──────────────────────────────────────────────────────────
    fig.suptitle("StaPep Training Set  |  MIC Distribution & Feature Analysis",
                 color="white", fontsize=14, fontweight="bold", y=0.97)

    out = save_path or "mic_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  ✅  Figure saved → {out}")
    return out


# ── Axis styling ───────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot MIC distribution of training data.")
    parser.add_argument("--save", default=None, metavar="FILE",
                        help="Output image path (default: mic_distribution.png)")
    args = parser.parse_args()

    print("\n  Loading MIC training data …")
    mic_df = build_mic_df()
    print(f"  Peptides with E. coli MIC + all features : {len(mic_df)}")
    print(f"  MIC range : {mic_df['mic_uM'].min():.2f} – {mic_df['mic_uM'].max():.1f} μM")
    for i, lbl in enumerate(SHORT):
        print(f"    {lbl:<12}: {int((mic_df['tier']==i).sum())}")

    print("\n  Loading AMP vs decoy data …")
    binary_df = build_binary_df()
    n_amp   = int(binary_df["label"].sum())
    n_decoy = len(binary_df) - n_amp
    print(f"  AMPs: {n_amp}  |  Decoys: {n_decoy}")

    print("\n  Loading PNAS permeability data …")
    pnas_df, pnas_feats = build_pnas_df()
    n_perm   = int(pnas_df["label"].sum())
    n_imperm = len(pnas_df) - n_perm
    print(f"  Permeable: {n_perm}  |  Impermeable: {n_imperm}")
    print(f"  Features available in PNAS: {pnas_feats}")

    out = make_figure(mic_df, binary_df, pnas_df, pnas_feats, save_path=args.save)
    print(f"\n  Open {out} to view the plot.\n")


if __name__ == "__main__":
    main()
