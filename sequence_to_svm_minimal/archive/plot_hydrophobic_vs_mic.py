#!/usr/bin/env python3
"""
plot_hydrophobic_vs_mic.py
==========================
Plot the correlation between hydrophobic surface area and MIC
for the stapled Buforin (F10W) variants.

Panel A — Hydrophobic SASA  (Total SASA − PSA)  vs  MIC (μg/mL)
Panel B — Hydrophobic Index (Kyte-Doolittle)    vs  MIC (μg/mL)

Data source: test_buf_specific_stapep_features.csv
  (7 F10W Buforin variants with StaPep MD features + literature MIC)
"""

import warnings, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

warnings.filterwarnings("ignore")

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"
FEAT_F10W = STAPEP / "test_buf_specific_stapep_features.csv"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(FEAT_F10W)

# Calculate hydrophobic SASA = total SASA − PSA
df["hydrophobic_sasa"] = df["sasa"] - df["psa"]

# Drop the variant with no MIC value (Buf_i7_1_F10W has MIC >100 → NaN)
df_valid = df.dropna(subset=["lit_mic_ugml"]).copy()

# Short labels for display
LABEL_MAP = {
    "Buf_i4_16_F10W": "Buf(i+4)16",
    "Buf_i4_14_F10W": "Buf(i+4)14",
    "Buf_i4_4_F10W":  "Buf(i+4)4",
    "Buf_i4_3_F10W":  "Buf(i+4)3",
    "Buf_i7_9_F10W":  "Buf(i+7)9",
    "Buf_i7_6_F10W":  "Buf(i+7)6",
    "Buf_i7_1_F10W":  "Buf(i+7)1",
}
df_valid["short_label"] = df_valid["peptide_id"].map(LABEL_MAP)

# Color by staple type
COLOR_MAP = {
    "Buf_i4_16_F10W": "#e6194b",
    "Buf_i4_14_F10W": "#3cb44b",
    "Buf_i4_4_F10W":  "#4363d8",
    "Buf_i4_3_F10W":  "#f58231",
    "Buf_i7_9_F10W":  "#911eb4",
    "Buf_i7_6_F10W":  "#42d4f4",
    "Buf_i7_1_F10W":  "#f032e6",
}
# Marker by staple type (i+4 vs i+7)
MARKER_MAP = {
    "Buf_i4_16_F10W": "o",
    "Buf_i4_14_F10W": "o",
    "Buf_i4_4_F10W":  "o",
    "Buf_i4_3_F10W":  "o",
    "Buf_i7_9_F10W":  "s",
    "Buf_i7_6_F10W":  "s",
    "Buf_i7_1_F10W":  "s",
}

print("=" * 70)
print("  Hydrophobic Surface Area vs MIC — Stapled Buforin (F10W) Variants")
print("=" * 70)

# Print table
print(f"\n  {'Peptide':<18}  {'Hydro SASA':>10}  {'Hydro Idx':>9}  {'MIC μg/mL':>9}  {'log₁₀(MIC)':>10}")
print("  " + "─" * 62)
for _, row in df.iterrows():
    mic_str = f"{row['lit_mic_ugml']:.1f}" if pd.notna(row["lit_mic_ugml"]) else ">100"
    log_mic = f"{np.log10(row['lit_mic_ugml']):.2f}" if pd.notna(row["lit_mic_ugml"]) else "—"
    print(f"  {LABEL_MAP[row['peptide_id']]:<18}  "
          f"{row['hydrophobic_sasa']:>10.1f}  "
          f"{row['hydrophobic_index']:>9.3f}  "
          f"{mic_str:>9}  "
          f"{log_mic:>10}")

# ── Prepare arrays ──────────────────────────────────────────────────────────────
hydro_sasa = df_valid["hydrophobic_sasa"].values
hydro_idx  = df_valid["hydrophobic_index"].values
mic_ugml   = df_valid["lit_mic_ugml"].values
log_mic    = np.log10(mic_ugml)
labels     = df_valid["short_label"].values
pids       = df_valid["peptide_id"].values

# ── Figure ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Hydrophobic Properties vs Antimicrobial Activity\n"
    "Stapled Buforin (F10W) Variants — E. coli MIC",
    fontsize=12, fontweight="bold",
)

# ════════════════════════════════════════════════════════════════════════════════
# Panel A: Hydrophobic SASA vs MIC
# ════════════════════════════════════════════════════════════════════════════════
for i, pid in enumerate(pids):
    ax1.scatter(hydro_sasa[i], log_mic[i],
                s=120,
                marker=MARKER_MAP[pid],
                color=COLOR_MAP[pid],
                edgecolors="k", linewidths=0.8, zorder=5)
    # offset label to avoid overlap
    x_off, y_off = 8, 5
    if labels[i] == "Buf(i+7)9":
        y_off = -14
    elif labels[i] == "Buf(i+4)16":
        y_off = -14
    ax1.annotate(labels[i], (hydro_sasa[i], log_mic[i]),
                 textcoords="offset points", xytext=(x_off, y_off),
                 fontsize=8, color=COLOR_MAP[pid], fontweight="bold")

# Regression line
coeffs1 = np.polyfit(hydro_sasa, log_mic, 1)
x_fit1  = np.linspace(hydro_sasa.min() - 20, hydro_sasa.max() + 20, 100)
ax1.plot(x_fit1, np.polyval(coeffs1, x_fit1),
         color="gray", lw=1.5, ls="--", zorder=2, alpha=0.7)

# Correlation stats
r_pearson1, p_pearson1 = pearsonr(hydro_sasa, log_mic)
r_spear1, p_spear1     = spearmanr(hydro_sasa, log_mic)

txt1 = (f"Pearson  r = {r_pearson1:+.3f}  (p={p_pearson1:.3f})\n"
        f"Spearman ρ = {r_spear1:+.3f}  (p={p_spear1:.3f})\n"
        f"N = {len(hydro_sasa)} variants")
ax1.text(0.03, 0.97, txt1, transform=ax1.transAxes,
         fontsize=8, va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa", alpha=0.92))

# Arrow annotation for the key insight
ax1.annotate(
    "Larger hydrophobic surface\n→ lower MIC (stronger activity)",
    xy=(max(hydro_sasa) - 20, min(log_mic) + 0.15),
    fontsize=8, color="#333", fontstyle="italic",
    bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#4caf50", alpha=0.9),
)

ax1.set_xlabel("Hydrophobic SASA  (Total SASA − PSA)  [Å²]", fontsize=10)
ax1.set_ylabel("log₁₀(MIC)  [μg/mL]", fontsize=10)
ax1.set_title("Panel A — Hydrophobic Surface Area vs MIC", fontsize=10, fontweight="bold")
ax1.grid(alpha=0.25, lw=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Add secondary y-axis showing actual MIC values
ax1_r = ax1.twinx()
y_ticks_log = ax1.get_yticks()
ax1_r.set_ylim(ax1.get_ylim())
mic_ticks = [1, 2, 5, 10, 20, 50, 100]
ax1_r.set_yticks([np.log10(m) for m in mic_ticks])
ax1_r.set_yticklabels([f"{m}" for m in mic_ticks], fontsize=8)
ax1_r.set_ylabel("MIC  [μg/mL]", fontsize=9)
ax1_r.spines["top"].set_visible(False)

# ════════════════════════════════════════════════════════════════════════════════
# Panel B: Hydrophobic Index vs MIC
# ════════════════════════════════════════════════════════════════════════════════
for i, pid in enumerate(pids):
    ax2.scatter(hydro_idx[i], log_mic[i],
                s=120,
                marker=MARKER_MAP[pid],
                color=COLOR_MAP[pid],
                edgecolors="k", linewidths=0.8, zorder=5)
    x_off, y_off = 8, 5
    if labels[i] == "Buf(i+7)9":
        y_off = -14
    elif labels[i] == "Buf(i+4)16":
        y_off = -14
    ax2.annotate(labels[i], (hydro_idx[i], log_mic[i]),
                 textcoords="offset points", xytext=(x_off, y_off),
                 fontsize=8, color=COLOR_MAP[pid], fontweight="bold")

# Regression line
coeffs2 = np.polyfit(hydro_idx, log_mic, 1)
x_fit2  = np.linspace(hydro_idx.min() - 0.05, hydro_idx.max() + 0.05, 100)
ax2.plot(x_fit2, np.polyval(coeffs2, x_fit2),
         color="gray", lw=1.5, ls="--", zorder=2, alpha=0.7)

# Correlation stats
r_pearson2, p_pearson2 = pearsonr(hydro_idx, log_mic)
r_spear2, p_spear2     = spearmanr(hydro_idx, log_mic)

txt2 = (f"Pearson  r = {r_pearson2:+.3f}  (p={p_pearson2:.3f})\n"
        f"Spearman ρ = {r_spear2:+.3f}  (p={p_spear2:.3f})\n"
        f"N = {len(hydro_idx)} variants")
ax2.text(0.03, 0.97, txt2, transform=ax2.transAxes,
         fontsize=8, va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#aaaaaa", alpha=0.92))

ax2.set_xlabel("Hydrophobic Index  (Kyte-Doolittle mean)", fontsize=10)
ax2.set_ylabel("log₁₀(MIC)  [μg/mL]", fontsize=10)
ax2.set_title("Panel B — Hydrophobic Index vs MIC", fontsize=10, fontweight="bold")
ax2.grid(alpha=0.25, lw=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Add secondary y-axis
ax2_r = ax2.twinx()
ax2_r.set_ylim(ax2.get_ylim())
ax2_r.set_yticks([np.log10(m) for m in mic_ticks])
ax2_r.set_yticklabels([f"{m}" for m in mic_ticks], fontsize=8)
ax2_r.set_ylabel("MIC  [μg/mL]", fontsize=9)
ax2_r.spines["top"].set_visible(False)

# ── Legend ──────────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#777",
           markeredgecolor="k", markersize=8, label="i+4 staple"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#777",
           markeredgecolor="k", markersize=8, label="i+7 staple"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.02), framealpha=0.9,
           title="Staple Type", title_fontsize=9)

# ── Footer ──────────────────────────────────────────────────────────────────────
fig.text(0.01, -0.06,
         "Hydrophobic SASA = Total SASA − PSA (from StaPep MD simulation, 10 ns).\n"
         "Hydrophobic Index = mean Kyte-Doolittle hydrophobicity of exposed residues (sequence-level).\n"
         "MIC values from literature (E. coli). Variant Buf(i+7)1 excluded (MIC >100 μg/mL).",
         fontsize=7.5, color="#555555", va="top")

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
out = BASE / "hydrophobic_vs_mic.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"\n  Figure saved → {out}")

# ── Print interpretation ──────────────────────────────────────────────────────
print(f"\n  Interpretation:")
print(f"  ───────────────")
if r_pearson1 < 0:
    print(f"  Panel A: Negative correlation (r={r_pearson1:+.3f}) → larger hydrophobic surface → lower MIC (stronger)")
else:
    print(f"  Panel A: Positive correlation (r={r_pearson1:+.3f}) → larger hydrophobic surface → higher MIC (weaker)")
if r_pearson2 < 0:
    print(f"  Panel B: Negative correlation (r={r_pearson2:+.3f}) → higher hydrophobic index → lower MIC (stronger)")
else:
    print(f"  Panel B: Positive correlation (r={r_pearson2:+.3f}) → higher hydrophobic index → higher MIC (weaker)")

print(f"\n  Advisor's observation: 'Larger hydrophobic surface → stronger interaction")
print(f"  with membrane → lower MIC' is {'SUPPORTED' if r_pearson1 < 0 else 'NOT SUPPORTED'} by Panel A (r={r_pearson1:+.3f}).")
print()
