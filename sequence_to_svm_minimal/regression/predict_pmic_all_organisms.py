#!/usr/bin/env python3
"""
predict_pmic_all_organisms.py

For EVERY target organism found in stapled_amps.csv:
  1. Extracts peptides with a parseable numeric MIC for that organism
  2. Trains a Random Forest regression model to predict pMIC
  3. Plots actual vs. predicted pMIC (Pearson R, n samples, linear fit)
  4. Predicts pMIC for Buforin II and Magainin II (native + stapled variants)
  5. Saves one PNG per organism to  pmic_by_organism/

Usage:
    python predict_pmic_all_organisms.py
    python predict_pmic_all_organisms.py --min-n 20 --out-dir pmic_by_organism
"""

import re, math, sys, argparse, warnings, os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# Make utils/ and features/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import STAPEP_DIR
from utils.mic_units import pmic_to_mic_uM as pmic_to_mic, mic_to_pmic_uM as mic_to_pmic
from features.stapep_columns import STAPEP_COLS_PAPER_14 as FEATURE_COLS

AMPS_META = STAPEP_DIR / "stapled_amps.csv"
AMPS_FEAT = STAPEP_DIR / "stapled_amps_features.csv"
TEST_FEAT = STAPEP_DIR / "test_stapled_features.csv"

# ─── Organisms to scan (name → regex pattern) ────────────────────────────────
# Each pattern anchors to the organism name immediately before "(MIC..."
ORGANISMS = {
    "E_coli":         ("E. coli",              r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+\w+)*"),
    "P_aeruginosa":   ("P. aeruginosa",         r"Pseudomonas\s+aeruginosa"),
    "S_aureus":       ("S. aureus (all)",        r"Staphylococcus\s+aureus"),
    "MRSA":           ("MRSA",                  r"methicillin.resistant\s+Staphylococcus\s+aureus|MRSA"),
    "L_monocytogenes":("L. monocytogenes",       r"Listeria\s+monocytogenes"),
    "B_subtilis":     ("B. subtilis",            r"Bacillus\s+subtilis"),
    "M_luteus":       ("M. luteus",              r"Micrococcus\s+luteus"),
    "Klebsiella":     ("Klebsiella spp.",        r"Klebsiella\s+\w+"),
    "Salmonella":     ("Salmonella spp.",        r"Salmonella\s+\w+"),
    "C_albicans":     ("C. albicans (fungus)",   r"Candida\s+albicans"),
}

# ─── Native peptide features (from 5 ns StaPep MD runs) ─────────────────────
NATIVE = {
    "Buforin II\n(native)": {
        "length": 21, "weight": 2473.829, "hydrophobic_index": -0.8142857142857142,
        "charge": 6.094, "aromaticity": 0.047619, "isoelectric_point": 11.9999,
        "fraction_arginine": 0.23810, "fraction_lysine": 0.04762,
        "helix_percent": 0.17819, "loop_percent": 0.82105,
        "mean_bfactor": 573.434, "mean_gyrate": 12.001,
        "num_hbonds": 0, "psa": 1064.217,
    },
    "Magainin II\n(native)": {
        "length": 23, "weight": 2466.832, "hydrophobic_index": 0.08261,
        "charge": 3.095, "aromaticity": 0.13043, "isoelectric_point": 10.0014,
        "fraction_arginine": 0.0, "fraction_lysine": 0.17391,
        "helix_percent": 0.16787, "loop_percent": 0.80809,
        "mean_bfactor": 895.020, "mean_gyrate": 12.213,
        "num_hbonds": 1, "psa": 918.001,
    },
}

TIER_COLOR = {
    "Very Strong": "#1f77b4",
    "Strong":      "#2ca02c",
    "Moderate":    "#ff7f0e",
    "Weak":        "#d62728",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def tier(mic_uM):
    if mic_uM < 2:  return "Very Strong"
    if mic_uM < 5:  return "Strong"
    if mic_uM < 10: return "Moderate"
    return "Weak"

def make_mic_regex(org_pattern):
    """
    Lenient regex:  ORG ... (MIC[suffix]= VALUE UNIT [,);]
    Skips values preceded by >, <, ≥, ≤  (those are bounds, not exact).
    Captures value and unit separately.
    Wraps org_pattern in (?:...) so any | inside stays local to that group.
    """
    return re.compile(
        r"(?:" + org_pattern + r")"
        + r"(?:\s+[\w\-]+)*\s*"               # optional strain / descriptor words
        + r"\(\s*MIC[\w.%]*\s*"               # (MIC / MIC99.9 / MIC99 / MIC50 …
        + r"([=≥>≤<]?)\s*"                    # optional operator  ← group 1
        + r"([\d.]+)\s*"                       # numeric value      ← group 2
        + r"([μu]g/mL|[μu]M)"                 # unit               ← group 3
        + r"\s*[,);]",
        re.IGNORECASE,
    )

def extract_mic_uM(text, mw, org_re):
    """Return MIC in μM for the first matching organism entry, or None."""
    if not isinstance(text, str):
        return None
    m = org_re.search(text)
    if m is None:
        return None
    op, val_str, unit = m.group(1).strip(), m.group(2), m.group(3).lower()
    if op in (">", "<", "≥", "≤"):       # skip bound-only entries
        return None
    try:
        val = float(val_str)
    except ValueError:
        return None
    if unit in ("μm", "um"):
        return val
    if unit in ("μg/ml", "ug/ml") and mw > 0:
        return val / mw * 1000.0          # μg/mL ÷ Da × 1000 → μM
    return None

# ─── Load base data ───────────────────────────────────────────────────────────
def load_base():
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    df   = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat[["DRAMP_ID"] + FEATURE_COLS],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=FEATURE_COLS)
    return df

def load_test():
    test = pd.read_csv(TEST_FEAT).dropna(subset=FEATURE_COLS)
    return test

# ─── Train RF + 5-fold CV ─────────────────────────────────────────────────────
def train_rf(X, y):
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv  = cross_val_predict(rf, X, y, cv=cv)
    r_cv, _ = stats.pearsonr(y, y_cv)
    rf.fit(X, y)                          # refit on all data
    return rf, y_cv, r_cv

# ─── Confidence-interval band for scatter ────────────────────────────────────
def _ci_band(x, slope, intercept, x_grid, n, alpha=0.05):
    y_fit  = slope * x + intercept
    y_grid = slope * x_grid + intercept
    s2     = np.sum((y_fit - (slope * x + intercept)) ** 2) / max(n - 2, 1)
    x_mean = np.mean(x)
    se     = np.sqrt(s2 * (1/n + (x_grid - x_mean)**2 / np.sum((x - x_mean)**2 + 1e-12)))
    t_val  = t_dist.ppf(1 - alpha / 2, df=max(n - 2, 1))
    return y_grid - t_val * se, y_grid + t_val * se

# ─── Build one figure ─────────────────────────────────────────────────────────
def make_figure(org_key, org_label, df_org, test, out_path):
    n    = len(df_org)
    X    = df_org[FEATURE_COLS].values.astype(float)
    y    = df_org["pMIC"].values

    rf, y_cv, r_cv = train_rf(X, y)

    # ── Predict query peptides ───────────────────────────────────────────────
    buf_ids = [r for r in test["peptide_id"] if r.lower().startswith("buf")]
    mag_ids = [r for r in test["peptide_id"] if r.lower().startswith("mag")]

    query_entries = []

    for group_ids, nat_key in [
        (buf_ids, "Buforin II\n(native)"),
        (mag_ids, "Magainin II\n(native)"),
    ]:
        feats = NATIVE[nat_key]
        mw    = feats["weight"]
        x_nat = np.array([[feats[f] for f in FEATURE_COLS]])
        pm    = float(rf.predict(x_nat)[0])
        mic   = pmic_to_mic(pm)
        query_entries.append({
            "id": nat_key.replace("\n", " "),
            "group": "Buforin" if "buf" in nat_key.lower() else "Magainin",
            "stapled": False, "pMIC": pm, "mic_uM": mic,
            "mic_ugml": mic * mw / 1000.0,
        })
        for pid in group_ids:
            row  = test[test["peptide_id"] == pid].iloc[0]
            x_st = np.array([[row[f] for f in FEATURE_COLS]])
            pm2  = float(rf.predict(x_st)[0])
            mic2 = pmic_to_mic(pm2)
            mw2  = float(row["weight"])
            query_entries.append({
                "id": pid, "group": "Buforin" if "buf" in pid.lower() else "Magainin",
                "stapled": True, "pMIC": pm2, "mic_uM": mic2,
                "mic_ugml": mic2 * mw2 / 1000.0,
            })

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.2],
                           left=0.07, right=0.97,
                           top=0.88, bottom=0.12, wspace=0.38)
    ax_scatter = fig.add_subplot(gs[0])
    ax_bars    = fig.add_subplot(gs[1])

    # ─── Panel 1: Actual vs Predicted pMIC ──────────────────────────────────
    ax_scatter.scatter(y, y_cv, alpha=0.6, s=40, color="#2c7bb6",
                       edgecolors="white", linewidths=0.4, zorder=3,
                       label=f"Peptides (n={n})")

    # Diagonal reference
    lo, hi = min(y.min(), y_cv.min()) - 0.1, max(y.max(), y_cv.max()) + 0.1
    ax_scatter.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    # Linear regression + CI
    slope, intercept, *_ = stats.linregress(y, y_cv)
    x_grid = np.linspace(lo, hi, 200)
    y_line = slope * x_grid + intercept
    ax_scatter.plot(x_grid, y_line, "-", color="#d7191c", lw=1.5, label="Linear fit")
    lo_ci, hi_ci = _ci_band(y, slope, intercept, x_grid, n)
    ax_scatter.fill_between(x_grid, lo_ci, hi_ci,
                            alpha=0.15, color="#d7191c", label="95% CI")

    # Pearson R annotation
    ax_scatter.text(0.05, 0.93,
                    f"Pearson R = {r_cv:.2f}\nn = {n}",
                    transform=ax_scatter.transAxes,
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="lightyellow", alpha=0.85))

    ax_scatter.set_xlabel("Actual pMIC", fontsize=11)
    ax_scatter.set_ylabel("Predicted pMIC (5-fold CV)", fontsize=11)
    ax_scatter.set_title(f"Actual vs. Predicted pMIC\n{org_label}  —  RF Regression",
                         fontsize=10, fontweight="bold")
    ax_scatter.set_xlim(lo, hi); ax_scatter.set_ylim(lo, hi)
    ax_scatter.legend(fontsize=8, loc="lower right")
    ax_scatter.set_aspect("equal", adjustable="box")

    # ─── Panel 2: Buforin & Magainin bar chart ───────────────────────────────
    all_labels  = [e["id"]       for e in query_entries]
    all_pmics   = [e["pMIC"]     for e in query_entries]
    all_mics    = [e["mic_uM"]   for e in query_entries]
    all_ugmls   = [e["mic_ugml"] for e in query_entries]
    all_stapled = [e["stapled"]  for e in query_entries]

    y_pos      = np.arange(len(all_labels))
    bar_colors = [TIER_COLOR[tier(m)] for m in all_mics]
    bars = ax_bars.barh(y_pos, all_pmics, color=bar_colors,
                        edgecolor="black", linewidth=0.7, height=0.55)

    for bar, is_st in zip(bars, all_stapled):
        if not is_st:
            bar.set_hatch("///")
            bar.set_edgecolor("#333")

    # Annotate: pMIC (μM / μg/mL)
    x_max = max(all_pmics)
    for y_i, pm, mic, ugml, is_st in zip(y_pos, all_pmics, all_mics, all_ugmls, all_stapled):
        ax_bars.text(pm + 0.02, y_i,
                     f" {pm:.2f}  ({mic:.1f} μM / {ugml:.1f} μg/mL)",
                     va="center", fontsize=7.5,
                     fontweight="bold" if not is_st else "normal")

    # Tier boundary lines (only if they fall in visible range)
    pmic_range = (min(all_pmics) - 0.2, max(all_pmics) + 1.9)
    for xv, lbl, lc in [(5.0, "10 μM", "#d62728"),
                          (5.3, "5 μM",  "#ff7f0e"),
                          (5.7, "2 μM",  "#1f77b4")]:
        if pmic_range[0] < xv < pmic_range[1]:
            ax_bars.axvline(xv, color=lc, linestyle="--", lw=0.9, alpha=0.7)
            ax_bars.text(xv, len(all_labels) - 0.05, lbl,
                         ha="center", va="bottom", fontsize=6, color=lc)

    # Separator line between Buforin / Magainin groups
    buf_count = sum(1 for e in query_entries if e["group"] == "Buforin")
    ax_bars.axhline(buf_count - 0.5, color="gray", lw=1.0, linestyle=":")
    ax_bars.text(pmic_range[0] + 0.05, buf_count - 0.5,
                 "  Magainin ▼", va="bottom", fontsize=7.5, color="gray")
    ax_bars.text(pmic_range[0] + 0.05, buf_count - 0.5,
                 "  Buforin ▲", va="top", fontsize=7.5, color="gray")

    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(all_labels, fontsize=8.5)
    ax_bars.invert_yaxis()
    ax_bars.set_xlabel("Predicted pMIC", fontsize=11)
    ax_bars.set_xlim(*pmic_range)
    ax_bars.set_title(f"Predicted pMIC for Query Peptides\n{org_label}",
                      fontsize=10, fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLOR.items()]
    patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                   hatch="///", label="Non-stapled (native)"))
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.03),
               framealpha=0.9, title="MIC Tier / Style", title_fontsize=8)

    fig.suptitle(
        f"pMIC Regression  —  {org_label}\n"
        f"RF Model (14 StaPep MD features, trained on n={n} stapled AMPs)",
        fontsize=11, fontweight="bold",
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-n",  type=int, default=15,
                        help="Minimum samples required to train a model (default 15)")
    parser.add_argument("--out-dir", default="pmic_by_organism",
                        help="Output folder for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("  pMIC Regression — All Organisms")
    print("=" * 70)
    print(f"\n  Loading data …")
    base = load_base()
    test = load_test()
    print(f"  Base dataset: {len(base)} peptides with full features")
    print(f"  Query peptides (stapled test set): {len(test)}\n")

    print(f"  {'Organism':<30}  {'n':>5}  {'Pearson R':>9}  Status")
    print("  " + "─" * 60)

    summary_rows = []

    for org_key, (org_label, org_pattern) in ORGANISMS.items():
        org_re = make_mic_regex(org_pattern)

        # Extract MIC
        base["mic_uM_tmp"] = base.apply(
            lambda r: extract_mic_uM(r["Target_Organism"], r["weight"], org_re), axis=1
        )
        df_org = base[base["mic_uM_tmp"].notna() & (base["mic_uM_tmp"] > 0)].copy()
        df_org["pMIC"] = df_org["mic_uM_tmp"].apply(mic_to_pmic)
        n = len(df_org)

        if n < args.min_n:
            print(f"  {org_label:<30}  {n:>5}  {'—':>9}  SKIPPED (n < {args.min_n})")
            summary_rows.append((org_label, n, None, "skipped"))
            continue

        # Train & evaluate
        X = df_org[FEATURE_COLS].values.astype(float)
        y = df_org["pMIC"].values
        rf_tmp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        cv = KFold(n_splits=min(5, n), shuffle=True, random_state=42)
        y_cv = cross_val_predict(rf_tmp, X, y, cv=cv)
        r_cv, _ = stats.pearsonr(y, y_cv)

        out_path = os.path.join(args.out_dir, f"{org_key}.png")
        make_figure(org_key, org_label, df_org, test, out_path)
        print(f"  {org_label:<30}  {n:>5}  {r_cv:>9.3f}  → {out_path}")
        summary_rows.append((org_label, n, r_cv, "ok"))

    # ── Summary bar chart of Pearson R by organism ──────────────────────────
    valid = [(lbl, n, r) for lbl, n, r, st in summary_rows if st == "ok"]
    if valid:
        labels_s = [f"{lbl}\n(n={n})" for lbl, n, _ in valid]
        rs       = [r for _, _, r in valid]
        colors   = ["#2ca02c" if r >= 0.7 else "#ff7f0e" if r >= 0.5 else "#d62728"
                    for r in rs]

        fig, ax = plt.subplots(figsize=(max(8, len(valid) * 1.4), 5))
        bars = ax.bar(labels_s, rs, color=colors, edgecolor="black", linewidth=0.7)
        for bar, r in zip(bars, rs):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                    f"R={r:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.axhline(0.7, color="green",  linestyle="--", lw=1, alpha=0.7, label="R=0.70")
        ax.axhline(0.84, color="navy", linestyle="--", lw=1, alpha=0.7, label="Paper R=0.84")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Pearson R (5-fold CV)", fontsize=11)
        ax.set_title("RF Regression Performance by Target Organism\n"
                     "(14 StaPep MD features, 5-fold cross-validation)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        summary_path = os.path.join(args.out_dir, "_summary_pearson_r.png")
        plt.savefig(summary_path, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"\n  Summary plot → {summary_path}")

    print(f"\n  All done. Plots saved to  {args.out_dir}/\n")


if __name__ == "__main__":
    main()
