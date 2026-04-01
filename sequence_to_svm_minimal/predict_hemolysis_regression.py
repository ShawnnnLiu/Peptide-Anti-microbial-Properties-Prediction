#!/usr/bin/env python3
"""
predict_hemolysis_regression.py

Builds a Random Forest regression model to predict % hemolysis of
hydrocarbon-stapled peptides from 14 StaPep MD features.

Data extraction:
  - Only uses "X% hemolysis at Y concentration" entries (not LC50/MHC)
  - For rows with multiple concentrations, takes the first match
  - Converts μg/mL test concentrations → μM using peptide MW
  - Normalises all % values to the median test concentration across
    the dataset so the target is always "% hemolysis at ~ref_conc μM"

Then predicts:
  - Buforin II (native + 4 stapled variants)
  - Magainin II (native + 4 stapled variants)

Usage:
    python predict_hemolysis_regression.py
    python predict_hemolysis_regression.py --save hemolysis_regression.png
"""

import re, math, argparse, warnings
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

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/training_dataset/StaPep"
AMPS_META = f"{DATA_DIR}/stapled_amps.csv"
AMPS_FEAT = f"{DATA_DIR}/stapled_amps_features.csv"
TEST_FEAT = f"{DATA_DIR}/test_stapled_features.csv"

FEATURE_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent", "mean_bfactor", "mean_gyrate",
    "num_hbonds", "psa",
]

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

# ─── Hemolysis extraction ──────────────────────────────────────────────────────
# Format A (single): "X% hemolysis ... at [peptide concentration of] Y unit"
_HEM_SINGLE_RE = re.compile(
    r"([<>≤]?\s*[\d.]+)\s*%\s*hemol"
    r"[^.;#\n]*?"
    r"\bat\s+(?:peptide\s+)?(?:concentration\s+of\s+)?"
    r"([<>≤]?\s*[\d.]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

# Format B (list): "X1%, X2%, ... Xn% hemolysis ... at C1, C2, ... Cn unit"
_HEM_LIST_PCTS_RE  = re.compile(r"([<>≤]?\s*[\d.]+)\s*%",   re.I)
_HEM_LIST_CONCS_RE = re.compile(r"([<>≤]?\s*[\d.]+)",        re.I)
# Allow "and" in the concentration block: "50.0 and 100.0 μM"
_HEM_LIST_AT_RE    = re.compile(
    r"\bhemol\w*[^.;#\n]*?\bat\s+(?:peptide\s+)?(?:concentration\s+of\s+)?"
    r"([\w\s.,<>]+?)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

def _to_uM(val, unit, mw_da):
    unit = unit.lower()
    if unit in ("μm", "um"):
        return val
    if unit in ("μg/ml", "ug/ml") and mw_da > 0:
        return val / mw_da * 1000.0
    return None

def _strip_op(s):
    return re.sub(r'^[<>≤≥\s]+', '', s)

def extract_hemolysis(text, mw_da):
    """
    Return (pct, conc_uM) for the best parseable % hemolysis entry, or None.

    Strategy:
      1. Try single-value regex (Format A) – use first real match
      2. Fall back to list regex (Format B) – pair the i-th % with i-th conc,
         take the entry at the concentration closest to 10 μM (reference)
    """
    if not isinstance(text, str):
        return None

    # ── Format A: single pair ────────────────────────────────────────────────
    for m in _HEM_SINGLE_RE.finditer(text):
        pct_s, conc_s, unit = m.group(1), m.group(2), m.group(3)
        try:
            pct  = float(_strip_op(pct_s))
            conc = float(_strip_op(conc_s))
        except ValueError:
            continue
        conc_uM = _to_uM(conc, unit, mw_da)
        if conc_uM and conc_uM > 0:
            return pct, conc_uM          # first valid single match

    # ── Format B: list  "X1%, X2%... hemolysis at C1, C2... unit" ────────────
    m_at = _HEM_LIST_AT_RE.search(text)
    if m_at is None:
        return None
    conc_block = m_at.group(1)
    unit       = m_at.group(2)

    # Find all % values BEFORE the "hemolysis" keyword that leads to this match
    before_hemol = text[:m_at.start()]
    pct_vals = []
    for mp in _HEM_LIST_PCTS_RE.finditer(before_hemol):
        try:
            pct_vals.append(float(_strip_op(mp.group(1))))
        except ValueError:
            pass

    conc_vals = []
    for mc in _HEM_LIST_CONCS_RE.finditer(conc_block):
        try:
            conc_vals.append(float(_strip_op(mc.group(1))))
        except ValueError:
            pass

    if not pct_vals or not conc_vals:
        return None

    # Pair i-th % with i-th concentration; take the last non-<1% real value,
    # or the value at the concentration closest to 10 μM reference
    n_pairs = min(len(pct_vals), len(conc_vals))
    pairs   = []
    for i in range(n_pairs):
        c_uM = _to_uM(conc_vals[i], unit, mw_da)
        if c_uM and c_uM > 0:
            pairs.append((pct_vals[i], c_uM))

    if not pairs:
        return None

    # Pick pair whose concentration is closest to 10 μM
    REF = 10.0
    best = min(pairs, key=lambda x: abs(x[1] - REF))
    return best


def hemolysis_label(pct):
    """Qualitative label for % hemolysis."""
    if pct < 5:   return "Non-hemolytic"
    if pct < 15:  return "Low"
    if pct < 40:  return "Moderate"
    return "High"

HEM_COLOR = {
    "Non-hemolytic": "#2ca02c",
    "Low":           "#ff7f0e",
    "Moderate":      "#d62728",
    "High":          "#7f0000",
}

# ─── CI band helper ───────────────────────────────────────────────────────────
def _ci_band(x, slope, intercept, x_grid, n, alpha=0.05):
    y_fit  = slope * x + intercept
    s2     = np.sum((y_fit - (slope * x + intercept)) ** 2) / max(n - 2, 1)
    x_mean = np.mean(x)
    se     = np.sqrt(s2 * (1/n + (x_grid - x_mean)**2 /
                           np.sum((x - x_mean)**2 + 1e-12)))
    from scipy.stats import t as t_dist
    t_val  = t_dist.ppf(1 - alpha / 2, df=max(n - 2, 1))
    y_line = slope * x_grid + intercept
    return y_line - t_val * se, y_line + t_val * se


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="hemolysis_regression.png")
    args = parser.parse_args()

    print("=" * 65)
    print("  Hemolysis Regression — % Hemolysis at Fixed Concentration")
    print("=" * 65)

    # ── Load & merge ─────────────────────────────────────────────────────────
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    df   = pd.merge(
        meta[["DRAMP_ID", "Hemolytic_Activity"]],
        feat[["DRAMP_ID"] + FEATURE_COLS],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=FEATURE_COLS)

    print(f"\n  Peptides with features: {len(df)}")

    # ── Extract hemolysis ──────────────────────────────────────────────────
    extracted = df.apply(
        lambda r: extract_hemolysis(r["Hemolytic_Activity"], r["weight"]), axis=1
    )
    df["pct_raw"]   = extracted.apply(lambda x: x[0] if x else None)
    df["conc_uM"]   = extracted.apply(lambda x: x[1] if x else None)
    df = df.dropna(subset=["pct_raw", "conc_uM"]).copy()
    print(f"  Rows with parseable % hemolysis at known concentration: {len(df)}")

    # ── Normalise to reference concentration ──────────────────────────────
    ref_conc = float(np.median(df["conc_uM"]))
    print(f"  Median test concentration: {ref_conc:.1f} μM  (used as reference)")
    # Linear normalisation: assume % ∝ conc at low hemolysis values
    df["pct_norm"] = df["pct_raw"] * (ref_conc / df["conc_uM"])
    # Clip at 0–100%
    df["pct_norm"] = df["pct_norm"].clip(0, 100)

    print(f"\n  Normalised % hemolysis at {ref_conc:.0f} μM:")
    print(f"    mean = {df['pct_norm'].mean():.1f}%")
    print(f"    std  = {df['pct_norm'].std():.1f}%")
    print(f"    min  = {df['pct_norm'].min():.1f}%  /  max = {df['pct_norm'].max():.1f}%")

    # ── Train RF ─────────────────────────────────────────────────────────────
    X = df[FEATURE_COLS].values.astype(float)
    y = df["pct_norm"].values
    n = len(df)

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv     = cross_val_predict(rf, X, y, cv=cv)
    y_cv     = np.clip(y_cv, 0, 100)
    r_cv, _  = stats.pearsonr(y, y_cv)
    r2_cv    = r2_score(y, y_cv)
    rf.fit(X, y)

    print(f"\n  5-fold CV Pearson R = {r_cv:.3f}   R² = {r2_cv:.3f}  (n={n})")

    # ── Load test (stapled variants) ─────────────────────────────────────────
    test = pd.read_csv(TEST_FEAT).dropna(subset=FEATURE_COLS)
    buf_ids = [r for r in test["peptide_id"] if r.lower().startswith("buf")]
    mag_ids = [r for r in test["peptide_id"] if r.lower().startswith("mag")]

    query_entries = []
    for group_ids, nat_key in [
        (buf_ids, "Buforin II\n(native)"),
        (mag_ids, "Magainin II\n(native)"),
    ]:
        feats = NATIVE[nat_key]
        x_nat = np.array([[feats[f] for f in FEATURE_COLS]])
        pred  = float(np.clip(rf.predict(x_nat)[0], 0, 100))
        group = "Buforin" if "buf" in nat_key.lower() else "Magainin"
        query_entries.append({
            "id": nat_key.replace("\n", " "),
            "group": group, "stapled": False, "pct": pred,
        })
        for pid in group_ids:
            row  = test[test["peptide_id"] == pid].iloc[0]
            x_st = np.array([[row[f] for f in FEATURE_COLS]])
            pred2 = float(np.clip(rf.predict(x_st)[0], 0, 100))
            query_entries.append({
                "id": pid,
                "group": group, "stapled": True, "pct": pred2,
            })

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n  Predicted % Hemolysis at {ref_conc:.0f} μM (reference):")
    print("  " + "─" * 55)
    print(f"  {'Peptide':<28}  {'% Hemolysis':>11}  Label")
    print("  " + "─" * 55)
    for group in ["Buforin", "Magainin"]:
        print(f"\n  — {group} —")
        for e in query_entries:
            if e["group"] != group: continue
            tag = "  [non-stapled]" if not e["stapled"] else ""
            print(f"  {'  '+e['id']:<26}  {e['pct']:>11.1f}%  {hemolysis_label(e['pct'])}{tag}")
    print("  " + "─" * 55)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 6))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.2],
                           left=0.08, right=0.97,
                           top=0.88, bottom=0.12, wspace=0.40)
    ax_sc  = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    # ─── Panel 1: Actual vs Predicted ────────────────────────────────────────
    ax_sc.scatter(y, y_cv, alpha=0.6, s=40, color="#9467bd",
                  edgecolors="white", linewidths=0.4, zorder=3,
                  label=f"Peptides (n={n})")

    lo = max(0, min(y.min(), y_cv.min()) - 2)
    hi = min(100, max(y.max(), y_cv.max()) + 2)
    ax_sc.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    slope, intercept, *_ = stats.linregress(y, y_cv)
    x_grid = np.linspace(lo, hi, 200)
    y_line = slope * x_grid + intercept
    ax_sc.plot(x_grid, y_line, "-", color="#d7191c", lw=1.5, label="Linear fit")
    lo_ci, hi_ci = _ci_band(y, slope, intercept, x_grid, n)
    ax_sc.fill_between(x_grid, lo_ci, hi_ci, alpha=0.15, color="#d7191c", label="95% CI")

    ax_sc.text(0.05, 0.28,
               f"Pearson R = {r_cv:.2f}\nR² = {r2_cv:.2f}\nn = {n}",
               transform=ax_sc.transAxes, fontsize=11, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="lightyellow", alpha=0.85))

    ax_sc.axhline(10, color="#ff7f0e", lw=0.8, ls="--", alpha=0.7, label="10% threshold")
    ax_sc.set_xlabel(f"Actual % Hemolysis (normalised to {ref_conc:.0f} μM)", fontsize=11)
    ax_sc.set_ylabel(f"Predicted % Hemolysis (5-fold CV)", fontsize=11)
    ax_sc.set_title("Actual vs. Predicted % Hemolysis\nRF Regression (14 StaPep MD features)",
                    fontsize=10, fontweight="bold")
    ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
    ax_sc.legend(fontsize=8, loc="lower right")
    ax_sc.set_aspect("equal", adjustable="box")

    # ─── Panel 2: Bar chart ───────────────────────────────────────────────────
    all_labels  = [e["id"]      for e in query_entries]
    all_pcts    = [e["pct"]     for e in query_entries]
    all_stapled = [e["stapled"] for e in query_entries]

    y_pos      = np.arange(len(all_labels))
    bar_colors = [HEM_COLOR[hemolysis_label(p)] for p in all_pcts]
    bars = ax_bar.barh(y_pos, all_pcts, color=bar_colors,
                       edgecolor="black", linewidth=0.7, height=0.55)

    for bar, is_st in zip(bars, all_stapled):
        if not is_st:
            bar.set_hatch("///")
            bar.set_edgecolor("#333")

    for y_i, pct, is_st in zip(y_pos, all_pcts, all_stapled):
        ax_bar.text(pct + 0.3, y_i,
                    f" {pct:.1f}%  {hemolysis_label(pct)}",
                    va="center", fontsize=8.5,
                    fontweight="bold" if not is_st else "normal")

    # Threshold lines
    for xv, lbl, lc in [(5, "5%", "#2ca02c"), (15, "15%", "#ff7f0e"), (40, "40%", "#d62728")]:
        if xv < max(all_pcts) + 5:
            ax_bar.axvline(xv, color=lc, linestyle="--", lw=0.9, alpha=0.7)
            ax_bar.text(xv, len(all_labels) - 0.1, lbl,
                        ha="center", va="bottom", fontsize=6.5, color=lc)

    # Separator Buforin / Magainin
    buf_count = sum(1 for e in query_entries if e["group"] == "Buforin")
    ax_bar.axhline(buf_count - 0.5, color="gray", lw=1.0, linestyle=":")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(all_labels, fontsize=8.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(f"Predicted % Hemolysis (at {ref_conc:.0f} μM ref.)", fontsize=11)
    ax_bar.set_xlim(0, max(all_pcts) + 8)
    ax_bar.set_title(f"Predicted % Hemolysis for Query Peptides\n"
                     f"(reference concentration ≈ {ref_conc:.0f} μM)",
                     fontsize=10, fontweight="bold")

    # Legend
    patches = [mpatches.Patch(color=c, label=t) for t, c in HEM_COLOR.items()]
    patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                   hatch="///", label="Non-stapled (native)"))
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.03),
               framealpha=0.9, title="Hemolysis Level / Style", title_fontsize=8)

    fig.suptitle(
        "% Hemolysis RF Regression  —  Stapled Peptide Dataset\n"
        f"14 StaPep MD features, n={n} peptides  |  "
        f"Normalised to reference {ref_conc:.0f} μM",
        fontsize=11, fontweight="bold",
    )

    plt.savefig(args.save, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved  →  {args.save}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
