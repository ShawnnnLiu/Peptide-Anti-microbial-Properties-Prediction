#!/usr/bin/env python3
"""
predict_buf_hemolysis.py
========================
RF regression for % hemolysis trained on the stapled AMP dataset
(StaPep MD features), then predicts hemolysis for ALL Buforin variants.

Test set (12 Buforin variants):
  - Buf WT (native, non-stapled)
  - 7 F10W variants  (from advisor table — have literature hemolysis)
  - 4 Buf12/13/Q9K/V15K,L19K  (from earlier NGC study)

Literature hemolysis values at 50 ug/mL (from advisor table) are
normalised to the model's reference concentration for fair comparison.

Figures:
  Panel A — 5-fold CV scatter (training set actual vs predicted)
  Panel B — Predicted vs literature hemolysis for Buf variants
  Panel C — Bar chart of predicted hemolysis for all Buf variants
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

FEATURE_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent", "mean_bfactor", "mean_gyrate",
    "num_hbonds", "psa",
]

# ── Buf WT features (from 5 ns StaPep MD) ──────────────────────────────────────
BUF_WT_FEATURES = {
    "length": 21, "weight": 2473.829,
    "hydrophobic_index": -0.8142857142857142, "charge": 6.094,
    "aromaticity": 0.047619, "isoelectric_point": 11.9999,
    "fraction_arginine": 0.23810, "fraction_lysine": 0.04762,
    "helix_percent": 0.17819, "loop_percent": 0.82105,
    "mean_bfactor": 573.434, "mean_gyrate": 12.001,
    "num_hbonds": 0, "psa": 1064.217,
}

# ── Literature hemolysis values (from advisor table, at 50 ug/mL) ──────────────
# MW needed to convert 50 ug/mL → uM for normalisation
LITERATURE_HEMO = {
    # peptide_id:         (hemo_pct, MW_Da, test_conc_ugml)
    "Buf_i4_16_F10W":     (12.6,  2429.9, 50.0),
    "Buf_i4_14_F10W":     (2.9,   2453.8, 50.0),
    "Buf_i4_4_F10W":      (2.4,   2523.0, 50.0),
    "Buf_i4_3_F10W":      (3.1,   2579.1, 50.0),
    "Buf_i7_9_F10W":      (57.0,  2500.0, 50.0),
    "Buf_i7_6_F10W":      (3.0,   2637.2, 50.0),
    "Buf_i7_1_F10W":      (2.3,   2551.0, 50.0),
    "Buf12":              (9.23,  2491.93, 50.0),
    "Buf13":              (3.84,  2514.96, 50.0),
    # Buf13_Q9K:  "low" in table → not numeric, skip
    # Buf12_V15K_L19K: no data
    # Buf WT: no hemolysis data
}

# ── Display labels ─────────────────────────────────────────────────────────────
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

# ── Hemolysis extraction (from training data text) ────────────────────────────
_HEM_SINGLE_RE = re.compile(
    r"([<>≤]?\s*[\d.]+)\s*%\s*hemol"
    r"[^.;#\n]*?"
    r"\bat\s+(?:peptide\s+)?(?:concentration\s+of\s+)?"
    r"([<>≤]?\s*[\d.]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

def _strip_op(s):
    return re.sub(r'^[<>≤≥\s]+', '', s)

def _to_uM(val, unit, mw_da):
    unit = unit.lower()
    if unit in ("μm", "um"):
        return val
    if unit in ("μg/ml", "ug/ml") and mw_da > 0:
        return val / mw_da * 1000.0
    return None

def extract_hemolysis(text, mw_da):
    """Return (pct, conc_uM) for first valid hemolysis entry, or None."""
    if not isinstance(text, str):
        return None
    for m in _HEM_SINGLE_RE.finditer(text):
        pct_s, conc_s, unit = m.group(1), m.group(2), m.group(3)
        try:
            pct  = float(_strip_op(pct_s))
            conc = float(_strip_op(conc_s))
        except ValueError:
            continue
        conc_uM = _to_uM(conc, unit, mw_da)
        if conc_uM and conc_uM > 0:
            return pct, conc_uM
    return None

def hemolysis_label(pct):
    if pct < 5:   return "Non-hemolytic"
    if pct < 15:  return "Low"
    if pct < 40:  return "Moderate"
    return "High"

HEM_COLORS = {
    "Non-hemolytic": "#2ca02c",
    "Low":           "#ff7f0e",
    "Moderate":      "#d62728",
    "High":          "#7f0000",
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  Hemolysis RF Regression — Buforin Variant Predictions")
    print("=" * 72)

    # ── 1. Load training data ────────────────────────────────────────────────
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    df   = pd.merge(
        meta[["DRAMP_ID", "Hemolytic_Activity"]],
        feat[["DRAMP_ID"] + FEATURE_COLS],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=FEATURE_COLS)
    print(f"\n  Stapled AMPs with complete features: {len(df)}")

    # ── 2. Extract hemolysis ─────────────────────────────────────────────────
    extracted = df.apply(
        lambda r: extract_hemolysis(r["Hemolytic_Activity"], r["weight"]), axis=1
    )
    df["pct_raw"]  = extracted.apply(lambda x: x[0] if x else None)
    df["conc_uM"]  = extracted.apply(lambda x: x[1] if x else None)
    df = df.dropna(subset=["pct_raw", "conc_uM"]).copy()
    print(f"  Parseable % hemolysis entries: {len(df)}")

    # ── 3. Normalise to reference concentration ──────────────────────────────
    ref_conc = float(np.median(df["conc_uM"]))
    df["pct_norm"] = (df["pct_raw"] * (ref_conc / df["conc_uM"])).clip(0, 100)
    print(f"  Median test concentration (reference): {ref_conc:.1f} uM")
    print(f"  Normalised % hemolysis: mean={df['pct_norm'].mean():.1f}%, "
          f"std={df['pct_norm'].std():.1f}%")

    # ── 4. Train RF ──────────────────────────────────────────────────────────
    X_train = df[FEATURE_COLS].values.astype(float)
    y_train = df["pct_norm"].values
    n_train = len(df)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = np.clip(cross_val_predict(rf, X_train, y_train, cv=cv), 0, 100)
    r_cv, _ = pearsonr(y_train, y_cv)
    r2_cv   = r2_score(y_train, y_cv)
    mae_cv  = mean_absolute_error(y_train, y_cv)
    rf.fit(X_train, y_train)

    print(f"\n  5-fold CV:  Pearson R = {r_cv:.3f}  |  R2 = {r2_cv:.3f}  "
          f"|  MAE = {mae_cv:.1f}%  |  n = {n_train}")

    # ── 5. Load ALL Buforin variant features ─────────────────────────────────
    test_entries = []

    # 5a. Buf WT (hardcoded)
    test_entries.append({
        "pid": "Buf_WT",
        "features": np.array([[BUF_WT_FEATURES[f] for f in FEATURE_COLS]]),
        "stapled": False,
        "group": "native",
    })

    # 5b. 7 F10W variants
    f10w_df = pd.read_csv(F10W_FEAT)
    for _, row in f10w_df.iterrows():
        pid = row["peptide_id"]
        test_entries.append({
            "pid": pid,
            "features": np.array([[row[f] for f in FEATURE_COLS]]),
            "stapled": True,
            "group": "F10W",
        })

    # 5c. 4 Buf variants from test_stapled_features.csv
    test_df = pd.read_csv(TEST_FEAT)
    buf_test = test_df[test_df["peptide_id"].str.lower().str.startswith("buf")]
    for _, row in buf_test.iterrows():
        pid = row["peptide_id"]
        test_entries.append({
            "pid": pid,
            "features": np.array([[row[f] for f in FEATURE_COLS]]),
            "stapled": True,
            "group": "NGC_variants",
        })

    print(f"\n  Total Buforin test variants: {len(test_entries)}")

    # ── 6. Predict hemolysis ─────────────────────────────────────────────────
    for e in test_entries:
        e["pred_pct"] = float(np.clip(rf.predict(e["features"])[0], 0, 100))

        # Literature value (normalised to model ref concentration)
        lit = LITERATURE_HEMO.get(e["pid"])
        if lit:
            lit_pct, mw, test_ugml = lit
            lit_conc_uM = test_ugml / mw * 1000.0
            e["lit_pct_raw"]  = lit_pct
            e["lit_conc_uM"]  = lit_conc_uM
            # Normalise literature value to model reference
            e["lit_pct_norm"] = lit_pct * (ref_conc / lit_conc_uM)
        else:
            e["lit_pct_raw"]  = None
            e["lit_conc_uM"]  = None
            e["lit_pct_norm"] = None

    # ── 7. Print results table ───────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print(f"  {'Variant':<22}  {'Pred %':>7}  {'Lit % (raw)':>11}  "
          f"{'Lit conc':>9}  {'Lit % (norm)':>12}  {'Label'}")
    print(f"{'─'*90}")
    for e in test_entries:
        dname = DISPLAY_LABELS.get(e["pid"], e["pid"]).replace("\n", " ")
        pred_s = f"{e['pred_pct']:.1f}%"
        if e["lit_pct_raw"] is not None:
            raw_s  = f"{e['lit_pct_raw']:.1f}%"
            conc_s = f"{e['lit_conc_uM']:.1f} uM"
            norm_s = f"{e['lit_pct_norm']:.1f}%"
        else:
            raw_s = conc_s = norm_s = "—"
        tag = "" if e["stapled"] else " [native]"
        print(f"  {dname:<22}  {pred_s:>7}  {raw_s:>11}  "
              f"{conc_s:>9}  {norm_s:>12}  {hemolysis_label(e['pred_pct'])}{tag}")
    print(f"{'─'*90}")
    print(f"  Note: Literature values at 50 ug/mL. Normalised to {ref_conc:.1f} uM "
          f"reference for fair comparison.")

    # ── 8. Correlation (predicted vs literature, normalised) ─────────────────
    paired = [(e["pred_pct"], e["lit_pct_norm"])
              for e in test_entries if e["lit_pct_norm"] is not None]
    pred_arr = np.array([p[0] for p in paired])
    lit_arr  = np.array([p[1] for p in paired])
    n_paired = len(paired)

    if n_paired >= 3:
        r_pl, p_pl = pearsonr(pred_arr, lit_arr)
        rho_pl, p_rho = spearmanr(pred_arr, lit_arr)
        print(f"\n  Predicted vs Literature (normalised):")
        print(f"    Pearson R  = {r_pl:.3f}  (p={p_pl:.3f})")
        print(f"    Spearman r = {rho_pl:.3f}  (p={p_rho:.3f})")
        print(f"    N = {n_paired} variants with both values")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(20, 7))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3],
                           left=0.05, right=0.97,
                           top=0.85, bottom=0.12, wspace=0.35)
    ax_cv  = fig.add_subplot(gs[0])
    ax_sc  = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    # ── Panel A: Training 5-fold CV scatter ──────────────────────────────────
    ax_cv.scatter(y_train, y_cv, alpha=0.55, s=45, color="#9467bd",
                  edgecolors="white", linewidths=0.4, zorder=3)

    lo = max(0, min(y_train.min(), y_cv.min()) - 2)
    hi = min(100, max(y_train.max(), y_cv.max()) + 2)
    ax_cv.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    slope, intercept, *_ = stats.linregress(y_train, y_cv)
    x_grid = np.linspace(lo, hi, 200)
    ax_cv.plot(x_grid, slope * x_grid + intercept,
               "-", color="#d7191c", lw=1.5, label="Linear fit")

    ax_cv.text(0.05, 0.95,
               f"Pearson R = {r_cv:.2f}\nR² = {r2_cv:.2f}\n"
               f"MAE = {mae_cv:.1f}%\nn = {n_train}",
               transform=ax_cv.transAxes, fontsize=9, va="top",
               fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="lightyellow", alpha=0.85))

    ax_cv.set_xlabel(f"Actual % Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=9)
    ax_cv.set_ylabel("Predicted % Hemolysis (5-fold CV)", fontsize=9)
    ax_cv.set_title("Panel A — Training Set\n5-fold Cross-Validation",
                    fontsize=10, fontweight="bold")
    ax_cv.set_xlim(lo, hi); ax_cv.set_ylim(lo, hi)
    ax_cv.legend(fontsize=7, loc="lower right")
    ax_cv.set_aspect("equal", adjustable="box")
    ax_cv.grid(alpha=0.2, lw=0.5)

    # ── Panel B: Predicted vs Literature (Buforin variants) ──────────────────
    if n_paired >= 3:
        # Color by group
        group_colors = {"F10W": "#4363d8", "NGC_variants": "#e6194b"}
        for e in test_entries:
            if e["lit_pct_norm"] is None:
                continue
            c = group_colors.get(e["group"], "#333")
            marker = "o" if e["group"] == "F10W" else "s"
            ax_sc.scatter(e["lit_pct_norm"], e["pred_pct"],
                          s=100, marker=marker, color=c,
                          edgecolors="k", linewidths=0.7, zorder=5)
            dname = DISPLAY_LABELS.get(e["pid"], e["pid"]).replace("\n", " ")
            # Smart label offset to avoid overlaps
            x_off, y_off = 6, 4
            if e["pred_pct"] < 5 and e["lit_pct_norm"] < 5:
                y_off = -10
            ax_sc.annotate(dname, (e["lit_pct_norm"], e["pred_pct"]),
                           textcoords="offset points", xytext=(x_off, y_off),
                           fontsize=6.5, color=c, fontweight="bold")

        # Perfect prediction line
        lo2 = 0
        hi2 = max(max(pred_arr), max(lit_arr)) * 1.15 + 2
        ax_sc.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.8, alpha=0.5, label="y = x")

        # Regression line
        sl2, ic2, *_ = stats.linregress(lit_arr, pred_arr)
        x_fit = np.linspace(lo2, hi2, 100)
        ax_sc.plot(x_fit, sl2 * x_fit + ic2,
                   color="#d7191c", lw=1.3, ls="--", alpha=0.7)

        ax_sc.text(0.05, 0.95,
                   f"Pearson R  = {r_pl:.2f}\n"
                   f"Spearman r = {rho_pl:.2f}\n"
                   f"N = {n_paired}",
                   transform=ax_sc.transAxes, fontsize=9, va="top",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="lightyellow", alpha=0.85))

        ax_sc.set_xlabel(f"Literature % Hemolysis\n(norm. to {ref_conc:.0f} uM)",
                         fontsize=9)
        ax_sc.set_ylabel("RF Predicted % Hemolysis", fontsize=9)
        ax_sc.set_title("Panel B — Predicted vs Literature\n"
                        "Buforin Variants (normalised to ref. conc.)",
                        fontsize=10, fontweight="bold")
        ax_sc.set_xlim(lo2, hi2); ax_sc.set_ylim(lo2, hi2)
        ax_sc.legend(fontsize=7, loc="lower right")
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.grid(alpha=0.2, lw=0.5)

        # Legend for variant groups
        leg_h = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#4363d8", markeredgecolor="k",
                   markersize=8, label="F10W variants"),
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor="#e6194b", markeredgecolor="k",
                   markersize=8, label="Buf12/13 variants"),
        ]
        ax_sc.legend(handles=leg_h, fontsize=7, loc="lower right")

    # ── Panel C: Bar chart — all variants ────────────────────────────────────
    # Order: Buf WT → i+4 F10W → i+7 F10W → Buf12/13 variants
    order = [
        "Buf_WT",
        "Buf_i4_16_F10W", "Buf_i4_14_F10W", "Buf_i4_4_F10W", "Buf_i4_3_F10W",
        "Buf_i7_9_F10W", "Buf_i7_6_F10W", "Buf_i7_1_F10W",
        "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    ]
    entry_map = {e["pid"]: e for e in test_entries}
    ordered = [entry_map[pid] for pid in order if pid in entry_map]

    bar_labels = [DISPLAY_LABELS.get(e["pid"], e["pid"]) for e in ordered]
    bar_pred   = [e["pred_pct"] for e in ordered]
    bar_lit    = [e["lit_pct_norm"] for e in ordered]
    bar_lit_raw = [e["lit_pct_raw"] for e in ordered]
    bar_stapled = [e["stapled"] for e in ordered]

    y_pos = np.arange(len(ordered))
    bar_h = 0.35

    # Predicted bars
    colors_pred = [HEM_COLORS[hemolysis_label(p)] for p in bar_pred]
    bars_p = ax_bar.barh(y_pos + bar_h/2, bar_pred, bar_h,
                         color=colors_pred, edgecolor="k", linewidth=0.6,
                         label="RF Predicted")
    # Literature bars (where available)
    lit_vals_plot = [v if v is not None else 0 for v in bar_lit]
    lit_colors = [HEM_COLORS[hemolysis_label(v)] if v is not None else "#eeeeee"
                  for v in bar_lit]
    bars_l = ax_bar.barh(y_pos - bar_h/2, lit_vals_plot, bar_h,
                         color=lit_colors, edgecolor="k", linewidth=0.6,
                         alpha=0.6, hatch="//",
                         label=f"Literature (norm. to {ref_conc:.0f} uM)")

    # Hide bars with no literature data
    for i, v in enumerate(bar_lit):
        if v is None:
            bars_l[i].set_visible(False)

    # Hatched bars for native
    for i, is_st in enumerate(bar_stapled):
        if not is_st:
            bars_p[i].set_hatch("xxx")

    # Annotations
    for i, (pred, lit, lit_raw, is_st) in enumerate(
            zip(bar_pred, bar_lit, bar_lit_raw, bar_stapled)):
        # Predicted value
        tag = " [native]" if not is_st else ""
        ax_bar.text(pred + 0.3, y_pos[i] + bar_h/2,
                    f" {pred:.1f}%{tag}",
                    va="center", fontsize=7, fontweight="bold")
        # Literature value
        if lit is not None:
            raw_str = f"(raw: {lit_raw:.1f}% @50ug/mL)"
            ax_bar.text(lit + 0.3, y_pos[i] - bar_h/2,
                        f" {lit:.1f}% {raw_str}",
                        va="center", fontsize=6, color="#555")

    # Threshold lines
    for xv, lbl, lc in [(5, "5%", "#2ca02c"), (15, "15%", "#ff7f0e"),
                         (40, "40%", "#d62728")]:
        ax_bar.axvline(xv, color=lc, linestyle="--", lw=0.9, alpha=0.7)
        ax_bar.text(xv, -0.7, lbl, ha="center", fontsize=6.5, color=lc)

    # Group dividers
    ax_bar.axhline(0.5, color="#555", lw=0.8, ls=":")   # WT | i+4 F10W
    ax_bar.axhline(4.5, color="#555", lw=0.8, ls=":")   # i+4 F10W | i+7 F10W
    ax_bar.axhline(7.5, color="#555", lw=0.8, ls=":")   # i+7 F10W | Buf12/13

    ax_bar.text(-0.5, 2.5, "i+4\nF10W", fontsize=7, color="#333",
                style="italic", ha="right", va="center")
    ax_bar.text(-0.5, 6.0, "i+7\nF10W", fontsize=7, color="#333",
                style="italic", ha="right", va="center")
    ax_bar.text(-0.5, 9.5, "Buf12/13\nvariants", fontsize=7, color="#333",
                style="italic", ha="right", va="center")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("% Hemolysis", fontsize=9)
    ax_bar.set_xlim(0, max(max(bar_pred), max(v for v in bar_lit if v)) * 1.25 + 3)
    ax_bar.set_title("Panel C — Predicted vs Literature Hemolysis\n"
                     "All Buforin Variants",
                     fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7.5, loc="lower right")
    ax_bar.grid(axis="x", alpha=0.2, lw=0.5)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "Hemolysis RF Regression — Stapled AMP Training → Buforin Variant Predictions\n"
        f"14 StaPep MD features  |  n_train = {n_train}  |  "
        f"ref. concentration = {ref_conc:.1f} uM",
        fontsize=12, fontweight="bold",
    )

    # ── Shared legend ────────────────────────────────────────────────────────
    hem_patches = [mpatches.Patch(color=c, label=t) for t, c in HEM_COLORS.items()]
    hem_patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                       hatch="xxx", label="Non-stapled (native)"))
    fig.legend(handles=hem_patches, loc="lower center", ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, title="Hemolysis Level", title_fontsize=8)

    out = BASE / "buf_hemolysis_regression.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
