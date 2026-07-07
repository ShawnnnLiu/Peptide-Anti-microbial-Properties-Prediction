#!/usr/bin/env python3
"""
predict_buf_hemolysis_v2.py
===========================
Improved hemolysis RF regression:

Improvements over v1:
  1. Expanded feature set: 18 features
     (added lyticity_index, sasa, sheet_percent, hydrophobic_sasa)
  2. Improved hemolysis parsing:
     - Single-value: "X% hemolysis at Y unit"          (~99 entries)
     - List-format: "X1%, X2%... hemolysis at C1,C2..."  (+22 entries)
     - LC50/HC50: converted via Hill equation (n=1)       (+17 entries)
     - List without 'hemolysis' keyword                   (+1 entry)
  3. Predicts for all 12 Buforin variants with literature comparison
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

# ── Feature columns (expanded from 14 → 18) ───────────────────────────────────
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

# ── Literature hemolysis values (from advisor table, at 50 ug/mL) ──────────────
LITERATURE_HEMO = {
    "Buf_i4_16_F10W":     (12.6,  2429.9, 50.0),
    "Buf_i4_14_F10W":     (2.9,   2453.8, 50.0),
    "Buf_i4_4_F10W":      (2.4,   2523.0, 50.0),
    "Buf_i4_3_F10W":      (3.1,   2579.1, 50.0),
    "Buf_i7_9_F10W":      (57.0,  2500.0, 50.0),
    "Buf_i7_6_F10W":      (3.0,   2637.2, 50.0),
    "Buf_i7_1_F10W":      (2.3,   2551.0, 50.0),
    "Buf12":              (9.23,  2491.93, 50.0),
    "Buf13":              (3.84,  2514.96, 50.0),
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
# HEMOLYSIS EXTRACTION (improved)
# ══════════════════════════════════════════════════════════════════════════════

def _strip_op(s):
    """Remove leading <, >, ≤, ≥ from numeric string."""
    return re.sub(r'^[<>≤≥\s]+', '', s)

def _to_uM(val, unit, mw_da):
    """Convert concentration value + unit to μM."""
    unit = unit.lower().strip()
    if unit in ("μm", "um"):
        return val
    if unit in ("μg/ml", "ug/ml", "μg/ml", "ug/ml") and mw_da > 0:
        return val / mw_da * 1000.0
    return None


# ── Format A: single "X% hemolysis at Y unit" ────────────────────────────────
_HEM_SINGLE_RE = re.compile(
    r"([<>≤]?\s*[\d.]+)\s*%\s*hemol"
    r"[^.;#\n]*?"
    r"\bat\s+(?:peptide\s+)?(?:concentration\s+of\s+)?"
    r"([<>≤]?\s*[\d.]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

# ── Format B: list "X1%, X2%,...Xn% hemolysis ... at C1, C2,...Cn unit" ──────
_LIST_PCT_RE  = re.compile(r"([<>≤]?\s*[\d.]+)\s*%")
_LIST_CONC_RE = re.compile(r"([\d.]+)")

# ── Format C: LC50 / HC50 ────────────────────────────────────────────────────
_LC50_RE = re.compile(
    r"(?:LC50|HC50|LC\s*50|HC\s*50)\s*[=:>< ]+\s*([><=]?\s*[\d.]+)\s*([μu]M|[μu]g/mL)",
    re.IGNORECASE,
)

# ── Format D: list WITHOUT "hemolysis" keyword ───────────────────────────────
# "0%, 0%, 2.1%, ... against ... at ... concentrations of C1, C2, ... unit"
_LIST_AGAINST_RE = re.compile(
    r"([\d.<>≤%, ]+)\s+against\s+.*?\bat\s+(?:peptide\s+)?(?:concentrations?\s+of\s+)?"
    r"([\d.,\s]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)


def extract_hemolysis(text, mw_da, ref_conc_uM):
    """
    Extract (pct, conc_uM) from hemolysis text, trying multiple formats.
    For dose-response lists, picks the data point closest to ref_conc_uM.
    For LC50/HC50, converts via Hill equation (n=1).

    Returns: (pct, conc_uM, method) or None
    """
    if not isinstance(text, str):
        return None

    # ── Format A: single match ────────────────────────────────────────────
    for m in _HEM_SINGLE_RE.finditer(text):
        pct_s, conc_s, unit = m.group(1), m.group(2), m.group(3)
        try:
            pct  = float(_strip_op(pct_s))
            conc = float(_strip_op(conc_s))
        except ValueError:
            continue
        conc_uM = _to_uM(conc, unit, mw_da)
        if conc_uM and conc_uM > 0:
            return pct, conc_uM, "single"

    # ── Format B: list "X1%,...Xn% hemolysis at C1,...Cn unit" ────────────
    # Find the "hemolysis" keyword and extract % values before it, conc after
    m_hemol = re.search(r"hemol\w*\s+against", text, re.I)
    if m_hemol:
        before = text[:m_hemol.start()]
        after  = text[m_hemol.end():]

        pct_vals = [float(_strip_op(m.group(1))) for m in _LIST_PCT_RE.finditer(before)
                    if _strip_op(m.group(1))]

        # Find "at C1, C2, ... unit" after "against"
        m_at = re.search(
            r"\bat\s+(?:peptide\s+)?(?:concentrations?\s+of\s+)?"
            r"([\d.,\s]+(?:\s+and\s+[\d.]+)?)\s*([μu]g/mL|[μu]M)",
            after, re.I
        )
        if m_at and pct_vals:
            conc_block = m_at.group(1)
            unit = m_at.group(2)
            # Parse concentration values (handle "and")
            conc_block = conc_block.replace(" and ", ",")
            conc_vals = []
            for mc in _LIST_CONC_RE.finditer(conc_block):
                try:
                    conc_vals.append(float(mc.group(1)))
                except ValueError:
                    pass

            if conc_vals:
                # Pair % with concentration
                n_pairs = min(len(pct_vals), len(conc_vals))
                pairs = []
                for i in range(n_pairs):
                    c_uM = _to_uM(conc_vals[i], unit, mw_da)
                    if c_uM and c_uM > 0:
                        pairs.append((pct_vals[i], c_uM))

                if pairs:
                    # Pick the data point closest to reference concentration
                    best = min(pairs, key=lambda x: abs(x[1] - ref_conc_uM))
                    return best[0], best[1], "list"

    # ── Format D: list WITHOUT "hemolysis" ────────────────────────────────
    m_ag = _LIST_AGAINST_RE.search(text)
    if m_ag:
        pct_block = m_ag.group(1)
        conc_block = m_ag.group(2)
        unit = m_ag.group(3)

        pct_vals = [float(_strip_op(m.group(1)))
                    for m in _LIST_PCT_RE.finditer(pct_block)
                    if _strip_op(m.group(1))]
        conc_block_clean = conc_block.replace(" and ", ",")
        conc_vals = [float(m.group(1))
                     for m in _LIST_CONC_RE.finditer(conc_block_clean)]

        if pct_vals and conc_vals:
            n_pairs = min(len(pct_vals), len(conc_vals))
            pairs = []
            for i in range(n_pairs):
                c_uM = _to_uM(conc_vals[i], unit, mw_da)
                if c_uM and c_uM > 0:
                    pairs.append((pct_vals[i], c_uM))
            if pairs:
                best = min(pairs, key=lambda x: abs(x[1] - ref_conc_uM))
                return best[0], best[1], "list_nohem"

    # ── Format C: LC50 / HC50 → Hill equation conversion ─────────────────
    m_lc = _LC50_RE.search(text)
    if m_lc:
        lc_s, unit = m_lc.group(1), m_lc.group(2)
        try:
            lc_val = float(_strip_op(lc_s))
        except ValueError:
            return None
        lc_uM = _to_uM(lc_val, unit, mw_da)
        if lc_uM and lc_uM > 0:
            # Hill equation (n=1):  hemolysis = 100 * C / (C + LC50)
            pct_at_ref = 100.0 * ref_conc_uM / (ref_conc_uM + lc_uM)
            return pct_at_ref, ref_conc_uM, "lc50_hill"

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
    print("  Hemolysis RF Regression v2 — Expanded Features + Improved Parsing")
    print("=" * 72)

    # ── 1. Load training data ────────────────────────────────────────────────
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)

    # Add hydrophobic_sasa to training features
    feat["hydrophobic_sasa"] = feat["sasa"] - feat["psa"]

    base_cols = [c for c in FEATURE_COLS if c != "hydrophobic_sasa"]
    df = pd.merge(
        meta[["DRAMP_ID", "Hemolytic_Activity"]],
        feat[["DRAMP_ID"] + base_cols + ["hydrophobic_sasa"]],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=FEATURE_COLS)
    print(f"\n  Stapled AMPs with complete features: {len(df)}")

    # ── 2. Extract hemolysis (improved parsing) ──────────────────────────────
    # First pass: determine median reference concentration from single-match
    first_pass = df.apply(
        lambda r: extract_hemolysis(str(r["Hemolytic_Activity"]), r["weight"], 10.0),
        axis=1,
    )
    temp_concs = [x[1] for x in first_pass if x is not None and x[2] == "single"]
    ref_conc = float(np.median(temp_concs)) if temp_concs else 10.0
    print(f"  Reference concentration (median of single-match): {ref_conc:.1f} uM")

    # Second pass: use ref_conc for list/LC50 parsing
    extracted = df.apply(
        lambda r: extract_hemolysis(str(r["Hemolytic_Activity"]), r["weight"], ref_conc),
        axis=1,
    )
    df["pct_raw"]  = extracted.apply(lambda x: x[0] if x else None)
    df["conc_uM"]  = extracted.apply(lambda x: x[1] if x else None)
    df["method"]   = extracted.apply(lambda x: x[2] if x else None)

    df_parsed = df.dropna(subset=["pct_raw", "conc_uM"]).copy()

    # Count by method
    method_counts = df_parsed["method"].value_counts()
    print(f"\n  Parsing results by method:")
    for method, count in method_counts.items():
        print(f"    {method:>15}: {count}")
    print(f"    {'TOTAL':>15}: {len(df_parsed)}  (was 98 in v1)")

    # ── 3. Normalise to reference concentration ──────────────────────────────
    # For single/list entries: linear normalisation
    # For lc50_hill entries: already at ref_conc (no normalisation needed)
    df_parsed["pct_norm"] = df_parsed.apply(
        lambda r: r["pct_raw"] if r["method"] == "lc50_hill"
                  else r["pct_raw"] * (ref_conc / r["conc_uM"]),
        axis=1,
    ).clip(0, 100)

    print(f"\n  Normalised % hemolysis at {ref_conc:.1f} uM:")
    print(f"    mean = {df_parsed['pct_norm'].mean():.1f}%")
    print(f"    std  = {df_parsed['pct_norm'].std():.1f}%")
    print(f"    min  = {df_parsed['pct_norm'].min():.1f}%  /  max = {df_parsed['pct_norm'].max():.1f}%")

    # ── 4. Train RF ──────────────────────────────────────────────────────────
    X_train = df_parsed[FEATURE_COLS].values.astype(float)
    y_train = df_parsed["pct_norm"].values
    n_train = len(df_parsed)

    rf = RandomForestRegressor(n_estimators=300, random_state=42,
                                max_depth=None, min_samples_leaf=3,
                                n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = np.clip(cross_val_predict(rf, X_train, y_train, cv=cv), 0, 100)
    r_cv, _ = pearsonr(y_train, y_cv)
    r2_cv   = r2_score(y_train, y_cv)
    mae_cv  = mean_absolute_error(y_train, y_cv)
    rf.fit(X_train, y_train)

    print(f"\n  RF (18 features, n={n_train}):")
    print(f"    5-fold CV:  Pearson R = {r_cv:.3f}  |  R2 = {r2_cv:.3f}  "
          f"|  MAE = {mae_cv:.1f}%")

    # ── Feature importance ───────────────────────────────────────────────────
    importances = rf.feature_importances_
    imp_order   = np.argsort(importances)[::-1]
    print(f"\n  Top 10 feature importances:")
    for rank, idx in enumerate(imp_order[:10]):
        print(f"    {rank+1:>2}. {FEATURE_COLS[idx]:<22}  {importances[idx]:.4f}")

    # ── 5. Load ALL Buforin variant features ─────────────────────────────────
    test_entries = []

    # 5a. Buf WT
    test_entries.append({
        "pid": "Buf_WT",
        "features": np.array([[BUF_WT_FEATURES[f] for f in FEATURE_COLS]]),
        "stapled": False, "group": "native",
    })

    # 5b. 7 F10W variants
    f10w_df = pd.read_csv(F10W_FEAT)
    f10w_df["hydrophobic_sasa"] = f10w_df["sasa"] - f10w_df["psa"]
    for _, row in f10w_df.iterrows():
        test_entries.append({
            "pid": row["peptide_id"],
            "features": np.array([[row[f] for f in FEATURE_COLS]]),
            "stapled": True, "group": "F10W",
        })

    # 5c. 4 Buf variants
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

    # ── 6. Predict hemolysis ─────────────────────────────────────────────────
    for e in test_entries:
        e["pred_pct"] = float(np.clip(rf.predict(e["features"])[0], 0, 100))

        lit = LITERATURE_HEMO.get(e["pid"])
        if lit:
            lit_pct, mw, test_ugml = lit
            lit_conc_uM = test_ugml / mw * 1000.0
            e["lit_pct_raw"]  = lit_pct
            e["lit_conc_uM"]  = lit_conc_uM
            e["lit_pct_norm"] = lit_pct * (ref_conc / lit_conc_uM)
        else:
            e["lit_pct_raw"]  = None
            e["lit_conc_uM"]  = None
            e["lit_pct_norm"] = None

    # ── 7. Results table ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  {'Variant':<22}  {'Pred %':>7}  {'Lit(raw)':>9}  "
          f"{'Lit(norm)':>9}  {'Delta':>7}  {'Label'}")
    print(f"{'='*90}")
    for e in test_entries:
        dname = DISPLAY_LABELS.get(e["pid"], e["pid"]).replace("\n", " ")
        pred_s = f"{e['pred_pct']:.1f}%"
        if e["lit_pct_raw"] is not None:
            raw_s  = f"{e['lit_pct_raw']:.1f}%"
            norm_s = f"{e['lit_pct_norm']:.1f}%"
            delta  = e['pred_pct'] - e['lit_pct_norm']
            delta_s = f"{delta:+.1f}%"
        else:
            raw_s = norm_s = delta_s = "  —"
        tag = " *" if not e["stapled"] else ""
        print(f"  {dname:<22}  {pred_s:>7}  {raw_s:>9}  "
              f"{norm_s:>9}  {delta_s:>7}  {hemolysis_label(e['pred_pct'])}{tag}")
    print(f"{'='*90}")
    print(f"  * = non-stapled  |  Lit. at 50 ug/mL, normalised to {ref_conc:.1f} uM")
    print(f"  LC50/HC50 entries converted via Hill equation (n=1)")

    # ── 8. Correlation ───────────────────────────────────────────────────────
    paired = [(e["pred_pct"], e["lit_pct_norm"])
              for e in test_entries if e["lit_pct_norm"] is not None]
    pred_arr = np.array([p[0] for p in paired])
    lit_arr  = np.array([p[1] for p in paired])
    n_paired = len(paired)

    if n_paired >= 3:
        r_pl, p_pl = pearsonr(pred_arr, lit_arr)
        rho_pl, p_rho = spearmanr(pred_arr, lit_arr)
        mae_test = mean_absolute_error(lit_arr, pred_arr)
        print(f"\n  Predicted vs Literature (normalised, N={n_paired}):")
        print(f"    Pearson R  = {r_pl:.3f}  (p={p_pl:.3f})")
        print(f"    Spearman r = {rho_pl:.3f}  (p={p_rho:.3f})")
        print(f"    MAE        = {mae_test:.1f}%")

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
    # Color by extraction method
    method_colors = {"single": "#9467bd", "list": "#17becf",
                     "lc50_hill": "#bcbd22", "list_nohem": "#e377c2"}
    for method, color in method_colors.items():
        mask = df_parsed["method"] == method
        if mask.any():
            ax_cv.scatter(y_train[mask], y_cv[mask], alpha=0.55, s=45,
                          color=color, edgecolors="white", linewidths=0.4,
                          zorder=3, label=f"{method} (n={mask.sum()})")

    lo = max(0, min(y_train.min(), y_cv.min()) - 2)
    hi = min(100, max(y_train.max(), y_cv.max()) + 2)
    ax_cv.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")

    slope, intercept, *_ = stats.linregress(y_train, y_cv)
    x_grid = np.linspace(lo, hi, 200)
    ax_cv.plot(x_grid, slope * x_grid + intercept,
               "-", color="#d7191c", lw=1.5, label="Linear fit")

    ax_cv.text(0.05, 0.95,
               f"Pearson R = {r_cv:.2f}\nR\u00b2 = {r2_cv:.2f}\n"
               f"MAE = {mae_cv:.1f}%\nn = {n_train}\n"
               f"features = {len(FEATURE_COLS)}",
               transform=ax_cv.transAxes, fontsize=9, va="top",
               fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3",
                         facecolor="lightyellow", alpha=0.85))

    ax_cv.set_xlabel(f"Actual % Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=9)
    ax_cv.set_ylabel("Predicted % Hemolysis (5-fold CV)", fontsize=9)
    ax_cv.set_title("Panel A \u2014 Training Set (5-fold CV)\n"
                    "Color = extraction method",
                    fontsize=10, fontweight="bold")
    ax_cv.set_xlim(lo, hi); ax_cv.set_ylim(lo, hi)
    ax_cv.legend(fontsize=6.5, loc="lower right")
    ax_cv.set_aspect("equal", adjustable="box")
    ax_cv.grid(alpha=0.2, lw=0.5)

    # ── Panel B: Predicted vs Literature scatter ─────────────────────────────
    if n_paired >= 3:
        group_colors = {"F10W": "#4363d8", "NGC_variants": "#e6194b"}
        for e in test_entries:
            if e["lit_pct_norm"] is None:
                continue
            c = group_colors.get(e["group"], "#333")
            marker = "o" if e["group"] == "F10W" else "s"
            ax_sc.scatter(e["lit_pct_norm"], e["pred_pct"],
                          s=110, marker=marker, color=c,
                          edgecolors="k", linewidths=0.7, zorder=5)
            dname = DISPLAY_LABELS.get(e["pid"], e["pid"]).replace("\n", " ")
            x_off, y_off = 6, 4
            if e["pred_pct"] < 8 and e["lit_pct_norm"] < 5:
                y_off = -10
            ax_sc.annotate(dname, (e["lit_pct_norm"], e["pred_pct"]),
                           textcoords="offset points", xytext=(x_off, y_off),
                           fontsize=6.5, color=c, fontweight="bold")

        lo2, hi2 = 0, max(max(pred_arr), max(lit_arr)) * 1.15 + 2
        ax_sc.plot([lo2, hi2], [lo2, hi2], "k--", lw=0.8, alpha=0.5, label="y = x")

        sl2, ic2, *_ = stats.linregress(lit_arr, pred_arr)
        x_fit = np.linspace(lo2, hi2, 100)
        ax_sc.plot(x_fit, sl2 * x_fit + ic2,
                   color="#d7191c", lw=1.3, ls="--", alpha=0.7)

        ax_sc.text(0.05, 0.95,
                   f"Pearson R  = {r_pl:.2f}\n"
                   f"Spearman \u03c1 = {rho_pl:.2f}\n"
                   f"MAE = {mae_test:.1f}%\n"
                   f"N = {n_paired}",
                   transform=ax_sc.transAxes, fontsize=9, va="top",
                   fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3",
                             facecolor="lightyellow", alpha=0.85))

        ax_sc.set_xlabel(f"Literature % Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=9)
        ax_sc.set_ylabel("RF Predicted % Hemolysis", fontsize=9)
        ax_sc.set_title("Panel B \u2014 Predicted vs Literature\n"
                        "Buforin Variants",
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

    bar_labels  = [DISPLAY_LABELS.get(e["pid"], e["pid"]) for e in ordered]
    bar_pred    = [e["pred_pct"] for e in ordered]
    bar_lit     = [e["lit_pct_norm"] for e in ordered]
    bar_lit_raw = [e["lit_pct_raw"] for e in ordered]
    bar_stapled = [e["stapled"] for e in ordered]

    y_pos = np.arange(len(ordered))
    bar_h = 0.35

    colors_pred = [HEM_COLORS[hemolysis_label(p)] for p in bar_pred]
    bars_p = ax_bar.barh(y_pos + bar_h/2, bar_pred, bar_h,
                         color=colors_pred, edgecolor="k", linewidth=0.6,
                         label="RF Predicted")

    lit_vals_plot = [v if v is not None else 0 for v in bar_lit]
    lit_colors = [HEM_COLORS[hemolysis_label(v)] if v is not None else "#eeeeee"
                  for v in bar_lit]
    bars_l = ax_bar.barh(y_pos - bar_h/2, lit_vals_plot, bar_h,
                         color=lit_colors, edgecolor="k", linewidth=0.6,
                         alpha=0.6, hatch="//",
                         label=f"Literature (norm. {ref_conc:.0f} uM)")

    for i, v in enumerate(bar_lit):
        if v is None:
            bars_l[i].set_visible(False)

    for i, is_st in enumerate(bar_stapled):
        if not is_st:
            bars_p[i].set_hatch("xxx")

    for i, (pred, lit, lit_raw, is_st) in enumerate(
            zip(bar_pred, bar_lit, bar_lit_raw, bar_stapled)):
        tag = " *" if not is_st else ""
        ax_bar.text(pred + 0.3, y_pos[i] + bar_h/2,
                    f" {pred:.1f}%{tag}",
                    va="center", fontsize=7, fontweight="bold")
        if lit is not None:
            ax_bar.text(lit + 0.3, y_pos[i] - bar_h/2,
                        f" {lit:.1f}% (raw:{lit_raw:.1f}%@50ug/mL)",
                        va="center", fontsize=6, color="#555")

    for xv, lbl, lc in [(5, "5%", "#2ca02c"), (15, "15%", "#ff7f0e"),
                         (40, "40%", "#d62728")]:
        ax_bar.axvline(xv, color=lc, linestyle="--", lw=0.9, alpha=0.7)
        ax_bar.text(xv, -0.7, lbl, ha="center", fontsize=6.5, color=lc)

    ax_bar.axhline(0.5, color="#555", lw=0.8, ls=":")
    ax_bar.axhline(4.5, color="#555", lw=0.8, ls=":")
    ax_bar.axhline(7.5, color="#555", lw=0.8, ls=":")

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_labels, fontsize=8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("% Hemolysis", fontsize=9)
    x_max = max(max(bar_pred), max(v for v in bar_lit if v is not None)) * 1.3 + 3
    ax_bar.set_xlim(0, x_max)
    ax_bar.set_title("Panel C \u2014 All Buforin Variants\n"
                     "Predicted (solid) vs Literature (hatched)",
                     fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7.5, loc="lower right")
    ax_bar.grid(axis="x", alpha=0.2, lw=0.5)

    # ── Suptitle ─────────────────────────────────────────────────────────────
    fig.suptitle(
        "Hemolysis RF Regression v2 \u2014 Expanded Features + Improved Parsing\n"
        f"{len(FEATURE_COLS)} StaPep features  |  n_train = {n_train} "
        f"(was 98)  |  ref = {ref_conc:.1f} uM  |  "
        f"LC50/HC50 via Hill eqn (n=1)",
        fontsize=12, fontweight="bold",
    )

    hem_patches = [mpatches.Patch(color=c, label=t) for t, c in HEM_COLORS.items()]
    hem_patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                       hatch="xxx", label="Non-stapled (native)"))
    fig.legend(handles=hem_patches, loc="lower center", ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, title="Hemolysis Level", title_fontsize=8)

    out = BASE / "buf_hemolysis_regression_v2.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # ── Comparison with v1 ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  v1 vs v2 comparison:")
    print(f"{'='*72}")
    print(f"  {'Metric':<30}  {'v1':>10}  {'v2':>10}  {'Change':>10}")
    print(f"  {'='*62}")
    print(f"  {'Features':.<30}  {'14':>10}  {f'{len(FEATURE_COLS)}':>10}  {'+4':>10}")
    print(f"  {'Training samples':.<30}  {'98':>10}  {f'{n_train}':>10}  {f'+{n_train-98}':>10}")
    print(f"  {'CV Pearson R':.<30}  {'0.844':>10}  {f'{r_cv:.3f}':>10}  "
          f"{f'{r_cv-0.844:+.3f}':>10}")
    if n_paired >= 3:
        print(f"  {'Test Pearson R':.<30}  {'0.596':>10}  {f'{r_pl:.3f}':>10}  "
              f"{f'{r_pl-0.596:+.3f}':>10}")
        print(f"  {'Test Spearman rho':.<30}  {'0.317':>10}  {f'{rho_pl:.3f}':>10}  "
              f"{f'{rho_pl-0.317:+.3f}':>10}")
    print()


if __name__ == "__main__":
    main()
