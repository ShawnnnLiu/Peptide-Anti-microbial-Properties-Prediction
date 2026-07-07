#!/usr/bin/env python3
"""
predict_buf_hemolysis_lyticity.py
==================================
Hemolysis regression using ONLY lyticity_index.

Trains quadratic and exponential regressions on lyticity_index → % hemolysis,
reports training CV performance and Buforin test accuracy.

Rationale: lyticity_index (amphipathic helical patterning score) is the
single most important feature (~40% importance in the 18-feature RF).
Simple parametric models on lyticity capture the dose-response
relationship and avoid overprediction caused by feature interactions
in tree-based models.
"""

import re, math, warnings, sys, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error
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

# ── Buf WT features ────────────────────────────────────────────────────────────
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
    "hydrophobic_sasa": 2038.292 - 1064.217,
}
BUF_WT_LYTICITY = BUF_WT_FEATURES["lyticity_index"]

# ── Literature hemolysis (raw % at 50 ug/mL) ─────────────────────────────────
#    name,               MW_Da,   raw_hemo%,  test_conc_ugml
LITERATURE_HEMO = {
    "Buf_i4_16_F10W": (12.6,  2429.9, 50.0),
    "Buf_i4_14_F10W": ( 2.9,  2453.8, 50.0),
    "Buf_i4_4_F10W":  ( 2.4,  2523.0, 50.0),
    "Buf_i4_3_F10W":  ( 3.1,  2579.1, 50.0),
    "Buf_i7_9_F10W":  (57.0,  2500.0, 50.0),
    "Buf_i7_6_F10W":  ( 3.0,  2637.2, 50.0),
    "Buf_i7_1_F10W":  ( 2.3,  2551.0, 50.0),
    "Buf12":          ( 9.23, 2491.93, 50.0),
    "Buf13":          ( 3.84, 2514.96, 50.0),
}

DISPLAY_LABELS = {
    "Buf_WT":          "Buf WT\n(native)",
    "Buf_i4_16_F10W":  "Buf(i+4)16\n(F10W)",
    "Buf_i4_14_F10W":  "Buf(i+4)14\n(F10W)",
    "Buf_i4_4_F10W":   "Buf(i+4)4\n(F10W)",
    "Buf_i4_3_F10W":   "Buf(i+4)3\n(F10W)",
    "Buf_i7_9_F10W":   "Buf(i+7)9\n(F10W)",
    "Buf_i7_6_F10W":   "Buf(i+7)6\n(F10W)",
    "Buf_i7_1_F10W":   "Buf(i+7)1\n(F10W)",
    "Buf12":           "Buf(i+4)12",
    "Buf13":           "Buf(i+4)13",
    "Buf13_Q9K":       "Buf(i+4)13\nQ9K",
    "Buf12_V15K_L19K": "Buf(i+4)12\nV15K,L19K",
}


# ══════════════════════════════════════════════════════════════════════════════
# HEMOLYSIS EXTRACTION (from v2)
# ══════════════════════════════════════════════════════════════════════════════
def _strip_op(s):
    return re.sub(r'^[<>≤≥\s]+', '', s)

def _to_uM(val, unit, mw_da):
    unit = unit.lower().strip()
    if unit in ("μm", "um"):
        return val
    if unit in ("μg/ml", "ug/ml") and mw_da > 0:
        return val / mw_da * 1000.0
    return None

_HEM_SINGLE_RE = re.compile(
    r"([<>≤]?\s*[\d.]+)\s*%\s*hemol"
    r"[^.;#\n]*?"
    r"\bat\s+(?:peptide\s+)?(?:concentration\s+of\s+)?"
    r"([<>≤]?\s*[\d.]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)
_LIST_PCT_RE  = re.compile(r"([<>≤]?\s*[\d.]+)\s*%")
_LIST_CONC_RE = re.compile(r"([\d.]+)")
_LC50_RE = re.compile(
    r"(?:LC50|HC50|LC\s*50|HC\s*50)\s*[=:>< ]+\s*([><=]?\s*[\d.]+)\s*([μu]M|[μu]g/mL)",
    re.IGNORECASE,
)
_LIST_AGAINST_RE = re.compile(
    r"([\d.<>≤%, ]+)\s+against\s+.*?\bat\s+(?:peptide\s+)?(?:concentrations?\s+of\s+)?"
    r"([\d.,\s]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

def extract_hemolysis(text, mw_da, ref_conc_uM):
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
            return pct, conc_uM, "single"

    m_hemol = re.search(r"hemol\w*\s+against", text, re.I)
    if m_hemol:
        before = text[:m_hemol.start()]
        after  = text[m_hemol.end():]
        pct_vals = [float(_strip_op(m.group(1))) for m in _LIST_PCT_RE.finditer(before)
                    if _strip_op(m.group(1))]
        m_at = re.search(
            r"\bat\s+(?:peptide\s+)?(?:concentrations?\s+of\s+)?"
            r"([\d.,\s]+(?:\s+and\s+[\d.]+)?)\s*([μu]g/mL|[μu]M)",
            after, re.I
        )
        if m_at and pct_vals:
            conc_block = m_at.group(1).replace(" and ", ",")
            unit = m_at.group(2)
            conc_vals = [float(mc.group(1)) for mc in _LIST_CONC_RE.finditer(conc_block)]
            if conc_vals:
                n_pairs = min(len(pct_vals), len(conc_vals))
                pairs = []
                for i in range(n_pairs):
                    c_uM = _to_uM(conc_vals[i], unit, mw_da)
                    if c_uM and c_uM > 0:
                        pairs.append((pct_vals[i], c_uM))
                if pairs:
                    best = min(pairs, key=lambda x: abs(x[1] - ref_conc_uM))
                    return best[0], best[1], "list"

    m_ag = _LIST_AGAINST_RE.search(text)
    if m_ag:
        pct_block = m_ag.group(1)
        conc_block = m_ag.group(2).replace(" and ", ",")
        unit = m_ag.group(3)
        pct_vals = [float(_strip_op(m.group(1))) for m in _LIST_PCT_RE.finditer(pct_block)
                    if _strip_op(m.group(1))]
        conc_vals = [float(m.group(1)) for m in _LIST_CONC_RE.finditer(conc_block)]
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

    m_lc = _LC50_RE.search(text)
    if m_lc:
        lc_s, unit = m_lc.group(1), m_lc.group(2)
        try:
            lc_val = float(_strip_op(lc_s))
        except ValueError:
            return None
        lc_uM = _to_uM(lc_val, unit, mw_da)
        if lc_uM and lc_uM > 0:
            pct_at_ref = 100.0 * ref_conc_uM / (ref_conc_uM + lc_uM)
            return pct_at_ref, ref_conc_uM, "lc50_hill"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 75)
    print("  Hemolysis Regression: Lyticity Index Only")
    print("  Quadratic fit vs 18-feature RF baseline")
    print("=" * 75)

    # ── 1. Load + parse training data ─────────────────────────────────────────
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    feat["hydrophobic_sasa"] = feat["sasa"] - feat["psa"]

    ALL_FEATS = [
        "length", "weight", "hydrophobic_index", "charge", "aromaticity",
        "isoelectric_point", "fraction_arginine", "fraction_lysine",
        "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
        "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
        "hydrophobic_sasa",
    ]

    df = pd.merge(
        meta[["DRAMP_ID", "Hemolytic_Activity"]],
        feat[["DRAMP_ID"] + ALL_FEATS],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=ALL_FEATS)

    # First pass for ref_conc
    first_pass = df.apply(
        lambda r: extract_hemolysis(str(r["Hemolytic_Activity"]), r["weight"], 10.0),
        axis=1,
    )
    temp_concs = [x[1] for x in first_pass if x is not None and x[2] == "single"]
    ref_conc = float(np.median(temp_concs)) if temp_concs else 10.0

    # Second pass
    extracted = df.apply(
        lambda r: extract_hemolysis(str(r["Hemolytic_Activity"]), r["weight"], ref_conc),
        axis=1,
    )
    df["pct_raw"]  = extracted.apply(lambda x: x[0] if x else None)
    df["conc_uM"]  = extracted.apply(lambda x: x[1] if x else None)
    df["method"]   = extracted.apply(lambda x: x[2] if x else None)
    df = df.dropna(subset=["pct_raw", "conc_uM"]).copy()

    df["pct_norm"] = df.apply(
        lambda r: r["pct_raw"] if r["method"] == "lc50_hill"
                  else r["pct_raw"] * (ref_conc / r["conc_uM"]),
        axis=1,
    ).clip(0, 100)

    n_train = len(df)
    lyti_train = df["lyticity_index"].values
    y_train = df["pct_norm"].values
    X_train_all = df[ALL_FEATS].values.astype(float)

    print(f"\n  Training set: n = {n_train}")
    print(f"  Reference concentration: {ref_conc:.1f} uM")
    print(f"  Hemolysis range: {y_train.min():.1f}% - {y_train.max():.1f}%")

    # ── 2. Train models ──────────────────────────────────────────────────────
    kf = KFold(5, shuffle=True, random_state=42)

    # Model A: Quadratic on lyticity
    quad_model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        Ridge(alpha=1.0),
    )
    cv_pred_quad = cross_val_predict(quad_model, lyti_train.reshape(-1, 1), y_train, cv=kf)
    cv_pred_quad = np.clip(cv_pred_quad, 0, 100)
    quad_model.fit(lyti_train.reshape(-1, 1), y_train)

    # Model B: Exponential on lyticity:  y = a * exp(b * x) + c
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    def exp_func_clipped(x, a, b, c):
        return np.clip(a * np.exp(b * x) + c, 0, 100)

    # Fit on full training data (with sensible initial guesses)
    try:
        popt_exp, _ = curve_fit(exp_func, lyti_train, y_train,
                                p0=[0.01, 0.005, 0.0],
                                maxfev=20000,
                                bounds=([0, 0, -50], [1e6, 0.05, 50]))
        exp_ok = True
        print(f"\n  Exponential fit: y = {popt_exp[0]:.4g} * exp({popt_exp[1]:.5f} * x) + ({popt_exp[2]:.2f})")
    except Exception as e:
        print(f"\n  [WARN] Exponential fit failed on full data: {e}")
        exp_ok = False

    # Manual 5-fold CV for exponential (can't use cross_val_predict with curve_fit)
    cv_pred_exp = np.full(len(y_train), np.nan)
    if exp_ok:
        for tr_idx, te_idx in kf.split(lyti_train):
            try:
                popt_fold, _ = curve_fit(exp_func, lyti_train[tr_idx], y_train[tr_idx],
                                         p0=popt_exp, maxfev=20000,
                                         bounds=([0, 0, -50], [1e6, 0.05, 50]))
                cv_pred_exp[te_idx] = exp_func_clipped(lyti_train[te_idx], *popt_fold)
            except Exception:
                cv_pred_exp[te_idx] = exp_func_clipped(lyti_train[te_idx], *popt_exp)
    cv_pred_exp = np.clip(np.nan_to_num(cv_pred_exp, nan=0), 0, 100)

    # Model C: RF on lyticity only (1 feature)
    rf_lyti_model = RandomForestRegressor(n_estimators=300, random_state=42,
                                          min_samples_leaf=3, n_jobs=-1)
    cv_pred_rf1 = cross_val_predict(rf_lyti_model, lyti_train.reshape(-1, 1), y_train, cv=kf)
    cv_pred_rf1 = np.clip(cv_pred_rf1, 0, 100)
    rf_lyti_model.fit(lyti_train.reshape(-1, 1), y_train)

    # Model D: Baseline 18-feature RF (for comparison)
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42,
                                      min_samples_leaf=3, n_jobs=-1)
    cv_pred_rf = cross_val_predict(rf_model, X_train_all, y_train, cv=kf)
    cv_pred_rf = np.clip(cv_pred_rf, 0, 100)
    rf_model.fit(X_train_all, y_train)

    # ── 3. Training CV metrics ────────────────────────────────────────────────
    def metrics(y_true, y_pred, label):
        r, p = pearsonr(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"\n  {label}:")
        print(f"    Pearson R  = {r:.3f}  (p = {p:.2e})")
        print(f"    Spearman r = {rho:.3f}")
        print(f"    MAE        = {mae:.2f}%")
        return r, rho, mae

    print(f"\n{'='*75}")
    print(f"  TRAINING 5-FOLD CV")
    print(f"{'='*75}")
    r_quad, rho_quad, mae_quad = metrics(y_train, cv_pred_quad, "Quadratic (lyticity only)")
    r_exp,  rho_exp,  mae_exp  = metrics(y_train, cv_pred_exp,  "Exponential (lyticity only)")
    r_rf1,  rho_rf1,  mae_rf1  = metrics(y_train, cv_pred_rf1,  "RF (lyticity only)")
    r_rf,   rho_rf,   mae_rf   = metrics(y_train, cv_pred_rf,   "RF (18 features, baseline)")

    # ── 4. Load Buforin test variants ─────────────────────────────────────────
    f10w = pd.read_csv(F10W_FEAT)
    f10w["hydrophobic_sasa"] = f10w["sasa"] - f10w["psa"]
    tst = pd.read_csv(TEST_FEAT)
    tst["hydrophobic_sasa"] = tst["sasa"] - tst["psa"]

    # Build test list: all 12 Buforin variants
    test_variants = []

    # Buf WT
    test_variants.append({
        "name": "Buf_WT",
        "lyticity": BUF_WT_LYTICITY,
        "features": np.array([BUF_WT_FEATURES[f] for f in ALL_FEATS]),
        "lit_raw": None, "lit_norm": None,
    })

    # F10W variants
    for _, row in f10w.iterrows():
        pid = row["peptide_id"]
        lit = LITERATURE_HEMO.get(pid)
        lit_raw, lit_norm = None, None
        if lit:
            hemo_raw, mw, conc_ugml = lit
            conc_uM = conc_ugml / mw * 1000.0
            lit_raw = hemo_raw
            lit_norm = hemo_raw * ref_conc / conc_uM
        test_variants.append({
            "name": pid,
            "lyticity": row["lyticity_index"],
            "features": np.array([row[f] for f in ALL_FEATS]),
            "lit_raw": lit_raw, "lit_norm": lit_norm,
        })

    # Buf12, Buf13, Q9K, V15K_L19K
    for _, row in tst.iterrows():
        pid = row["peptide_id"]
        if not pid.lower().startswith("buf"):
            continue
        lit = LITERATURE_HEMO.get(pid)
        lit_raw, lit_norm = None, None
        if lit:
            hemo_raw, mw, conc_ugml = lit
            conc_uM = conc_ugml / mw * 1000.0
            lit_raw = hemo_raw
            lit_norm = hemo_raw * ref_conc / conc_uM
        test_variants.append({
            "name": pid,
            "lyticity": row["lyticity_index"],
            "features": np.array([row[f] for f in ALL_FEATS]),
            "lit_raw": lit_raw, "lit_norm": lit_norm,
        })

    # ── 5. Predict ────────────────────────────────────────────────────────────
    X_test_lyti = np.array([v["lyticity"] for v in test_variants]).reshape(-1, 1)
    X_test_all  = np.array([v["features"] for v in test_variants])

    pred_quad = np.clip(quad_model.predict(X_test_lyti), 0, 100)
    pred_rf1  = np.clip(rf_lyti_model.predict(X_test_lyti), 0, 100)
    pred_rf   = np.clip(rf_model.predict(X_test_all), 0, 100)
    if exp_ok:
        pred_exp = np.clip(exp_func(X_test_lyti.ravel(), *popt_exp), 0, 100)
    else:
        pred_exp = np.full(len(test_variants), np.nan)

    for i, v in enumerate(test_variants):
        v["pred_quad"] = pred_quad[i]
        v["pred_exp"]  = pred_exp[i]
        v["pred_rf1"]  = pred_rf1[i]
        v["pred_rf"]   = pred_rf[i]

    # ── 6. Buforin test metrics (variants with literature only) ──────────────
    with_lit = [v for v in test_variants if v["lit_norm"] is not None]
    lit_arr  = np.array([v["lit_norm"] for v in with_lit])
    pq_arr   = np.array([v["pred_quad"] for v in with_lit])
    pe_arr   = np.array([v["pred_exp"]  for v in with_lit])
    pr1_arr  = np.array([v["pred_rf1"]  for v in with_lit])
    pr_arr   = np.array([v["pred_rf"] for v in with_lit])

    print(f"\n{'='*75}")
    print(f"  BUFORIN TEST SET (n = {len(with_lit)} variants with literature)")
    print(f"{'='*75}")
    r_bq, rho_bq, mae_bq   = metrics(lit_arr, pq_arr,  "Quadratic (lyticity only)")
    r_be, rho_be, mae_be    = metrics(lit_arr, pe_arr,  "Exponential (lyticity only)")
    r_b1, rho_b1, mae_b1    = metrics(lit_arr, pr1_arr, "RF (lyticity only)")
    r_br, rho_br, mae_br    = metrics(lit_arr, pr_arr,  "RF (18 features, baseline)")

    # ── 7. Per-variant table ─────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  PER-VARIANT PREDICTIONS")
    print(f"{'='*75}")
    print(f"\n  {'Variant':<22} {'Lyticity':>8} {'Lit(norm)':>10}"
          f" {'Quad':>7} {'Err_Q':>7} {'Exp':>7} {'Err_E':>7} {'RF-1':>7} {'Err_1':>7} {'RF-18':>7} {'Err_18':>7}")
    print(f"  {'-'*112}")

    for v in test_variants:
        name = v["name"]
        lyti = v["lyticity"]
        ln = f"{v['lit_norm']:.1f}%" if v["lit_norm"] is not None else "---"
        pq = v["pred_quad"]
        pe = v["pred_exp"]
        p1 = v["pred_rf1"]
        pr = v["pred_rf"]

        if v["lit_norm"] is not None:
            eq = f"{pq - v['lit_norm']:+.1f}%"
            ee = f"{pe - v['lit_norm']:+.1f}%" if not np.isnan(pe) else "---"
            e1 = f"{p1 - v['lit_norm']:+.1f}%"
            er = f"{pr - v['lit_norm']:+.1f}%"
        else:
            eq, ee, e1, er = "---", "---", "---", "---"

        pe_s = f"{pe:.1f}%" if not np.isnan(pe) else "---"
        print(f"  {name:<22} {lyti:>7.0f}  {ln:>10}  {pq:>6.1f}% {eq:>7}  {pe_s:>6} {ee:>7}  {p1:>6.1f}% {e1:>7}  {pr:>6.1f}% {er:>7}")

    # ── 8. Figure (3 panels) ─────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 7))

    # ── Panel A: Training CV — Actual vs Predicted ────────────────────────────
    ax1.scatter(y_train, cv_pred_quad, alpha=0.5, s=40, color="#2c7bb6",
                edgecolors="white", linewidths=0.3, zorder=3,
                label=f"Quadratic (R={r_quad:.2f}, MAE={mae_quad:.1f}%)")
    if exp_ok:
        ax1.scatter(y_train, cv_pred_exp, alpha=0.5, s=40, color="#ff7f0e",
                    edgecolors="white", linewidths=0.3, zorder=3, marker="s",
                    label=f"Exponential (R={r_exp:.2f}, MAE={mae_exp:.1f}%)")
    ax1.scatter(y_train, cv_pred_rf1, alpha=0.4, s=35, color="#9467bd",
                edgecolors="white", linewidths=0.3, zorder=2, marker="D",
                label=f"RF-lyti (R={r_rf1:.2f}, MAE={mae_rf1:.1f}%)")
    ax1.scatter(y_train, cv_pred_rf, alpha=0.3, s=30, color="#d62728",
                edgecolors="white", linewidths=0.3, zorder=2, marker="^",
                label=f"RF-18 (R={r_rf:.2f}, MAE={mae_rf:.1f}%)")

    lim = [0, 105]
    ax1.plot(lim, lim, "--", color="#333", lw=1, alpha=0.5)
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("Actual % Hemolysis", fontsize=11)
    ax1.set_ylabel("Predicted % Hemolysis (5-fold CV)", fontsize=11)
    ax1.set_title(f"Panel A \u2014 Training CV (n={n_train})", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.2)
    ax1.set_aspect("equal")

    # ── Panel B: Lyticity curve with training data ────────────────────────────
    x_grid = np.linspace(100, 850, 500)
    y_curve_quad = np.clip(quad_model.predict(x_grid.reshape(-1, 1)), 0, 100)

    ax2.scatter(lyti_train, y_train, alpha=0.4, s=35, color="#888888",
                edgecolors="white", linewidths=0.3, zorder=3,
                label=f"Training (n={n_train})")
    ax2.plot(x_grid, y_curve_quad, "-", color="#2c7bb6", lw=2.5, zorder=4,
             label="Quadratic fit")

    if exp_ok:
        y_curve_exp = np.clip(exp_func(x_grid, *popt_exp), 0, 100)
        ax2.plot(x_grid, y_curve_exp, "-", color="#ff7f0e", lw=2.5, zorder=4,
                 label="Exponential fit")

    y_curve_rf1 = np.clip(rf_lyti_model.predict(x_grid.reshape(-1, 1)), 0, 100)
    ax2.plot(x_grid, y_curve_rf1, "-", color="#9467bd", lw=2.5, zorder=4,
             label="RF (lyticity only)", alpha=0.8)

    # Overlay Buforin test points
    for v in test_variants:
        if v["lit_norm"] is not None:
            marker = "^" if "i7" in v["name"] else "o"
            color = "#e6194b" if "i7" in v["name"] else "#4363d8"
            ax2.scatter(v["lyticity"], v["lit_norm"], s=100, marker=marker,
                        color=color, edgecolors="k", linewidths=0.8, zorder=6)

    # Zone shading
    ax2.axvspan(80, 400, alpha=0.06, color="#2ca02c", zorder=1)
    ax2.axvspan(400, 700, alpha=0.06, color="#ff7f0e", zorder=1)
    ax2.axvspan(700, 860, alpha=0.06, color="#d62728", zorder=1)
    ax2.axvline(400, color="#333", lw=0.8, ls="--", alpha=0.4)
    ax2.axvline(700, color="#333", lw=0.8, ls="--", alpha=0.4)

    ax2.set_xlabel("Lyticity Index", fontsize=11)
    ax2.set_ylabel(f"% Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=11)
    ax2.set_title("Panel B \u2014 Quadratic vs Exponential\n"
                  "Training + Buforin overlay", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.2)
    ax2.set_ylim(-5, 105)
    ax2.set_xlim(80, 860)

    # Custom legend for Buforin markers
    from matplotlib.lines import Line2D
    custom = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4363d8",
               markeredgecolor="k", markersize=8, label="Buforin i+4 (lit)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6194b",
               markeredgecolor="k", markersize=8, label="Buforin i+7 (lit)"),
    ]
    base_handles, base_labels = ax2.get_legend_handles_labels()
    ax2.legend(handles=base_handles + custom,
               labels=base_labels + ["Buforin i+4 (lit)", "Buforin i+7 (lit)"],
               fontsize=7.5, loc="upper left")

    # ── Panel C: Buforin predictions bar chart ────────────────────────────────
    n_vars = len(test_variants)
    y_pos = np.arange(n_vars)
    bar_h = 0.19  # narrower for 4 models

    labels = [DISPLAY_LABELS.get(v["name"], v["name"]) for v in test_variants]

    ax3.barh(y_pos - 1.5*bar_h, [v["pred_quad"] for v in test_variants],
             bar_h, color="#2c7bb6", edgecolor="k", linewidth=0.5,
             label="Quadratic", zorder=3)
    if exp_ok:
        ax3.barh(y_pos - 0.5*bar_h, [v["pred_exp"] for v in test_variants],
                 bar_h, color="#ff7f0e", edgecolor="k", linewidth=0.5,
                 label="Exponential", zorder=3)
    ax3.barh(y_pos + 0.5*bar_h, [v["pred_rf1"] for v in test_variants],
             bar_h, color="#9467bd", edgecolor="k", linewidth=0.5,
             label="RF-lyti", zorder=3)
    ax3.barh(y_pos + 1.5*bar_h, [v["pred_rf"] for v in test_variants],
             bar_h, color="#d62728", edgecolor="k", linewidth=0.5,
             alpha=0.5, label="RF-18", zorder=3)

    # Literature diamonds
    for i, v in enumerate(test_variants):
        if v["lit_norm"] is not None:
            ax3.plot(v["lit_norm"], i, "D", color="#2ca02c", ms=9,
                     markeredgecolor="k", markeredgewidth=0.8, zorder=5)
    ax3.plot([], [], "D", color="#2ca02c", ms=9, markeredgecolor="k",
             markeredgewidth=0.8, label="Literature (norm)")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel(f"% Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=10)
    ax3.set_title(f"Panel C \u2014 Buforin Predictions (n={len(with_lit)})",
                  fontsize=11, fontweight="bold")
    ax3.legend(fontsize=7, loc="lower right")
    ax3.grid(axis="x", alpha=0.2)

    # Accuracy box for Buforin test set
    acc_text = (
        f"Buforin Test Accuracy (n={len(with_lit)})\n"
        f"{'Model':<14} {'R':>5}  {'MAE':>6}\n"
        f"{'-'*28}\n"
        f"{'Quadratic':<14} {r_bq:>5.3f}  {mae_bq:>5.1f}%\n"
        f"{'Exponential':<14} {r_be:>5.3f}  {mae_be:>5.1f}%\n"
        f"{'RF-lyti':<14} {r_b1:>5.3f}  {mae_b1:>5.1f}%\n"
        f"{'RF-18':<14} {r_br:>5.3f}  {mae_br:>5.1f}%"
    )
    ax3.text(0.98, 0.02, acc_text, transform=ax3.transAxes,
             fontsize=7.5, fontfamily="monospace",
             verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                       edgecolor="#999", alpha=0.9),
             zorder=10)

    fig.suptitle(
        "Hemolysis Prediction: Lyticity-Only Models vs RF-18 Baseline\n"
        f"Training n={n_train} | Buforin test n={len(with_lit)}",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out = BASE / "buf_hemolysis_lyticity_only.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  SUMMARY")
    print(f"{'='*75}")
    print(f"                     {'Quadratic':>12}  {'Exponential':>12}  {'RF-lyti':>12}  {'RF-18':>12}")
    print(f"  Training CV R      {r_quad:>12.3f}  {r_exp:>12.3f}  {r_rf1:>12.3f}  {r_rf:>12.3f}")
    print(f"  Training CV MAE    {mae_quad:>11.1f}%  {mae_exp:>11.1f}%  {mae_rf1:>11.1f}%  {mae_rf:>11.1f}%")
    print(f"  Buforin R          {r_bq:>12.3f}  {r_be:>12.3f}  {r_b1:>12.3f}  {r_br:>12.3f}")
    print(f"  Buforin MAE        {mae_bq:>11.1f}%  {mae_be:>11.1f}%  {mae_b1:>11.1f}%  {mae_br:>11.1f}%")
    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
