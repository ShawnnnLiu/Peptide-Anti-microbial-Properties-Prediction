#!/usr/bin/env python3
"""
run_mic_classifier.py
=====================
4-class MIC-tier Random Forest classifier for stapled AMPs vs E. coli.
Compares three feature sets:
  1. StaPep      — 17 MD / sequence features
  2. QSAR        — 12 descriptors from the 2016 PNAS SVM paper
  3. QSAR+StaPep — all 29 features combined

Tier definitions (μM):
  0 – Very strong : MIC <  2
  1 – Strong      : MIC  2–5
  2 – Moderate    : MIC  5–10
  3 – Weak        : MIC > 10

NOTE: Decoys are excluded — they have no measured MIC.
"""

from __future__ import annotations

import re, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.metrics         import (classification_report, confusion_matrix,
                                     balanced_accuracy_score, make_scorer)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

# ── Feature columns ───────────────────────────────────────────────────────────
SP_COLS = [
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
QSAR_TEST_NAME_MAP = {
    "Buf(i+4)_12"          : "Buf12",
    "Buf(i+4)_13"          : "Buf13",
    "Buf(i+4)_13_Q9K"      : "Buf13_Q9K",
    "Buf(i+4)_12_V15K_L19K": "Buf12_V15K_L19K",
    "Mag_20": "Mag20", "Mag_25": "Mag25",
    "Mag_31": "Mag31", "Mag_36": "Mag36",
}
TEST_ORDER = ["Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
              "Mag20",  "Mag25", "Mag31",     "Mag36"]

BINS        = [0, 2, 10, np.inf]
TIER_LABELS = ["Very strong (<2 μM)", "Moderate (2–10 μM)", "Weak (>10 μM)"]
SHORT       = ["VeryStrong", "Moderate", "Weak"]


# ── MIC parsing ───────────────────────────────────────────────────────────────
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


def _attach_mic(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw MIC → mic_uM and assign tier. df must have mic_raw, unit, weight."""
    df = df.copy().reset_index(drop=True)
    df["mic_uM"] = df["mic_raw"].astype(float)
    mask = df["unit"].str.lower().str.contains("g/ml").values
    df.loc[mask, "mic_uM"] = df["mic_raw"].values[mask] * 1000 / df["weight"].values[mask]
    df = df.dropna(subset=["mic_uM"]).reset_index(drop=True)
    df["tier"] = pd.cut(df["mic_uM"], bins=BINS, labels=[0, 1, 2]).astype(int)
    return df


# ── Data loading ──────────────────────────────────────────────────────────────
def load_feature_sets() -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Returns dict: { feature_set_name : (X, y) }
    Keys: "StaPep", "QSAR", "QSAR+StaPep"
    """
    mic   = parse_ecoli_mic(STAPEP / "stapled_amps.csv")
    sp    = pd.read_csv(STAPEP / "stapled_amps_features.csv")
    qsar  = pd.read_csv(STAPEP / "qsar_stapled_amps.csv").rename(
                columns={"peptide_id": "DRAMP_ID"})

    # StaPep — join MIC ↔ StaPep features
    sp_merge_cols = list(dict.fromkeys(["DRAMP_ID"] + SP_COLS))
    df_sp = _attach_mic(
        mic.merge(sp[sp_merge_cols], on="DRAMP_ID", how="inner"))

    # QSAR — join MIC ↔ QSAR features  (weight needed for ug/mL conversion)
    df_qsar = _attach_mic(
        mic.merge(sp[["DRAMP_ID", "weight"]], on="DRAMP_ID", how="inner")
           .merge(qsar[["DRAMP_ID"] + QSAR_COLS], on="DRAMP_ID", how="inner"))

    # QSAR+StaPep — triple join
    df_both = _attach_mic(
        mic.merge(sp[sp_merge_cols], on="DRAMP_ID", how="inner")
           .merge(qsar[["DRAMP_ID"] + QSAR_COLS], on="DRAMP_ID", how="inner"))

    return {
        "StaPep"     : (df_sp[SP_COLS].astype(float),   df_sp["tier"]),
        "QSAR"       : (df_qsar[QSAR_COLS].astype(float), df_qsar["tier"]),
        "QSAR+StaPep": (df_both[SP_COLS + QSAR_COLS].astype(float), df_both["tier"]),
    }


def load_test_sets() -> dict[str, pd.DataFrame]:
    """Returns { feature_set_name : X_test (8 × n_feat) aligned to TEST_ORDER }."""
    sp_test   = pd.read_csv(STAPEP / "test_stapled_features.csv")
    qsar_test = pd.read_csv(STAPEP / "qsar_stapled_test.csv")
    qsar_test["peptide_id"] = (qsar_test["peptide_id"]
                                .map(QSAR_TEST_NAME_MAP)
                                .fillna(qsar_test["peptide_id"]))
    # Align both to TEST_ORDER
    sp_test   = sp_test.set_index("peptide_id").reindex(TEST_ORDER)
    qsar_test = qsar_test.set_index("peptide_id").reindex(TEST_ORDER)

    return {
        "StaPep"     : sp_test[SP_COLS].astype(float),
        "QSAR"       : qsar_test[QSAR_COLS].astype(float),
        "QSAR+StaPep": pd.concat([sp_test[SP_COLS],
                                   qsar_test[QSAR_COLS]], axis=1).astype(float),
    }


# ── Model ─────────────────────────────────────────────────────────────────────
def make_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])


# ── Main ──────────────────────────────────────────────────────────────────────
def main(cv_folds: int = 5) -> None:
    print("\n" + "=" * 72)
    print("  MIC-Tier Classifier  [Random Forest]   —   E. coli  |  4-class")
    print("  Feature sets: StaPep  |  QSAR  |  QSAR+StaPep")
    print("=" * 72)

    feat_sets  = load_feature_sets()
    test_sets  = load_test_sets()

    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=42)

    # ── CV comparison table ───────────────────────────────────────────────
    print(f"\n  {'Feature Set':<15}  {'n':>4}  {'Bal-Acc':>9}  {'Accuracy':>9}")
    print(f"  {'─'*15}  {'─'*4}  {'─'*9}  {'─'*9}")

    cv_results: dict = {}
    for name, (X, y) in feat_sets.items():
        bal = cross_val_score(make_pipe(), X, y, scoring="balanced_accuracy",
                              cv=skf, n_jobs=-1)
        acc = cross_val_score(make_pipe(), X, y, scoring="accuracy",
                              cv=skf, n_jobs=-1)
        cv_results[name] = {"X": X, "y": y, "bal": bal, "acc": acc}
        print(f"  {name:<15}  {len(y):>4}  "
              f"{bal.mean():.3f}±{bal.std():.3f}  "
              f"{acc.mean():.3f}±{acc.std():.3f}")

    # ── Per-class report + confusion matrix per feature set ───────────────
    for name, res in cv_results.items():
        X, y = res["X"], res["y"]
        y_true_all, y_pred_all = [], []
        for tr, te in skf.split(X, y):
            p = make_pipe()
            p.fit(X.iloc[tr], y.iloc[tr])
            y_true_all.extend(y.iloc[te].tolist())
            y_pred_all.extend(p.predict(X.iloc[te]).tolist())

        print(f"\n  ── {name} per-class report ──")
        print(classification_report(y_true_all, y_pred_all,
                                    target_names=SHORT, zero_division=0))
        cm  = confusion_matrix(y_true_all, y_pred_all)
        hdr = "".join(f"{s:>12}" for s in SHORT)
        print(f"  {'':15}{hdr}")
        for i, rv in enumerate(cm):
            print(f"  {SHORT[i]:<15}" + "".join(f"{v:>12}" for v in rv))

    # ── Test predictions — side-by-side ───────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  Test Peptide Predictions  (refit on full training data)")
    print(f"{'=' * 72}")

    all_probs: dict[str, np.ndarray] = {}
    all_preds: dict[str, np.ndarray] = {}
    for name, (X, y) in feat_sets.items():
        pipe = make_pipe()
        pipe.fit(X, y)
        X_te = test_sets[name]
        all_probs[name] = pipe.predict_proba(X_te)
        all_preds[name] = pipe.predict(X_te)

    # Header
    col_w = 12
    hdr = f"\n  {'Peptide':<22}"
    for fs in feat_sets:
        hdr += f"  {fs:^{col_w*4+6}}"
    print(hdr)

    sub = f"  {'':22}"
    for _ in feat_sets:
        sub += "  " + "  ".join(f"{s:>{col_w}}" for s in SHORT)
    print(sub)
    print("  " + "─" * (22 + len(feat_sets) * (col_w * 4 + 8)))

    for i, pid in enumerate(TEST_ORDER):
        row = f"  {pid:<22}"
        for name in feat_sets:
            probs = all_probs[name][i]
            pred  = all_preds[name][i]
            parts = []
            for j, p in enumerate(probs):
                cell = f"{p:.3f}"
                if j == pred:
                    cell = f"[{p:.3f}]"   # bracket the predicted class
                parts.append(f"{cell:>{col_w}}")
            row += "  " + "  ".join(parts)
        print(row)

    print(f"\n  [x.xxx] = predicted tier  |  columns: {' | '.join(SHORT)}")
    print(f"  Feature sets: {' | '.join(feat_sets.keys())}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cv-folds", type=int, default=5)
    args = p.parse_args()
    main(cv_folds=args.cv_folds)
