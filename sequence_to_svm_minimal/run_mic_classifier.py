#!/usr/bin/env python3
"""
run_mic_classifier.py
=====================
4-class MIC-tier classifier for stapled AMPs against E. coli.

Tier definitions (μM):
  0 – Very strong : MIC <  2 μM
  1 – Strong      : MIC  2–5 μM
  2 – Moderate    : MIC  5–10 μM
  3 – Weak        : MIC > 10 μM

Training data  : stapled_amps_features.csv  (StaPep MD features)
Labels source  : stapled_amps.csv           (Target_Organism MIC field)
Model          : Random Forest (sklearn), stratified 5-fold CV
Test peptides  : test_stapled_features.csv  (8 novel stapled peptides)

NOTE: Decoys are entirely excluded — they have no measured MIC.
      This model answers: "Given an AMP, how potent is it vs E. coli?"
"""

from __future__ import annotations

import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, make_scorer,
)

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

AMP_CSV   = STAPEP / "stapled_amps.csv"
FEAT_CSV  = STAPEP / "stapled_amps_features.csv"
TEST_CSV  = STAPEP / "test_stapled_features.csv"

# ── StaPep feature columns ───────────────────────────────────────────────────
FEATURES = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]

# ── MIC tier definitions ─────────────────────────────────────────────────────
BINS   = [0, 2, 10, np.inf]
TIER_LABELS = [
    "Very strong (<2 μM)",
    "Moderate (2–10 μM)",
    "Weak (>10 μM)",
]
SHORT = ["VeryStrong", "Moderate", "Weak"]


# ════════════════════════════════════════════════════════════════════════════
# 1.  Parse E. coli MIC from stapled_amps.csv
# ════════════════════════════════════════════════════════════════════════════
def parse_ecoli_mic(amp_csv: Path) -> pd.DataFrame:
    """
    Return DataFrame with columns [DRAMP_ID, mic_raw, unit].
    Catches 'E.coli', 'E. coli', 'Escherichia coli' + any strain suffix.
    Skips censored values (MIC > X).
    """
    amp = pd.read_csv(amp_csv)
    # Unicode μ  = \u03bc  or Latin-1 \xb5  — handle both
    pat = re.compile(
        r'(?:Escherichia\s+coli|E\.?\s*coli)[^(]*'
        r'\(MIC[^=]*=\s*([\d.]+)\s*([\u03bcμ\xb5]M|[\u03bcμ\xb5]g/mL)',
        re.I,
    )
    rows = []
    for _, row in amp.iterrows():
        txt = str(row.get("Target_Organism", ""))
        m = pat.search(txt)
        if m:
            rows.append({
                "DRAMP_ID": row["DRAMP_ID"],
                "mic_raw" : float(m.group(1)),
                "unit"    : m.group(2),
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
# 2.  Build labelled training set
# ════════════════════════════════════════════════════════════════════════════
def build_training_data(
    amp_csv: Path, feat_csv: Path
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns:
        X       – feature matrix (n_samples × n_features)
        y       – integer tier labels (0-3)
        y_names – human-readable tier name per sample
    """
    mic_df = parse_ecoli_mic(amp_csv)
    feat   = pd.read_csv(feat_csv)

    # Merge on DRAMP_ID
    df = mic_df.merge(feat[["DRAMP_ID"] + FEATURES], on="DRAMP_ID", how="inner")

    # Convert μg/mL → μM  (μM = μg/mL × 1000 / MW_Da)
    # For unit conversion we need MW; use 'weight' column already in feat
    feat_mw = pd.read_csv(feat_csv)[["DRAMP_ID","weight"]].rename(
        columns={"weight": "MW_Da"})
    df = df.merge(feat_mw, on="DRAMP_ID", how="left")

    df["mic_uM"] = df["mic_raw"].copy().astype(float)
    mask_ug = df["unit"].str.lower().str.contains("g/ml")
    df.loc[mask_ug, "mic_uM"] = (
        df.loc[mask_ug, "mic_raw"] * 1000 / df.loc[mask_ug, "MW_Da"]
    )

    # Drop rows where conversion failed
    df = df.dropna(subset=["mic_uM"] + FEATURES)

    # Assign tiers
    df["tier"] = pd.cut(df["mic_uM"], bins=BINS, labels=[0, 1, 2]).astype(int)
    df["tier_name"] = pd.cut(df["mic_uM"], bins=BINS, labels=TIER_LABELS)

    X = df[FEATURES].astype(float)
    y = df["tier"]
    return X, y, df[["DRAMP_ID","mic_uM","tier","tier_name"]]


# ════════════════════════════════════════════════════════════════════════════
# 3.  Model pipeline
# ════════════════════════════════════════════════════════════════════════════
def make_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(
            n_estimators   = 500,
            max_features   = "sqrt",
            min_samples_leaf = 2,
            class_weight   = "balanced",
            random_state   = 42,
            n_jobs         = -1,
        )),
    ])


# ════════════════════════════════════════════════════════════════════════════
# 4.  Cross-validation
# ════════════════════════════════════════════════════════════════════════════
def run_cv(X: pd.DataFrame, y: pd.Series) -> dict:
    pipe = make_pipeline()
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "balanced_acc": make_scorer(balanced_accuracy_score),
        "accuracy":     "accuracy",
    }
    res = cross_validate(pipe, X, y, cv=cv, scoring=scoring,
                         return_train_score=True)
    return res


# ════════════════════════════════════════════════════════════════════════════
# 5.  Feature importance
# ════════════════════════════════════════════════════════════════════════════
def feature_importance(pipe: Pipeline) -> pd.DataFrame:
    rf = pipe.named_steps["rf"]
    imp = pd.DataFrame({
        "feature"   : FEATURES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp


# ════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("\n" + "=" * 65)
    print("  MIC-Tier Classifier  (E. coli, StaPep features, Random Forest)")
    print("=" * 65)

    # ── Build training data ──────────────────────────────────────────────
    X, y, meta = build_training_data(AMP_CSV, FEAT_CSV)

    print(f"\n  Training set: {len(X)} AMPs with E. coli MIC")
    print(f"  MIC range   : {meta['mic_uM'].min():.2f} – {meta['mic_uM'].max():.1f} μM  "
          f"(median {meta['mic_uM'].median():.2f} μM)")
    print("\n  Class distribution:")
    vc = meta["tier_name"].value_counts().sort_index()
    for name, cnt in vc.items():
        bar = "█" * int(cnt / max(vc) * 25)
        print(f"    {name:<25}  {cnt:>3}  {bar}")

    # ── Cross-validation ─────────────────────────────────────────────────
    print("\n  Running stratified 5-fold cross-validation …")
    cv_res = run_cv(X, y)

    print(f"\n  {'Metric':<30}  {'Mean':>7}  {'±':>1}  {'Std':>6}")
    print(f"  {'─'*30}  {'─'*7}  {'─'*1}  {'─'*6}")
    for key, label in [
        ("test_balanced_acc", "Balanced Accuracy (CV)"),
        ("train_balanced_acc","Balanced Accuracy (train)"),
        ("test_accuracy",     "Accuracy (CV)"),
    ]:
        vals = cv_res[key]
        print(f"  {label:<30}  {vals.mean():>7.3f}  ±  {vals.std():>6.3f}")

    # ── Full-data fit for predictions & importances ──────────────────────
    pipe = make_pipeline()
    pipe.fit(X, y)

    # Per-class CV classification report (refit needed, do hold-out style)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for tr, te in cv.split(X, y):
        p = make_pipeline()
        p.fit(X.iloc[tr], y.iloc[tr])
        y_true_all.extend(y.iloc[te].tolist())
        y_pred_all.extend(p.predict(X.iloc[te]).tolist())

    print("\n  Per-class report (aggregated over 5 folds):")
    print(classification_report(
        y_true_all, y_pred_all,
        target_names=SHORT,
        zero_division=0,
    ))

    print("  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true_all, y_pred_all)
    hdr = "".join(f"{s:>12}" for s in SHORT)
    print(f"  {'':15}{hdr}")
    for i, row_vals in enumerate(cm):
        row_str = "".join(f"{v:>12}" for v in row_vals)
        print(f"  {SHORT[i]:<15}{row_str}")

    # ── Feature importance ───────────────────────────────────────────────
    imp = feature_importance(pipe)
    print("\n  Feature importances (full-data model):")
    print(f"  {'Feature':<22}  {'Importance':>10}")
    print(f"  {'─'*22}  {'─'*10}")
    for _, r in imp.iterrows():
        bar = "█" * int(r["importance"] / imp["importance"].max() * 20)
        print(f"  {r['feature']:<22}  {r['importance']:>10.4f}  {bar}")

    # ── Predict on 8 test peptides ───────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Predictions for 8 Novel Test Peptides")
    print("=" * 65)

    test = pd.read_csv(TEST_CSV)
    X_test = test[FEATURES].astype(float)

    probs     = pipe.predict_proba(X_test)      # (8, 4)
    preds     = pipe.predict(X_test)            # (8,)

    print(f"\n  {'Peptide':<22}  {'Predicted Tier':<25}  "
          + "  ".join(f"{s:>10}" for s in SHORT))
    print(f"  {'─'*22}  {'─'*25}  " + "  ".join(["─"*10]*4))
    for i, row in test.iterrows():
        pid  = row["peptide_id"]
        pred = preds[i]
        name = TIER_LABELS[pred]
        conf = probs[i].max()
        prob_str = "  ".join(f"{p:>10.3f}" for p in probs[i])
        flag = " ★" if conf >= 0.70 else ""
        print(f"  {pid:<22}  {name:<25}  {prob_str}{flag}")

    print(f"\n  ★ = top-class confidence ≥ 0.70")
    print(f"  Column order: {' | '.join(SHORT)}")
    print(f"\n  Note: These 8 peptides have no measured E. coli MIC.")
    print(f"        The model predicts which potency tier they are MOST LIKELY")
    print(f"        to fall into based on their StaPep feature profile.\n")


if __name__ == "__main__":
    main()
