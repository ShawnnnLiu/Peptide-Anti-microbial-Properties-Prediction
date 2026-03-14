#!/usr/bin/env python3
"""
run_mic_svm.py
==============
4-class MIC-tier classifier for stapled AMPs vs E. coli using RBF-SVM.

Tiers (μM):
  0 – Very strong : MIC <  2 μM
  1 – Moderate    : MIC  2–10 μM  (Strong + Moderate combined)
  2 – Weak        : MIC > 10 μM

Model : RBF-SVM  + Platt scaling  + GridSearchCV (C × γ)
Data  : 147 AMPs with E. coli MIC from stapled_amps.csv (decoys excluded)

Usage
-----
  conda run -n esm_env python run_mic_svm.py
"""

from __future__ import annotations

import re, warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                      cross_val_score)
from sklearn.pipeline       import Pipeline
from sklearn.impute         import SimpleImputer
from sklearn.metrics        import (classification_report, confusion_matrix,
                                    balanced_accuracy_score)

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]

BINS        = [0, 2, 10, np.inf]
TIER_LABELS = ["Very strong (<2 μM)", "Moderate (2–10 μM)", "Weak (>10 μM)"]
SHORT       = ["VeryStrong", "Moderate", "Weak"]

SVM_GRID = {
    "svc__C"    : [0.1, 1, 10, 100, 1000],
    "svc__gamma": ["scale", 1e-3, 1e-2, 0.1],
}

TEST_NAMES = ["Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
              "Mag20", "Mag25", "Mag31", "Mag36"]


# ── MIC parsing ───────────────────────────────────────────────────────────────
def parse_ecoli_mic(amp_csv: Path) -> pd.DataFrame:
    amp = pd.read_csv(amp_csv)
    pat = re.compile(
        r'(?:Escherichia\s+coli|E\.?\s*coli)[^(]*'
        r'\(MIC[^=]*=\s*([\d.]+)\s*([\u03bcμ\xb5]M|[\u03bcμ\xb5]g/mL)',
        re.I)
    rows = []
    for _, row in amp.iterrows():
        m = pat.search(str(row.get("Target_Organism", "")))
        if m:
            rows.append({"DRAMP_ID": row["DRAMP_ID"],
                         "mic_raw" : float(m.group(1)),
                         "unit"    : m.group(2)})
    return pd.DataFrame(rows)


def build_training(amp_csv: Path, feat_csv: Path):
    """Returns (X, y, meta_df)."""
    mic   = parse_ecoli_mic(amp_csv)
    feat  = pd.read_csv(feat_csv)
    # weight is already in FEATURES — avoid duplicating it in the merge select
    merge_cols = list(dict.fromkeys(["DRAMP_ID"] + FEATURES))
    df    = mic.merge(feat[merge_cols], on="DRAMP_ID", how="inner").reset_index(drop=True)

    df["mic_uM"] = df["mic_raw"].astype(float)
    mask = df["unit"].str.lower().str.contains("g/ml").values
    df.loc[mask, "mic_uM"] = (df["mic_raw"].values[mask] * 1000
                               / df["weight"].values[mask])

    df = df.dropna(subset=["mic_uM"] + FEATURES).reset_index(drop=True)
    df["tier"] = pd.cut(df["mic_uM"], bins=BINS,
                        labels=[0, 1, 2]).astype(int)
    return (df[FEATURES].astype(float),
            df["tier"],
            df[["DRAMP_ID", "mic_uM", "tier"]])


# ── Pipeline ──────────────────────────────────────────────────────────────────
def make_pipe(C=1, gamma="scale") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        C=C, gamma=gamma, random_state=42)),
    ])


def make_gs(cv: int) -> GridSearchCV:
    return GridSearchCV(
        make_pipe(), SVM_GRID,
        cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        scoring="balanced_accuracy",
        n_jobs=-1, refit=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main(cv_folds: int = 5) -> None:
    print("\n" + "=" * 65)
    print("  MIC-Tier Classifier  [RBF-SVM + Platt Scaling]")
    print("  E. coli   |  StaPep features  |  4-class")
    print("=" * 65)

    X, y, meta = build_training(
        STAPEP / "stapled_amps.csv",
        STAPEP / "stapled_amps_features.csv",
    )

    print(f"\n  Training : {len(X)} AMPs with E. coli MIC")
    print(f"  MIC range: {meta['mic_uM'].min():.2f} – "
          f"{meta['mic_uM'].max():.1f} μM  "
          f"(median {meta['mic_uM'].median():.2f} μM)")
    print("\n  Class distribution:")
    for i, lbl in enumerate(TIER_LABELS):
        cnt = (y == i).sum()
        bar = "█" * int(cnt / len(y) * 30)
        print(f"    {lbl:<25}  {cnt:>3}  {bar}")

    # ── GridSearchCV ─────────────────────────────────────────────────────
    print(f"\n  Running GridSearchCV ({cv_folds}-fold, balanced accuracy) …")
    gs = make_gs(cv_folds)
    gs.fit(X, y)
    print(f"  Best C={gs.best_params_['svc__C']}  "
          f"γ={gs.best_params_['svc__gamma']}")
    print(f"  Best CV balanced accuracy : {gs.best_score_:.3f}")

    # ── Per-fold report (refit with best params) ──────────────────────────
    cv_bal  = cross_val_score(
        make_pipe(gs.best_params_["svc__C"], gs.best_params_["svc__gamma"]),
        X, y, scoring="balanced_accuracy",
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), n_jobs=-1)
    cv_acc  = cross_val_score(
        make_pipe(gs.best_params_["svc__C"], gs.best_params_["svc__gamma"]),
        X, y, scoring="accuracy",
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42), n_jobs=-1)

    print(f"\n  {'Metric':<30}  {'Mean':>7}  ±  {'Std':>6}")
    print(f"  {'─'*30}  {'─'*7}     {'─'*6}")
    print(f"  {'Balanced Accuracy (CV)':<30}  {cv_bal.mean():>7.3f}  ±  {cv_bal.std():>6.3f}")
    print(f"  {'Accuracy (CV)':<30}  {cv_acc.mean():>7.3f}  ±  {cv_acc.std():>6.3f}")

    # ── Per-class report ──────────────────────────────────────────────────
    skf = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []
    for tr, te in skf.split(X, y):
        p = make_pipe(gs.best_params_["svc__C"], gs.best_params_["svc__gamma"])
        p.fit(X.iloc[tr], y.iloc[tr])
        y_true_all.extend(y.iloc[te].tolist())
        y_pred_all.extend(p.predict(X.iloc[te]).tolist())

    print("\n  Per-class report (aggregated over 5 folds):")
    print(classification_report(y_true_all, y_pred_all,
                                target_names=SHORT, zero_division=0))

    print("  Confusion matrix (rows=true, cols=pred):")
    cm  = confusion_matrix(y_true_all, y_pred_all)
    hdr = "".join(f"{s:>12}" for s in SHORT)
    print(f"  {'':15}{hdr}")
    for i, rv in enumerate(cm):
        print(f"  {SHORT[i]:<15}" + "".join(f"{v:>12}" for v in rv))

    # ── Test predictions ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Predictions — 8 Novel Test Peptides")
    print("=" * 65)

    test  = pd.read_csv(STAPEP / "test_stapled_features.csv")
    X_te  = test[FEATURES].astype(float)

    # Refit on full training data with best params
    best_pipe = make_pipe(gs.best_params_["svc__C"],
                          gs.best_params_["svc__gamma"])
    best_pipe.fit(X, y)

    probs = best_pipe.predict_proba(X_te)   # (8, 4)
    preds = best_pipe.predict(X_te)

    hdr = (f"\n  {'Peptide':<22}  {'Predicted Tier':<25}  "
           + "  ".join(f"{s:>10}" for s in SHORT))
    print(hdr)
    print(f"  {'─'*22}  {'─'*25}  " + "  ".join(["─"*10]*4))
    for i, row in test.iterrows():
        pid      = row["peptide_id"]
        tier_str = TIER_LABELS[preds[i]]
        conf     = probs[i].max()
        p_str    = "  ".join(f"{p:>10.3f}" for p in probs[i])
        flag     = " ★" if conf >= 0.70 else ""
        print(f"  {pid:<22}  {tier_str:<25}  {p_str}{flag}")

    print(f"\n  ★ = top-class confidence ≥ 0.70")
    print(f"  Column order: {' | '.join(SHORT)}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cv-folds", type=int, default=5)
    args = p.parse_args()
    main(cv_folds=args.cv_folds)
