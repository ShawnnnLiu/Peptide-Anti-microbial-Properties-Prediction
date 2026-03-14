#!/usr/bin/env python3
"""
run_stapep_svm.py
=================
RBF-SVM classifier trained on stapled peptides (AMP vs Decoy), following
the approach of the 2016 PNAS SVM paper:

  • Kernel        : RBF  (non-linear; evidenced by 46.5% support-vector ratio)
  • Normalisation : Z-score per feature (StandardScaler — paper-faithful)
  • Probabilities : Platt scaling  (SVC probability=True → P(+1) output)
  • Tuning        : GridSearchCV over C and gamma, 5-fold stratified CV

Three feature-set combinations:
  1. StaPep      — StaPep MD / sequence features (17)
  2. QSAR        — 12 descriptors from 2016 PNAS SVM paper
  3. QSAR+StaPep — QSAR ∪ StaPep

Usage
-----
  conda run -n esm_env python run_stapep_svm.py [--cv-folds N]
"""

import warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.svm           import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                     cross_val_score)
from sklearn.pipeline  import Pipeline
from sklearn.impute    import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

PATHS = {
    "amp_stapep"  : STAPEP / "stapled_amps_features.csv",
    "decoy_stapep": STAPEP / "stapled_decoys.csv",
    "amp_qsar"    : STAPEP / "qsar_stapled_amps.csv",
    "decoy_qsar"  : STAPEP / "qsar_stapled_decoys.csv",
    "test_stapep" : STAPEP / "test_stapled_features.csv",
    "test_qsar"   : STAPEP / "qsar_stapled_test.csv",
}

# ── Feature column lists ───────────────────────────────────────────────────────
STAPEP_COLS = [
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

TEST_NAMES = [
    "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    "Mag20",  "Mag25", "Mag31",     "Mag36",
]

QSAR_TEST_NAME_MAP = {
    "Buf(i+4)_12"          : "Buf12",
    "Buf(i+4)_13"          : "Buf13",
    "Buf(i+4)_13_Q9K"      : "Buf13_Q9K",
    "Buf(i+4)_12_V15K_L19K": "Buf12_V15K_L19K",
    "Mag_20"               : "Mag20",
    "Mag_25"               : "Mag25",
    "Mag_31"               : "Mag31",
    "Mag_36"               : "Mag36",
}

# ── SVM hyperparameter grid (paper-style RBF tuning) ──────────────────────────
# C     : regularisation strength — wider range covers hard- and soft-margin regimes
# gamma : RBF bandwidth — 'scale' = 1/(n_features*X.var()), covers smooth→sharp boundaries
SVM_PARAM_GRID = {
    "svc__C"    : [0.1, 1, 10, 100, 1000],
    "svc__gamma": ["scale", 1e-3, 1e-2, 0.1],
}


# ── Utilities ─────────────────────────────────────────────────────────────────
def _present(df: pd.DataFrame, cols: list) -> list:
    return [c for c in cols if c in df.columns]


def _dedup(primary: list, secondary: list) -> list:
    seen = set(primary)
    return [c for c in secondary if c not in seen]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_training():
    """Return (amp_sp, dec_sp, amp_qsar, dec_qsar)."""
    # StaPep
    amp_sp = pd.read_csv(PATHS["amp_stapep"]).rename(
        columns={"DRAMP_ID": "peptide_id", "Hiden_Sequence": "sequence"})
    dec_sp = pd.read_csv(PATHS["decoy_stapep"]).rename(
        columns={"COMPOUND_ID": "peptide_id", "SEQUENCE": "sequence"})
    amp_sp["label"]   = 1;  dec_sp["label"]   = 0
    amp_sp["join_id"] = amp_sp["peptide_id"].astype(str)
    dec_sp["join_id"] = (dec_sp.reset_index(drop=True).index + 1).astype(str)
    amp_sp = amp_sp.dropna(subset=_present(amp_sp, STAPEP_COLS), how="all")
    dec_sp = dec_sp.dropna(subset=_present(dec_sp, STAPEP_COLS), how="all")

    # QSAR
    amp_qsar = pd.DataFrame(); dec_qsar = pd.DataFrame()
    if PATHS["amp_qsar"].exists():
        amp_qsar = pd.read_csv(PATHS["amp_qsar"])
        amp_qsar["label"]   = 1
        amp_qsar["join_id"] = amp_qsar["peptide_id"].astype(str)
    else:
        print("  ⚠  AMP QSAR file missing — run extract_stapep_qsar.py first")

    if PATHS["decoy_qsar"].exists():
        dec_qsar = pd.read_csv(PATHS["decoy_qsar"])
        dec_qsar["label"]   = 0
        dec_qsar["join_id"] = (dec_qsar.reset_index(drop=True).index + 1).astype(str)
    else:
        print("  ⚠  Decoy QSAR file missing — run extract_stapep_qsar.py first")

    return amp_sp, dec_sp, amp_qsar, dec_qsar


def load_test():
    """Return (test_sp, test_qsar)."""
    test_sp = pd.DataFrame(); test_qsar = pd.DataFrame()

    if PATHS["test_stapep"].exists():
        test_sp = pd.read_csv(PATHS["test_stapep"])
    else:
        print("  ⚠  Test StaPep missing — run run_test_stapep_md.py in WSL")

    if PATHS["test_qsar"].exists():
        test_qsar = pd.read_csv(PATHS["test_qsar"])
        test_qsar["peptide_id"] = (test_qsar["peptide_id"]
                                   .map(QSAR_TEST_NAME_MAP)
                                   .fillna(test_qsar["peptide_id"]))
    else:
        print("  ⚠  Test QSAR missing — run extract_stapep_qsar.py first")

    return test_sp, test_qsar


# ── Feature matrix builders ────────────────────────────────────────────────────
def _concat_merge(amp_frames, dec_frames, col_sets):
    all_cols = []
    for cs in col_sets:
        all_cols += _dedup(all_cols, cs)

    def _merge_group(frames, col_sets_local):
        b_cols = _present(frames[0], _dedup([], col_sets_local[0]))
        merged = frames[0][["join_id", "label"] + b_cols].copy()
        used   = set(b_cols)
        for df, cs in zip(frames[1:], col_sets_local[1:]):
            new_cs = [c for c in cs if c not in used and c in df.columns]
            if new_cs:
                merged = merged.merge(df[["join_id"] + new_cs],
                                      on="join_id", how="inner")
            used.update(new_cs)
        return merged

    train    = pd.concat([_merge_group(amp_frames, col_sets),
                          _merge_group(dec_frames, col_sets)], ignore_index=True)
    fin_cols = [c for c in all_cols if c in train.columns]
    return train[fin_cols].values.astype(float), train["label"].values, pd.Index(fin_cols)


def build_sp_matrix(amp_sp, dec_sp):
    cols  = list(dict.fromkeys(_present(amp_sp, STAPEP_COLS) +
                               _present(dec_sp, STAPEP_COLS)))
    train = pd.concat([amp_sp[["label"] + cols],
                       dec_sp[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, pd.Index(cols)


def build_qsar_matrix(amp_qsar, dec_qsar):
    cols  = list(dict.fromkeys(_present(amp_qsar, QSAR_COLS) +
                               _present(dec_qsar, QSAR_COLS)))
    train = pd.concat([amp_qsar[["label"] + cols],
                       dec_qsar[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, pd.Index(cols)


def build_qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar):
    sp_cols   = _present(amp_sp, STAPEP_COLS)
    qsar_only = _dedup(sp_cols, QSAR_COLS)
    return _concat_merge([amp_sp, amp_qsar], [dec_sp, dec_qsar],
                         [sp_cols, qsar_only])


# ── SVM pipeline & tuning ─────────────────────────────────────────────────────
def _base_pipe(feature_names: pd.Index) -> Pipeline:
    """Impute → Z-score → RBF-SVM (paper-faithful normalisation)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf",
                        probability=True,        # Platt scaling → P(+1)
                        class_weight="balanced",
                        random_state=42)),
    ])


def make_svm_gs(folds: int, feature_names: pd.Index) -> GridSearchCV:
    """GridSearchCV wrapping the SVM pipeline — mirrors paper's grid search."""
    return GridSearchCV(
        _base_pipe(feature_names),
        SVM_PARAM_GRID,
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────
def cv_score_svm(X: np.ndarray, y: np.ndarray,
                 label: str, folds: int,
                 feature_names: pd.Index) -> dict:
    """
    Fit GridSearchCV on (X, y); report CV-AUC for best params
    plus Average Precision recomputed with those params.
    """
    gs = make_svm_gs(folds, feature_names)
    gs.fit(X, y)

    best_idx  = gs.best_index_
    fold_aucs = np.array([
        gs.cv_results_[f"split{i}_test_score"][best_idx]
        for i in range(folds)
    ])

    # Recompute AP with best (C, gamma)
    best_C     = gs.best_params_["svc__C"]
    best_gamma = gs.best_params_["svc__gamma"]
    pipe_ap    = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42,
                        C=best_C, gamma=best_gamma)),
    ])
    ap_scores = cross_val_score(
        pipe_ap, X, y,
        scoring="average_precision",
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        n_jobs=-1,
    )

    return {
        "label"    : label,
        "n"        : len(y),
        "auc_mean" : fold_aucs.mean(),
        "auc_std"  : fold_aucs.std(),
        "ap_mean"  : ap_scores.mean(),
        "ap_std"   : ap_scores.std(),
        "best_C"   : best_C,
        "best_gamma": best_gamma,
    }


def fit_predict_svm(X_train: np.ndarray, y_train: np.ndarray,
                    X_test:  np.ndarray, folds: int,
                    feature_names: pd.Index) -> np.ndarray:
    """GridSearch → refit on all training data → P(AMP=1) for test."""
    gs = make_svm_gs(folds, feature_names)
    gs.fit(X_train, y_train)
    return gs.predict_proba(X_test)[:, 1]


# ── Test-feature lookup ───────────────────────────────────────────────────────
def build_test_matrix(feature_names: pd.Index, *dfs) -> np.ndarray:
    rows = []
    for name in TEST_NAMES:
        vals = []
        for col in feature_names:
            v = np.nan
            for df in dfs:
                if df.empty or "peptide_id" not in df.columns or col not in df.columns:
                    continue
                row = df[df["peptide_id"] == name]
                if not row.empty:
                    candidate = row.iloc[0].get(col, np.nan)
                    if not pd.isna(candidate):
                        v = float(candidate)
                        break
            vals.append(v)
        rows.append(vals)
    return np.array(rows)


# ── Display helpers ───────────────────────────────────────────────────────────
def print_cv_table(results: list) -> None:
    col1_w = max(len(r["label"]) for r in results) + 2
    print(f"\n  {'Model':<{col1_w}}  {'AUC':>12}   {'Avg Prec':>12}   "
          f"{'n':>5}   {'Best C':>8}  {'Best γ':>10}")
    print(f"  {'─'*col1_w}  {'─'*12}   {'─'*12}   {'─'*5}   {'─'*8}  {'─'*10}")
    for r in results:
        gamma_str = (f"{r['best_gamma']:.4g}"
                     if isinstance(r["best_gamma"], float)
                     else str(r["best_gamma"]))
        print(f"  {r['label']:<{col1_w}}  "
              f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}   "
              f"{r['ap_mean']:.3f} ± {r['ap_std']:.3f}   "
              f"{r['n']:>5}   {r['best_C']:>8}  {gamma_str:>10}")


def print_prediction_table(probs_dict: dict) -> None:
    models  = list(probs_dict.keys())
    pep_w   = max(len(n) for n in TEST_NAMES) + 1
    mod_w   = max(max(len(m) for m in models), 9) + 2
    sep     = "─" * (pep_w + 1 + len(models) * (mod_w + 1) + 18)

    # Header
    hdr = f"  {'Peptide':<{pep_w}}"
    for m in models:
        hdr += f" {m:^{mod_w}}"
    print(f"\n  {sep}")
    print(hdr)
    print(f"  {sep}")

    # Rows
    votes = [0] * len(TEST_NAMES)
    for i, name in enumerate(TEST_NAMES):
        row   = f"  {name:<{pep_w}}"
        n_pos = 0
        for m in models:
            p    = probs_dict[m][i]
            flag = "✓" if p >= 0.5 else " "
            cell = f"{p:.3f}{flag}"
            row += f" {cell:^{mod_w}}"
            if p >= 0.5:
                n_pos += 1
                votes[i] += 1
        row += f"   ← {n_pos}/{len(models)}"
        print(row)

    print(f"  {sep}")
    print(f"  {'':.<{pep_w}}", end="")
    for _ in models:
        print(f" {'↑ P(AMP)':^{mod_w}}", end="")
    print(f"\n  ✓ = P(AMP) ≥ 0.50   |   rightmost column = # models voting AMP\n")

    # Majority-vote summary
    half      = len(models) / 2
    amp_pred  = [TEST_NAMES[i] for i, v in enumerate(votes) if v > half]
    namp_pred = [TEST_NAMES[i] for i, v in enumerate(votes) if v <= half]
    print(f"  Majority vote ({len(models)} models, threshold > {half:.0f}):")
    if amp_pred:
        print(f"    Predicted AMP     : {', '.join(amp_pred)}")
    if namp_pred:
        print(f"    Predicted non-AMP : {', '.join(namp_pred)}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main(cv_folds: int = 5):
    print("\n" + "=" * 72)
    print("  Stapled Peptide RBF-SVM — Seven Feature-Set Comparison")
    print("  (Following 2016 PNAS paper: RBF kernel + Z-score + Platt probs)")
    print("=" * 72)

    amp_sp, dec_sp, amp_qsar, dec_qsar = load_training()
    test_sp, test_qsar = load_test()

    qsar_avail = not amp_qsar.empty and not dec_qsar.empty

    print(f"\n  Training data:")
    print(f"    StaPep — AMPs: {len(amp_sp):>3}   Decoys: {len(dec_sp):>3}")
    if qsar_avail:
        print(f"    QSAR   — AMPs: {len(amp_qsar):>3}   Decoys: {len(dec_qsar):>3}")
    print(f"  Test peptides — StaPep:{len(test_sp)}  QSAR:{len(test_qsar)}")

    # ── Build feature matrices ─────────────────────────────────────────────
    models_train: dict = {}
    models_train["StaPep"] = build_sp_matrix(amp_sp, dec_sp)

    if qsar_avail:
        models_train["QSAR"]    = build_qsar_matrix(amp_qsar, dec_qsar)
        models_train["QSAR+SP"] = build_qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar)

    # ── Cross-validation (GridSearchCV per feature set) ────────────────────
    print(f"\n{'─'*72}")
    print(f"  GridSearchCV ({cv_folds}-fold Stratified)  —  RBF-SVM, C×γ grid search")
    print(f"  C ∈ {SVM_PARAM_GRID['svc__C']}   γ ∈ {SVM_PARAM_GRID['svc__gamma']}")
    print(f"{'─'*72}")

    cv_results = []
    for name, (X, y, feat_names) in models_train.items():
        print(f"  Tuning [{name}]  (n={len(y)}) ...", end=" ", flush=True)
        res = cv_score_svm(X, y, name, folds=cv_folds, feature_names=feat_names)
        cv_results.append(res)
        print(f"done  AUC={res['auc_mean']:.3f}  C={res['best_C']}  γ={res['best_gamma']}")

    print_cv_table(cv_results)

    # ── Final predictions ──────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Final Predictions on Test Peptides  (GridSearchCV → refit on all data)")
    print(f"{'─'*72}")

    probs_dict: dict = {}
    for name, (X_tr, y_tr, feat_names) in models_train.items():
        dfs = []
        if "SP"   in name or name == "StaPep": dfs.append(test_sp)
        if "QSAR" in name:                     dfs.append(test_qsar)
        if not dfs:
            dfs = [test_sp]

        X_te = build_test_matrix(feat_names, *dfs)
        probs_dict[name] = fit_predict_svm(X_tr, y_tr, X_te,
                                           folds=cv_folds,
                                           feature_names=feat_names)

    print_prediction_table(probs_dict)

    # ── Save ───────────────────────────────────────────────────────────────
    out_df = pd.DataFrame({"peptide_id": TEST_NAMES})
    for m, probs in probs_dict.items():
        out_df[f"P_AMP_{m}"] = np.round(probs, 4)
    out_path = STAPEP / "test_stapled_svm_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of CV folds (default: 5)")
    args = p.parse_args()
    main(cv_folds=args.cv_folds)
