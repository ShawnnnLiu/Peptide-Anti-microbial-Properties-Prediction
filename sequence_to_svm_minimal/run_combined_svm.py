#!/usr/bin/env python3
"""
run_combined_svm.py
===================
Unified SVM comparison across three feature sets:

  1. QSAR only    — pretrained 2016 PNAS SVM, PURE INFERENCE (no retraining)
                    Linear kernel, Z-scored with original training statistics,
                    Platt scaling from the original published model.

  2. QSAR+StaPep  — NEW RBF-SVM trained on our stapled AMP/decoy dataset
                    (12 QSAR + 17 StaPep features, GridSearchCV tuning)

  3. StaPep only  — NEW RBF-SVM trained on our stapled AMP/decoy dataset
                    (17 StaPep MD/sequence features, GridSearchCV tuning)

Usage
-----
  conda run -n esm_env python run_combined_svm.py [--cv-folds N]
"""

import warnings, argparse, os, sys
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"
PP_DIR = BASE / ".." / "pretrained_svm" / "sequence_to_svm_minimal" / "predictionsParameters"

PATHS = {
    "amp_stapep"  : STAPEP / "stapled_amps_features.csv",
    "decoy_stapep": STAPEP / "stapled_decoys.csv",
    "amp_qsar"    : STAPEP / "qsar_stapled_amps.csv",
    "decoy_qsar"  : STAPEP / "qsar_stapled_decoys.csv",
    "test_stapep" : STAPEP / "test_stapled_features.csv",
    "test_qsar"   : STAPEP / "qsar_stapled_test.csv",
    "z_file"      : PP_DIR  / "Z_score_mean_std__intersect_noflip.csv",
    "svc_pkl"     : PP_DIR  / "svc.pkl",
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

# Canonical test-peptide names (order must match test CSV rows)
TEST_NAMES = [
    "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    "Mag20",  "Mag25", "Mag31",     "Mag36",
]
QSAR_ID_MAP = {
    "Buf(i+4)_12"          : "Buf12",
    "Buf(i+4)_13"          : "Buf13",
    "Buf(i+4)_13_Q9K"      : "Buf13_Q9K",
    "Buf(i+4)_12_V15K_L19K": "Buf12_V15K_L19K",
    "Mag_20": "Mag20", "Mag_25": "Mag25",
    "Mag_31": "Mag31", "Mag_36": "Mag36",
}

# RBF-SVM hyperparameter grid (same as original run_stapep_svm.py)
SVM_GRID = {
    "svc__C"    : [0.1, 1, 10, 100, 1000],
    "svc__gamma": ["scale", 1e-3, 1e-2, 0.1],
}


# ════════════════════════════════════════════════════════════════════════════════
# 1.  PRETRAINED SVM — QSAR ONLY (pure inference)
# ════════════════════════════════════════════════════════════════════════════════

def _npy(n: int) -> np.ndarray:
    return np.load(PP_DIR / f"svc.pkl_{n:02d}.npy", allow_pickle=False)


def pretrained_qsar_inference(test_qsar: pd.DataFrame) -> np.ndarray:
    """
    Load the 2016 PNAS pretrained SVM and run inference on test QSAR features.
    Returns P(AMP) array ordered by TEST_NAMES.

    Verified pkl mapping (cross-checked against exp1 reference predictions):
      pkl_03 → dual_coef_      pkl_04 → probA_
      pkl_06 → n_support_      pkl_07 → support_vectors_
      pkl_10 → intercept_      pkl_11 → probB_
    """
    # -- Load Z-score params from the original training statistics --
    with open(PATHS["z_file"]) as f:
        desc_names = f.readline().strip().split(",")
        z_means    = np.array([float(x) for x in f.readline().strip().split(",")])
        z_stds     = np.array([float(x) for x in f.readline().strip().split(",")])

    # -- Build test feature matrix aligned to TEST_NAMES --
    prob_arr = np.full(len(TEST_NAMES), np.nan)
    tq = test_qsar.copy()
    tq["peptide_id"] = tq["peptide_id"].map(QSAR_ID_MAP).fillna(tq["peptide_id"])

    # -- Load pretrained SVM arrays from .npy sidecar files --
    support_vectors_ = _npy(7)   # (225, 12)
    dual_coef_       = _npy(3)   # (1, 225)
    intercept_       = _npy(10)  # (1,)  = -0.01187876 (VERIFIED)
    probA_           = _npy(4)   # (1,)  = -3.29162142 (VERIFIED)
    probB_           = _npy(11)  # (1,)  = +0.03014156 (VERIFIED)

    # Linear kernel weight vector
    w = support_vectors_.T @ dual_coef_[0]   # (12,)

    for i, name in enumerate(TEST_NAMES):
        row = tq[tq["peptide_id"] == name]
        if row.empty:
            continue
        feat = row.iloc[0][desc_names].values.astype(float)
        x_z  = (feat - z_means) / z_stds
        decision = float(x_z @ w + intercept_[0])
        fval     = probA_[0] * decision + probB_[0]
        prob_arr[i] = 1.0 / (1.0 + np.exp(fval))

    return prob_arr


# ════════════════════════════════════════════════════════════════════════════════
# 2.  TRAINED SVMs — QSAR+StaPep and StaPep only
# ════════════════════════════════════════════════════════════════════════════════

def _present(df, cols):
    return [c for c in cols if c in df.columns]


def load_training():
    amp_sp = pd.read_csv(PATHS["amp_stapep"]).rename(
        columns={"DRAMP_ID": "peptide_id", "Hiden_Sequence": "sequence"})
    dec_sp = pd.read_csv(PATHS["decoy_stapep"]).rename(
        columns={"COMPOUND_ID": "peptide_id", "SEQUENCE": "sequence"})
    amp_sp["label"] = 1;  dec_sp["label"] = 0
    amp_sp["join_id"] = amp_sp["peptide_id"].astype(str)
    dec_sp["join_id"] = (dec_sp.reset_index(drop=True).index + 1).astype(str)
    amp_sp = amp_sp.dropna(subset=_present(amp_sp, STAPEP_COLS), how="all")
    dec_sp = dec_sp.dropna(subset=_present(dec_sp, STAPEP_COLS), how="all")

    amp_qsar = pd.DataFrame(); dec_qsar = pd.DataFrame()
    if PATHS["amp_qsar"].exists():
        amp_qsar = pd.read_csv(PATHS["amp_qsar"])
        amp_qsar["label"]   = 1
        amp_qsar["join_id"] = amp_qsar["peptide_id"].astype(str)
    if PATHS["decoy_qsar"].exists():
        dec_qsar = pd.read_csv(PATHS["decoy_qsar"])
        dec_qsar["label"]   = 0
        dec_qsar["join_id"] = (dec_qsar.reset_index(drop=True).index + 1).astype(str)

    return amp_sp, dec_sp, amp_qsar, dec_qsar


def load_test():
    test_sp = pd.DataFrame()
    if PATHS["test_stapep"].exists():
        test_sp = pd.read_csv(PATHS["test_stapep"])
    test_qsar = pd.DataFrame()
    if PATHS["test_qsar"].exists():
        test_qsar = pd.read_csv(PATHS["test_qsar"])
        test_qsar["peptide_id"] = (test_qsar["peptide_id"]
                                   .map(QSAR_ID_MAP)
                                   .fillna(test_qsar["peptide_id"]))
    return test_sp, test_qsar


def build_sp_matrix(amp_sp, dec_sp):
    cols  = list(dict.fromkeys(_present(amp_sp, STAPEP_COLS)))
    train = pd.concat([amp_sp[["label"] + cols],
                       dec_sp[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, pd.Index(cols)


def build_qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar):
    sp_cols  = _present(amp_sp, STAPEP_COLS)
    q_only   = [c for c in QSAR_COLS if c not in set(sp_cols)]

    amp_mrg = amp_sp[["join_id","label"] + sp_cols].merge(
        amp_qsar[["join_id"] + q_only], on="join_id", how="inner")
    dec_mrg = dec_sp[["join_id","label"] + sp_cols].merge(
        dec_qsar[["join_id"] + q_only], on="join_id", how="inner")

    all_feat = sp_cols + q_only
    train    = pd.concat([amp_mrg, dec_mrg], ignore_index=True)
    return (train[all_feat].values.astype(float),
            train["label"].values,
            pd.Index(all_feat))


def _base_pipe():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42)),
    ])


def make_gs(folds):
    return GridSearchCV(_base_pipe(), SVM_GRID,
                        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
                        scoring="roc_auc", n_jobs=-1, refit=True)


def cv_score(X, y, label, folds):
    gs = make_gs(folds)
    gs.fit(X, y)
    bi = gs.best_index_
    fold_aucs = np.array([gs.cv_results_[f"split{i}_test_score"][bi]
                          for i in range(folds)])
    best_C, best_g = gs.best_params_["svc__C"], gs.best_params_["svc__gamma"]
    ap_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True, class_weight="balanced",
                        random_state=42, C=best_C, gamma=best_g)),
    ])
    ap = cross_val_score(ap_pipe, X, y, scoring="average_precision",
                         cv=StratifiedKFold(folds, shuffle=True, random_state=42),
                         n_jobs=-1)
    return {"label": label, "n": len(y),
            "auc_mean": fold_aucs.mean(), "auc_std": fold_aucs.std(),
            "ap_mean": ap.mean(), "ap_std": ap.std(),
            "best_C": best_C, "best_gamma": best_g, "gs": gs}


def build_test_row(feat_names, *dfs):
    rows = []
    for name in TEST_NAMES:
        vals = []
        for col in feat_names:
            v = np.nan
            for df in dfs:
                if df.empty or "peptide_id" not in df.columns or col not in df.columns:
                    continue
                r = df[df["peptide_id"] == name]
                if not r.empty:
                    cand = r.iloc[0].get(col, np.nan)
                    if not pd.isna(cand):
                        v = float(cand); break
            vals.append(v)
        rows.append(vals)
    return np.array(rows)


# ════════════════════════════════════════════════════════════════════════════════
# Display helpers
# ════════════════════════════════════════════════════════════════════════════════

def print_cv_table(results):
    col1_w = max(len(r["label"]) for r in results) + 2
    print(f"\n  {'Model':<{col1_w}}  {'AUC':>12}   {'Avg Prec':>12}   "
          f"{'n':>5}   {'Best C':>8}  {'Best γ':>10}   {'Note'}")
    print(f"  {'─'*col1_w}  {'─'*12}   {'─'*12}   {'─'*5}   {'─'*8}  {'─'*10}   {'─'*22}")
    for r in results:
        gs     = r.get("gs")
        gamma  = r.get("best_gamma", "—")
        g_str  = f"{gamma:.4g}" if isinstance(gamma, float) else str(gamma)
        note   = r.get("note", "RBF-SVM, trained on stapled set")
        auc    = r.get("auc_mean")
        ap     = r.get("ap_mean")
        auc_s  = f"{auc:.3f} ± {r['auc_std']:.3f}" if auc is not None else "N/A (pretrained)"
        ap_s   = f"{ap:.3f} ± {r['ap_std']:.3f}"   if ap  is not None else "N/A (pretrained)"
        c_str  = str(r.get("best_C", "—"))
        print(f"  {r['label']:<{col1_w}}  {auc_s:>12}   {ap_s:>12}   "
              f"{r['n']:>5}   {c_str:>8}  {g_str:>10}   {note}")


def print_prediction_table(probs_dict, qsar_seqs=None):
    models = list(probs_dict.keys())
    pep_w  = max(len(n) for n in TEST_NAMES) + 1
    mod_w  = max(max(len(m) for m in models), 10) + 2
    sep    = "─" * (pep_w + len(models) * (mod_w + 1) + 10)

    hdr = f"  {'Peptide':<{pep_w}}"
    for m in models:
        hdr += f"  {m:^{mod_w}}"
    print(f"\n  {sep}")
    print(hdr)
    print(f"  {sep}")

    votes = [0] * len(TEST_NAMES)
    for i, name in enumerate(TEST_NAMES):
        row = f"  {name:<{pep_w}}"
        npos = 0
        for m in models:
            p    = probs_dict[m][i]
            flag = "✓" if (not np.isnan(p) and p >= 0.5) else " "
            cell = f"{p:.3f}{flag}" if not np.isnan(p) else "  N/A "
            row += f"  {cell:^{mod_w}}"
            if not np.isnan(p) and p >= 0.5:
                npos += 1; votes[i] += 1
        row += f"   ← {npos}/{len(models)}"
        print(row)

    print(f"  {sep}")
    foot = f"  {'':.<{pep_w}}"
    for _ in models:
        foot += f"  {'↑ P(AMP)':^{mod_w}}"
    print(foot)
    print(f"  ✓ = P(AMP) ≥ 0.50   |   rightmost column = # models voting AMP\n")

    # Notes on QSAR source
    if qsar_seqs:
        print("  QSAR source sequences (parent residues at staple positions):")
        for name, seq in qsar_seqs.items():
            print(f"    {name:<24} {seq}")
        print()


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main(cv_folds: int = 5):
    print("\n" + "=" * 72)
    print("  Stapled Peptide SVM — Three Feature-Set Comparison")
    print("  QSAR: pretrained 2016 PNAS model (NO retraining)")
    print("  QSAR+StaPep & StaPep: RBF-SVM trained on stapled AMP/decoy set")
    print("=" * 72)

    amp_sp, dec_sp, amp_qsar, dec_qsar = load_training()
    test_sp, test_qsar = load_test()

    print(f"\n  Training data:")
    print(f"    StaPep — AMPs: {len(amp_sp):>3}   Decoys: {len(dec_sp):>3}")
    if not amp_qsar.empty:
        print(f"    QSAR   — AMPs: {len(amp_qsar):>3}   Decoys: {len(dec_qsar):>3}")
    print(f"  Test peptides — StaPep: {len(test_sp)}   QSAR: {len(test_qsar)}")

    probs_dict = {}
    cv_results = []

    # ── 1. QSAR only — pretrained SVM, no CV score (not our model) ────────────
    print(f"\n{'─'*72}")
    print("  [1/3] QSAR only — pretrained 2016 PNAS SVM (pure inference) ...")
    if test_qsar.empty:
        print("  ⚠  Test QSAR file missing — run extract_stapep_qsar.py first")
        probs_dict["QSAR\n(pretrained)"] = np.full(len(TEST_NAMES), np.nan)
    else:
        probs = pretrained_qsar_inference(test_qsar)
        probs_dict["QSAR\n(pretrained)"] = probs
        print(f"  Done. Results for {np.sum(~np.isnan(probs))} peptides.")

    cv_results.append({
        "label"     : "QSAR (pretrained)",
        "n"         : 486,                   # shape_fit_ from the original pickle
        "auc_mean"  : None, "auc_std"  : 0,
        "ap_mean"   : None, "ap_std"   : 0,
        "best_C"    : 0.01267,
        "best_gamma": "0 (linear)",
        "note"      : "Pretrained 2016 PNAS SVM — linear kernel",
    })

    # ── 2. QSAR + StaPep — retrain ────────────────────────────────────────────
    if not amp_qsar.empty and not dec_qsar.empty:
        print(f"\n{'─'*72}")
        print(f"  [2/3] QSAR+StaPep — training RBF-SVM ({cv_folds}-fold CV) ...")
        X_qs, y_qs, feat_qs = build_qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar)
        res_qs = cv_score(X_qs, y_qs, "QSAR+StaPep", cv_folds)
        cv_results.append({**res_qs,
                           "note": "RBF-SVM, trained on stapled set"})
        print(f"  Done. AUC={res_qs['auc_mean']:.3f}  C={res_qs['best_C']}  γ={res_qs['best_gamma']}")

        X_te_qs = build_test_row(feat_qs, test_sp, test_qsar)
        probs_dict["QSAR+StaPep\n(trained)"] = res_qs["gs"].predict_proba(X_te_qs)[:, 1]
    else:
        print("\n  ⚠  Skipping QSAR+StaPep — QSAR files missing")

    # ── 3. StaPep only — retrain ──────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  [3/3] StaPep only — training RBF-SVM ({cv_folds}-fold CV) ...")
    X_sp, y_sp, feat_sp = build_sp_matrix(amp_sp, dec_sp)
    res_sp = cv_score(X_sp, y_sp, "StaPep", cv_folds)
    cv_results.append({**res_sp,
                       "note": "RBF-SVM, trained on stapled set"})
    print(f"  Done. AUC={res_sp['auc_mean']:.3f}  C={res_sp['best_C']}  γ={res_sp['best_gamma']}")

    X_te_sp = build_test_row(feat_sp, test_sp)
    probs_dict["StaPep\n(trained)"] = res_sp["gs"].predict_proba(X_te_sp)[:, 1]

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  CV Performance Summary")
    print(f"{'='*72}")
    print_cv_table(cv_results)

    print(f"\n{'='*72}")
    print("  Test Peptide Predictions — P(AMP)")
    print(f"{'='*72}")

    # Gather QSAR sequences for reference
    qsar_seqs = {}
    if not test_qsar.empty:
        tq_disp = test_qsar.copy()
        tq_disp["peptide_id"] = (tq_disp["peptide_id"]
                                 .map(QSAR_ID_MAP)
                                 .fillna(tq_disp["peptide_id"]))
        for _, r in tq_disp.iterrows():
            qsar_seqs[r["peptide_id"]] = r.get("sequence", "?")

    # Rename keys for display (strip newlines)
    disp = {k.replace("\n", " "): v for k, v in probs_dict.items()}
    print_prediction_table(disp, qsar_seqs)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame({"peptide_id": TEST_NAMES})
    for m, probs in probs_dict.items():
        col = m.replace("\n", "_").replace(" ", "")
        out_df[f"P_AMP_{col}"] = np.round(probs, 4)
    out_path = STAPEP / "test_combined_svm_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="Number of CV folds for trained SVMs (default: 5)")
    args = ap.parse_args()
    main(cv_folds=args.cv_folds)
