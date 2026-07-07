#!/usr/bin/env python3
"""
predict_mic_mlp.py
==================
Optimised MLP Classifier (AMP vs Decoy) — P(AMP) vs MIC analysis.

Improvements over baseline:
  • Ensemble of 20 MLPs (bootstrap bagging) for stable, well-spread P(AMP)
  • 10 000 max epochs with adaptive learning rate & patience=50
  • Wider architecture search (up to 3 hidden layers)
  • Isotonic probability calibration (CalibratedClassifierCV)
  • Cross-validated P(AMP) for training set (no overfit bias in Panel A)
  • Extra panels: convergence curve + permutation feature importance

Usage:
    conda run -n esm_env python predict_mic_mlp.py
"""

import io, re, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, cross_val_predict,
                                     train_test_split)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             confusion_matrix, matthews_corrcoef,
                             balanced_accuracy_score, classification_report)
from sklearn.base import clone

# Make utils/ and features/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import PROJECT_ROOT, STAPEP_DIR
from utils.mic_units import mic_to_pmic_ugml as mic_to_pmic
from features.stapep_columns import STAPEP_COLS_WITH_HSASA as FEATURE_COLS
from features.reference_peptides import LITERATURE_MIC_ECOLI as LITERATURE_MIC

warnings.filterwarnings("ignore")

# Path aliases kept for any in-file legacy references.
BASE = PROJECT_ROOT
DATA = STAPEP_DIR

N_ENSEMBLE   = 20
N_BOOT_FRAC  = 0.85

BUF_WT_FEATURES = {
    "length": 21, "weight": 2473.829,
    "hydrophobic_index": -0.8142857142857142, "charge": 6.094,
    "aromaticity": 0.047619047619047616,
    "isoelectric_point": 11.999967765808105,
    "fraction_arginine": 0.23809523809523808,
    "fraction_lysine": 0.047619047619047616,
    "lyticity_index": 300.106,
    "helix_percent": 0.17819, "sheet_percent": 0.00076,
    "loop_percent": 0.82105,
    "mean_bfactor": 573.434, "mean_gyrate": 12.001,
    "num_hbonds": 0,
    "psa": 1064.217, "sasa": 2038.292,
    "hydrophobic_sasa": 2038.292 - 1064.217,
}

# ── MIC parsing helpers ─────────────────────────────────────────────────────
_COLI = r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+\w+)*\s*"
_MIC_UM   = r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M"
_MIC_UGML = r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL"
_RE_UM    = re.compile(_COLI + _MIC_UM,   re.IGNORECASE)
_RE_UGML  = re.compile(_COLI + _MIC_UGML, re.IGNORECASE)
_GENERIC_UM   = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M", re.I)
_GENERIC_UGML = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL", re.I)


def _parse(m):
    if m is None: return None
    mod = m.group(1).strip()
    if mod in (">", "<", "\u2265"): return None
    try: return float(m.group(2))
    except: return None


def get_mic_ugml(text, mw):
    if not isinstance(text, str): return None
    v = _parse(_RE_UGML.search(text))
    if v is not None: return v
    v = _parse(_RE_UM.search(text))
    if v is not None and mw > 0: return v * mw / 1000.0
    v = _parse(_GENERIC_UGML.search(text))
    if v is not None: return v
    v = _parse(_GENERIC_UM.search(text))
    if v is not None and mw > 0: return v * mw / 1000.0
    return None


def pearson_safe(x, y):
    if len(x) < 3: return np.nan, np.nan
    return stats.pearsonr(x, y)


def spearman_safe(x, y):
    if len(x) < 3: return np.nan, np.nan
    return stats.spearmanr(x, y)


def _make_mlp(hidden, alpha, lr_init, seed):
    """Build a single MLP pipeline with convergence-friendly settings."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("mlp",     MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=lr_init,
            learning_rate="adaptive",
            max_iter=10000,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=50,
            tol=1e-5,
            solver="adam",
            batch_size="auto",
            random_state=seed,
        )),
    ])


class EnsembleMLP:
    """Bootstrap-bagged ensemble of calibrated MLPs."""

    def __init__(self, base_pipe, n_models=20, boot_frac=0.85):
        self.base_pipe = base_pipe
        self.n_models = n_models
        self.boot_frac = boot_frac
        self.models_ = []
        self.oob_aucs_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(42)
        n = len(X)
        n_boot = int(n * self.boot_frac)

        for i in range(self.n_models):
            idx = rng.choice(n, size=n_boot, replace=True)
            oob_idx = np.setdiff1d(np.arange(n), np.unique(idx))

            pipe = clone(self.base_pipe)
            pipe.set_params(mlp__random_state=42 + i)
            pipe.fit(X[idx], y[idx])

            cal = CalibratedClassifierCV(pipe, cv="prefit", method="isotonic")
            if len(oob_idx) >= 10:
                cal.fit(X[oob_idx], y[oob_idx])
                oob_pred = cal.predict_proba(X[oob_idx])[:, 1]
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y[oob_idx], oob_pred)
                except ValueError:
                    auc = np.nan
                self.oob_aucs_.append(auc)
            else:
                cal.fit(X[idx], y[idx])
                self.oob_aucs_.append(np.nan)

            self.models_.append(cal)
            sys.stdout.write(f"\r    Ensemble member {i+1}/{self.n_models} "
                             f"(OOB AUC={self.oob_aucs_[-1]:.3f})")
            sys.stdout.flush()
        print()
        return self

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)
        return np.column_stack([1 - probs.mean(axis=0), probs.mean(axis=0)])

    def predict_proba_individual(self, X):
        """Return (n_models, n_samples) array of individual P(AMP)."""
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)

    def convergence_curves(self):
        """Extract loss curves from each member's inner MLP."""
        curves = []
        for cal_model in self.models_:
            inner = cal_model.estimator
            mlp = inner.named_steps["mlp"]
            if hasattr(mlp, "loss_curve_"):
                curves.append(mlp.loss_curve_)
        return curves


# ═════════════════════════════════════════════════════════════════════════════
def main():
    # Unicode-safe stdout for Windows console. Done inside main() so the
    # module remains importable without side effects on sys.stdout.
    if sys.stdout is not None and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  Optimised MLP Ensemble — P(AMP) vs MIC")
    print("  20 bootstrap-bagged, isotonic-calibrated MLPs")
    print("  Train AMP/Decoy on ALL stapled peptides | Test on Buforin variants")
    print("=" * 72)

    # ── 1. Load ALL training data ────────────────────────────────────────────
    meta = pd.read_csv(DATA / "stapled_amps.csv")
    feat = pd.read_csv(DATA / "stapled_amps_features.csv")
    dec  = pd.read_csv(DATA / "stapled_decoys.csv")

    amp_feat = feat.copy()
    amp_feat["hydrophobic_sasa"] = amp_feat["sasa"] - amp_feat["psa"]
    amp_feat["label"] = 1

    dec["hydrophobic_sasa"] = dec["sasa"] - dec["psa"]
    dec["label"] = -1

    clf_df = pd.concat([
        amp_feat[FEATURE_COLS + ["label"]],
        dec[FEATURE_COLS + ["label"]],
    ], ignore_index=True).dropna(subset=FEATURE_COLS)

    X_clf = clf_df[FEATURE_COLS].values.astype(float)
    y_clf = clf_df["label"].values
    n_amp = (y_clf == 1).sum()
    n_dec = (y_clf == -1).sum()

    print(f"\n  Classifier training: {len(clf_df)} total  "
          f"(AMPs={n_amp}, Decoys={n_dec})")

    # ── 2. Training AMPs with E. coli MIC ────────────────────────────────────
    df_amp = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat, on="DRAMP_ID", how="inner",
    )
    df_amp["hydrophobic_sasa"] = df_amp["sasa"] - df_amp["psa"]
    df_amp["mic_ugml"] = df_amp.apply(
        lambda r: get_mic_ugml(r["Target_Organism"], r["weight"]), axis=1
    )
    df_mic = df_amp[df_amp["mic_ugml"].notna() & (df_amp["mic_ugml"] > 0)].copy()
    df_mic["pMIC"] = df_mic.apply(
        lambda r: mic_to_pmic(r["mic_ugml"], r["weight"]), axis=1
    )
    df_mic = df_mic.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    X_mic_train = df_mic[FEATURE_COLS].values.astype(float)
    y_mic_ugml  = df_mic["mic_ugml"].values
    y_mic_pmic  = df_mic["pMIC"].values
    n_mic = len(df_mic)

    print(f"  Training AMPs with E. coli MIC: {n_mic}")
    print(f"  MIC range: {y_mic_ugml.min():.1f} - {y_mic_ugml.max():.1f} ug/mL")

    # ── 3. Hyperparameter search (single MLP, find best arch) ────────────────
    print(f"\n{'='*72}")
    print("  Phase 1: GridSearch for best architecture")
    print(f"{'='*72}")

    search_pipe = _make_mlp((128,), 1e-3, 1e-3, 42)
    mlp_grid = {
        "mlp__hidden_layer_sizes": [
            (64,), (128,), (256,),
            (128, 64), (256, 128), (128, 128),
            (256, 128, 64), (128, 64, 32),
        ],
        "mlp__alpha":              [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 5e-4, 2e-4],
    }
    gs = GridSearchCV(
        search_pipe, mlp_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1, refit=True,
    )
    gs.fit(X_clf, y_clf)

    best_p = gs.best_params_
    best_hidden = best_p["mlp__hidden_layer_sizes"]
    best_alpha  = best_p["mlp__alpha"]
    best_lr     = best_p["mlp__learning_rate_init"]

    print(f"  Best params: hidden={best_hidden}, alpha={best_alpha}, lr={best_lr}")
    print(f"  Best CV AUC: {gs.best_score_:.4f}")

    # Convergence info from the refit model
    refit_mlp = gs.best_estimator_.named_steps["mlp"]
    n_iters = refit_mlp.n_iter_
    final_loss = refit_mlp.loss_curve_[-1] if hasattr(refit_mlp, "loss_curve_") else None
    print(f"  Refit converged in {n_iters} epochs (final loss={final_loss:.5f})")

    # ── 3b. Hold-out test evaluation (80/20 stratified split) ────────────────
    print(f"\n{'='*72}")
    print("  Phase 1b: Hold-out test evaluation (80/20 stratified)")
    print(f"{'='*72}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clf, y_clf, test_size=0.20, stratify=y_clf, random_state=42,
    )
    eval_pipe = _make_mlp(best_hidden, best_alpha, best_lr, 42)
    eval_pipe.fit(X_tr, y_tr)

    y_te_pred  = eval_pipe.predict(X_te)
    y_te_proba = eval_pipe.predict_proba(X_te)[:, 1]
    y_te_bin   = np.where(y_te == 1, 1, 0)
    y_pred_bin = np.where(y_te_pred == 1, 1, 0)

    test_acc   = accuracy_score(y_te_bin, y_pred_bin)
    test_bacc  = balanced_accuracy_score(y_te_bin, y_pred_bin)
    test_f1    = f1_score(y_te_bin, y_pred_bin)
    test_prec  = precision_score(y_te_bin, y_pred_bin)
    test_rec   = recall_score(y_te_bin, y_pred_bin)
    test_auc   = roc_auc_score(y_te_bin, y_te_proba)
    test_mcc   = matthews_corrcoef(y_te_bin, y_pred_bin)
    test_cm    = confusion_matrix(y_te_bin, y_pred_bin)
    fpr, tpr, _ = roc_curve(y_te_bin, y_te_proba)

    n_te_amp = (y_te == 1).sum()
    n_te_dec = (y_te == -1).sum()

    print(f"  Split:  Train={len(X_tr)}  Test={len(X_te)}  "
          f"(AMPs={n_te_amp}, Decoys={n_te_dec})")
    print(f"\n  {'Metric':<25} {'Value':>8}")
    print(f"  {'-'*25} {'-'*8}")
    print(f"  {'Accuracy':<25} {test_acc:>8.3f}")
    print(f"  {'Balanced Accuracy':<25} {test_bacc:>8.3f}")
    print(f"  {'F1 Score':<25} {test_f1:>8.3f}")
    print(f"  {'Precision':<25} {test_prec:>8.3f}")
    print(f"  {'Recall (Sensitivity)':<25} {test_rec:>8.3f}")
    print(f"  {'AUC-ROC':<25} {test_auc:>8.3f}")
    print(f"  {'MCC':<25} {test_mcc:>8.3f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred Decoy  Pred AMP")
    print(f"  Act Decoy   {test_cm[0,0]:>9}  {test_cm[0,1]:>8}")
    print(f"  Act AMP     {test_cm[1,0]:>9}  {test_cm[1,1]:>8}")

    # Also compute per-fold CV metrics for robustness
    cv_folds = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clone(eval_pipe), X_clf, (y_clf == 1).astype(int),
                            cv=cv_folds, scoring="f1")
    cv_prec = cross_val_score(clone(eval_pipe), X_clf, (y_clf == 1).astype(int),
                              cv=cv_folds, scoring="precision")
    cv_rec = cross_val_score(clone(eval_pipe), X_clf, (y_clf == 1).astype(int),
                             cv=cv_folds, scoring="recall")
    cv_auc_scores = cross_val_score(clone(eval_pipe), X_clf, (y_clf == 1).astype(int),
                                    cv=cv_folds, scoring="roc_auc")

    print(f"\n  5-fold CV (mean +/- std):")
    print(f"    F1:        {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")
    print(f"    Precision: {cv_prec.mean():.3f} +/- {cv_prec.std():.3f}")
    print(f"    Recall:    {cv_rec.mean():.3f} +/- {cv_rec.std():.3f}")
    print(f"    AUC-ROC:   {cv_auc_scores.mean():.3f} +/- {cv_auc_scores.std():.3f}")

    # ── 4. Build ensemble with best hyperparams ──────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Phase 2: Training {N_ENSEMBLE}-member bootstrap ensemble")
    print(f"  (isotonic calibration on OOB samples)")
    print(f"{'='*72}\n")

    base_pipe = _make_mlp(best_hidden, best_alpha, best_lr, 42)
    ensemble = EnsembleMLP(base_pipe, n_models=N_ENSEMBLE, boot_frac=N_BOOT_FRAC)
    ensemble.fit(X_clf, y_clf)

    valid_aucs = [a for a in ensemble.oob_aucs_ if not np.isnan(a)]
    print(f"  Mean OOB AUC: {np.mean(valid_aucs):.4f} +/- {np.std(valid_aucs):.4f}")

    # ── 5. Cross-validated P(AMP) for training AMPs (avoids overfit bias) ────
    print(f"\n{'='*72}")
    print("  Phase 3: 5-fold CV probabilities for training set")
    print(f"{'='*72}")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    cv_pipe = _make_mlp(best_hidden, best_alpha, best_lr, 42)
    train_probs_cv = cross_val_predict(cv_pipe, X_clf, y_clf, cv=cv,
                                       method="predict_proba")[:, 1]

    amp_mask = (y_clf == 1)
    mic_idx_in_clf = []
    for i, row in df_mic.iterrows():
        x_row = row[FEATURE_COLS].values.astype(float)
        dists = np.linalg.norm(X_clf[amp_mask] - x_row, axis=1)
        best_match = np.where(amp_mask)[0][np.argmin(dists)]
        mic_idx_in_clf.append(best_match)
    mic_idx_in_clf = np.array(mic_idx_in_clf)
    train_probs_cv_mic = train_probs_cv[mic_idx_in_clf]

    cv_acc = cross_val_score(
        clone(cv_pipe), X_clf, y_clf,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    print(f"  CV Accuracy: {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
    print(f"  CV P(AMP) range (train AMPs w/ MIC): "
          f"{train_probs_cv_mic.min():.3f} - {train_probs_cv_mic.max():.3f}")

    # ── 6. Ensemble P(AMP) for training AMPs with MIC ────────────────────────
    train_probs_ens = ensemble.predict_proba(X_mic_train)[:, 1]
    train_probs_indiv = ensemble.predict_proba_individual(X_mic_train)
    train_probs_std = train_probs_indiv.std(axis=0)

    print(f"\n{'='*72}")
    print(f"  TRAINING AMPs WITH MIC — Ensemble P(AMP) vs MIC")
    print(f"{'='*72}")

    log_mic = np.log10(y_mic_ugml)

    r_train_logmic_cv, _  = pearson_safe(train_probs_cv_mic, log_mic)
    rho_train_logmic_cv, _ = spearman_safe(train_probs_cv_mic, log_mic)
    r_train_pmic_cv, _    = pearson_safe(train_probs_cv_mic, y_mic_pmic)
    rho_train_pmic_cv, _  = spearman_safe(train_probs_cv_mic, y_mic_pmic)

    r_train_logmic_ens, _  = pearson_safe(train_probs_ens, log_mic)
    rho_train_logmic_ens, _ = spearman_safe(train_probs_ens, log_mic)
    r_train_pmic_ens, _    = pearson_safe(train_probs_ens, y_mic_pmic)
    rho_train_pmic_ens, _  = spearman_safe(train_probs_ens, y_mic_pmic)

    print(f"\n  {'Metric':<40} {'CV (unbiased)':>14} {'Ensemble':>10}")
    print(f"  {'-'*40} {'-'*14} {'-'*10}")
    print(f"  {'R vs log10(MIC)':<40} {r_train_logmic_cv:>14.3f} {r_train_logmic_ens:>10.3f}")
    print(f"  {'rho vs log10(MIC)':<40} {rho_train_logmic_cv:>14.3f} {rho_train_logmic_ens:>10.3f}")
    print(f"  {'R vs pMIC':<40} {r_train_pmic_cv:>14.3f} {r_train_pmic_ens:>10.3f}")
    print(f"  {'rho vs pMIC':<40} {rho_train_pmic_cv:>14.3f} {rho_train_pmic_ens:>10.3f}")

    # ── 7. Load Buforin test variants ────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BUFORIN TEST SET")
    print(f"{'='*72}")

    test_f10w = pd.read_csv(DATA / "test_buf_specific_stapep_features.csv")
    test_orig = pd.read_csv(DATA / "test_stapled_features.csv")
    test_orig = test_orig[test_orig["peptide_id"].str.startswith("Buf")].copy()

    test_f10w["hydrophobic_sasa"] = test_f10w["sasa"] - test_f10w["psa"]
    test_orig["hydrophobic_sasa"] = test_orig["sasa"] - test_orig["psa"]

    test_variants = []

    x_wt = np.array([[BUF_WT_FEATURES[f] for f in FEATURE_COLS]])
    prob_wt_ens = ensemble.predict_proba(x_wt)[0, 1]
    prob_wt_std = ensemble.predict_proba_individual(x_wt).std()
    test_variants.append({
        "name": "Buf WT", "short": "Buf WT",
        "prob_amp": prob_wt_ens, "prob_std": prob_wt_std,
        "lit_ugml": None, "mw": BUF_WT_FEATURES["weight"],
        "group": "WT",
    })

    for _, row in test_f10w.iterrows():
        pid = row["peptide_id"]
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        prob_ens = ensemble.predict_proba(x)[0, 1]
        prob_std = ensemble.predict_proba_individual(x).std()
        lit = LITERATURE_MIC.get(pid, {})
        short = pid.replace("Buf_", "").replace("_F10W", "")
        group = "i+7" if "i7" in pid else "i+4"
        test_variants.append({
            "name": pid, "short": short,
            "prob_amp": prob_ens, "prob_std": prob_std,
            "lit_ugml": lit.get("mic_ugml"), "mw": row["weight"],
            "group": group,
        })

    for _, row in test_orig.iterrows():
        pid = row["peptide_id"]
        x = row[FEATURE_COLS].values.astype(float).reshape(1, -1)
        prob_ens = ensemble.predict_proba(x)[0, 1]
        prob_std = ensemble.predict_proba_individual(x).std()
        lit = LITERATURE_MIC.get(pid, {})
        test_variants.append({
            "name": pid, "short": pid,
            "prob_amp": prob_ens, "prob_std": prob_std,
            "lit_ugml": lit.get("mic_ugml"), "mw": row["weight"],
            "group": "Original",
        })

    with_lit = [v for v in test_variants if v["lit_ugml"] is not None]
    all_test = test_variants

    if len(with_lit) >= 3:
        buf_probs   = np.array([v["prob_amp"] for v in with_lit])
        buf_mic     = np.array([v["lit_ugml"] for v in with_lit])
        buf_logmic  = np.log10(buf_mic)
        buf_pmic    = np.array([mic_to_pmic(v["lit_ugml"], v["mw"]) for v in with_lit])

        r_buf_logmic, _   = pearson_safe(buf_probs, buf_logmic)
        rho_buf_logmic, _ = spearman_safe(buf_probs, buf_logmic)
        r_buf_pmic, _     = pearson_safe(buf_probs, buf_pmic)
        rho_buf_pmic, _   = spearman_safe(buf_probs, buf_pmic)

        print(f"\n  Buforin test (n={len(with_lit)} with literature MIC):")
        print(f"  {'Correlation':<35} {'Pearson R':>10} {'Spearman rho':>13}")
        print(f"  {'-'*35} {'-'*10} {'-'*13}")
        print(f"  {'P(AMP) vs log10(MIC)':<35} {r_buf_logmic:>10.3f} {rho_buf_logmic:>13.3f}")
        print(f"  {'P(AMP) vs pMIC':<35} {r_buf_pmic:>10.3f} {rho_buf_pmic:>13.3f}")
    else:
        r_buf_logmic = rho_buf_logmic = np.nan
        r_buf_pmic = rho_buf_pmic = np.nan

    # ── 8. Per-variant table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PER-VARIANT RESULTS (Ensemble mean +/- std)")
    print(f"{'='*80}")
    print(f"\n  {'Variant':<22} {'P(AMP)':>7} {'  +/-':>6} {'Lit MIC':>10} {'Group':>8}")
    print(f"  {'-'*56}")

    for v in test_variants:
        lit_s = f"{v['lit_ugml']:.1f}" if v["lit_ugml"] is not None else "---"
        print(f"  {v['name']:<22} {v['prob_amp']:>7.3f} {v['prob_std']:>6.3f} "
              f"{lit_s:>10} {v['group']:>8}")

    # ── 9. Permutation feature importance ────────────────────────────────────
    print(f"\n{'='*72}")
    print("  Permutation Feature Importance (refit model)")
    print(f"{'='*72}")

    perm_result = permutation_importance(
        gs.best_estimator_, X_clf, y_clf,
        n_repeats=15, random_state=42, scoring="roc_auc", n_jobs=-1,
    )
    feat_imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": perm_result.importances_mean,
        "std": perm_result.importances_std,
    }).sort_values("importance", ascending=False)

    for _, r in feat_imp.iterrows():
        bar = "+" * max(1, int(r["importance"] * 200))
        print(f"  {r['feature']:<22} {r['importance']:>8.4f} +/- {r['std']:.4f}  {bar}")

    # ═════════════════════════════════════════════════════════════════════════
    #  FIGURE  (4 x 2 grid)
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(4, 2, figsize=(17, 28))
    fig.suptitle(
        "Optimised MLP Ensemble: P(AMP) vs MIC (E. coli)\n"
        f"Training: {n_amp} AMPs + {n_dec} Decoys | "
        f"{N_ENSEMBLE}-member bagged ensemble, isotonic calibration\n"
        f"Test: {len(all_test)} Buforin variants ({len(with_lit)} with MIC)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    colors_map  = {"i+4": "#4363d8", "i+7": "#e6194b",
                   "Original": "#3cb44b", "WT": "#f58231"}
    markers_map = {"i+4": "o", "i+7": "^", "Original": "s", "WT": "D"}

    # ── Panel A: Training AMPs — CV P(AMP) vs MIC ────────────────────────────
    ax = axes[0, 0]
    sc = ax.scatter(train_probs_cv_mic, y_mic_ugml, c=y_mic_pmic,
                    cmap="RdYlGn", s=50, alpha=0.6,
                    edgecolors="white", linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=ax, label="pMIC (higher = more potent)")

    slope, intercept, _, _, _ = stats.linregress(train_probs_cv_mic, log_mic)
    x_fit = np.linspace(max(0, train_probs_cv_mic.min() - 0.02),
                        min(1, train_probs_cv_mic.max() + 0.02), 100)
    y_fit = 10 ** (slope * x_fit + intercept)
    ax.plot(x_fit, y_fit, "--", color="#d62728", lw=2, alpha=0.7,
            label="log-linear fit")

    ax.set_yscale("log")
    ax.set_xlabel("P(AMP) — 5-fold CV probability (unbiased)", fontsize=10)
    ax.set_ylabel("MIC (ug/mL) [log scale]", fontsize=10)
    ax.set_title(f"Panel A: Training AMPs (n={n_mic}) — CV P(AMP)\n"
                 f"R={r_train_logmic_cv:.3f} | rho={rho_train_logmic_cv:.3f}  "
                 f"(P(AMP) vs log10 MIC)",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2, which="both")

    # ── Panel B: Buforin Test — Ensemble P(AMP) vs MIC ───────────────────────
    ax = axes[0, 1]

    ax.scatter(train_probs_ens, y_mic_ugml, alpha=0.15, s=20,
               color="#bbbbbb", edgecolors="none", zorder=1,
               label=f"Training AMPs (n={n_mic})")

    for v in with_lit:
        ax.scatter(v["prob_amp"], v["lit_ugml"], s=120,
                   marker=markers_map.get(v["group"], "o"),
                   color=colors_map.get(v["group"], "#333"),
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.errorbar(v["prob_amp"], v["lit_ugml"], xerr=v["prob_std"],
                    fmt="none", color="gray", alpha=0.5, capsize=3, zorder=4)
        ax.annotate(v["short"], (v["prob_amp"], v["lit_ugml"]),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=7.5, fontweight="bold", alpha=0.9)

    no_lit = [v for v in all_test if v["lit_ugml"] is None]
    for v in no_lit:
        ax.scatter(v["prob_amp"], 1.0, s=80, alpha=0.5,
                   marker=markers_map.get(v["group"], "o"),
                   color=colors_map.get(v["group"], "#333"),
                   edgecolors="k", linewidths=0.5, zorder=4)
        ax.annotate(v["short"] + "\n(no MIC)", (v["prob_amp"], 1.0),
                    textcoords="offset points", xytext=(7, -12),
                    fontsize=6.5, alpha=0.7, style="italic")

    if len(with_lit) >= 3:
        sl2, int2, _, _, _ = stats.linregress(buf_probs, buf_logmic)
        all_buf_probs = [v["prob_amp"] for v in all_test]
        x_f2 = np.linspace(
            max(0, min(buf_probs.min(), min(all_buf_probs)) - 0.02),
            min(1, max(buf_probs.max(), max(all_buf_probs)) + 0.02), 100)
        y_f2 = 10 ** (sl2 * x_f2 + int2)
        ax.plot(x_f2, y_f2, "--", color="#d62728", lw=1.5, alpha=0.6,
                label="Buf trend (log-linear)")

    buf_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4363d8",
               markeredgecolor="k", markersize=9, label="i+4 (F10W)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6194b",
               markeredgecolor="k", markersize=9, label="i+7 (F10W)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3cb44b",
               markeredgecolor="k", markersize=9, label="Original Buf"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#f58231",
               markeredgecolor="k", markersize=9, label="Buf WT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbb",
               markeredgecolor="none", markersize=7, label="Training AMPs"),
    ]
    ax.legend(handles=buf_legend, fontsize=7.5, loc="upper left")

    ax.set_yscale("log")
    ax.set_xlabel("P(AMP) — Ensemble mean (error bars = ensemble std)", fontsize=10)
    ax.set_ylabel("Literature MIC (ug/mL) [log scale]", fontsize=10)
    ax.set_title(f"Panel B: Buforin Test (n={len(with_lit)} with MIC)\n"
                 f"R={r_buf_logmic:.3f} | rho={rho_buf_logmic:.3f}  "
                 f"(P(AMP) vs log10 MIC)",
                 fontsize=9.5, fontweight="bold")
    ax.grid(alpha=0.2, which="both")

    # ── Panel C: P(AMP) vs pMIC (combined) ───────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(train_probs_cv_mic, y_mic_pmic, alpha=0.4, s=40,
               color="#2c7bb6", edgecolors="white", linewidths=0.2,
               zorder=2, label=f"Training AMPs (n={n_mic})")

    if len(with_lit) >= 3:
        for v in with_lit:
            pmic_v = mic_to_pmic(v["lit_ugml"], v["mw"])
            ax.scatter(v["prob_amp"], pmic_v, s=120,
                       marker=markers_map.get(v["group"], "o"),
                       color=colors_map.get(v["group"], "#333"),
                       edgecolors="k", linewidths=0.8, zorder=5)
            ax.errorbar(v["prob_amp"], pmic_v, xerr=v["prob_std"],
                        fmt="none", color="gray", alpha=0.5, capsize=3, zorder=4)
            ax.annotate(v["short"], (v["prob_amp"], pmic_v),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, fontweight="bold", alpha=0.9)

    sl3, int3, _, _, _ = stats.linregress(train_probs_cv_mic, y_mic_pmic)
    all_probs = np.concatenate([train_probs_cv_mic,
                                [v["prob_amp"] for v in all_test]])
    x_f3 = np.linspace(max(0, all_probs.min() - 0.02),
                       min(1, all_probs.max() + 0.02), 100)
    y_f3 = sl3 * x_f3 + int3
    ax.plot(x_f3, y_f3, "--", color="#d62728", lw=1.5, alpha=0.6,
            label=f"Training trend (slope={sl3:.2f})")

    ax.set_xlabel("P(AMP)", fontsize=10)
    ax.set_ylabel("pMIC = -log10(MIC_M)  [higher = more potent]", fontsize=10)
    ax.set_title(f"Panel C: P(AMP) vs pMIC (combined)\n"
                 f"Train: R={r_train_pmic_cv:.3f} | "
                 f"Buf: R={r_buf_pmic:.3f}",
                 fontsize=9.5, fontweight="bold")
    ax.legend(handles=buf_legend, fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.2)

    # ── Panel D: Bar chart with error bars ───────────────────────────────────
    ax = axes[1, 1]
    names = [v["short"] for v in all_test]
    probs = [v["prob_amp"] for v in all_test]
    stds  = [v["prob_std"] for v in all_test]

    x_pos = np.arange(len(all_test))
    bar_colors = [colors_map.get(v["group"], "#333") for v in all_test]

    bars = ax.bar(x_pos, probs, yerr=stds, capsize=4,
                  color=bar_colors, alpha=0.75,
                  edgecolor="k", linewidth=0.5,
                  error_kw={"elinewidth": 1.2, "capthick": 1.2})

    for i, (b, p, s) in enumerate(zip(bars, probs, stds)):
        y_pos = b.get_height() + s
        ax.text(b.get_x() + b.get_width()/2, y_pos + 0.015,
                f"{p:.2f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold")

    for i, v in enumerate(all_test):
        if v["lit_ugml"] is not None:
            ax.text(i, -0.03, f"{v['lit_ugml']:.1f}",
                    ha="center", va="top", fontsize=6.5, color="#d62728",
                    fontweight="bold")

    ax.axhline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P(AMP) — Ensemble mean +/- std", fontsize=10)
    ax.set_xlabel("Buforin Variant", fontsize=10)
    ax.set_title("Panel D: Ensemble P(AMP) per Buforin Variant\n"
                 "(error bars = ensemble std; red = lit MIC ug/mL)",
                 fontsize=9.5, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.15, axis="y")

    if not np.isnan(r_buf_logmic):
        box_text = (f"Buf P(AMP)-MIC\n"
                    f"R = {r_buf_logmic:.3f} (vs log10 MIC)\n"
                    f"rho = {rho_buf_logmic:.3f}\n"
                    f"n = {len(with_lit)}")
        ax.text(0.98, 0.98, box_text, transform=ax.transAxes,
                fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#ffffcc", edgecolor="#999", alpha=0.9))

    # ── Panel E: Convergence curves ──────────────────────────────────────────
    ax = axes[2, 0]
    curves = ensemble.convergence_curves()
    cmap_conv = plt.cm.viridis(np.linspace(0.2, 0.9, len(curves)))
    max_epoch = 0
    for i, lc in enumerate(curves):
        ax.plot(lc, color=cmap_conv[i], alpha=0.5, lw=0.8)
        max_epoch = max(max_epoch, len(lc))

    mean_len = int(np.median([len(c) for c in curves]))
    padded = []
    for c in curves:
        if len(c) >= mean_len:
            padded.append(c[:mean_len])
        else:
            padded.append(c + [c[-1]] * (mean_len - len(c)))
    mean_curve = np.mean(padded, axis=0)
    ax.plot(mean_curve, color="#d62728", lw=2.5, label="Ensemble mean", zorder=5)

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Training Loss", fontsize=10)
    ax.set_title(f"Panel E: Convergence ({N_ENSEMBLE} ensemble members)\n"
                 f"Median convergence: {mean_len} epochs | "
                 f"Max: {max_epoch}",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, max_epoch * 1.02)

    # ── Panel F: Feature importance ──────────────────────────────────────────
    ax = axes[2, 1]
    feat_imp_sorted = feat_imp.sort_values("importance", ascending=True)
    y_fi = np.arange(len(feat_imp_sorted))
    bars_fi = ax.barh(y_fi, feat_imp_sorted["importance"],
                      xerr=feat_imp_sorted["std"],
                      capsize=3, color="#5499C7", alpha=0.8,
                      edgecolor="k", linewidth=0.5,
                      error_kw={"elinewidth": 1, "capthick": 1})

    top3 = feat_imp_sorted.tail(3)["feature"].values
    for bar, fname in zip(bars_fi, feat_imp_sorted["feature"]):
        if fname in top3:
            bar.set_color("#E74C3C")
            bar.set_alpha(0.85)

    ax.set_yticks(y_fi)
    ax.set_yticklabels(feat_imp_sorted["feature"], fontsize=8.5)
    ax.set_xlabel("Permutation Importance (AUC drop)", fontsize=10)
    ax.set_title("Panel F: Feature Importance\n"
                 "(permutation, 15 repeats, top-3 highlighted)",
                 fontsize=9.5, fontweight="bold")
    ax.grid(alpha=0.2, axis="x")

    # ── Panel G: Confusion Matrix ────────────────────────────────────────────
    ax = axes[3, 0]
    cm_labels = ["Decoy", "AMP"]
    im = ax.imshow(test_cm, cmap="Blues", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(2):
        for j in range(2):
            val = test_cm[i, j]
            total_row = test_cm[i].sum()
            pct = val / total_row * 100 if total_row > 0 else 0
            color = "white" if val > test_cm.max() * 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.0f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(cm_labels, fontsize=11)
    ax.set_yticklabels(cm_labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Panel G: Confusion Matrix (20% hold-out test)\n"
                 f"n={len(X_te)} (AMPs={n_te_amp}, Decoys={n_te_dec})",
                 fontsize=9.5, fontweight="bold")

    # ── Panel H: ROC curve + metrics table ───────────────────────────────────
    ax = axes[3, 1]

    ax.plot(fpr, tpr, color="#2c7bb6", lw=2.5, label=f"MLP (AUC = {test_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2c7bb6")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Panel H: ROC Curve + Classification Metrics\n"
                 "(20% hold-out test set)",
                 fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    metrics_text = (
        f"{'Hold-out Test (20%)'}\n"
        f"{'─' * 28}\n"
        f"{'Accuracy':.<20} {test_acc:.3f}\n"
        f"{'Balanced Acc':.<20} {test_bacc:.3f}\n"
        f"{'F1 Score':.<20} {test_f1:.3f}\n"
        f"{'Precision':.<20} {test_prec:.3f}\n"
        f"{'Recall':.<20} {test_rec:.3f}\n"
        f"{'AUC-ROC':.<20} {test_auc:.3f}\n"
        f"{'MCC':.<20} {test_mcc:.3f}\n"
        f"{'─' * 28}\n"
        f"{'5-fold CV (mean)'}\n"
        f"{'─' * 28}\n"
        f"{'F1':.<20} {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}\n"
        f"{'Precision':.<20} {cv_prec.mean():.3f} +/- {cv_prec.std():.3f}\n"
        f"{'Recall':.<20} {cv_rec.mean():.3f} +/- {cv_rec.std():.3f}\n"
        f"{'AUC-ROC':.<20} {cv_auc_scores.mean():.3f} +/- {cv_auc_scores.std():.3f}"
    )
    ax.text(0.55, 0.45, metrics_text, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5",
                      facecolor="#f0f8ff", edgecolor="#4a90d9", alpha=0.95))

    plt.tight_layout()
    out = BASE / "buf_mic_mlp_prob.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  SUMMARY")
    print(f"{'='*72}")
    print(f"\n  Architecture:   hidden={best_hidden}, alpha={best_alpha}, lr={best_lr}")
    print(f"  GridSearch AUC: {gs.best_score_:.4f}")
    print(f"  CV Accuracy:    {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")
    print(f"  Ensemble OOB AUC: {np.mean(valid_aucs):.4f} +/- {np.std(valid_aucs):.4f}")
    print(f"\n  Hold-out Test (20%, n={len(X_te)}):")
    print(f"    Accuracy:     {test_acc:.3f}")
    print(f"    F1 Score:     {test_f1:.3f}")
    print(f"    Precision:    {test_prec:.3f}")
    print(f"    Recall:       {test_rec:.3f}")
    print(f"    AUC-ROC:      {test_auc:.3f}")
    print(f"    MCC:          {test_mcc:.3f}")
    print(f"\n  Training AMPs with MIC (n={n_mic}) — CV P(AMP):")
    print(f"    R  vs log10(MIC): {r_train_logmic_cv:.3f}  |  rho: {rho_train_logmic_cv:.3f}")
    print(f"    R  vs pMIC:       {r_train_pmic_cv:.3f}  |  rho: {rho_train_pmic_cv:.3f}")
    if not np.isnan(r_buf_logmic):
        print(f"\n  Buforin test (n={len(with_lit)}) — Ensemble P(AMP):")
        print(f"    R  vs log10(MIC): {r_buf_logmic:.3f}  |  rho: {rho_buf_logmic:.3f}")
        print(f"    R  vs pMIC:       {r_buf_pmic:.3f}  |  rho: {rho_buf_pmic:.3f}")
    print(f"\n  P(AMP) spread (Buforin): "
          f"{min(v['prob_amp'] for v in all_test):.3f} – "
          f"{max(v['prob_amp'] for v in all_test):.3f}")
    print(f"  Done.\n")


if __name__ == "__main__":
    main()
