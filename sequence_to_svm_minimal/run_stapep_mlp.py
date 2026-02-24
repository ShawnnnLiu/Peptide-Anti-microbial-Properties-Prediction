#!/usr/bin/env python3
"""
MLP classifier trained on stapled peptides (AMP vs Decoy).

Three feature sets:
  1. geometric  — ESMFold-derived geometric descriptors
  2. stapep     — StaPep MD/sequence features
  3. combined   — geometric ∪ stapep (concatenated)

Usage:
  conda run -n esm_env python run_stapep_mlp.py [--no-scale] [--cv-folds N]
"""

import warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

PATHS = {
    # training data
    "amp_stapep"    : STAPEP / "stapled_amps_features.csv",
    "decoy_stapep"  : STAPEP / "stapled_decoys.csv",
    "amp_geo"       : STAPEP / "stapep_amp_geometric.csv",
    "decoy_geo"     : STAPEP / "stapep_decoy_geometric.csv",
    # test data
    "test_stapep"   : STAPEP / "test_stapled_features.csv",
    "test_geo"      : STAPEP / "test_stapled_geometric.csv",
    # AMP index→DRAMP_ID map (from seqs file ordering)
    "amp_csv"       : STAPEP / "stapled_amps.csv",
}

# ── StaPep feature columns (common to AMP and Decoy CSVs) ─────────────────────
STAPEP_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]

# Geometric feature columns produced by extract_all_features()
GEO_COLS = [
    "plddt_mean", "plddt_std", "plddt_min", "plddt_max",
    "radius_gyration", "end_to_end_distance", "max_pairwise_distance",
    "centroid_distance_mean", "centroid_distance_std",
    "fraction_helix", "fraction_sheet", "fraction_coil",
    "total_sasa", "hydrophobic_sasa", "fraction_hydrophobic_sasa",
    "length", "net_charge", "mean_hydrophobicity", "hydrophobic_moment",
    "curvature_mean", "curvature_std", "curvature_max",
    "torsion_mean", "torsion_std",
]

TEST_NAMES = [
    "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    "Mag20", "Mag25", "Mag31", "Mag36",
]

# ── Data loading helpers ───────────────────────────────────────────────────────

def _keep_present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def load_training() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_amp_sp, df_dec_sp, df_amp_geo, df_dec_geo) as loaded DataFrames.

    Each DataFrame gets a 'join_id' column that can be used to merge sp ↔ geo:
      • AMPs   : join_id = DRAMP_ID
      • Decoys : join_id = row position (1-indexed) so it aligns with seq_index
                 in the geometry CSV (seq_index 1 = row 0 in stapled_decoys.csv).
    """
    # ── StaPep ──
    amp_sp = pd.read_csv(PATHS["amp_stapep"])
    dec_sp = pd.read_csv(PATHS["decoy_stapep"])

    amp_sp = amp_sp.rename(columns={"DRAMP_ID": "peptide_id",
                                     "Hiden_Sequence": "sequence"})
    dec_sp = dec_sp.rename(columns={"COMPOUND_ID": "peptide_id",
                                     "SEQUENCE": "sequence"})
    amp_sp["label"] = 1
    dec_sp["label"] = 0

    # join key for merging with geometry
    amp_sp["join_id"] = amp_sp["peptide_id"].astype(str)        # DRAMP_ID
    dec_sp["join_id"] = (dec_sp.index + 1).astype(str)          # 1-based row pos

    sp_cols_present_amp = _keep_present(amp_sp, STAPEP_COLS)
    amp_sp = amp_sp.dropna(subset=sp_cols_present_amp, how="all")
    sp_cols_present_dec = _keep_present(dec_sp, STAPEP_COLS)
    dec_sp = dec_sp.dropna(subset=sp_cols_present_dec, how="all")

    # ── Geometric ──
    amp_geo = pd.DataFrame()
    dec_geo = pd.DataFrame()

    if PATHS["amp_geo"].exists():
        amp_geo = pd.read_csv(PATHS["amp_geo"])
        # external_id column holds DRAMP_ID (added by build_stapep_geo.py)
        if "external_id" in amp_geo.columns:
            amp_geo["join_id"] = amp_geo["external_id"].astype(str)
        else:
            # fallback: use seq_index → map to DRAMP_ID via AMP CSV
            amp_csv = pd.read_csv(PATHS["amp_csv"])
            idx2id  = {str(i + 1): str(row["DRAMP_ID"]) for i, row in amp_csv.iterrows()}
            amp_geo["join_id"] = amp_geo["seq_index"].astype(str).map(idx2id)
        amp_geo["label"] = 1
    else:
        print("  ⚠️  AMP geometric features not found — run build_stapep_geo.py first")

    if PATHS["decoy_geo"].exists():
        dec_geo = pd.read_csv(PATHS["decoy_geo"])
        dec_geo["label"] = 0
        # seq_index in dec_geo = 1-based row number in stapled_decoys.csv
        dec_geo["join_id"] = dec_geo["seq_index"].astype(str)
    else:
        print("  ⚠️  Decoy geometric features not found — run build_stapep_geo.py first")

    return amp_sp, dec_sp, amp_geo, dec_geo


def load_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (test_stapep, test_geo)."""
    test_sp  = pd.DataFrame()
    test_geo = pd.DataFrame()

    if PATHS["test_stapep"].exists():
        test_sp = pd.read_csv(PATHS["test_stapep"])
    else:
        print("  ⚠️  Test StaPep features not found — run run_test_stapep_md.py in WSL first")

    if PATHS["test_geo"].exists():
        test_geo = pd.read_csv(PATHS["test_geo"])
    else:
        print("  ⚠️  Test geometric features not found — run build_stapep_geo.py first")

    return test_sp, test_geo


# ── feature matrix builders ────────────────────────────────────────────────────

def build_sp_matrix(amp_sp: pd.DataFrame, dec_sp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Build (X, y, feature_names) for StaPep features."""
    cols = list(dict.fromkeys(_keep_present(amp_sp, STAPEP_COLS) +
                               _keep_present(dec_sp, STAPEP_COLS)))
    train = pd.concat([amp_sp[["peptide_id", "label"] + cols],
                       dec_sp[["peptide_id", "label"] + cols]], ignore_index=True)
    X = train[cols].values.astype(float)
    y = train["label"].values
    return X, y, pd.Index(cols)


def build_geo_matrix(amp_geo: pd.DataFrame, dec_geo: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Build (X, y, feature_names) for geometric features."""
    cols = list(dict.fromkeys(_keep_present(amp_geo, GEO_COLS) +
                               _keep_present(dec_geo, GEO_COLS)))
    train = pd.concat([amp_geo[["peptide_id", "label"] + cols],
                       dec_geo[["peptide_id", "label"] + cols]], ignore_index=True)
    X = train[cols].values.astype(float)
    y = train["label"].values
    return X, y, pd.Index(cols)


def build_combined_matrix(amp_sp: pd.DataFrame, dec_sp: pd.DataFrame,
                           amp_geo: pd.DataFrame, dec_geo: pd.DataFrame
                           ) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Inner-join on join_id so only peptides with BOTH feature sets are used.
    join_id is DRAMP_ID for AMPs and 1-based row index for Decoys.
    """
    sp_cols  = _keep_present(amp_sp, STAPEP_COLS)
    geo_cols = (_keep_present(amp_geo, GEO_COLS) if not amp_geo.empty
                else _keep_present(dec_geo, GEO_COLS))

    # Only take geo columns that don't duplicate a StaPep column name
    geo_only = [c for c in geo_cols if c not in sp_cols]

    # Merge AMPs on join_id (DRAMP_ID)
    amp_geo_cols = [c for c in geo_only if c in amp_geo.columns]
    amps = (amp_sp[["join_id", "label"] + sp_cols]
            .merge(amp_geo[["join_id"] + amp_geo_cols], on="join_id", how="inner"))

    # Merge Decoys on join_id (1-based row index)
    dec_geo_cols = [c for c in geo_only if c in dec_geo.columns]
    decs = (dec_sp[["join_id", "label"] + _keep_present(dec_sp, STAPEP_COLS)]
            .merge(dec_geo[["join_id"] + dec_geo_cols], on="join_id", how="inner"))

    train = pd.concat([amps, decs], ignore_index=True)
    all_cols = sp_cols + [c for c in geo_only if c in train.columns]
    X = train[all_cols].values.astype(float)
    y = train["label"].values
    return X, y, pd.Index(all_cols)


# ── MLP pipeline builder ───────────────────────────────────────────────────────
def make_mlp() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("mlp",     MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=3e-4,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
        )),
    ])


# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate_cv(X: np.ndarray, y: np.ndarray, model_name: str, cv: int = 5) -> None:
    clf = make_mlp()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, scoring="roc_auc", cv=skf, n_jobs=-1)
    aps  = cross_val_score(clf, X, y, scoring="average_precision", cv=skf, n_jobs=-1)
    print(f"  {model_name:<22s}: AUC={aucs.mean():.3f}±{aucs.std():.3f}  "
          f"AP={aps.mean():.3f}±{aps.std():.3f}  (n={len(y)})")


def fit_and_predict(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray) -> np.ndarray:
    """Return P(AMP=1) for each test peptide."""
    clf = make_mlp()
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# ── test vector builders ───────────────────────────────────────────────────────
def test_sp_vector(test_sp: pd.DataFrame, feature_names: pd.Index) -> np.ndarray:
    """Align test StaPep rows to training feature order."""
    rows = []
    for name in TEST_NAMES:
        row = test_sp[test_sp["peptide_id"] == name]
        if row.empty:
            rows.append([np.nan] * len(feature_names))
        else:
            rows.append([float(row.iloc[0].get(c, np.nan)) for c in feature_names])
    return np.array(rows)


def test_geo_vector(test_geo: pd.DataFrame, feature_names: pd.Index) -> np.ndarray:
    rows = []
    for name in TEST_NAMES:
        row = test_geo[test_geo["peptide_id"] == name]
        if row.empty:
            rows.append([np.nan] * len(feature_names))
        else:
            rows.append([float(row.iloc[0].get(c, np.nan)) for c in feature_names])
    return np.array(rows)


def test_combined_vector(test_sp: pd.DataFrame, test_geo: pd.DataFrame,
                          feature_names: pd.Index) -> np.ndarray:
    rows = []
    for name in TEST_NAMES:
        sp_row  = test_sp[test_sp["peptide_id"] == name]
        geo_row = test_geo[test_geo["peptide_id"] == name]
        vals = []
        for c in feature_names:
            v = np.nan
            if not sp_row.empty and c in sp_row.columns:
                v2 = sp_row.iloc[0].get(c, np.nan)
                if not pd.isna(v2):
                    v = float(v2)
            if pd.isna(v) and not geo_row.empty and c in geo_row.columns:
                v2 = geo_row.iloc[0].get(c, np.nan)
                if not pd.isna(v2):
                    v = float(v2)
            vals.append(v)
        rows.append(vals)
    return np.array(rows)


# ── display helpers ────────────────────────────────────────────────────────────
def print_predictions(probs_dict: dict[str, np.ndarray]) -> None:
    """Print a table of P(AMP) for each test peptide across all models."""
    models = list(probs_dict.keys())
    headers = ["Peptide"] + [f"P(+1) {m}" for m in models]
    col_w   = max(len(h) for h in headers) + 2

    header_line = "".join(h.ljust(col_w) for h in headers)
    print("\n" + "─" * len(header_line))
    print(header_line)
    print("─" * len(header_line))

    for i, name in enumerate(TEST_NAMES):
        parts = [name.ljust(col_w)]
        for m in models:
            p = probs_dict[m][i]
            flag = " ✓" if p >= 0.5 else "  "
            parts.append(f"{p:.3f}{flag}".ljust(col_w))
        print("".join(parts))
    print("─" * len(header_line))
    print("✓ = predicted AMP  (P ≥ 0.50)\n")


# ── main ───────────────────────────────────────────────────────────────────────
def main(cv_folds: int = 5):
    print("\n" + "=" * 65)
    print("  Stapled Peptide MLP — Three Feature-Set Comparison")
    print("=" * 65)

    amp_sp, dec_sp, amp_geo, dec_geo = load_training()
    test_sp, test_geo                = load_test()

    sp_ok  = not amp_geo.empty or not dec_geo.empty
    geo_ok = not amp_geo.empty or not dec_geo.empty

    print(f"\n  Training AMPs  : {len(amp_sp)} StaPep  /  {len(amp_geo)} Geo")
    print(f"  Training Decoys: {len(dec_sp)} StaPep  /  {len(dec_geo)} Geo")
    print(f"  Test peptides  : {len(test_sp)} StaPep  /  {len(test_geo)} Geo")

    # ── Cross-validated AUC ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Cross-validation ({cv_folds}-fold StratifiedKFold)")
    print(f"{'─'*65}")

    X_sp, y_sp, sp_names   = build_sp_matrix(amp_sp, dec_sp)
    evaluate_cv(X_sp, y_sp, "StaPep only", cv=cv_folds)

    if not amp_geo.empty and not dec_geo.empty:
        X_geo, y_geo, geo_names = build_geo_matrix(amp_geo, dec_geo)
        evaluate_cv(X_geo, y_geo, "Geometric only", cv=cv_folds)

        X_comb, y_comb, comb_names = build_combined_matrix(amp_sp, dec_sp, amp_geo, dec_geo)
        evaluate_cv(X_comb, y_comb, "Combined (Geo+StaPep)", cv=cv_folds)
    else:
        print("  Geometric or combined models skipped (missing geo CSVs).")
        geo_names  = pd.Index([])
        comb_names = pd.Index([])
        X_geo = y_geo = X_comb = y_comb = None

    # ── Fit final models on ALL training data, predict test ───────────────────
    print(f"\n{'─'*65}")
    print("  Final predictions on test peptides (trained on all data)")
    print(f"{'─'*65}")

    probs_dict = {}

    # StaPep
    if not test_sp.empty:
        Xt = test_sp_vector(test_sp, sp_names)
        probs_dict["StaPep"] = fit_and_predict(X_sp, y_sp, Xt)
    else:
        print("  StaPep test predictions skipped (no test_stapled_features.csv).")

    # Geometric
    if X_geo is not None and not test_geo.empty:
        Xt_geo = test_geo_vector(test_geo, geo_names)
        probs_dict["Geometric"] = fit_and_predict(X_geo, y_geo, Xt_geo)
    elif not test_geo.empty:
        Xt_geo = test_geo_vector(test_geo, geo_names)

    # Combined
    if X_comb is not None and not test_sp.empty and not test_geo.empty:
        Xt_comb = test_combined_vector(test_sp, test_geo, comb_names)
        probs_dict["Geo+StaPep"] = fit_and_predict(X_comb, y_comb, Xt_comb)

    if probs_dict:
        print_predictions(probs_dict)

        # Save predictions
        out_df = pd.DataFrame({"peptide_id": TEST_NAMES})
        for m, probs in probs_dict.items():
            out_df[f"P_AMP_{m}"] = probs
        out_path = STAPEP / "test_stapled_mlp_predictions.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Results saved → {out_path}\n")
    else:
        print("  No predictions made (missing feature files).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cv-folds", type=int, default=5)
    args = p.parse_args()
    main(cv_folds=args.cv_folds)
