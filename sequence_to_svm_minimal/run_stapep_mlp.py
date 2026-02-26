#!/usr/bin/env python3
"""
run_stapep_mlp.py
=================
MLP classifier trained on stapled peptides (AMP vs Decoy).

Seven feature sets compared:
  1. StaPep      — StaPep MD / sequence features (17 features)
  2. Geo         — ESMFold geometric descriptors  (24 features)
  3. QSAR        — 12 descriptors from 2016 PNAS SVM paper
  4. Geo+StaPep  — Geometric ∪ StaPep
  5. QSAR+Geo    — QSAR ∪ Geometric
  6. QSAR+StaPep — QSAR ∪ StaPep
  7. QSAR+G+SP   — All three combined

Usage
-----
  conda run -n esm_env python run_stapep_mlp.py [--cv-folds N]
"""

import warnings, argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

PATHS = {
    # training — StaPep MD features
    "amp_stapep"    : STAPEP / "stapled_amps_features.csv",
    "decoy_stapep"  : STAPEP / "stapled_decoys.csv",
    # training — Geometric
    "amp_geo"       : STAPEP / "stapep_amp_geometric.csv",
    "decoy_geo"     : STAPEP / "stapep_decoy_geometric.csv",
    # training — QSAR (2016 PNAS paper)
    "amp_qsar"      : STAPEP / "qsar_stapled_amps.csv",
    "decoy_qsar"    : STAPEP / "qsar_stapled_decoys.csv",
    # test
    "test_stapep"   : STAPEP / "test_stapled_features.csv",
    "test_geo"      : STAPEP / "test_stapled_geometric.csv",
    "test_qsar"     : STAPEP / "qsar_stapled_test.csv",
    # metadata
    "amp_csv"       : STAPEP / "stapled_amps.csv",
}

# ── Feature column lists ───────────────────────────────────────────────────────
STAPEP_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]

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

QSAR_COLS = [
    "netCharge", "FC", "LW", "DP", "NK", "AE", "pcMK",
    "_SolventAccessibilityD1025",
    "tau2_GRAR740104", "tau4_GRAR740104",
    "QSO50_GRAR740104", "QSO29_GRAR740104",
]

# Short display names for each model (used in table header)
MODEL_SHORT = {
    "StaPep"     : "StaPep",
    "Geo"        : "Geo",
    "QSAR"       : "QSAR",
    "Geo+SP"     : "Geo+SP",
    "QSAR+Geo"   : "QSAR+Geo",
    "QSAR+SP"    : "QSAR+SP",
    "QSAR+G+SP"  : "QSAR+G+SP",
}

# Ordered display name for 8 test peptides
TEST_NAMES = [
    "Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
    "Mag20",  "Mag25", "Mag31",     "Mag36",
]

# QSAR CSV uses longer names — map them back to TEST_NAMES
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


# ── Utilities ─────────────────────────────────────────────────────────────────
def _present(df: pd.DataFrame, cols: list) -> list:
    """Return only columns that exist in df."""
    return [c for c in cols if c in df.columns]


def _dedup_cols(primary: list, secondary: list) -> list:
    """Return secondary columns that don't duplicate a primary column."""
    seen = set(primary)
    return [c for c in secondary if c not in seen]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_training():
    """
    Load all three feature-set DataFrames for AMPs and Decoys.

    join_id convention
    ------------------
    AMPs   : DRAMP_ID  (identical across all three DFs)
    Decoys : 1-based row number in stapled_decoys.csv
             (consistent across SP, Geo, QSAR because all were generated
              from the same row-ordered CSV)
    """
    # ── StaPep ──────────────────────────────────────────────────────────────
    amp_sp  = pd.read_csv(PATHS["amp_stapep"])
    dec_sp  = pd.read_csv(PATHS["decoy_stapep"])
    amp_sp  = amp_sp.rename(columns={"DRAMP_ID": "peptide_id",
                                      "Hiden_Sequence": "sequence"})
    dec_sp  = dec_sp.rename(columns={"COMPOUND_ID": "peptide_id",
                                      "SEQUENCE": "sequence"})
    amp_sp["label"]   = 1
    dec_sp["label"]   = 0
    amp_sp["join_id"] = amp_sp["peptide_id"].astype(str)
    dec_sp["join_id"] = (dec_sp.reset_index(drop=True).index + 1).astype(str)

    sp_ok_amp = _present(amp_sp, STAPEP_COLS)
    sp_ok_dec = _present(dec_sp, STAPEP_COLS)
    amp_sp = amp_sp.dropna(subset=sp_ok_amp, how="all")
    dec_sp = dec_sp.dropna(subset=sp_ok_dec, how="all")

    # ── Geometric ───────────────────────────────────────────────────────────
    amp_geo = pd.DataFrame()
    dec_geo = pd.DataFrame()

    if PATHS["amp_geo"].exists():
        amp_geo = pd.read_csv(PATHS["amp_geo"])
        if "external_id" in amp_geo.columns:
            amp_geo["join_id"] = amp_geo["external_id"].astype(str)
        else:
            amp_csv = pd.read_csv(PATHS["amp_csv"])
            idx2id  = {str(i + 1): str(r["DRAMP_ID"]) for i, r in amp_csv.iterrows()}
            amp_geo["join_id"] = amp_geo["seq_index"].astype(str).map(idx2id)
        amp_geo["label"] = 1
    else:
        print("  ⚠  AMP geometric file not found — run build_stapep_geo.py first")

    if PATHS["decoy_geo"].exists():
        dec_geo = pd.read_csv(PATHS["decoy_geo"])
        dec_geo["label"]   = 0
        dec_geo["join_id"] = dec_geo["seq_index"].astype(str)
    else:
        print("  ⚠  Decoy geometric file not found — run build_stapep_geo.py first")

    # ── QSAR ────────────────────────────────────────────────────────────────
    amp_qsar = pd.DataFrame()
    dec_qsar = pd.DataFrame()

    if PATHS["amp_qsar"].exists():
        amp_qsar = pd.read_csv(PATHS["amp_qsar"])
        amp_qsar["label"]   = 1
        amp_qsar["join_id"] = amp_qsar["peptide_id"].astype(str)   # DRAMP_ID
    else:
        print("  ⚠  AMP QSAR file not found — run extract_stapep_qsar.py first")

    if PATHS["decoy_qsar"].exists():
        dec_qsar = pd.read_csv(PATHS["decoy_qsar"])
        dec_qsar["label"]   = 0
        dec_qsar["join_id"] = (dec_qsar.reset_index(drop=True).index + 1).astype(str)
    else:
        print("  ⚠  Decoy QSAR file not found — run extract_stapep_qsar.py first")

    return amp_sp, dec_sp, amp_geo, dec_geo, amp_qsar, dec_qsar


def load_test():
    """Load all three feature-set DataFrames for the 8 test peptides."""
    test_sp   = pd.DataFrame()
    test_geo  = pd.DataFrame()
    test_qsar = pd.DataFrame()

    if PATHS["test_stapep"].exists():
        test_sp = pd.read_csv(PATHS["test_stapep"])
    else:
        print("  ⚠  Test StaPep features not found — run run_test_stapep_md.py in WSL")

    if PATHS["test_geo"].exists():
        test_geo = pd.read_csv(PATHS["test_geo"])
    else:
        print("  ⚠  Test geometric features not found — run build_stapep_geo.py first")

    if PATHS["test_qsar"].exists():
        test_qsar = pd.read_csv(PATHS["test_qsar"])
        # Normalise peptide_id → short TEST_NAMES format
        test_qsar["peptide_id"] = (test_qsar["peptide_id"]
                                   .map(QSAR_TEST_NAME_MAP)
                                   .fillna(test_qsar["peptide_id"]))
    else:
        print("  ⚠  Test QSAR features not found — run extract_stapep_qsar.py first")

    return test_sp, test_geo, test_qsar


# ── Feature matrix builders ───────────────────────────────────────────────────
def _concat_merge(amp_frames: list, dec_frames: list,
                  col_sets:  list) -> tuple:
    """
    Inner-join multiple feature-set DataFrames on join_id.

    amp_frames / dec_frames : list of DataFrames (all must have join_id + label)
    col_sets                : list of column-name lists (one per DataFrame)

    Returns (X, y, feature_names).
    """
    # Build de-duplicated combined column list
    all_cols: list = []
    for cols in col_sets:
        all_cols += _dedup_cols(all_cols, cols)

    def _merge_group(frames: list, col_sets_local: list) -> pd.DataFrame:
        base   = frames[0]
        b_cols = _present(base, _dedup_cols([], col_sets_local[0]))
        merged = base[["join_id", "label"] + b_cols].copy()
        used   = set(b_cols)
        for df, cs in zip(frames[1:], col_sets_local[1:]):
            new_cs = [c for c in cs if c not in used and c in df.columns]
            if new_cs:
                merged = merged.merge(df[["join_id"] + new_cs],
                                      on="join_id", how="inner")
            used.update(new_cs)
        return merged

    amp_merged = _merge_group(amp_frames, col_sets)
    dec_merged = _merge_group(dec_frames, col_sets)

    train    = pd.concat([amp_merged, dec_merged], ignore_index=True)
    fin_cols = [c for c in all_cols if c in train.columns]
    X = train[fin_cols].values.astype(float)
    y = train["label"].values
    return X, y, pd.Index(fin_cols)


def build_sp_matrix(amp_sp, dec_sp):
    sp_amp = _present(amp_sp, STAPEP_COLS)
    sp_dec = _present(dec_sp, STAPEP_COLS)
    cols   = list(dict.fromkeys(sp_amp + sp_dec))
    train  = pd.concat([amp_sp[["label"] + cols],
                        dec_sp[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, pd.Index(cols)


def build_geo_matrix(amp_geo, dec_geo):
    g_amp = _present(amp_geo, GEO_COLS)
    g_dec = _present(dec_geo, GEO_COLS)
    cols  = list(dict.fromkeys(g_amp + g_dec))
    train = pd.concat([amp_geo[["label"] + cols],
                       dec_geo[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, pd.Index(cols)


def build_qsar_matrix(amp_qsar, dec_qsar):
    q_amp = _present(amp_qsar, QSAR_COLS)
    q_dec = _present(dec_qsar, QSAR_COLS)
    cols  = list(dict.fromkeys(q_amp + q_dec))
    train = pd.concat([amp_qsar[["label"] + cols],
                       dec_qsar[["label"] + cols]], ignore_index=True)
    return train[cols].values.astype(float), train["label"].values, pd.Index(cols)


def build_geo_sp_matrix(amp_sp, dec_sp, amp_geo, dec_geo):
    sp_cols  = _present(amp_sp, STAPEP_COLS)
    geo_only = _dedup_cols(sp_cols, _present(amp_geo, GEO_COLS))
    return _concat_merge(
        [amp_sp,  amp_geo],
        [dec_sp,  dec_geo],
        [sp_cols, geo_only],
    )


def build_qsar_geo_matrix(amp_geo, dec_geo, amp_qsar, dec_qsar):
    geo_cols  = _present(amp_geo, GEO_COLS)
    qsar_only = _dedup_cols(geo_cols, QSAR_COLS)
    return _concat_merge(
        [amp_geo,  amp_qsar],
        [dec_geo,  dec_qsar],
        [geo_cols, qsar_only],
    )


def build_qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar):
    sp_cols   = _present(amp_sp, STAPEP_COLS)
    qsar_only = _dedup_cols(sp_cols, QSAR_COLS)
    return _concat_merge(
        [amp_sp,  amp_qsar],
        [dec_sp,  dec_qsar],
        [sp_cols, qsar_only],
    )


def build_qsar_geo_sp_matrix(amp_sp, dec_sp, amp_geo, dec_geo,
                              amp_qsar, dec_qsar):
    sp_cols   = _present(amp_sp, STAPEP_COLS)
    geo_only  = _dedup_cols(sp_cols,          _present(amp_geo, GEO_COLS))
    qsar_only = _dedup_cols(sp_cols + geo_only, QSAR_COLS)
    return _concat_merge(
        [amp_sp,  amp_geo,  amp_qsar],
        [dec_sp,  dec_geo,  dec_qsar],
        [sp_cols, geo_only, qsar_only],
    )


# ── MLP pipeline ──────────────────────────────────────────────────────────────
def make_mlp() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("mlp",     MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=80,
            tol=1e-6,
            random_state=42,
        )),
    ])


# ── Evaluation helpers ─────────────────────────────────────────────────────────
def cv_score(X, y, label: str, folds: int) -> dict:
    clf = make_mlp()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, scoring="roc_auc",           cv=skf, n_jobs=-1)
    aps  = cross_val_score(clf, X, y, scoring="average_precision", cv=skf, n_jobs=-1)
    return {"label": label, "n": len(y),
            "auc_mean": aucs.mean(), "auc_std": aucs.std(),
            "ap_mean":  aps.mean(),  "ap_std":  aps.std()}


def fit_predict(X_train, y_train, X_test) -> np.ndarray:
    clf = make_mlp()
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


# ── Test-set vector builder ───────────────────────────────────────────────────
def _lookup_row(df: pd.DataFrame, name: str, cols) -> list:
    """Return feature values for test peptide `name` from df."""
    row = df[df["peptide_id"] == name]
    if row.empty:
        return [np.nan] * len(cols)
    return [float(row.iloc[0].get(c, np.nan)) for c in cols]


def build_test_matrix(feature_names: pd.Index, *dfs: pd.DataFrame) -> np.ndarray:
    """
    Build test matrix (8 × n_features) by looking up each feature in each df
    in order — the first df that has a non-NaN value wins.
    """
    rows = []
    for name in TEST_NAMES:
        vals = []
        for col in feature_names:
            v = np.nan
            for df in dfs:
                if df.empty or "peptide_id" not in df.columns:
                    continue
                if col not in df.columns:
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


# ── Pretty results table ──────────────────────────────────────────────────────
def print_cv_table(results: list[dict]) -> None:
    """Print cross-validation results as a tidy table."""
    col1_w = max(len(r["label"]) for r in results) + 2
    print(f"\n  {'Model':<{col1_w}}  {'AUC':>12}   {'Avg Prec':>12}   {'n':>5}")
    print(f"  {'─'*col1_w}  {'─'*12}   {'─'*12}   {'─'*5}")
    for r in results:
        print(f"  {r['label']:<{col1_w}}  "
              f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}   "
              f"{r['ap_mean']:.3f} ± {r['ap_std']:.3f}   "
              f"{r['n']:>5}")


def print_prediction_table(probs_dict: dict) -> None:
    """Print P(AMP) predictions for each test peptide in a clean grid."""
    models = list(probs_dict.keys())

    # Column widths
    pep_w    = max(len(n) for n in TEST_NAMES) + 1          # peptide column
    mod_w    = max(max(len(m) for m in models), 8) + 2      # model columns

    # ── Header ────────────────────────────────────────────────────────────
    sep  = "─" * (pep_w + 1 + len(models) * (mod_w + 1))
    hdr  = f"  {'Peptide':<{pep_w}}"
    for m in models:
        hdr += f" {m:^{mod_w}}"
    note = f"  {'':.<{pep_w}}"
    for m in models:
        note += f" {'(n_feat=' + str(len(probs_dict[m])) + ')':^{mod_w}}"

    print(f"\n  {sep}")
    print(hdr)
    print(f"  {sep}")

    # ── Rows ──────────────────────────────────────────────────────────────
    votes = [0] * len(TEST_NAMES)
    for i, name in enumerate(TEST_NAMES):
        row = f"  {name:<{pep_w}}"
        n_pos = 0
        for m in models:
            p    = probs_dict[m][i]
            flag = "✓" if p >= 0.5 else " "
            cell = f"{p:.3f}{flag}"
            row += f" {cell:^{mod_w}}"
            if p >= 0.5:
                n_pos += 1
                votes[i] += 1
        row += f"   ← {n_pos}/{len(models)} models"
        print(row)

    print(f"  {sep}")
    print(f"  {'':.<{pep_w}}", end="")
    for m in models:
        print(f" {'↑ P(AMP)':^{mod_w}}", end="")
    print(f"\n  ✓ = P(AMP) ≥ 0.50   |   rightmost column = # models voting AMP\n")

    # ── Majority-vote summary ─────────────────────────────────────────────
    half = len(models) / 2
    amps_predicted = [TEST_NAMES[i] for i, v in enumerate(votes) if v > half]
    non_amps       = [TEST_NAMES[i] for i, v in enumerate(votes) if v <= half]
    print(f"  Majority vote  ({len(models)} models, threshold > {half:.0f}):")
    if amps_predicted:
        print(f"    Predicted AMP : {', '.join(amps_predicted)}")
    if non_amps:
        print(f"    Predicted non-AMP : {', '.join(non_amps)}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
def main(cv_folds: int = 5):
    print("\n" + "=" * 70)
    print("  Stapled Peptide MLP — Seven Feature-Set Comparison")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    amp_sp, dec_sp, amp_geo, dec_geo, amp_qsar, dec_qsar = load_training()
    test_sp, test_geo, test_qsar = load_test()

    geo_avail  = not amp_geo.empty  and not dec_geo.empty
    qsar_avail = not amp_qsar.empty and not dec_qsar.empty

    print(f"\n  Training data:")
    print(f"    StaPep  — AMPs: {len(amp_sp):>3}   Decoys: {len(dec_sp):>3}")
    if geo_avail:
        print(f"    Geo     — AMPs: {len(amp_geo):>3}   Decoys: {len(dec_geo):>3}")
    if qsar_avail:
        print(f"    QSAR    — AMPs: {len(amp_qsar):>3}   Decoys: {len(dec_qsar):>3}")
    print(f"\n  Test peptides  — StaPep:{len(test_sp)}  Geo:{len(test_geo)}  QSAR:{len(test_qsar)}")

    # ── Build matrices ─────────────────────────────────────────────────────
    models_train: dict[str, tuple] = {}

    models_train["StaPep"] = build_sp_matrix(amp_sp, dec_sp)

    if geo_avail:
        models_train["Geo"]      = build_geo_matrix(amp_geo, dec_geo)
        models_train["Geo+SP"]   = build_geo_sp_matrix(amp_sp, dec_sp, amp_geo, dec_geo)

    if qsar_avail:
        models_train["QSAR"]     = build_qsar_matrix(amp_qsar, dec_qsar)
        if geo_avail:
            models_train["QSAR+Geo"] = build_qsar_geo_matrix(amp_geo, dec_geo, amp_qsar, dec_qsar)
        models_train["QSAR+SP"]  = build_qsar_sp_matrix(amp_sp, dec_sp, amp_qsar, dec_qsar)
        if geo_avail:
            models_train["QSAR+G+SP"] = build_qsar_geo_sp_matrix(
                amp_sp, dec_sp, amp_geo, dec_geo, amp_qsar, dec_qsar)

    # ── Cross-validation ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Cross-Validation ({cv_folds}-fold Stratified)  —  AUC & Average Precision")
    print(f"{'─'*70}")

    cv_results = []
    for name, (X, y, _) in models_train.items():
        res = cv_score(X, y, name, folds=cv_folds)
        cv_results.append(res)

    print_cv_table(cv_results)

    # ── Final predictions on test set ─────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Final Predictions on Test Peptides  (fit on all training data)")
    print(f"{'─'*70}")

    probs_dict: dict[str, np.ndarray] = {}

    for name, (X_tr, y_tr, feat_names) in models_train.items():
        # Decide which test DFs to pull features from
        dfs = []
        if "SP"     in name or name == "StaPep":  dfs.append(test_sp)
        if "Geo"    in name or name == "Geo":     dfs.append(test_geo)
        if "QSAR"   in name:                      dfs.append(test_qsar)
        if not dfs:
            dfs = [test_sp]

        X_te = build_test_matrix(feat_names, *dfs)
        probs_dict[name] = fit_predict(X_tr, y_tr, X_te)

    # ── Display + save ─────────────────────────────────────────────────────
    print_prediction_table(probs_dict)

    out_df = pd.DataFrame({"peptide_id": TEST_NAMES})
    for m, probs in probs_dict.items():
        out_df[f"P_AMP_{m}"] = np.round(probs, 4)
    out_path = STAPEP / "test_stapled_mlp_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of CV folds (default: 5)")
    args = p.parse_args()
    main(cv_folds=args.cv_folds)
