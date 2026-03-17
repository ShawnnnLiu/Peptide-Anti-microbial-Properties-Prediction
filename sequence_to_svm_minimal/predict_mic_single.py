#!/usr/bin/env python3
"""
predict_mic_single.py
=====================
Predict MIC tier for a single peptide sequence using three models
(Random Forest, RBF-SVM, MLP) trained on StaPep features.

Feature lookup strategy
-----------------------
  1. Normalize the input sequence (strip staple markers: S5, R8, X, etc.)
     and search stapled_amps_features.csv and test_stapled_features.csv
     for an exact amino-acid match.
  2. If found  → use the pre-computed StaPep features directly.
  3. If not found → compute the 9 sequence-level features with
     BioPython ProteinAnalysis; leave the 8 MD/structure features as NaN
     (the pipeline's median imputer fills them from training data).

Tiers (μM E. coli MIC)
-----------------------
  Very strong : < 2      Strong   : 2–5
  Moderate    : 5–10     Weak     : > 10

Usage
-----
  conda run -n esm_env python predict_mic_single.py TRSSRAGLQWPVGRVHRLLRK
  conda run -n esm_env python predict_mic_single.py TRSSRAGLQWPVGRVHRLLRK --id BufParent

  # Supply all 17 pre-computed StaPep features directly (skips DB lookup + BioPython):
  conda run -n esm_env python predict_mic_single.py TRSSRAGLQWPVGRVHRLLRK --id BufNative \\
    --full-features '{"length":21,"weight":2473.829,"hydrophobic_index":-0.8143,
    "charge":6.094,"aromaticity":0.04762,"isoelectric_point":12.0,
    "fraction_arginine":0.2381,"fraction_lysine":0.04762,"lyticity_index":300.106,
    "helix_percent":0.0,"sheet_percent":0.0,"loop_percent":1.0,
    "mean_bfactor":176.184,"mean_gyrate":18.249,"num_hbonds":0,
    "psa":1137.781,"sasa":2334.656}'
"""

from __future__ import annotations

import json
import re
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from Bio.SeqUtils.ProtParam import ProteinAnalysis

from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

FEAT_CSV  = STAPEP / "stapled_amps_features.csv"
AMP_CSV   = STAPEP / "stapled_amps.csv"
TEST_CSV  = STAPEP / "test_stapled_features.csv"

# ── Feature columns ───────────────────────────────────────────────────────────
SEQ_FEATURES = [                 # computable from sequence alone
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index",
]
MD_FEATURES = [                  # require MD simulation — imputed if missing
    "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]
FEATURES = SEQ_FEATURES + MD_FEATURES

# Kyte–Doolittle hydrophobicity scale
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
       "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,  "L": 3.8,  "K": -3.9,
       "M": 1.9,  "F": 2.8,  "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
       "Y": -1.3, "V": 4.2}

# Hydrophobic-moment scale (Eisenberg consensus, used for helical moment)
_EISEN = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
          "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
          "L": 1.06,  "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
          "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

BINS        = [0, 2, 5, 10, np.inf]
TIER_LABELS = ["Very strong (<2 μM)", "Strong (2–5 μM)", "Moderate (5–10 μM)", "Weak (>10 μM)"]
SHORT       = ["VeryStrong", "Strong", "Moderate", "Weak"]


# ════════════════════════════════════════════════════════════════════════════
# Sequence utilities
# ════════════════════════════════════════════════════════════════════════════
_STAPLE_MARKER = re.compile(r'S5|R8|[BX]', re.I)

def normalize_seq(raw: str) -> str:
    """Strip staple markers and non-standard residues → plain AA string."""
    cleaned = _STAPLE_MARKER.sub("", raw.upper())
    return "".join(c for c in cleaned if c in STANDARD_AA)


def helical_moment(seq: str, delta_deg: float = 100.0) -> float:
    """
    Eisenberg hydrophobic moment (μH) for an ideal α-helix.
    δ = 100° ≈ 360° / 3.6 residues per turn.
    Returns the absolute μH (Eisenberg et al. 1982).
    """
    delta = np.radians(delta_deg)
    angles = np.arange(len(seq)) * delta
    H = np.array([_EISEN.get(aa, 0.0) for aa in seq])
    sin_sum = np.sum(H * np.sin(angles))
    cos_sum = np.sum(H * np.cos(angles))
    return float(np.sqrt(sin_sum**2 + cos_sum**2))


def compute_seq_features(seq: str) -> dict:
    """
    Compute the 9 sequence-level StaPep features from a plain AA string.
    lyticity_index is approximated as the Eisenberg helical hydrophobic moment.
    """
    analysis = ProteinAnalysis(seq)
    n = len(seq)
    return {
        "length"           : n,
        "weight"           : analysis.molecular_weight(),
        "hydrophobic_index": sum(_KD.get(aa, 0.0) for aa in seq) / n,
        "charge"           : analysis.charge_at_pH(7.0),
        "aromaticity"      : analysis.aromaticity(),
        "isoelectric_point": analysis.isoelectric_point(),
        "fraction_arginine": seq.count("R") / n,
        "fraction_lysine"  : seq.count("K") / n,
        "lyticity_index"   : helical_moment(seq),   # ≈ helical hydrophobic moment
    }


# ════════════════════════════════════════════════════════════════════════════
# Feature lookup
# ════════════════════════════════════════════════════════════════════════════
def _load_feature_db() -> pd.DataFrame:
    """Combine training + test feature CSVs into one lookup table."""
    dfs = []
    for path in [FEAT_CSV, TEST_CSV]:
        if path.exists():
            df = pd.read_csv(path)
            # normalise whichever column holds the raw sequence
            seq_col = next((c for c in ["stapep_seq", "sequence", "Sequence"]
                            if c in df.columns), None)
            if seq_col:
                df["_norm_seq"] = df[seq_col].apply(normalize_seq)
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def lookup_features(plain_seq: str, db: pd.DataFrame) -> pd.Series | None:
    """Return the first row whose normalised sequence matches plain_seq."""
    if "_norm_seq" not in db.columns:
        return None
    hits = db[db["_norm_seq"] == plain_seq.upper()]
    if hits.empty:
        return None
    row = hits.iloc[0]
    print(f"  ✅ Exact feature match found  "
          f"({row.get('peptide_id', row.get('DRAMP_ID', '?'))})")
    return row


def get_feature_row(raw_seq: str, db: pd.DataFrame,
                    full_features: dict | None = None) -> pd.DataFrame:
    """
    Return a 1-row DataFrame of FEATURES for the query peptide.

    Priority:
      1. full_features dict supplied directly (e.g. from StaPep MD pipeline)
      2. Exact sequence lookup in the feature database CSVs
      3. BioPython sequence features + NaN for MD features (imputed)
    """
    plain = normalize_seq(raw_seq)

    if full_features is not None:
        missing = [f for f in FEATURES if f not in full_features]
        if missing:
            print(f"  ⚠  --full-features missing keys: {missing}")
            print(f"     Those will be NaN → imputed with training medians.")
        row = {f: full_features.get(f, np.nan) for f in FEATURES}
        source = "StaPep MD pipeline (all 17 features supplied)"
        print(f"  ✅ Using supplied StaPep feature vector directly.")
    else:
        hit = lookup_features(plain, db)
        if hit is not None:
            row = {f: hit.get(f, np.nan) for f in FEATURES}
            source = "pre-computed"
        else:
            print(f"  ⚠  Sequence not in feature database.")
            print(f"     Computing sequence-level features via BioPython.")
            print(f"     MD features (helix/sheet/loop/SASA/…) set to NaN")
            print(f"     → imputed with training-set medians.\n")
            row = {f: np.nan for f in FEATURES}
            row.update(compute_seq_features(plain))
            source = "BioPython (MD features imputed)"

    df = pd.DataFrame([row])[FEATURES].astype(float)
    return df, source


# ════════════════════════════════════════════════════════════════════════════
# MIC training data
# ════════════════════════════════════════════════════════════════════════════
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


def build_training() -> tuple[pd.DataFrame, pd.Series]:
    mic  = parse_ecoli_mic(AMP_CSV)
    feat = pd.read_csv(FEAT_CSV)
    mc   = list(dict.fromkeys(["DRAMP_ID"] + FEATURES))
    df   = mic.merge(feat[mc], on="DRAMP_ID", how="inner").reset_index(drop=True)

    df["mic_uM"] = df["mic_raw"].astype(float)
    mask = df["unit"].str.lower().str.contains("g/ml").values
    df.loc[mask, "mic_uM"] = (df["mic_raw"].values[mask] * 1000
                               / df["weight"].values[mask])

    df = df.dropna(subset=["mic_uM"] + FEATURES).reset_index(drop=True)
    df["tier"] = pd.cut(df["mic_uM"], bins=BINS, labels=[0, 1, 2, 3]).astype(int)
    return df[FEATURES].astype(float), df["tier"]


# ════════════════════════════════════════════════════════════════════════════
# Model pipelines
# ════════════════════════════════════════════════════════════════════════════
def _rf_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestClassifier(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])


def _svm_pipe(C=1, gamma="scale") -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("svc",     SVC(kernel="rbf", probability=True,
                        class_weight="balanced",
                        C=C, gamma=gamma, random_state=42)),
    ])


_SVM_GRID = {"svc__C": [0.1, 1, 10, 100, 1000],
             "svc__gamma": ["scale", 1e-3, 1e-2, 0.1]}


def _mlp_pipe(hidden=(64, 32), alpha=0.01, lr=1e-3) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("mlp",     MLPClassifier(
            hidden_layer_sizes=hidden, alpha=alpha,
            learning_rate_init=lr, activation="relu",
            solver="adam", max_iter=1000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=20, random_state=42)),
    ])


_MLP_GRID = {
    "mlp__hidden_layer_sizes": [(64, 32), (32, 16), (128, 64), (64,)],
    "mlp__alpha"             : [1e-3, 1e-2, 0.1, 1.0],
    "mlp__learning_rate_init": [1e-3, 5e-4],
}


def train_all(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit RF, SVM (tuned), MLP (tuned) on the full training set."""
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    print("  Training Random Forest …", end=" ", flush=True)
    rf = _rf_pipe()
    rf.fit(X, y)
    print("done")

    print("  Training RBF-SVM (GridSearchCV) …", end=" ", flush=True)
    gs_svm = GridSearchCV(_svm_pipe(), _SVM_GRID, cv=skf,
                          scoring="balanced_accuracy", n_jobs=-1, refit=True)
    gs_svm.fit(X, y)
    print(f"done  (C={gs_svm.best_params_['svc__C']}, "
          f"γ={gs_svm.best_params_['svc__gamma']})")

    print("  Training MLP (GridSearchCV) …", end=" ", flush=True)
    gs_mlp = GridSearchCV(_mlp_pipe(), _MLP_GRID, cv=skf,
                          scoring="balanced_accuracy", n_jobs=-1, refit=True)
    gs_mlp.fit(X, y)
    bp = gs_mlp.best_params_
    print(f"done  (hidden={bp['mlp__hidden_layer_sizes']}, "
          f"α={bp['mlp__alpha']})")

    return {"RF": rf, "SVM": gs_svm.best_estimator_, "MLP": gs_mlp.best_estimator_}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Predict MIC tier for a single peptide (StaPep features, "
                    "RF / SVM / MLP).")
    parser.add_argument("sequence",
                        help="Amino-acid sequence (standard AA, 1-letter code). "
                             "Non-standard residues and staple markers are stripped.")
    parser.add_argument("--id", default=None,
                        help="Label for the peptide (default: first 20 chars of sequence)")
    parser.add_argument("--full-features", default=None, metavar="JSON",
                        help="JSON string with all 17 pre-computed StaPep features. "
                             "When supplied, skips DB lookup and BioPython fallback entirely.")
    args = parser.parse_args()

    raw_seq   = args.sequence.strip()
    plain_seq = normalize_seq(raw_seq)
    pep_id    = args.id or raw_seq[:20]

    # Parse --full-features JSON if supplied
    full_features = None
    if args.full_features:
        try:
            full_features = json.loads(args.full_features)
        except json.JSONDecodeError as e:
            sys.exit(f"ERROR: --full-features is not valid JSON: {e}")

    if len(plain_seq) < 5:
        sys.exit(f"ERROR: sequence too short after normalisation: {plain_seq!r}")

    print("\n" + "=" * 65)
    print("  Single-Peptide MIC-Tier Predictor")
    print("  Models: Random Forest | RBF-SVM | MLP")
    print("=" * 65)
    print(f"\n  Input sequence : {raw_seq}")
    print(f"  Normalised     : {plain_seq}")
    print(f"  Peptide ID     : {pep_id}")
    print(f"  Length         : {len(plain_seq)} residues\n")

    # ── Load feature database (training + test CSVs) ─────────────────────
    db = _load_feature_db()

    # ── Build feature row for query peptide ──────────────────────────────
    print("  Feature extraction …")
    X_query, source = get_feature_row(raw_seq, db, full_features=full_features)

    # ── Build MIC training set ───────────────────────────────────────────
    print(f"  Feature source : {source}\n")
    print("  Loading MIC training data …")
    X_train, y_train = build_training()
    print(f"  Training set   : {len(X_train)} AMPs with E. coli MIC")
    print(f"  Classes        : "
          + "  ".join(f"{SHORT[i]}={int((y_train==i).sum())}"
                      for i in range(len(SHORT))))

    # Show computed sequence features for the query
    print("\n  Query StaPep feature vector:")
    print(f"  {'Feature':<22}  {'Value':>10}  {'Source'}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*12}")
    for feat in FEATURES:
        val = X_query.iloc[0][feat]
        src = "sequence" if feat in SEQ_FEATURES else (
              "pre-computed" if not np.isnan(val) else "imputed (median)")
        val_str = f"{val:>10.4f}" if not np.isnan(val) else f"{'NaN':>10}"
        print(f"  {feat:<22}  {val_str}  {src}")

    # ── Train models ─────────────────────────────────────────────────────
    print("\n  Fitting models on full training data …")
    models = train_all(X_train, y_train)

    # ── Predict ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  MIC-Tier Predictions")
    print("=" * 65)
    print(f"\n  Peptide  : {pep_id}")
    print(f"  Sequence : {plain_seq}")

    print(f"\n  {'Model':<8}  {'Predicted Tier':<25}  "
          + "  ".join(f"{s:>10}" for s in SHORT))
    print(f"  {'─'*8}  {'─'*25}  " + "  ".join(["─"*10]*len(SHORT)))

    for name, model in models.items():
        probs = model.predict_proba(X_query)[0]
        pred  = model.predict(X_query)[0]
        tier  = TIER_LABELS[pred]
        conf  = probs.max()
        flag  = " ★" if conf >= 0.70 else ""
        prob_str = "  ".join(f"{p:>10.3f}" for p in probs)
        print(f"  {name:<8}  {tier:<25}  {prob_str}{flag}")

    print(f"\n  ★ = top-class confidence ≥ 0.70")
    print(f"  Column order  : {' | '.join(SHORT)}")
    if source.startswith("BioPython"):
        print(f"\n  ⚠  MD features were not available for this sequence.")
        print(f"     Predictions rely on sequence features + median-imputed")
        print(f"     MD features. For higher accuracy, run the full StaPep")
        print(f"     pipeline (ESMFold + AMBER MD) to obtain all 17 features.")
    print()


if __name__ == "__main__":
    main()
