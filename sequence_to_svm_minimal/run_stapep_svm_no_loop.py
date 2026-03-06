"""
run_stapep_svm_no_loop.py
--------------------------
Mirrors run_pretrained_svm_inference.py but uses our RETRAINED StaPep RBF-SVM
with loop_percent WITHHELD from the feature set.

Purpose: ablation study — quantify how much loop_percent alone is responsible
for the SVM scoring Buf13 lower than Buf12.

Comparison:
  Full StaPep SVM  (17 features)        → from run_stapep_svm.py
  No-loop StaPep SVM (16 features)      → this script
  Pretrained PNAS SVM (QSAR, no StaPep) → from run_pretrained_svm_inference.py
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import platform
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
STAPEP = os.path.join(BASE, "data", "training_dataset", "StaPep")

# ── Feature sets ──────────────────────────────────────────────────────────────
ALL_SP_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "lyticity_index", "helix_percent", "sheet_percent", "loop_percent",
    "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa",
]
NO_LOOP_COLS = [c for c in ALL_SP_COLS if c != "loop_percent"]

# ── Load training data ────────────────────────────────────────────────────────
amp = pd.read_csv(os.path.join(STAPEP, "stapled_amps_features.csv"))
dec = pd.read_csv(os.path.join(STAPEP, "stapled_decoys.csv"))
amp["label"] = 1;  dec["label"] = -1
train = pd.concat([amp[ALL_SP_COLS + ["label"]],
                   dec[ALL_SP_COLS + ["label"]]], ignore_index=True).dropna()

X_full    = train[ALL_SP_COLS].values
X_noloop  = train[NO_LOOP_COLS].values
y         = train["label"].values

# ── Load test peptides ────────────────────────────────────────────────────────
test = pd.read_csv(os.path.join(STAPEP, "test_stapled_features.csv"))
test = test.set_index("peptide_id")

TEST_NAMES = ["Buf12", "Buf13", "Buf13_Q9K", "Buf12_V15K_L19K",
              "Mag20",  "Mag25",  "Mag31",     "Mag36"]
test = test.loc[TEST_NAMES]

X_test_full   = test[ALL_SP_COLS].values
X_test_noloop = test[NO_LOOP_COLS].values

# ── Train FULL StaPep SVM (baseline) ─────────────────────────────────────────
pipe_full = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, random_state=42)),
])
pipe_full.fit(X_full, y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_full = cross_val_score(pipe_full, X_full, y, cv=cv,
                           scoring="roc_auc").mean()

# ── Train NO-LOOP StaPep SVM (ablation) ───────────────────────────────────────
pipe_noloop = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                   probability=True, random_state=42)),
])
pipe_noloop.fit(X_noloop, y)

auc_noloop = cross_val_score(pipe_noloop, X_noloop, y, cv=cv,
                             scoring="roc_auc").mean()

# ── Predict on test peptides ──────────────────────────────────────────────────
prob_full   = pipe_full.predict_proba(X_test_full)[:, 1]
prob_noloop = pipe_noloop.predict_proba(X_test_noloop)[:, 1]

# ── Print CV performance ──────────────────────────────────────────────────────
W = 74
print()
print("=" * W)
print("  StaPep RBF-SVM Ablation: Effect of Withholding loop_percent")
print("=" * W)
print(f"  Full model   (17 features, incl. loop_percent) :  CV AUC = {auc_full:.4f}")
print(f"  No-loop model (16 features, excl. loop_percent):  CV AUC = {auc_noloop:.4f}")
print(f"  AUC change from removing loop_percent          :  {auc_noloop - auc_full:+.4f}")
print("=" * W)

# ── Print side-by-side predictions ───────────────────────────────────────────
print()
print("=" * W)
print("  Test Peptide Predictions — Full vs No-Loop Model")
print(f"  {'Peptide':<26} {'Full P(AMP)':>12} {'No-Loop P(AMP)':>15} {'Δ P(AMP)':>10}")
print(f"  {'─'*26} {'─'*12} {'─'*15} {'─'*10}")

for i, name in enumerate(TEST_NAMES):
    pf  = prob_full[i]
    pnl = prob_noloop[i]
    delta = pnl - pf
    arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
    print(f"  {name:<26} {pf:>12.4f} {pnl:>15.4f} {delta:>+10.4f} {arrow}")

print("=" * W)

# ── Highlight Buf12 vs Buf13 gap ──────────────────────────────────────────────
i12 = TEST_NAMES.index("Buf12")
i13 = TEST_NAMES.index("Buf13")

gap_full   = prob_full[i12]   - prob_full[i13]
gap_noloop = prob_noloop[i12] - prob_noloop[i13]

print()
print(f"  Buf12 − Buf13 gap (Full model)   : {gap_full:+.4f}")
print(f"  Buf12 − Buf13 gap (No-loop model): {gap_noloop:+.4f}")
if abs(gap_noloop) < abs(gap_full):
    print(f"  → Removing loop_percent REDUCES the gap by "
          f"{abs(gap_full - gap_noloop):.4f} — confirming it contributes to Buf13's penalty.")
else:
    print(f"  → Removing loop_percent does NOT reduce the gap — other features dominate.")
print()
