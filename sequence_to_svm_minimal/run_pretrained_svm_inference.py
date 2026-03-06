"""
run_pretrained_svm_inference.py
-------------------------------
Run the pretrained 2016 PNAS SVM on our 8 test peptides (QSAR features only).
No retraining — pure inference.

Strategy to handle the pre-0.18 sklearn pickle:
  • Load pickle for scalar hyperparameters (kernel='linear', C, etc.)
  • Load numpy arrays from the joblib .npy sidecar files (clean np.load)
    bypassing the NDArrayWrapper incompatibility in modern sklearn/joblib.
  • Implement decision_function + Platt probability directly.

Confirmed array mapping (verified against exp1 reference predictions):
  pkl_01  (2,) int64     → classes_           [-1, 1]
  pkl_02  (225,) int32   → support_           (SV indices)
  pkl_03  (1,225) f64    → dual_coef_         (alpha_i * y_i per SV)
  pkl_04  (1,) f64       → probA_             Platt A  = -3.29162142
  pkl_05  (2,) f64       → class_weight_      [1., 1.]
  pkl_06  (2,) int32     → n_support_         [113, 112]
  pkl_07  (225,12) f64   → support_vectors_   Z-scored training SVs
  pkl_08  (1,225) f64    → _dual_coef_        internal (negated copy)
  pkl_09  (1,) f64       → _intercept_        internal libsvm rho (+ve)
  pkl_10  (1,) f64       → intercept_         decision bias = -0.01187876
  pkl_11  (1,) f64       → probB_             Platt B  = +0.03014156

Verification: decision(exp1 seq1460) = X@w + pkl_10 = +3.4182  (known: 3.4183 ✓)
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
QSAR_TEST = os.path.join(BASE, "data", "training_dataset", "StaPep",
                         "qsar_stapled_test.csv")
PP_DIR    = os.path.join(BASE, "..", "pretrained_svm",
                         "sequence_to_svm_minimal", "predictionsParameters")
Z_FILE    = os.path.join(PP_DIR, "Z_score_mean_std__intersect_noflip.csv")
SVC_PKL   = os.path.join(PP_DIR, "svc.pkl")
FINAL_OUT = os.path.join(BASE, "pretrained_svm_test_predictions.csv")

# ── Step 1: Load & Z-score normalise QSAR features ───────────────────────────
df    = pd.read_csv(QSAR_TEST)
names = df["peptide_id"].tolist()
seqs  = df["sequence"].tolist()
df_desc = df.drop(columns=["peptide_id", "sequence"], errors="ignore")

with open(Z_FILE) as f:
    desc_names = f.readline().strip().split(",")
    z_means    = np.array([float(x) for x in f.readline().strip().split(",")])
    z_stds     = np.array([float(x) for x in f.readline().strip().split(",")])

X = df_desc[desc_names].values.astype(float)
X = (X - z_means) / z_stds          # same Z-scoring used at training time
print(f"Input matrix: {X.shape}  ({len(names)} peptides × {X.shape[1]} descriptors)")

# ── Step 2: Load scalar hyperparameters from pickle ───────────────────────────
sys.modules.setdefault('sklearn.externals.joblib', joblib)
try:
    import sklearn.svm._classes as _c
    sys.modules.setdefault('sklearn.svm.classes', _c)
except Exception:
    pass

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    svc = joblib.load(SVC_PKL)

kernel = getattr(svc, 'kernel', 'linear')
C      = getattr(svc, 'C',      getattr(svc, '_C', 1.0))
print(f"Kernel: {kernel}  |  C = {C:.6f}")

# ── Step 3: Load arrays from .npy sidecar files ───────────────────────────────
def _npy(n):
    return np.load(os.path.join(PP_DIR, f"svc.pkl_{n:02d}.npy"), allow_pickle=False)

support_vectors_ = _npy(7)    # (225, 12) — Z-scored training support vectors
dual_coef_       = _npy(3)    # (1, 225)  — alpha_i * y_i
intercept_       = _npy(10)   # (1,)      — decision boundary bias (VERIFIED: -0.01187876)
probA_           = _npy(4)    # (1,)      — Platt A (VERIFIED: -3.29162142)
probB_           = _npy(11)   # (1,)      — Platt B (VERIFIED: +0.03014156)
n_support_       = _npy(6)    # (2,) int32

print(f"Support vectors: {support_vectors_.shape}  n_support: {n_support_}")
print(f"intercept = {intercept_[0]:.8f}")
print(f"probA = {probA_[0]:.8f}  probB = {probB_[0]:.8f}")

# ── Step 4: Linear kernel decision function ───────────────────────────────────
# f(x) = x @ w + b   where w = SV.T @ dual_coef_[0]
w             = support_vectors_.T @ dual_coef_[0]   # (12,)
decision_vals = X @ w + intercept_[0]                # (8,)

print(f"\nWeight vector norm: {np.linalg.norm(w):.6f}")
print(f"Decision values:    {np.round(decision_vals, 4)}")

# ── Step 5: Platt scaling → class probabilities ───────────────────────────────
# P(+1|x) = 1 / (1 + exp(A*f(x) + B))
fval     = probA_[0] * decision_vals + probB_[0]
prob_pos = 1.0 / (1.0 + np.exp(fval))    # P(AMP)
prob_neg = 1.0 - prob_pos

pred_labels = np.where(decision_vals > 0, 1, -1)

# ── Step 6: Print sequences used ─────────────────────────────────────────────
print("\n=== Sequences fed into pretrained SVM (QSAR / parent sequences) ===")
for name, seq in zip(names, seqs):
    print(f"  {name:<30} {seq}")

# ── Step 7: Print results ─────────────────────────────────────────────────────
print("\n" + "=" * 74)
print("  Pretrained 2016 PNAS SVM — Inference on 8 Test Peptides")
print("  (linear kernel, Z-score normalised, Platt-scaled probabilities)")
print("=" * 74)
print(f"{'Peptide':<30} {'Prediction':>12} {'P(AMP)':>8} {'Decision f(x)':>14}")
print("-" * 74)

order = np.argsort(prob_pos)[::-1]
for i in order:
    label = "+1 (AMP)" if pred_labels[i] == 1 else "-1 (non)"
    print(f"{names[i]:<30} {label:>12} {prob_pos[i]:>8.4f} {decision_vals[i]:>14.4f}")
print("=" * 74)

# ── Step 8: Save clean CSV ────────────────────────────────────────────────────
rows = []
for i in order:
    rows.append({
        "peptide":        names[i],
        "sequence":       seqs[i],
        "prediction":     int(pred_labels[i]),
        "P(-1)":          round(float(prob_neg[i]), 6),
        "P(+1)":          round(float(prob_pos[i]), 6),
        "decision_f(x)":  round(float(decision_vals[i]), 6),
    })
pd.DataFrame(rows).to_csv(FINAL_OUT, index=False)
print(f"\nResults saved to:\n  {FINAL_OUT}\n")
