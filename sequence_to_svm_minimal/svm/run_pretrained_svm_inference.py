"""
run_pretrained_svm_inference.py
-------------------------------
Run the pretrained 2016 PNAS SVM on our 8 test peptides (QSAR features only).
No retraining — pure inference.

Strategy to handle the pre-0.18 sklearn pickle:
  • Optionally load pickle for diagnostic display of kernel/C only.
  • Reconstruct the model entirely from the joblib .npy sidecar files via
    ``utils.pretrained_svm.load_pretrained_svm()`` — bypasses NDArrayWrapper.
  • Decision function + Platt probability live in ``PretrainedSVM``.

Confirmed array mapping (verified against exp1 reference predictions):
  pkl_03 (1,225) f64    → dual_coef_         (alpha_i * y_i per SV)
  pkl_04 (1,)    f64    → probA_             Platt A  = -3.29162142
  pkl_06 (2,)    int32  → n_support_         [113, 112]
  pkl_07 (225,12) f64   → support_vectors_   Z-scored training SVs
  pkl_10 (1,)    f64    → intercept_         decision bias = -0.01187876
  pkl_11 (1,)    f64    → probB_             Platt B  = +0.03014156

Verification: decision(exp1 seq1460) = X@w + pkl_10 = +3.4182  (known: 3.4183)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Make utils/ importable regardless of cwd. Also fixes a pre-refactor bug
# where BASE = svm/ (not project root), causing data paths to resolve to
# svm/data/... which doesn't exist.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import STAPEP_DIR, PROJECT_ROOT, PRETRAINED_SVC_PKL
from utils.pretrained_svm import load_pretrained_svm

QSAR_TEST = STAPEP_DIR / "qsar_stapled_test.csv"
FINAL_OUT = PROJECT_ROOT / "pretrained_svm_test_predictions.csv"


def _print_pickle_diagnostics() -> None:
    """Best-effort diagnostic: read kernel + C off the legacy pickle.

    The pickle was made with sklearn 0.19.2; loading under modern sklearn
    requires the legacy-namespace shims below. Failures are non-fatal — we
    fall back to advertised values, since the .npy sidecars are the source
    of truth for actual computation.
    """
    try:
        import joblib
        sys.modules.setdefault("sklearn.externals.joblib", joblib)
        try:
            import sklearn.svm._classes as _c
            sys.modules.setdefault("sklearn.svm.classes", _c)
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            svc = joblib.load(PRETRAINED_SVC_PKL)
        kernel = getattr(svc, "kernel", "linear")
        C      = getattr(svc, "C",      getattr(svc, "_C", 1.0))
        print(f"Kernel: {kernel}  |  C = {C:.6f}")
    except Exception as exc:
        print(f"Kernel: linear (pickle diagnostic skipped: {exc.__class__.__name__})")


def main() -> None:
    # ── Step 1: Load test descriptors ────────────────────────────────────────
    df    = pd.read_csv(QSAR_TEST)
    names = df["peptide_id"].tolist()
    seqs  = df["sequence"].tolist()

    # ── Step 2: Load pretrained SVM (from .npy sidecars only) ───────────────
    model = load_pretrained_svm()

    print(f"Input matrix: ({len(df)}, {len(model.desc_names)})  "
          f"({len(names)} peptides × {len(model.desc_names)} descriptors)")

    _print_pickle_diagnostics()
    print(f"Support-vector weight norm: {np.linalg.norm(model.w):.6f}")
    print(f"intercept = {model.intercept:.8f}")
    print(f"probA = {model.probA:.8f}  probB = {model.probB:.8f}")

    # ── Step 3: Inference ────────────────────────────────────────────────────
    pred          = model.predict_from_descriptors(df)
    decision_vals = pred.decision
    prob_pos      = pred.prob_amp
    prob_neg      = 1.0 - prob_pos
    pred_labels   = np.where(decision_vals > 0, 1, -1)

    print(f"\nDecision values: {np.round(decision_vals, 4)}")

    # ── Step 4: Print sequences used ─────────────────────────────────────────
    print("\n=== Sequences fed into pretrained SVM (QSAR / parent sequences) ===")
    for name, seq in zip(names, seqs):
        print(f"  {name:<30} {seq}")

    # ── Step 5: Print results ────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  Pretrained 2016 PNAS SVM — Inference on 8 Test Peptides")
    print("  (linear kernel, Z-score normalised, Platt-scaled probabilities)")
    print("=" * 74)
    print(f"{'Peptide':<30} {'Prediction':>12} {'P(AMP)':>8} {'Decision f(x)':>14}")
    print("-" * 74)

    order = np.argsort(prob_pos)[::-1]
    for i in order:
        label = "+1 (AMP)" if pred_labels[i] == 1 else "-1 (non)"
        print(f"{names[i]:<30} {label:>12} {prob_pos[i]:>8.4f} "
              f"{decision_vals[i]:>14.4f}")
    print("=" * 74)

    # ── Step 6: Save clean CSV ───────────────────────────────────────────────
    rows = []
    for i in order:
        rows.append({
            "peptide":       names[i],
            "sequence":      seqs[i],
            "prediction":    int(pred_labels[i]),
            "P(-1)":         round(float(prob_neg[i]), 6),
            "P(+1)":         round(float(prob_pos[i]), 6),
            "decision_f(x)": round(float(decision_vals[i]), 6),
        })
    pd.DataFrame(rows).to_csv(FINAL_OUT, index=False)
    print(f"\nResults saved to:\n  {FINAL_OUT}\n")


if __name__ == "__main__":
    main()
