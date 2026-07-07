"""Pure-numpy wrapper around the 2016 PNAS pretrained SVM.

The original ``svc.pkl`` was serialized with scikit-learn 0.19.2 and cannot
be unpickled cleanly under modern sklearn (NDArrayWrapper, sklearn.externals
namespace, etc.). The companion ``svc.pkl_NN.npy`` files hold the
reconstructable arrays. Verified mapping (cross-checked against exp1
reference predictions in run_pretrained_svm_inference.py):

    pkl_03 → dual_coef_       pkl_04 → probA_   (= -3.29162142)
    pkl_06 → n_support_       pkl_07 → support_vectors_   (225, 12)
    pkl_10 → intercept_       (= -0.01187876)
    pkl_11 → probB_           (= +0.03014156)

The model uses a linear kernel, so f(x) = x_z @ w + b where
w = SV.T @ dual_coef_[0]. Class probability is Platt-scaled:
P(+1|x) = 1 / (1 + exp(probA * f(x) + probB)).

This module replaces the verbatim copies of this loader that previously
appeared in svm/run_pretrained_svm_inference.py, svm/run_pretrained_svm_low_loop.py,
and svm/run_combined_svm.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from utils.paths import PRETRAINED_PARAMS_DIR, PRETRAINED_ZSCORE_CSV


@dataclass(frozen=True)
class PretrainedSVM:
    """In-memory reconstruction of the 2016 PNAS linear-kernel SVM."""

    desc_names: List[str]  # ordered descriptor names from Z-score CSV
    z_means:    np.ndarray  # shape (n_features,)
    z_stds:     np.ndarray  # shape (n_features,)
    w:          np.ndarray  # shape (n_features,) linear-kernel weight vector
    intercept:  float       # decision bias
    probA:      float       # Platt A
    probB:      float       # Platt B

    def z_score(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply original training-time Z-score normalisation."""
        return (X_raw - self.z_means) / self.z_stds

    def decision(self, X_z: np.ndarray) -> np.ndarray:
        """Decision function f(x) on already-Z-scored input matrix."""
        return X_z @ self.w + self.intercept

    def proba_amp(self, X_z: np.ndarray) -> np.ndarray:
        """P(AMP=+1) via Platt scaling. ``X_z`` must already be Z-scored."""
        dec = self.decision(X_z)
        return 1.0 / (1.0 + np.exp(self.probA * dec + self.probB))

    def predict_from_descriptors(
        self, df: pd.DataFrame
    ) -> "PretrainedPrediction":
        """Convenience: given a DataFrame containing the 12 descriptor columns,
        return Z-scored matrix, decision values, and P(AMP) arrays.
        """
        X_raw = df[self.desc_names].values.astype(float)
        X_z   = self.z_score(X_raw)
        dec   = self.decision(X_z)
        prob  = 1.0 / (1.0 + np.exp(self.probA * dec + self.probB))
        return PretrainedPrediction(X_z=X_z, decision=dec, prob_amp=prob)


@dataclass(frozen=True)
class PretrainedPrediction:
    """Bundle of arrays returned by ``PretrainedSVM.predict_from_descriptors``."""
    X_z:      np.ndarray  # (n, n_features)
    decision: np.ndarray  # (n,)
    prob_amp: np.ndarray  # (n,) P(+1) via Platt scaling


def _load_zscore_csv(z_file: Path) -> tuple[List[str], np.ndarray, np.ndarray]:
    with open(z_file) as fh:
        names = fh.readline().strip().split(",")
        means = np.array([float(x) for x in fh.readline().strip().split(",")])
        stds  = np.array([float(x) for x in fh.readline().strip().split(",")])
    return names, means, stds


def load_pretrained_svm(
    params_dir: Path = PRETRAINED_PARAMS_DIR,
    z_file:     Path = PRETRAINED_ZSCORE_CSV,
) -> PretrainedSVM:
    """Reconstruct the 2016 PNAS SVM from its .npy sidecars.

    Bypasses ``svc.pkl`` entirely — only reads the ``svc.pkl_NN.npy`` files
    and the Z-score CSV, so it works under any modern sklearn (or with no
    sklearn at all).
    """
    names, z_means, z_stds = _load_zscore_csv(z_file)

    def _npy(n: int) -> np.ndarray:
        return np.load(params_dir / f"svc.pkl_{n:02d}.npy", allow_pickle=False)

    sv        = _npy(7)   # (225, 12) support vectors (already Z-scored)
    dual_coef = _npy(3)   # (1, 225)  alpha_i * y_i
    intercept = _npy(10)  # (1,)
    probA     = _npy(4)   # (1,)
    probB     = _npy(11)  # (1,)

    w = sv.T @ dual_coef[0]  # (n_features,) linear-kernel weight vector

    return PretrainedSVM(
        desc_names=list(names),
        z_means=z_means,
        z_stds=z_stds,
        w=w,
        intercept=float(intercept[0]),
        probA=float(probA[0]),
        probB=float(probB[0]),
    )
