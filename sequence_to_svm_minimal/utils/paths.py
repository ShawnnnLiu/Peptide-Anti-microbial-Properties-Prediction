"""Shared project path constants.

Replaces the ``Path(__file__).resolve().parent.parent`` / ``os.path.dirname(
os.path.abspath(__file__))`` pattern that previously appeared in 15+ scripts,
and fixes a recurring bug where scripts under ``svm/`` resolved BASE to their
own folder (``.../svm/``) instead of the project root.

Importers should read from these constants rather than rebuild paths
themselves. All values are ``pathlib.Path`` objects; pass ``str(...)`` if a
caller needs a string.
"""
from pathlib import Path

# This file lives at sequence_to_svm_minimal/utils/paths.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Training data
DATA_DIR        = PROJECT_ROOT / "data" / "training_dataset"
STAPEP_DIR      = DATA_DIR / "StaPep"
STRUCTURES_DIR  = DATA_DIR / "structures"

# Legacy 2016 PNAS QSAR descriptor generator
DESCRIPTORS_DIR = PROJECT_ROOT / "descriptors"
AAINDEX_DIR     = DESCRIPTORS_DIR / "aaindex"

# Output destinations
RESULTS_DIR  = PROJECT_ROOT / "results"
FIGURES_DIR  = PROJECT_ROOT / "figures"

# Pre-trained 2016 PNAS SVM bundle. The live svm/ scripts read from the
# frozen snapshot copy under pretrained_svm/, not the live predictionsParameters/.
PRETRAINED_PARAMS_DIR = (
    PROJECT_ROOT.parent / "pretrained_svm" / "sequence_to_svm_minimal"
    / "predictionsParameters"
)
PRETRAINED_SVC_PKL    = PRETRAINED_PARAMS_DIR / "svc.pkl"
PRETRAINED_ZSCORE_CSV = PRETRAINED_PARAMS_DIR / "Z_score_mean_std__intersect_noflip.csv"

# Live predictionsParameters/ (used by scripts/run_sequence_svm.py)
LIVE_PARAMS_DIR = PROJECT_ROOT / "predictionsParameters"
