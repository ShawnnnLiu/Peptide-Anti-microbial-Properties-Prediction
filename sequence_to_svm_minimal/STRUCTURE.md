# Project Structure Documentation

## Overview

This project combines traditional biochemical descriptors with ESMFold structural predictions for antimicrobial peptide analysis and MIC (Minimum Inhibitory Concentration) prediction.

## Directory Structure

```
sequence_to_svm_minimal/
├── descriptors/                  # Traditional biochemical features
│   ├── descripGen_12_py3.py     # 12-descriptor generator
│   └── aaindex/                  # AAIndex database
│
├── predictionsParameters/        # SVM model and inference
│   ├── svc.pkl                   # Pre-trained SVM
│   ├── predictSVC.py            # SVM inference script
│   └── Z_score_*.csv            # Normalization parameters
│
├── scripts/                      # Pipeline scripts
│   ├── run_sequence_svm.py      # Original SVM pipeline
│   └── make_seqs_windows.py     # Sequence windowing
│
├── structure/                    # NEW: ESMFold integration
│   ├── esmfold_predictor.py     # ESMFold wrapper with caching
│   ├── structure_features.py    # Extract geometric features
│   └── pdb_parser.py            # PDB file parsing
│
├── features/                     # NEW: Feature fusion
│   ├── feature_extractor.py     # Unified pipeline (Branch A + B)
│   └── feature_utils.py         # Helper functions
│
├── models/                       # NEW: ML models
│   ├── mic_predictor.py         # Neural network architectures
│   ├── train_mic.py             # Training script
│   └── checkpoints/             # Saved model weights
│
├── cli/                          # NEW: Command-line tools
│   ├── predict_mic.py           # MIC prediction interface
│   ├── extract_features.py      # Feature extraction tool
│   └── train_model.py           # Training interface
│
├── utils/                        # NEW: Shared utilities
│   ├── caching.py               # Cache management
│   ├── data_prep.py             # Dataset preprocessing
│   └── evaluation.py            # Metrics and plotting
│
├── data/                         # External datasets
│   ├── mic_datasets/            # MIC-labeled data (DBAASP, etc.)
│   └── examples/                # Test sequences
│
├── cache/                        # Runtime cache (not in git)
│   └── esmfold_cache/           # Cached structure predictions
│
└── experiments/                  # Experimental runs
    ├── exp1/
    └── exp2/
```

## Module Descriptions

### `structure/` - ESMFold Integration
- **esmfold_predictor.py**: Loads ESMFold, predicts structures, manages caching
- **structure_features.py**: Extracts geometric and secondary structure features
- **pdb_parser.py**: Simple PDB parsing (alternative to biotite)

### `features/` - Feature Fusion
- **feature_extractor.py**: Combines traditional + SVM + structural features
- **feature_utils.py**: Normalization, validation, transformation helpers

### `models/` - Machine Learning
- **mic_predictor.py**: PyTorch NN models for MIC regression
- **train_mic.py**: Training loop with baselines and evaluation
- **checkpoints/**: Saved model weights (excluded from git)

### `cli/` - User Interfaces
- **predict_mic.py**: Predict MIC for sequences
- **extract_features.py**: Extract features without prediction
- **train_model.py**: Train new models

### `utils/` - Utilities
- **caching.py**: ESMFold cache management (MD5 hashing)
- **data_prep.py**: MIC data cleaning, CD-HIT clustering, splitting
- **evaluation.py**: Metrics (Spearman, RMSE) and visualization

## Data Flow

### Stage 1: Feature Extraction
```
Sequence → [Traditional Descriptors] → SVM → σ, p(+1)
          ↓
          [ESMFold] → Structure → Geometric Features
          ↓
          [Fusion] → Combined Feature Vector (~24 dims)
```

### Stage 2: MIC Prediction
```
Feature Vector → Neural Network → log(MIC) → MIC (μg/mL)
```

## Import Patterns

```python
# Traditional features
from descriptors.descripGen_12_py3 import descripGen_bespoke

# ESMFold and structure
from structure.esmfold_predictor import ESMFoldPredictor
from structure.structure_features import extract_geometric_features

# Feature fusion
from features.feature_extractor import UnifiedFeatureExtractor

# Models
from models.mic_predictor import MICPredictor

# CLI usage
from cli.predict_mic import predict_single_sequence
```

## File Naming Conventions

- **Module files**: Lowercase with underscores (`esmfold_predictor.py`)
- **Class names**: PascalCase (`ESMFoldPredictor`)
- **Function names**: Lowercase with underscores (`extract_features`)
- **Constants**: UPPERCASE (`CACHE_DIR`)

## Configuration

Key paths and settings should be configurable:
- Cache directory: `cache/esmfold_cache/`
- Model checkpoints: `models/checkpoints/`
- Data directory: `data/mic_datasets/`

## Development Workflow

1. **Prototype**: Start in top-level files or notebooks
2. **Refactor**: Move working code into appropriate modules
3. **Test**: Create unit tests for each module
4. **Integrate**: Connect modules in CLI scripts
5. **Document**: Update docstrings and this file

## Notes

- Old code in `descriptors/`, `predictionsParameters/`, `scripts/` remains unchanged
- New modules are independent and can be developed separately
- Cache directory excluded from git (regenerable)
- Model checkpoints excluded from git (too large)
- Example datasets should be small (<1MB)

---

**Last Updated**: January 2026
**Version**: 0.1.0

