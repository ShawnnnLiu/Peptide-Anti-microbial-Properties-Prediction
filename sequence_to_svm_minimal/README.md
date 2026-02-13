# AMP Classification with ESMFold & Graph Neural Networks

**Binary classification of Antimicrobial Peptides (AMP) vs Decoys using ESMFold-predicted structures and Graph Neural Networks.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.3+-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)

---

## Overview

This project implements a machine learning pipeline for classifying antimicrobial peptides using:

1. **ESMFold** - Predict 3D structures from amino acid sequences
2. **Geometric Features** - Extract 24 coordinate-invariant structural descriptors
3. **Graph Neural Networks** - GCN, GAT, EGNN architectures operating on residue graphs
4. **Feature Fusion** - Combine structural, QSAR, and graph features

### Dataset
- **572 peptides**: 286 AMP (antimicrobial) + 286 Decoy (non-antimicrobial)
- ESMFold-predicted PDB structures for all peptides
- Cluster-based cross-validation to prevent data leakage

### Results
| Model | Features | AUC-ROC | F1 Score |
|-------|----------|---------|----------|
| GCN | Graph-only | ~0.97 | ~0.92 |
| GAT | Graph-only | ~0.98 | ~0.93 |
| EGNN | Graph-only | ~0.97 | ~0.92 |
| MLP | Geo-24 | ~0.95 | ~0.90 |
| SVM | QSAR-12 | ~0.93 | ~0.87 |

---

## Quick Start

### Prerequisites
- Windows 10/11 with WSL2
- NVIDIA GPU with CUDA support
- Python 3.10+

### Installation

```bash
# 1. Open WSL
wsl

# 2. Navigate to project
cd /mnt/c/Users/YOUR_USERNAME/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install PyTorch Geometric
pip install torch-geometric

# 6. Install remaining dependencies
pip install -r requirements.txt
```

> 📘 **First time setup?** See [SETUP.md](SETUP.md) for complete WSL, CUDA, and environment configuration.

---

## Usage

### Run GNN Comparison (GCN vs GAT vs EGNN)

```bash
python run_gnn_comparison.py
```

This trains all 3 architectures × 3 feature sets = 9 models with 5-fold CV.

### Run MLP Feature Fusion Experiments

```bash
python run_feature_fusion_experiments.py
```

Compares QSAR-12, Geometric-24, and Combined-36 features.

### Visualize Training Curves

```bash
jupyter notebook plot_training_curves.ipynb
```

---

## Project Structure

```
sequence_to_svm_minimal/
│
├── data/training_dataset/
│   ├── AMP/                      # ESMFold PDB structures (286 files)
│   ├── DECOY/                    # ESMFold PDB structures (286 files)
│   ├── geometric_features_clustered.csv
│   ├── seqs_AMP.txt              # AMP sequences
│   └── seqs_decoy_subsample.txt  # Decoy sequences
│
├── features/
│   └── geometric_features.py     # Extract 24 structural features from PDB
│
├── gnn/
│   ├── data_utils.py             # PDB → PyG graph conversion
│   ├── models.py                 # GCN, GAT, EGNN architectures
│   └── train.py                  # Training loop with early stopping
│
├── nn_pipeline/
│   ├── feature_dataset.py        # Feature loading & preprocessing
│   ├── models.py                 # MLP classifier
│   └── train.py                  # NN training utilities
│
├── models/
│   ├── run_esmfold_peptides.py   # ESMFold inference script
│   └── batch_esmfold.py          # Batch ESMFold processing
│
├── results/
│   └── gnn/
│       ├── curves/               # Training curves per run
│       └── gnn_comparison_*.json # Results
│
├── run_gnn_comparison.py         # Main: Compare GNN architectures
├── run_gnn_training.py           # Train single GNN model
├── run_nn_training.py            # Train MLP on geometric features
├── run_feature_fusion_experiments.py  # Compare feature combinations
├── build_geometric_features.py   # Generate geometric_features.csv
├── plot_training_curves.ipynb    # Visualization notebook
│
├── requirements.txt              # Python dependencies
├── SETUP.md                      # Detailed setup guide
└── README.md                     # This file
```

---

## Features

### Geometric Features (24 dimensions)
Extracted from ESMFold PDB structures:

| Category | Features |
|----------|----------|
| **pLDDT Confidence** | mean, std, min, max |
| **Compactness** | Radius of gyration, end-to-end distance, max pairwise, centroid stats |
| **Secondary Structure** | Fraction helix, sheet, coil |
| **SASA** | Total, hydrophobic, fraction hydrophobic |
| **Sequence** | Length, net charge, mean hydrophobicity, hydrophobic moment |
| **Curvature** | Backbone curvature (mean, std, max), torsion (mean, std) |

### Graph Features (per-node)
For GNN models, each residue is a node with:
- One-hot amino acid type (20 dims)
- pLDDT confidence
- Hydrophobicity
- Charge
- Secondary structure encoding
- Relative sequence position

### Edges
- Sequential bonds (i → i+1)
- Spatial contacts (Cα distance < 8Å)

---

## Evaluation Protocol

### Cluster-Based Cross-Validation
- Sequences clustered at 40% identity (CD-HIT-like)
- GroupKFold ensures no similar sequences leak between train/test
- More realistic generalization estimate

### PNAS-Style Evaluation
- 15% strict blind test set (never touched during training)
- 15 rounds stratified shuffle CV on remaining 85%
- Matches Lee et al. PNAS 2016 protocol

---

## Citation

If you use this code, please cite:

```bibtex
@software{amp_esm_gnn,
  title={AMP Classification with ESMFold and Graph Neural Networks},
  year={2026},
  url={https://github.com/...}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Troubleshooting

See [SETUP.md](SETUP.md#troubleshooting) for common issues with:
- CUDA not detected in WSL
- PyTorch Geometric installation
- Out of GPU memory
- Import errors
