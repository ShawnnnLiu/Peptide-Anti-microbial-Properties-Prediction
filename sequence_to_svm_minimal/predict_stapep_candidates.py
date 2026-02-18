#!/usr/bin/env python3
"""
Stapled Peptide AMP Candidate Prediction

Trains GNN ensemble models (GCN, GAT, EGNN) on the StaPep training dataset
using 5-fold cross-validation, then predicts AMP activity for novel stapled
peptide candidates.

Reports for each candidate:
  - AMP classification (Yes / No)
  - P(AMP): ensemble-averaged probability
  - Three confidence scores derived from calibrated logit-margin distributions:
    * Z-Conf:   How many SDs the logit margin is from the decision boundary
    * LR-Conf:  Bayesian posterior P(predicted_class | logit_margin)
    * Pct-Conf: Percentile rank within the predicted class's distribution

Usage:
    python predict_stapep_candidates.py
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add parent directory for gnn imports
sys.path.insert(0, str(Path(__file__).parent))

from gnn.models import PeptideGNN
from gnn.train import run_training
from torch_geometric.loader import DataLoader


# =============================================================================
# CANDIDATE DEFINITIONS
# =============================================================================
# X = olefinic amino acid for hydrocarbon stapling → replaced with Alanine
# 8 = R8 stapling notation → removed

CANDIDATES = [
    {
        "name": "Buf(i+4) 12",
        "raw": "TRSSRAGLQWPXGRVXRLLRK",
        "clean": "TRSSRAGLQWPAGRVARLLRK",
        "notes": "Buforin II, i+4 staple at pos 12,16",
    },
    {
        "name": "Buf(i+4) 13",
        "raw": "TRSSRAGLQWPVXRVHXLLRK",
        "clean": "TRSSRAGLQWPVARVHALLRK",
        "notes": "Buforin II, i+4 staple at pos 13,17",
    },
    {
        "name": "Buf(i+4) 13 Q9K",
        "raw": "TRSSRAGLKWPVXRVHXLLRK",
        "clean": "TRSSRAGLKWPVARVHALLRK",
        "notes": "Buforin II, i+4 staple + Q9K mutation",
    },
    {
        "name": "Buf(i+4) 12 V15K,L19K",
        "raw": "TRSSRAGLQWPXGRKXRLKRK",
        "clean": "TRSSRAGLQWPAGRKARLKRK",
        "notes": "Buforin II, i+4 staple + V15K, L19K mutations",
    },
    {
        "name": "Mag 20",
        "raw": "GIGKFLHSKK8FGKAFVXEIAKK",
        "clean": "GIGKFLHSKKFGKAFVAEIAKK",
        "notes": "Magainin 2, staple variant 20",
    },
    {
        "name": "Mag 25",
        "raw": "XKGKXLHSKKKFGKAXVGEXAKK",
        "clean": "AKGKALHSKKKFGKAAVGEAAKK",
        "notes": "Magainin 2, staple variant 25",
    },
    {
        "name": "Mag 31",
        "raw": "GXGKFXHSKKKKGKAXVGEXAKK",
        "clean": "GAGKFAHSKKKKGKAAVGEAAKK",
        "notes": "Magainin 2, staple variant 31",
    },
    {
        "name": "Mag 36",
        "raw": "GXGKFXHSKKKK8KAFKGEXAKK",
        "clean": "GAGKFAHSKKKKKAFKGEAAKK",
        "notes": "Magainin 2, staple variant 36",
    },
]


# =============================================================================
# CONFIGURATION (matches training script)
# =============================================================================

CONFIG = {
    'amp_seqs': 'data/training_dataset/StaPep/seqs_AMP_stapep.txt',
    'decoy_seqs': 'data/training_dataset/StaPep/seqs_DECOY_stapep.txt',
    'amp_pdb_dir': 'data/training_dataset/StaPep/structures/AMP',
    'decoy_pdb_dir': 'data/training_dataset/StaPep/structures/DECOY',
    'test_seqs': 'data/training_dataset/StaPep/seqs_test_stapled.txt',
    'test_pdb_dir': 'data/training_dataset/StaPep/structures/TEST',
    'seed': 42,
    'n_folds': 5,
    'epochs': 500,
    'batch_size': 16,
    'lr': 1e-3,
    'patience': 50,
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.3,
    'distance_threshold': 8.0,
}

ARCHITECTURES = ['gcn', 'gat', 'egnn']


# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_sequence_file(input_file):
    """Parse sequence file (index sequence per line)."""
    sequences = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                idx, seq = parts
                sequences.append((idx.strip(), seq.strip()))
            elif len(parts) == 1:
                sequences.append((str(len(sequences) + 1), parts[0].strip()))
    return sequences


# =============================================================================
# STRUCTURE GENERATION FOR TEST CANDIDATES
# =============================================================================

def generate_test_structures(candidates, output_dir, device="cuda"):
    """
    Generate PDB structures for test candidates using ESMFold.
    Only generates missing structures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which need generation
    to_generate = []
    for i, cand in enumerate(candidates):
        pdb_path = output_dir / f"structure_{i+1}.pdb"
        if not pdb_path.exists():
            to_generate.append((i, cand))

    if not to_generate:
        print("  All test structures already exist!")
        return True

    print(f"  Generating {len(to_generate)} test structures with ESMFold...")

    from transformers import EsmForProteinFolding

    local_model_path = Path(__file__).parent / "models" / "esmfold_v1_local"
    load_dtype = torch.float16 if device == "cuda" else torch.float32

    if local_model_path.exists():
        print(f"  Loading ESMFold from local: {local_model_path}")
        model = EsmForProteinFolding.from_pretrained(
            str(local_model_path),
            local_files_only=True,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )
    else:
        print("  Downloading ESMFold from HuggingFace...")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )

    model = model.to(device)
    model.eval()

    success = 0
    for i, cand in tqdm(to_generate, desc="  ESMFold"):
        pdb_path = output_dir / f"structure_{i+1}.pdb"
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                pdb_string = model.infer_pdb(cand['clean'])
            with open(pdb_path, 'w') as f:
                f.write(pdb_string)
            success += 1
        except Exception as e:
            print(f"  Failed for {cand['name']}: {e}")

    # Free GPU memory for GNN training
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Generated {success}/{len(to_generate)} structures")
    return success == len(to_generate)


# =============================================================================
# DATASET
# =============================================================================

class StaPepDataset:
    """Dataset for converting PDB files to PyTorch Geometric graphs."""

    def __init__(self, df, distance_threshold=8.0):
        self.df = df.reset_index(drop=True)
        self.distance_threshold = distance_threshold

        from gnn.data_utils import parse_pdb, compute_node_features, compute_edges
        self.parse_pdb = parse_pdb
        self.compute_node_features = compute_node_features
        self.compute_edges = compute_edges

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from torch_geometric.data import Data

        row = self.df.iloc[idx]
        pdb_path = Path(row['pdb_path'])

        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")

        aa_sequence, ca_coords, plddt_values = self.parse_pdb(str(pdb_path))
        n_residues = len(aa_sequence)

        if n_residues < 2:
            raise ValueError(f"Too short: {row.get('peptide_id', idx)}")

        x = self.compute_node_features(aa_sequence, plddt_values, n_residues)
        edge_index, edge_attr = self.compute_edges(ca_coords, self.distance_threshold)
        pos = torch.tensor(ca_coords, dtype=torch.float32)
        label = int(row['label'])

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_residues,
        )


# =============================================================================
# PREDICTION WITH σ_GNN SCORING
# =============================================================================

@torch.no_grad()
def get_predictions(model, data_list, device, batch_size=32):
    """
    Get predictions from a trained model.

    Returns:
        p_amp:  array of P(AMP) probabilities
        sigma:  array of σ_GNN scores (logit_AMP − logit_nonAMP)
        labels: array of ground-truth labels
    """
    model.eval()
    model = model.to(device)

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    all_p_amp = []
    all_sigma = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)

        all_p_amp.extend(probs[:, 1].cpu().numpy())
        all_sigma.extend((logits[:, 1] - logits[:, 0]).cpu().numpy())
        all_labels.extend(batch.y.view(-1).cpu().numpy())

    return np.array(all_p_amp), np.array(all_sigma), np.array(all_labels)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("  STAPLED PEPTIDE AMP CANDIDATE PREDICTION")
    print("  GNN Ensemble (GCN + GAT + EGNN) with sigma_GNN Scoring")
    print("=" * 80)

    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Architectures: {', '.join(a.upper() for a in ARCHITECTURES)}")
    print(f"  CV Folds: {CONFIG['n_folds']}")

    # Print candidate info
    print(f"\n  Candidates ({len(CANDIDATES)} peptides):")
    for c in CANDIDATES:
        print(f"    {c['name']:<30} {c['clean']}  (len={len(c['clean'])})")

    # ------------------------------------------------------------------
    # Step 1: Prepare test candidate structures
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Step 1: Test Candidate Structures")
    print("=" * 80)

    test_pdb_dir = Path(CONFIG['test_pdb_dir'])
    all_exist = all(
        (test_pdb_dir / f"structure_{i+1}.pdb").exists()
        for i in range(len(CANDIDATES))
    )

    if not all_exist:
        generate_test_structures(CANDIDATES, test_pdb_dir, device.type)
    else:
        print("  All 8 test structures already exist!")

    # Verify
    test_found = sum(
        1 for i in range(len(CANDIDATES))
        if (test_pdb_dir / f"structure_{i+1}.pdb").exists()
    )
    print(f"  Test structures: {test_found}/{len(CANDIDATES)}")

    # Build test dataframe
    test_records = []
    test_valid_idx = []
    for i, cand in enumerate(CANDIDATES):
        pdb_path = test_pdb_dir / f"structure_{i+1}.pdb"
        if pdb_path.exists():
            test_records.append({
                'peptide_id': cand['name'],
                'sequence': cand['clean'],
                'label': 0,  # dummy label for graph construction
                'pdb_path': str(pdb_path),
            })
            test_valid_idx.append(i)
    test_df = pd.DataFrame(test_records)

    # ------------------------------------------------------------------
    # Step 2: Load training data
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Step 2: Loading Training Data")
    print("=" * 80)

    amp_seqs = parse_sequence_file(CONFIG['amp_seqs'])
    decoy_seqs = parse_sequence_file(CONFIG['decoy_seqs'])

    train_records = []
    for idx, seq in amp_seqs:
        pdb_path = Path(CONFIG['amp_pdb_dir']) / f"structure_{idx}.pdb"
        if pdb_path.exists():
            train_records.append({
                'peptide_id': f"AMP_{idx}",
                'sequence': seq,
                'label': 1,
                'pdb_path': str(pdb_path),
            })
    for idx, seq in decoy_seqs:
        pdb_path = Path(CONFIG['decoy_pdb_dir']) / f"structure_{idx}.pdb"
        if pdb_path.exists():
            train_records.append({
                'peptide_id': f"DECOY_{idx}",
                'sequence': seq,
                'label': 0,
                'pdb_path': str(pdb_path),
            })

    train_df = pd.DataFrame(train_records)
    n_amp = (train_df['label'] == 1).sum()
    n_decoy = (train_df['label'] == 0).sum()
    print(f"  Training set: {len(train_df)} peptides ({n_amp} AMPs, {n_decoy} Decoys)")

    # ------------------------------------------------------------------
    # Step 3: Build graphs
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Step 3: Building Graphs")
    print("=" * 80)

    # Training graphs
    print("  Loading training graphs...")
    train_dataset = StaPepDataset(train_df, CONFIG['distance_threshold'])
    train_graphs = []
    train_labels_list = []
    failed = 0
    for i in tqdm(range(len(train_dataset)), desc="  Training"):
        try:
            g = train_dataset[i]
            train_graphs.append(g)
            train_labels_list.append(int(train_df.iloc[i]['label']))
        except Exception as e:
            failed += 1
            if failed <= 3:
                tqdm.write(f"    Warning: {train_df.iloc[i]['peptide_id']}: {e}")

    train_labels = np.array(train_labels_list)
    print(f"  Loaded {len(train_graphs)} training graphs (failed: {failed})")

    # Test graphs
    print("  Loading test candidate graphs...")
    test_dataset = StaPepDataset(test_df, CONFIG['distance_threshold'])
    test_graphs = []
    test_graph_indices = []  # maps position in test_graphs to CANDIDATES index
    for i in range(len(test_dataset)):
        try:
            g = test_dataset[i]
            test_graphs.append(g)
            test_graph_indices.append(test_valid_idx[i])
        except Exception as e:
            print(f"    Failed: {CANDIDATES[test_valid_idx[i]]['name']}: {e}")

    print(f"  Loaded {len(test_graphs)}/{len(CANDIDATES)} test graphs")

    if len(test_graphs) == 0:
        print("\n  No test graphs could be loaded. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 4: Train GNN models and predict
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("Step 4: Training GNN Models & Predicting")
    print("=" * 80)

    cv = StratifiedKFold(
        n_splits=CONFIG['n_folds'], shuffle=True, random_state=CONFIG['seed']
    )
    splits = list(cv.split(np.arange(len(train_labels)), train_labels))

    # Storage
    all_cv_metrics = {}        # arch -> list of fold metrics
    all_test_predictions = {}  # arch -> list of {p_amp, sigma} per fold
    all_val_sigma = {}         # arch -> {amp: [...], decoy: [...]}

    total_start = time.time()

    for arch in ARCHITECTURES:
        print(f"\n{'='*60}")
        print(f"  Architecture: {arch.upper()}")
        print(f"{'='*60}")

        def make_model(a=arch):
            return PeptideGNN(
                architecture=a,
                in_channels=26,
                hidden_channels=CONFIG['hidden_channels'],
                num_layers=CONFIG['num_layers'],
                dropout=CONFIG['dropout'],
                num_classes=2,
                pooling='mean_max',
                geo_feature_dim=0,
            )

        arch_cv = []
        arch_test = []
        arch_val_amp_sigma = []
        arch_val_decoy_sigma = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            fold_start = time.time()

            fold_train = [train_graphs[i] for i in train_idx]
            fold_val = [train_graphs[i] for i in val_idx]

            train_loader = DataLoader(
                fold_train, batch_size=CONFIG['batch_size'], shuffle=True
            )
            val_loader = DataLoader(
                fold_val, batch_size=CONFIG['batch_size'], shuffle=False
            )

            model = make_model()

            _, best_metrics = run_training(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=CONFIG['epochs'],
                lr=CONFIG['lr'],
                patience=CONFIG['patience'],
                verbose=False,
            )

            arch_cv.append(best_metrics)
            fold_time = time.time() - fold_start
            print(
                f"    Fold {fold+1}: "
                f"AUC={best_metrics['auc_roc']:.4f}, "
                f"F1={best_metrics['f1']:.4f}  "
                f"({fold_time:.0f}s)"
            )

            # Validation σ_GNN for calibration (unbiased — model didn't train on these)
            val_p, val_sigma, val_labels = get_predictions(
                model, fold_val, device, CONFIG['batch_size']
            )
            arch_val_amp_sigma.extend(val_sigma[val_labels == 1].tolist())
            arch_val_decoy_sigma.extend(val_sigma[val_labels == 0].tolist())

            # Predict on test candidates
            test_p, test_sigma, _ = get_predictions(
                model, test_graphs, device, CONFIG['batch_size']
            )
            arch_test.append({'p_amp': test_p, 'sigma': test_sigma})

        # Store per-architecture results
        all_cv_metrics[arch] = arch_cv
        all_test_predictions[arch] = arch_test
        all_val_sigma[arch] = {
            'amp': np.array(arch_val_amp_sigma),
            'decoy': np.array(arch_val_decoy_sigma),
        }

        mean_auc = np.mean([m['auc_roc'] for m in arch_cv])
        std_auc = np.std([m['auc_roc'] for m in arch_cv])
        mean_f1 = np.mean([m['f1'] for m in arch_cv])
        std_f1 = np.std([m['f1'] for m in arch_cv])
        print(
            f"  --> {arch.upper()} CV: "
            f"AUC={mean_auc:.4f}+/-{std_auc:.4f}, "
            f"F1={mean_f1:.4f}+/-{std_f1:.4f}"
        )

    total_time = time.time() - total_start
    print(f"\n  Total training time: {timedelta(seconds=int(total_time))}")

    # ------------------------------------------------------------------
    # Step 5: Report results
    # ------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("  PREDICTION RESULTS")
    print("=" * 80)

    # ---- Per-architecture results ----
    for arch in ARCHITECTURES:
        preds = all_test_predictions[arch]

        mean_p = np.mean([p['p_amp'] for p in preds], axis=0)
        std_p = np.std([p['p_amp'] for p in preds], axis=0)

        cv_auc = np.mean([m['auc_roc'] for m in all_cv_metrics[arch]])
        cv_f1 = np.mean([m['f1'] for m in all_cv_metrics[arch]])

        print(f"\n  {'~'*60}")
        print(f"  {arch.upper()} (5-fold ensemble)  |  CV AUC={cv_auc:.4f}, F1={cv_f1:.4f}")
        print(f"  {'~'*60}")

        print(f"  {'Peptide':<30} {'AMP?':<6} {'P(+1) = sigmoid(z)'}")
        print(f"  {'-'*60}")

        for j in range(len(test_graphs)):
            ci = test_graph_indices[j]
            cand = CANDIDATES[ci]
            p, ps = mean_p[j], std_p[j]

            is_amp = "Yes" if p > 0.5 else "No"
            print(f"  {cand['name']:<30} {is_amp:<6} {p:.4f} +/- {ps:.4f}")

    # ---- Meta-ensemble (all architectures × all folds = 15 models) ----
    print(f"\n{'='*80}")
    print("  META-ENSEMBLE (GCN + GAT + EGNN, 15 models total)")
    print("=" * 80)

    all_preds_p = []
    for arch in ARCHITECTURES:
        for p in all_test_predictions[arch]:
            all_preds_p.append(p['p_amp'])

    meta_p = np.mean(all_preds_p, axis=0)
    meta_p_std = np.std(all_preds_p, axis=0)

    print(f"\n  {'Peptide':<30} {'AMP?':<6} {'P(+1) = sigmoid(z)'}")
    print(f"  {'-'*60}")

    meta_results = []
    for j in range(len(test_graphs)):
        ci = test_graph_indices[j]
        cand = CANDIDATES[ci]
        p, ps = meta_p[j], meta_p_std[j]

        is_amp = "Yes" if p > 0.5 else "No"
        print(f"  {cand['name']:<30} {is_amp:<6} {p:.4f} +/- {ps:.4f}")

        meta_results.append({
            'name': cand['name'],
            'raw_sequence': cand['raw'],
            'clean_sequence': cand['clean'],
            'notes': cand['notes'],
            'is_amp': bool(p > 0.5),
            'p_plus1': float(p),
            'p_plus1_std': float(ps),
        })

    # ---- Interpretation guide ----
    print(f"\n{'='*80}")
    print("  INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
  P(+1) = sigmoid(z)
             The model's direct output probability for the AMP class.
             z = logit_AMP - logit_nonAMP  (raw pre-softmax difference)
             sigmoid(z) = 1 / (1 + exp(-z))  =  P(AMP | graph)

             > 0.5  = classified as AMP
             > 0.98 = high-confidence AMP (SVM paper threshold analog)

             +/- shows fold-to-fold variability across the 5-fold ensemble.
             Low std = consistent prediction; high std = fold-dependent.
""")

    # ------------------------------------------------------------------
    # Step 6: Save results
    # ------------------------------------------------------------------
    os.makedirs('results/stapep_predictions', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_json = {
        'description': 'StaPep AMP candidate predictions using GNN ensemble',
        'timestamp': timestamp,
        'config': CONFIG,
        'candidates': CANDIDATES,
        'training_set': {
            'total': len(train_graphs),
            'amps': int((train_labels == 1).sum()),
            'decoys': int((train_labels == 0).sum()),
        },
        'cv_performance': {
            arch: {
                'auc_roc_mean': float(np.mean([m['auc_roc'] for m in all_cv_metrics[arch]])),
                'auc_roc_std': float(np.std([m['auc_roc'] for m in all_cv_metrics[arch]])),
                'f1_mean': float(np.mean([m['f1'] for m in all_cv_metrics[arch]])),
                'f1_std': float(np.std([m['f1'] for m in all_cv_metrics[arch]])),
                'mcc_mean': float(np.mean([m['mcc'] for m in all_cv_metrics[arch]])),
                'mcc_std': float(np.std([m['mcc'] for m in all_cv_metrics[arch]])),
            }
            for arch in ARCHITECTURES
        },
        'predictions': {
            'meta_ensemble': meta_results,
            'per_architecture': {
                arch: [
                    {
                        'name': CANDIDATES[test_graph_indices[j]]['name'],
                        'p_plus1': float(np.mean([p_['p_amp'][j] for p_ in all_test_predictions[arch]])),
                        'p_plus1_std': float(np.std([p_['p_amp'][j] for p_ in all_test_predictions[arch]])),
                    }
                    for j in range(len(test_graphs))
                ]
                for arch in ARCHITECTURES
            },
        },
    }

    json_path = f'results/stapep_predictions/predictions_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Results saved to: {json_path}")

    print(f"\n  Prediction complete!")


if __name__ == '__main__':
    main()
