#!/usr/bin/env python3
"""
StaPep GNN Architecture Comparison Script

Compares GCN, GAT, and EGNN architectures on the Stapled Peptide dataset.
This script handles the unique format of StaPep data:
- AMPs from stapled_amps.csv (188 peptides)
- Decoys from stapled_decoys.csv (355 stapled non-AMPs)

Pipeline:
1. Prepare combined dataset with clean sequences
2. Generate PDB structures using ESMFold (if not exists)
3. Compute geometric features
4. Run GNN comparison with stratified 5-fold CV
"""

import os
import sys
import re
import warnings
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from gnn.models import PeptideGNN
from gnn.train import run_training
from torch_geometric.loader import DataLoader


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # New txt file format (matching generate_stapep_structures.py)
    'amp_seqs': 'data/training_dataset/StaPep/seqs_AMP_stapep.txt',
    'decoy_seqs': 'data/training_dataset/StaPep/seqs_DECOY_stapep.txt',
    'amp_pdb_dir': 'data/training_dataset/StaPep/structures/AMP',
    'decoy_pdb_dir': 'data/training_dataset/StaPep/structures/DECOY',
    'output_dir': 'data/training_dataset/StaPep',
    'seed': 42,
    'n_folds': 5,
    'epochs': 500,
    'batch_size': 16,  # Smaller batch for smaller dataset
    'lr': 1e-3,
    'patience': 50,  # More patience for smaller dataset
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.3,  # Slightly higher dropout for smaller dataset
    'distance_threshold': 8.0,
}

# Feature configurations - Graph-only for StaPep (txt files don't have extra features)
FEATURE_CONFIGS = {
    'Graph-only': {'use_geo': False, 'geo_dim': 0},
}

# GNN architectures
ARCHITECTURES = ['gcn', 'gat', 'egnn']


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_sequence(seq):
    """
    Clean sequence by removing non-standard amino acid codes.
    Handles various stapling notations:
    - Circled letters (Ⓚ, Ⓧ, etc.) -> standard AA
    - Stapling codes (S5, R8, X, etc.) -> remove or replace
    """
    if pd.isna(seq):
        return None
    
    # Map circled letters to standard AAs
    circled_map = {
        'Ⓚ': 'K', 'Ⓐ': 'A', 'Ⓡ': 'R', 'Ⓝ': 'N', 'Ⓓ': 'D',
        'Ⓒ': 'C', 'Ⓔ': 'E', 'Ⓠ': 'Q', 'Ⓖ': 'G', 'Ⓗ': 'H',
        'Ⓘ': 'I', 'Ⓛ': 'L', 'Ⓜ': 'M', 'Ⓕ': 'F', 'Ⓟ': 'P',
        'Ⓢ': 'S', 'Ⓣ': 'T', 'Ⓦ': 'W', 'Ⓨ': 'Y', 'Ⓥ': 'V',
        'Ⓧ': 'A',  # Non-natural AA placeholder -> Alanine
    }
    
    for circled, standard in circled_map.items():
        seq = seq.replace(circled, standard)
    
    # Remove stapling codes (S5, S8, R5, R8, X, numbers, etc.)
    # Keep only standard amino acid letters
    seq = re.sub(r'[SR][0-9]+', '', seq)  # Remove S5, R8, etc.
    seq = re.sub(r'\*', '', seq)  # Remove asterisks
    seq = re.sub(r'\?', '', seq)  # Remove question marks
    seq = re.sub(r'[0-9]+', '', seq)  # Remove numbers
    seq = re.sub(r'-', '', seq)  # Remove dashes
    
    # Keep only standard amino acids
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    cleaned = ''.join([aa for aa in seq.upper() if aa in standard_aa])
    
    return cleaned if len(cleaned) >= 5 else None  # Minimum length 5


def parse_sequence_file(input_file):
    """
    Parse sequence file in format:
        1 MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN
        2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG
    
    Returns list of (index, sequence) tuples
    """
    sequences = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split(None, 1)  # Split on whitespace, max 2 parts
            if len(parts) == 2:
                idx, seq = parts
                sequences.append((idx.strip(), seq.strip()))
            elif len(parts) == 1:
                seq = parts[0]
                idx = len(sequences) + 1
                sequences.append((str(idx), seq.strip()))
    
    return sequences


def prepare_stapep_dataset(config):
    """
    Prepare combined StaPep dataset from txt files.
    
    Returns:
        DataFrame with columns: peptide_id, sequence, label, length, pdb_path
    """
    print("Loading StaPep data from txt files...")
    
    amp_path = Path(config['amp_seqs'])
    decoy_path = Path(config['decoy_seqs'])
    
    if not amp_path.exists():
        raise FileNotFoundError(f"AMP file not found: {amp_path}")
    if not decoy_path.exists():
        raise FileNotFoundError(f"Decoy file not found: {decoy_path}")
    
    # Load AMPs
    amp_seqs = parse_sequence_file(amp_path)
    print(f"  AMPs loaded: {len(amp_seqs)}")
    
    # Load Decoys
    decoy_seqs = parse_sequence_file(decoy_path)
    print(f"  Decoys loaded: {len(decoy_seqs)}")
    
    # Process AMPs
    amp_pdb_dir = Path(config['amp_pdb_dir'])
    amp_records = []
    for idx, seq in amp_seqs:
        pdb_path = amp_pdb_dir / f"structure_{idx}.pdb"
        amp_records.append({
            'peptide_id': f"AMP_{idx}",
            'sequence': seq,
            'label': 1,  # AMP = positive
            'length': len(seq),
            'pdb_path': str(pdb_path),
        })
    
    # Process Decoys
    decoy_pdb_dir = Path(config['decoy_pdb_dir'])
    decoy_records = []
    for idx, seq in decoy_seqs:
        pdb_path = decoy_pdb_dir / f"structure_{idx}.pdb"
        decoy_records.append({
            'peptide_id': f"DECOY_{idx}",
            'sequence': seq,
            'label': 0,  # Decoy = negative
            'length': len(seq),
            'pdb_path': str(pdb_path),
        })
    
    # Combine
    all_records = amp_records + decoy_records
    combined_df = pd.DataFrame(all_records)
    
    print(f"\nCombined dataset: {len(combined_df)} peptides")
    print(f"  AMPs: {(combined_df['label'] == 1).sum()}")
    print(f"  Decoys: {(combined_df['label'] == 0).sum()}")
    print(f"  Length range: {combined_df['length'].min()} - {combined_df['length'].max()}")
    
    return combined_df


def check_structures_and_filter(df, config):
    """
    Check which PDB structures exist, drop missing ones and proceed.
    Returns filtered DataFrame with only available structures.
    """
    has_structure = []
    missing = 0
    for _, row in df.iterrows():
        if Path(row['pdb_path']).exists():
            has_structure.append(True)
        else:
            has_structure.append(False)
            missing += 1
    
    df_filtered = df[has_structure].reset_index(drop=True)
    
    if missing == 0:
        print(f"✅ All {len(df)} structures found")
    else:
        print(f"⚠️  Skipping {missing} peptides with missing structures")
        print(f"✅ Proceeding with {len(df_filtered)} peptides")
    
    print(f"   AMPs: {(df_filtered['label'] == 1).sum()}")
    print(f"   Decoys: {(df_filtered['label'] == 0).sum()}")
    
    return df_filtered


def generate_structures_esmfold(df, config):
    """
    Generate structures - redirects to the standalone script.
    """
    print("⚠️  Use the dedicated structure generation script:")
    print("    python generate_stapep_structures.py")
    return False


class StaPepDataset:
    """
    Custom dataset for StaPep peptide graphs.
    """
    
    def __init__(self, df, feature_cols=None, distance_threshold=8.0):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.distance_threshold = distance_threshold
        
        # Import graph construction functions
        from gnn.data_utils import parse_pdb, compute_node_features, compute_edges
        self.parse_pdb = parse_pdb
        self.compute_node_features = compute_node_features
        self.compute_edges = compute_edges
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        from torch_geometric.data import Data
        
        row = self.df.iloc[idx]
        
        # Get PDB path from dataframe
        pdb_path = Path(row['pdb_path'])
        
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB not found: {pdb_path}")
        
        # Parse PDB
        aa_sequence, ca_coords, plddt_values = self.parse_pdb(str(pdb_path))
        n_residues = len(aa_sequence)
        
        if n_residues < 2:
            raise ValueError(f"Peptide too short: {row['peptide_id']}")
        
        # Compute node features (26 dims)
        x = self.compute_node_features(aa_sequence, plddt_values, n_residues)
        
        # Compute edges
        edge_index, edge_attr = self.compute_edges(ca_coords, self.distance_threshold)
        
        # Coordinates
        pos = torch.tensor(ca_coords, dtype=torch.float32)
        
        # Label
        label = int(row['label'])
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([label], dtype=torch.long),
            num_nodes=n_residues
        )
        
        # Add extra features if specified
        if self.feature_cols:
            extra_feats = []
            for col in self.feature_cols:
                val = row.get(col, 0.0)
                extra_feats.append(float(val) if pd.notna(val) else 0.0)
            extra_feats = np.array(extra_feats, dtype=np.float32)
            data.geo_features = torch.tensor(extra_feats, dtype=torch.float32).unsqueeze(0)
        
        return data


def run_single_experiment(arch, feature_name, feature_config, all_data, labels, device, config, curves_base_dir):
    """Run a single experiment configuration and save training curves."""
    
    geo_dim = feature_config['geo_dim']
    
    # Create folder for this model's curves
    model_name = f"{arch.upper()}_{feature_name.replace('+', '_plus_')}"
    curves_dir = Path(curves_base_dir) / model_name
    curves_dir.mkdir(parents=True, exist_ok=True)
    
    # Model factory
    def model_fn():
        return PeptideGNN(
            architecture=arch,
            in_channels=26,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_classes=2,
            pooling='mean_max',
            geo_feature_dim=geo_dim
        )
    
    # Create CV splits (Stratified, no clustering for StaPep)
    cv = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=config['seed'])
    splits = list(cv.split(np.arange(len(labels)), labels))
    
    # Cross-validation
    cv_results = {
        'auc_roc': [], 'auc_pr': [], 'f1': [], 
        'accuracy': [], 'mcc': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        # Create dataloaders
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        
        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)
        
        # Fresh model
        model = model_fn()
        
        # Train and capture history
        history, best_metrics = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=config['epochs'],
            lr=config['lr'],
            patience=config['patience'],
            verbose=False
        )
        
        # Save training curves for this fold
        fold_df = pd.DataFrame({
            'epoch': list(range(1, len(history['train_loss']) + 1)),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_auc_roc': history['val_auc_roc'],
            'val_f1': history['val_f1'],
        })
        fold_csv_path = curves_dir / f"fold_{fold+1}.csv"
        fold_df.to_csv(fold_csv_path, index=False)
        
        # Also save as JSON
        fold_json = {
            'fold': fold + 1,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'best_epoch': int(np.argmax(history['val_auc_roc']) + 1),
            'best_metrics': {k: float(v) for k, v in best_metrics.items()},
            'history': {k: [float(x) for x in v] for k, v in history.items()}
        }
        fold_json_path = curves_dir / f"fold_{fold+1}.json"
        with open(fold_json_path, 'w') as f:
            json.dump(fold_json, f, indent=2)
        
        # Record results
        for metric in cv_results:
            cv_results[metric].append(best_metrics.get(metric, 0.0))
        
        print(f"      Fold {fold+1}: AUC={best_metrics['auc_roc']:.4f}, F1={best_metrics['f1']:.4f}")
    
    # Save summary
    summary = {
        'model': model_name,
        'architecture': arch,
        'feature_set': feature_name,
        'geo_dim': geo_dim,
        'config': {
            'epochs': config['epochs'],
            'patience': config['patience'],
            'lr': config['lr'],
            'hidden_channels': config['hidden_channels'],
            'num_layers': config['num_layers'],
        },
        'cv_results': {
            'auc_roc': {'mean': float(np.mean(cv_results['auc_roc'])), 'std': float(np.std(cv_results['auc_roc'])), 'values': [float(x) for x in cv_results['auc_roc']]},
            'f1': {'mean': float(np.mean(cv_results['f1'])), 'std': float(np.std(cv_results['f1'])), 'values': [float(x) for x in cv_results['f1']]},
            'mcc': {'mean': float(np.mean(cv_results['mcc'])), 'std': float(np.std(cv_results['mcc'])), 'values': [float(x) for x in cv_results['mcc']]},
        }
    }
    
    summary_path = curves_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"      📁 Curves saved to: {curves_dir}")
    
    return cv_results


def main():
    print("="*80)
    print("StaPep GNN Architecture Comparison: GCN vs GAT vs EGNN")
    print("="*80)
    print("Dataset: Stapled Peptides (AMPs vs Decoys)")
    
    # Setup
    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\n📊 Configuration:")
    print(f"   Epochs: {CONFIG['epochs']}")
    print(f"   Patience: {CONFIG['patience']}")
    print(f"   Hidden channels: {CONFIG['hidden_channels']}")
    print(f"   Num layers: {CONFIG['num_layers']}")
    print(f"   CV folds: {CONFIG['n_folds']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    
    # Prepare dataset
    print("\n" + "-"*80)
    print("Step 1: Preparing StaPep Dataset")
    print("-"*80)
    
    combined_df = prepare_stapep_dataset(CONFIG)
    
    # Save combined dataset
    combined_path = Path(CONFIG['output_dir']) / 'stapep_combined.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined dataset saved to: {combined_path}")
    
    # Check structures - skip any missing ones
    print("\n" + "-"*80)
    print("Step 2: Checking PDB Structures")
    print("-"*80)
    
    combined_df = check_structures_and_filter(combined_df, CONFIG)
    
    if len(combined_df) == 0:
        print("❌ No structures found! Run: python generate_stapep_structures.py")
        return
    
    # Get labels
    labels = combined_df['label'].values
    
    # Results storage
    all_results = {}
    
    # Create base directory for training curves
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    curves_base_dir = Path('results/stapep_gnn/curves') / f'run_{timestamp}'
    curves_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Training curves will be saved to: {curves_base_dir}")
    
    # Run experiments
    print("\n" + "="*80)
    print("Step 3: Running GNN Experiments (Stratified 5-Fold CV)")
    print("="*80)
    
    for feature_name, feature_config in FEATURE_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Feature Set: {feature_name}")
        print(f"{'='*60}")
        
        # Determine feature columns (Graph-only = no extra features)
        feature_cols = None
        print(f"Extra features: 0 dims")
        
        # Create dataset
        dataset = StaPepDataset(
            combined_df,
            feature_cols=feature_cols,
            distance_threshold=CONFIG['distance_threshold']
        )
        
        # Load all data
        print("Loading graphs...")
        all_data = []
        failed = 0
        for i in tqdm(range(len(dataset)), desc="Loading"):
            try:
                all_data.append(dataset[i])
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Warning: Failed to load {combined_df.iloc[i]['peptide_id']}: {e}")
        
        if failed > 0:
            print(f"  ⚠️  Failed to load {failed} graphs")
        print(f"Loaded {len(all_data)} graphs")
        
        # Update labels to match loaded data
        loaded_labels = labels[:len(all_data)]
        
        for arch in ARCHITECTURES:
            print(f"\n   🔬 {arch.upper()}...")
            
            cv_results = run_single_experiment(
                arch, feature_name, feature_config,
                all_data, loaded_labels, device, CONFIG,
                curves_base_dir
            )
            
            # Store results
            key = f"{arch.upper()} ({feature_name})"
            all_results[key] = {
                'auc_roc_mean': np.mean(cv_results['auc_roc']),
                'auc_roc_std': np.std(cv_results['auc_roc']),
                'f1_mean': np.mean(cv_results['f1']),
                'f1_std': np.std(cv_results['f1']),
                'mcc_mean': np.mean(cv_results['mcc']),
                'mcc_std': np.std(cv_results['mcc']),
            }
            
            print(f"      → AUC-ROC: {all_results[key]['auc_roc_mean']:.4f} ± {all_results[key]['auc_roc_std']:.4f}")
            print(f"      → F1:      {all_results[key]['f1_mean']:.4f} ± {all_results[key]['f1_std']:.4f}")
            print(f"      → MCC:     {all_results[key]['mcc_mean']:.4f} ± {all_results[key]['mcc_std']:.4f}")
    
    # Print final comparison table
    print("\n" + "="*80)
    print("RESULTS SUMMARY: StaPep GNN Architecture Comparison")
    print("="*80)
    
    print(f"\n📊 Dataset: {len(all_data)} stapled peptides")
    print(f"   AMPs: {(loaded_labels == 1).sum()}, Decoys: {(loaded_labels == 0).sum()}")
    
    print("\n" + "-"*100)
    print(f"{'Model':<35} {'AUC-ROC':<20} {'F1':<20} {'MCC':<20}")
    print("-"*100)
    
    for feature_name in FEATURE_CONFIGS.keys():
        for arch in ARCHITECTURES:
            key = f"{arch.upper()} ({feature_name})"
            if key in all_results:
                r = all_results[key]
                print(f"{key:<35} "
                      f"{r['auc_roc_mean']:.4f} ± {r['auc_roc_std']:.4f}   "
                      f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}   "
                      f"{r['mcc_mean']:.4f} ± {r['mcc_std']:.4f}")
        print()
    
    # Save results
    os.makedirs('results/stapep_gnn', exist_ok=True)
    
    results_dict = {
        'config': CONFIG,
        'dataset': {
            'total': len(all_data),
            'amps': int((loaded_labels == 1).sum()),
            'decoys': int((loaded_labels == 0).sum())
        },
        'results': all_results,
        'timestamp': timestamp,
        'curves_dir': str(curves_base_dir)
    }
    
    json_path = f'results/stapep_gnn/stapep_gnn_comparison_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    print(f"📁 Training curves saved to: {curves_base_dir}")
    
    print("\n✅ StaPep GNN comparison complete!")


if __name__ == '__main__':
    main()
