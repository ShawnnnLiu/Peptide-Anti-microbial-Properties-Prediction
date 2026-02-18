#!/usr/bin/env python3
"""
Single GCN — Full Training → Stapled Peptide Prediction
========================================================
Trains ONE GCN model on the complete StaPep training dataset
(no hold-out), then predicts on 8 novel stapled-peptide candidates.

Output per candidate:
  AMP          Yes / No  (threshold P > 0.5)
  P(+1)        sigmoid(z) — the model's class-1 probability
  z            raw logit margin  logit_AMP − logit_nonAMP

Usage:
    python predict_gcn_single.py
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

sys.path.insert(0, str(Path(__file__).parent))

from gnn.models import PeptideGNN
from gnn.train import run_training
from torch_geometric.loader import DataLoader

import pandas as pd


# =============================================================================
# CANDIDATES
# =============================================================================
CANDIDATES = [
    {"name": "Buf(i+4) 12",          "clean": "TRSSRAGLQWPAGRVARLLRK"},
    {"name": "Buf(i+4) 13",          "clean": "TRSSRAGLQWPVARVHALLRK"},
    {"name": "Buf(i+4) 13 Q9K",      "clean": "TRSSRAGLKWPVARVHALLRK"},
    {"name": "Buf(i+4) 12 V15K,L19K","clean": "TRSSRAGLQWPAGRKARLKRK"},
    {"name": "Mag 20",               "clean": "GIGKFLHSKKFGKAFVAEIAKK"},
    {"name": "Mag 25",               "clean": "AKGKALHSKKKFGKAAVGEAAKK"},
    {"name": "Mag 31",               "clean": "GAGKFAHSKKKKGKAAVGEAAKK"},
    {"name": "Mag 36",               "clean": "GAGKFAHSKKKKKAFKGEAAKK"},
]

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    'amp_seqs':     'data/training_dataset/StaPep/seqs_AMP_stapep.txt',
    'decoy_seqs':   'data/training_dataset/StaPep/seqs_DECOY_stapep.txt',
    'amp_pdb_dir':  'data/training_dataset/StaPep/structures/AMP',
    'decoy_pdb_dir':'data/training_dataset/StaPep/structures/DECOY',
    'test_pdb_dir': 'data/training_dataset/StaPep/structures/TEST',
    'seed':          42,
    'epochs':        500,
    'batch_size':    16,
    'lr':            1e-3,
    'patience':      50,
    'hidden_channels': 64,
    'num_layers':    3,
    'dropout':       0.3,
    'distance_threshold': 8.0,
}


# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_sequence_file(path):
    sequences = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                sequences.append((parts[0].strip(), parts[1].strip()))
            elif len(parts) == 1:
                sequences.append((str(len(sequences) + 1), parts[0].strip()))
    return sequences


# =============================================================================
# STRUCTURE GENERATION (ESMFold) — only for missing TEST PDBs
# =============================================================================

def generate_test_structures(candidates, output_dir, device_str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    to_generate = [
        (i, c) for i, c in enumerate(candidates)
        if not (output_dir / f"structure_{i+1}.pdb").exists()
    ]
    if not to_generate:
        print("  All test structures already exist.")
        return

    print(f"  Generating {len(to_generate)} structures with ESMFold...")
    from transformers import EsmForProteinFolding

    load_dtype = torch.float16 if device_str == "cuda" else torch.float32
    local_path = Path(__file__).parent / "models" / "esmfold_v1_local"
    if local_path.exists():
        model = EsmForProteinFolding.from_pretrained(
            str(local_path), local_files_only=True,
            dtype=load_dtype, low_cpu_mem_usage=True)
    else:
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            dtype=load_dtype, low_cpu_mem_usage=True)

    model = model.to(device_str)
    model.eval()

    for i, cand in tqdm(to_generate, desc="  ESMFold"):
        pdb_path = output_dir / f"structure_{i+1}.pdb"
        try:
            if device_str == "cuda":
                torch.cuda.empty_cache()
            with torch.no_grad():
                pdb_str = model.infer_pdb(cand['clean'])
            pdb_path.write_text(pdb_str)
        except Exception as e:
            print(f"  ✗ {cand['name']}: {e}")

    del model
    if device_str == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# DATASET (PDB → PyG graph)
# =============================================================================

class PeptideGraphDataset:
    def __init__(self, df, distance_threshold=8.0):
        self.df = df.reset_index(drop=True)
        self.dt = distance_threshold
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

        aa_seq, ca_coords, plddt = self.parse_pdb(str(pdb_path))
        n = len(aa_seq)
        if n < 2:
            raise ValueError(f"Sequence too short: {n}")

        x = self.compute_node_features(aa_seq, plddt, n)
        edge_index, edge_attr = self.compute_edges(ca_coords, self.dt)
        pos = torch.tensor(ca_coords, dtype=torch.float32)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=torch.tensor([int(row['label'])], dtype=torch.long),
            num_nodes=n,
        )


def build_graphs(df, distance_threshold, desc=""):
    ds = PeptideGraphDataset(df, distance_threshold)
    graphs, failed = [], 0
    for i in tqdm(range(len(ds)), desc=f"  {desc}", leave=False):
        try:
            graphs.append(ds[i])
        except Exception as e:
            failed += 1
            if failed <= 3:
                tqdm.write(f"  ✗ {df.iloc[i].get('peptide_id','?')}: {e}")
    if failed:
        print(f"  Skipped {failed} graphs")
    return graphs


# =============================================================================
# PREDICTION
# =============================================================================

@torch.no_grad()
def predict(model, graphs, device, batch_size=32):
    """Return P(+1) and z for each graph."""
    model.eval()
    model = model.to(device)
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_p, all_z = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)                             # [N, 2]
        probs  = F.softmax(logits, dim=1)
        z      = logits[:, 1] - logits[:, 0]             # logit margin

        all_p.extend(probs[:, 1].cpu().numpy())
        all_z.extend(z.cpu().numpy())

    return np.array(all_p), np.array(all_z)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  SINGLE GCN — FULL TRAINING → CANDIDATE PREDICTION")
    print("=" * 70)

    set_seed(CONFIG['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device : {device}")
    if device.type == 'cuda':
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── Step 1 : Ensure test PDB structures exist ─────────────────────────
    print(f"\n{'─'*70}")
    print("  Step 1 · Test Structures")
    print(f"{'─'*70}")
    generate_test_structures(CANDIDATES, CONFIG['test_pdb_dir'], device.type)

    # Build test dataframe
    test_pdb_dir = Path(CONFIG['test_pdb_dir'])
    test_records, valid_idx = [], []
    for i, cand in enumerate(CANDIDATES):
        pdb = test_pdb_dir / f"structure_{i+1}.pdb"
        if pdb.exists():
            test_records.append({'peptide_id': cand['name'], 'sequence': cand['clean'],
                                  'label': 0, 'pdb_path': str(pdb)})
            valid_idx.append(i)
        else:
            print(f"  ✗ Missing PDB for {cand['name']} — skipped")

    test_df = pd.DataFrame(test_records)
    print(f"  Test candidates ready : {len(test_df)}/{len(CANDIDATES)}")

    # ── Step 2 : Load training data ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Step 2 · Training Data")
    print(f"{'─'*70}")

    amp_seqs   = parse_sequence_file(CONFIG['amp_seqs'])
    decoy_seqs = parse_sequence_file(CONFIG['decoy_seqs'])

    train_records = []
    for idx, seq in amp_seqs:
        pdb = Path(CONFIG['amp_pdb_dir']) / f"structure_{idx}.pdb"
        if pdb.exists():
            train_records.append({'peptide_id': f"AMP_{idx}", 'sequence': seq,
                                   'label': 1, 'pdb_path': str(pdb)})
    for idx, seq in decoy_seqs:
        pdb = Path(CONFIG['decoy_pdb_dir']) / f"structure_{idx}.pdb"
        if pdb.exists():
            train_records.append({'peptide_id': f"DECOY_{idx}", 'sequence': seq,
                                   'label': 0, 'pdb_path': str(pdb)})

    train_df = pd.DataFrame(train_records)
    n_amp   = (train_df['label'] == 1).sum()
    n_decoy = (train_df['label'] == 0).sum()
    print(f"  Peptides : {len(train_df)}  ({n_amp} AMPs, {n_decoy} Decoys)")

    # ── Step 3 : Build graphs ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Step 3 · Building Graphs")
    print(f"{'─'*70}")

    print("  Training graphs …")
    train_graphs = build_graphs(train_df, CONFIG['distance_threshold'], "Train")
    print(f"  Loaded {len(train_graphs)} training graphs")

    print("  Test graphs …")
    test_graphs = build_graphs(test_df, CONFIG['distance_threshold'], "Test")
    print(f"  Loaded {len(test_graphs)}/{len(CANDIDATES)} test graphs")

    if not test_graphs:
        print("\n  No test graphs loaded — exiting.")
        return

    # ── Step 4 : Train single GCN on ALL data ─────────────────────────────
    print(f"\n{'─'*70}")
    print("  Step 4 · Training GCN (full dataset, no hold-out)")
    print(f"{'─'*70}")

    model = PeptideGNN(
        architecture='gcn',
        in_channels=26,
        hidden_channels=CONFIG['hidden_channels'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        num_classes=2,
        pooling='mean_max',
        geo_feature_dim=0,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")

    # Use full dataset as both train and val so the model sees all data;
    # val metrics here are in-sample (optimistic) but the goal is full
    # utilisation, not generalisation estimation.
    full_loader = DataLoader(train_graphs, batch_size=CONFIG['batch_size'], shuffle=True)

    t0 = time.time()
    _, metrics = run_training(
        model=model,
        train_loader=full_loader,
        val_loader=full_loader,   # in-sample val — trains until convergence
        device=device,
        epochs=CONFIG['epochs'],
        lr=CONFIG['lr'],
        patience=CONFIG['patience'],
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed:.0f}s")
    print(f"  In-sample AUC : {metrics['auc_roc']:.4f}   F1 : {metrics['f1']:.4f}")

    # ── Step 5 : Predict ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Step 5 · Predicting on Candidates")
    print(f"{'─'*70}")

    p_plus1, z_scores = predict(model, test_graphs, device, CONFIG['batch_size'])

    # ── Step 6 : Display results ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS — Single GCN (trained on full StaPep dataset)")
    print(f"{'='*70}")
    print(f"\n  {'Peptide':<30} {'AMP?':<6} {'P(+1)':<10} {'z (logit margin)'}")
    print(f"  {'─'*60}")

    for j, (p, z) in enumerate(zip(p_plus1, z_scores)):
        ci   = valid_idx[j]
        name = CANDIDATES[ci]['name']
        amp  = "Yes" if p > 0.5 else "No"
        print(f"  {name:<30} {amp:<6} {p:.4f}     {z:+.4f}")

    print(f"\n  P(+1) = sigmoid(z) = 1 / (1 + exp(−z))")
    print(f"  z > 0  → classified as AMP   (P > 0.5)")
    print(f"  z < 0  → classified as non-AMP (P < 0.5)")
    print(f"  Magnitude of z reflects distance from the decision boundary.")


if __name__ == '__main__':
    main()
