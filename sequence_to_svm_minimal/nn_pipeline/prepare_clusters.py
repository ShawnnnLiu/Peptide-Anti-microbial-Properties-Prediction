#!/usr/bin/env python3
"""
Prepare sequences for CD-HIT clustering and parse cluster results.

This module:
1. Converts sequences to FASTA format for CD-HIT
2. Parses CD-HIT cluster output
3. Assigns cluster IDs for GroupKFold splitting

Usage:
    # Step 1: Generate FASTA
    python prepare_clusters.py --generate-fasta \
        --input data/training_dataset/geometric_features.csv \
        --output data/training_dataset/sequences.fasta
    
    # Step 2: Run CD-HIT externally
    # cd-hit -i sequences.fasta -o clusters -c 0.40 -n 2 -M 16000
    
    # Step 3: Parse clusters
    python prepare_clusters.py --parse-clusters \
        --clstr-file data/training_dataset/clusters.clstr \
        --features-csv data/training_dataset/geometric_features.csv \
        --output data/training_dataset/geometric_features_clustered.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re


def generate_fasta(features_csv: Path, output_fasta: Path) -> int:
    """
    Convert features CSV to FASTA format for CD-HIT.
    
    Args:
        features_csv: Path to geometric_features.csv
        output_fasta: Output FASTA file path
        
    Returns:
        Number of sequences written
    """
    df = pd.read_csv(features_csv)
    
    with open(output_fasta, 'w') as f:
        for _, row in df.iterrows():
            peptide_id = row['peptide_id']
            sequence = row['sequence']
            f.write(f">{peptide_id}\n{sequence}\n")
    
    print(f"✅ Wrote {len(df)} sequences to {output_fasta}")
    return len(df)


def parse_cdhit_clusters(clstr_file: Path) -> Dict[str, int]:
    """
    Parse CD-HIT .clstr file to extract cluster assignments.
    
    Args:
        clstr_file: Path to .clstr file from CD-HIT
        
    Returns:
        Dictionary mapping peptide_id → cluster_id
    """
    cluster_map = {}
    current_cluster = -1
    
    with open(clstr_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('>Cluster'):
                # New cluster: ">Cluster 0"
                current_cluster = int(line.split()[1])
            elif line:
                # Sequence line: "0	22aa, >AMP_123... *"
                # Extract the sequence ID between > and ...
                match = re.search(r'>(\S+)\.\.\.', line)
                if match:
                    peptide_id = match.group(1)
                    cluster_map[peptide_id] = current_cluster
    
    return cluster_map


def add_clusters_to_features(features_csv: Path, clstr_file: Path, 
                              output_csv: Path) -> pd.DataFrame:
    """
    Add cluster assignments to features CSV.
    
    Args:
        features_csv: Original geometric_features.csv
        clstr_file: CD-HIT cluster file
        output_csv: Output path for clustered features
        
    Returns:
        DataFrame with cluster_id column
    """
    df = pd.read_csv(features_csv)
    cluster_map = parse_cdhit_clusters(clstr_file)
    
    # Add cluster IDs
    df['cluster_id'] = df['peptide_id'].map(cluster_map)
    
    # Check for unmapped sequences
    unmapped = df['cluster_id'].isna().sum()
    if unmapped > 0:
        print(f"⚠️  {unmapped} sequences not found in cluster file")
        # Assign to new clusters
        max_cluster = df['cluster_id'].max()
        unmapped_mask = df['cluster_id'].isna()
        df.loc[unmapped_mask, 'cluster_id'] = range(
            int(max_cluster) + 1, 
            int(max_cluster) + 1 + unmapped
        )
    
    df['cluster_id'] = df['cluster_id'].astype(int)
    
    # Save
    df.to_csv(output_csv, index=False)
    
    n_clusters = df['cluster_id'].nunique()
    print(f"✅ Added cluster IDs to {len(df)} sequences")
    print(f"   Total clusters: {n_clusters}")
    print(f"   Cluster sizes: min={df['cluster_id'].value_counts().min()}, "
          f"max={df['cluster_id'].value_counts().max()}, "
          f"median={df['cluster_id'].value_counts().median():.0f}")
    print(f"   Saved to: {output_csv}")
    
    return df


def create_simple_clusters(features_csv: Path, output_csv: Path, 
                           identity_threshold: float = 0.80) -> pd.DataFrame:
    """
    Create simple sequence-based clusters without CD-HIT.
    
    Uses a greedy approach: for each unclustered sequence, create a new cluster
    and add all sequences with >threshold identity to it.
    
    This is slower than CD-HIT but works without external dependencies.
    
    Args:
        features_csv: Input features CSV
        output_csv: Output path
        identity_threshold: Sequence identity threshold (0-1)
        
    Returns:
        DataFrame with cluster_id column
    """
    from difflib import SequenceMatcher
    
    df = pd.read_csv(features_csv)
    sequences = df['sequence'].tolist()
    n = len(sequences)
    
    print(f"🔄 Creating simple clusters at {identity_threshold*100:.0f}% identity...")
    print(f"   This may take a while for {n} sequences...")
    
    cluster_ids = [-1] * n
    current_cluster = 0
    
    for i in range(n):
        if cluster_ids[i] >= 0:
            continue  # Already assigned
            
        # Start new cluster with this sequence
        cluster_ids[i] = current_cluster
        seq_i = sequences[i]
        
        # Find all similar sequences
        for j in range(i + 1, n):
            if cluster_ids[j] >= 0:
                continue
            
            seq_j = sequences[j]
            
            # Quick length filter
            len_ratio = min(len(seq_i), len(seq_j)) / max(len(seq_i), len(seq_j))
            if len_ratio < identity_threshold:
                continue
            
            # Compute sequence identity
            identity = SequenceMatcher(None, seq_i, seq_j).ratio()
            
            if identity >= identity_threshold:
                cluster_ids[j] = current_cluster
        
        current_cluster += 1
        
        if current_cluster % 50 == 0:
            print(f"   Processed {current_cluster} clusters...")
    
    df['cluster_id'] = cluster_ids
    df.to_csv(output_csv, index=False)
    
    n_clusters = df['cluster_id'].nunique()
    print(f"✅ Created {n_clusters} clusters")
    print(f"   Cluster sizes: min={df['cluster_id'].value_counts().min()}, "
          f"max={df['cluster_id'].value_counts().max()}")
    print(f"   Saved to: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare sequence clusters for training")
    
    # Mode selection
    parser.add_argument('--generate-fasta', action='store_true',
                        help='Generate FASTA file for CD-HIT')
    parser.add_argument('--parse-clusters', action='store_true',
                        help='Parse CD-HIT output and add to features')
    parser.add_argument('--simple-clusters', action='store_true',
                        help='Create simple clusters without CD-HIT')
    
    # Input/output
    parser.add_argument('--input', '-i', type=Path,
                        help='Input features CSV')
    parser.add_argument('--output', '-o', type=Path,
                        help='Output file path')
    parser.add_argument('--clstr-file', type=Path,
                        help='CD-HIT .clstr file (for --parse-clusters)')
    parser.add_argument('--identity', type=float, default=0.80,
                        help='Sequence identity threshold (default: 0.80)')
    
    args = parser.parse_args()
    
    if args.generate_fasta:
        if not args.input or not args.output:
            parser.error("--generate-fasta requires --input and --output")
        generate_fasta(args.input, args.output)
        
    elif args.parse_clusters:
        if not args.input or not args.output or not args.clstr_file:
            parser.error("--parse-clusters requires --input, --output, and --clstr-file")
        add_clusters_to_features(args.input, args.clstr_file, args.output)
        
    elif args.simple_clusters:
        if not args.input or not args.output:
            parser.error("--simple-clusters requires --input and --output")
        create_simple_clusters(args.input, args.output, args.identity)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
