"""
Extract numerical features from ESMFold PDB structures
Outputs feature CSV compatible with SVM pipeline
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_pdb_file(pdb_file):
    """
    Parse PDB file and extract CA (alpha carbon) coordinates
    Returns: numpy array of shape (N, 3) with x, y, z coordinates
    """
    coords = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # Extract atom name
                atom_name = line[12:16].strip()
                
                # Only use CA (alpha carbon) atoms
                if atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    
    return np.array(coords)


def extract_plddt_scores(pdb_file):
    """
    Extract pLDDT confidence scores from B-factor column
    ESMFold stores confidence in the B-factor field
    """
    plddts = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name == 'CA':
                    # B-factor is in columns 60-66
                    bfactor = float(line[60:66])
                    plddts.append(bfactor)
    
    return np.array(plddts)


def calculate_radius_of_gyration(coords):
    """
    Calculate radius of gyration (Rg)
    Measure of structural compactness
    """
    center = coords.mean(axis=0)
    distances_sq = np.sum((coords - center)**2, axis=1)
    rg = np.sqrt(distances_sq.mean())
    return rg


def calculate_end_to_end_distance(coords):
    """
    Distance between first and last CA atoms
    """
    if len(coords) < 2:
        return 0.0
    
    distance = np.linalg.norm(coords[0] - coords[-1])
    return distance


def calculate_asphericity(coords):
    """
    Measure of how non-spherical the structure is
    0 = perfect sphere, higher = more elongated
    """
    center = coords.mean(axis=0)
    centered = coords - center
    
    # Calculate moment of inertia tensor
    I = np.dot(centered.T, centered) / len(coords)
    
    # Get eigenvalues
    eigvals = np.linalg.eigvalsh(I)
    eigvals = np.sort(eigvals)[::-1]  # Sort descending
    
    # Asphericity
    if eigvals.sum() == 0:
        return 0.0
    
    asphericity = eigvals[0] - 0.5 * (eigvals[1] + eigvals[2])
    return asphericity


def calculate_contact_order(coords, threshold=8.0):
    """
    Calculate relative contact order
    Measures long-range vs short-range contacts
    """
    n = len(coords)
    if n < 2:
        return 0.0
    
    contacts = 0
    contact_distance = 0
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                contacts += 1
                contact_distance += abs(j - i)
    
    if contacts == 0:
        return 0.0
    
    # Relative contact order
    rco = contact_distance / (contacts * n)
    return rco


def calculate_secondary_structure_simple(coords):
    """
    Simple helix detection based on CA distances
    More sophisticated: use DSSP, but this works without external tools
    """
    n = len(coords)
    if n < 4:
        return 0.0, 0.0
    
    # Helix: CA(i) to CA(i+3) distance ~5.5 Å
    # Sheet: CA(i) to CA(i+2) distance ~6.5 Å
    
    helix_count = 0
    sheet_count = 0
    
    for i in range(n - 3):
        dist_i_i3 = np.linalg.norm(coords[i] - coords[i+3])
        # Helix signature
        if 4.5 < dist_i_i3 < 6.5:
            helix_count += 1
    
    for i in range(n - 2):
        dist_i_i2 = np.linalg.norm(coords[i] - coords[i+2])
        # Sheet signature
        if 6.0 < dist_i_i2 < 7.5:
            sheet_count += 1
    
    helix_fraction = helix_count / (n - 3) if n > 3 else 0.0
    sheet_fraction = sheet_count / (n - 2) if n > 2 else 0.0
    
    return helix_fraction, sheet_fraction


def extract_features_from_pdb(pdb_file):
    """
    Extract all structural features from a single PDB file
    Returns: dictionary of features
    """
    features = {}
    
    try:
        # Parse coordinates
        coords = parse_pdb_file(pdb_file)
        
        if len(coords) == 0:
            return None
        
        # Parse confidence scores
        plddts = extract_plddt_scores(pdb_file)
        
        # Basic features
        features['length'] = len(coords)
        
        # Confidence features
        features['plddt_mean'] = plddts.mean() if len(plddts) > 0 else 0.0
        features['plddt_min'] = plddts.min() if len(plddts) > 0 else 0.0
        features['plddt_std'] = plddts.std() if len(plddts) > 0 else 0.0
        features['plddt_q25'] = np.percentile(plddts, 25) if len(plddts) > 0 else 0.0
        features['plddt_q75'] = np.percentile(plddts, 75) if len(plddts) > 0 else 0.0
        
        # Geometric features
        features['radius_gyration'] = calculate_radius_of_gyration(coords)
        features['end_to_end_dist'] = calculate_end_to_end_distance(coords)
        features['asphericity'] = calculate_asphericity(coords)
        features['contact_order'] = calculate_contact_order(coords)
        
        # Compactness
        features['compactness'] = features['radius_gyration'] / features['length'] if features['length'] > 0 else 0.0
        
        # Secondary structure (simple)
        helix_frac, sheet_frac = calculate_secondary_structure_simple(coords)
        features['helix_fraction'] = helix_frac
        features['sheet_fraction'] = sheet_frac
        features['coil_fraction'] = 1.0 - helix_frac - sheet_frac
        
        # Shape descriptors
        center = coords.mean(axis=0)
        centered = coords - center
        distances = np.linalg.norm(centered, axis=1)
        features['max_distance_from_center'] = distances.max()
        features['mean_distance_from_center'] = distances.mean()
        features['std_distance_from_center'] = distances.std()
        
        return features
        
    except Exception as e:
        print(f"Error processing {pdb_file}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract structural features from ESMFold PDB files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python extract_structure_features.py --input structures/ --output structure_features.csv

Input: Directory containing PDB files from ESMFold
Output: CSV file with numerical features for each structure
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input directory containing PDB files')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for features')
    
    args = parser.parse_args()
    
    # Find all PDB files
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    pdb_files = sorted(input_dir.glob('*.pdb'))
    
    if len(pdb_files) == 0:
        print(f"Error: No PDB files found in {input_dir}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  Extracting Structural Features")
    print(f"{'='*60}")
    print(f"Found {len(pdb_files)} PDB files")
    
    # Process all PDB files
    all_features = []
    all_indices = []
    
    for pdb_file in tqdm(pdb_files, desc="Processing structures"):
        # Extract sequence index from filename (e.g., structure_1.pdb -> 1)
        filename = pdb_file.stem
        if filename.startswith('structure_'):
            seq_idx = filename.replace('structure_', '')
        else:
            seq_idx = filename
        
        features = extract_features_from_pdb(pdb_file)
        
        if features is not None:
            all_indices.append(seq_idx)
            all_features.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df.insert(0, 'seqIndex', all_indices)
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    
    print(f"\n✅ Feature extraction complete")
    print(f"   Processed: {len(all_features)}/{len(pdb_files)} structures")
    print(f"   Features: {len(df.columns)-1} descriptors")
    print(f"   Output: {args.output}")
    
    # Print feature summary
    print(f"\n{'='*60}")
    print(f"  Feature Summary")
    print(f"{'='*60}")
    print(df.describe())
    
    print()


if __name__ == "__main__":
    main()
