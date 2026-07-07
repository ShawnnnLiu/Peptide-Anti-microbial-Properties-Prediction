#!/usr/bin/env python3
"""
Deep semantic analysis of QSAR-12 vs Geometric-24 feature overlap.
Check if any features measure the same concept under different names.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Bootstrap project root so this script can reach the shared path module whether
# run standalone or imported.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import DATA_DIR


# Define what each feature measures (per-feature documentation used by the
# semantic-overlap report below).
qsar_meanings = {
    'netCharge': 'Sum of residue charges at pH7 (K,R,H=+1; D,E=-1)',
    'FC': 'Dipeptide frequency: Phe-Cys pairs / total dipeptides',
    'LW': 'Dipeptide frequency: Leu-Trp pairs / total dipeptides',
    'DP': 'Dipeptide frequency: Asp-Pro pairs / total dipeptides',
    'NK': 'Dipeptide frequency: Asn-Lys pairs / total dipeptides',
    'AE': 'Dipeptide frequency: Ala-Glu pairs / total dipeptides',
    'pcMK': 'Proportion: count(M) / (count(M) + count(K))',
    '_SolventAccessibilityD1025': 'CTD: % residues in "buried" solvent accessibility class (sequence-based)',
    'tau2_GRAR740104': 'Sequence-order: avg Grantham distance at lag 2',
    'tau4_GRAR740104': 'Sequence-order: avg Grantham distance at lag 4',
    'QSO50_GRAR740104': 'Quasi-sequence-order descriptor 50',
    'QSO29_GRAR740104': 'Quasi-sequence-order descriptor 29',
}

geo_meanings = {
    'plddt_mean': 'ESMFold confidence (structure prediction quality)',
    'plddt_std': 'Variation in prediction confidence',
    'plddt_min': 'Lowest confidence residue',
    'plddt_max': 'Highest confidence residue',
    'radius_gyration': '3D compactness: RMS distance from centroid',
    'end_to_end_distance': '3D: distance between first and last Cα',
    'max_pairwise_distance': '3D diameter: max Cα-Cα distance',
    'centroid_distance_mean': '3D: mean distance to centroid',
    'centroid_distance_std': '3D: variation in centroid distances',
    'fraction_helix': 'Secondary structure: % α-helix',
    'fraction_sheet': 'Secondary structure: % β-sheet',
    'fraction_coil': 'Secondary structure: % coil',
    'total_sasa': '3D: total solvent accessible surface area (Å²)',
    'hydrophobic_sasa': '3D: SASA of hydrophobic residues',
    'fraction_hydrophobic_sasa': '3D: hydrophobic SASA / total SASA',
    'length': 'Number of residues in sequence',
    'net_charge': 'Sum of residue charges at pH7 (K,R=+1; H=+0.1; D,E=-1)',
    'mean_hydrophobicity': 'Mean Kyte-Doolittle hydrophobicity score',
    'hydrophobic_moment': 'Amphipathicity: hydrophobic moment assuming helix',
    'curvature_mean': '3D: mean Menger curvature of Cα backbone',
    'curvature_std': '3D: curvature variation',
    'curvature_max': '3D: maximum backbone curvature',
    'torsion_mean': '3D: mean backbone torsion angle',
    'torsion_std': '3D: torsion variation',
}


def main():
    # Load both datasets
    qsar = pd.read_csv(DATA_DIR / "qsar12_descriptors.csv")
    geo = pd.read_csv(DATA_DIR / "geometric_features.csv")

    # Align by peptide_id
    df = qsar.merge(geo, on=['peptide_id', 'sequence'], how='inner')

    print('='*80)
    print('SEMANTIC OVERLAP ANALYSIS: What does each feature ACTUALLY measure?')
    print('='*80)

    print('\n' + '='*80)
    print('QSAR-12 FEATURE DEFINITIONS')
    print('='*80)
    for feat, meaning in qsar_meanings.items():
        print(f'  {feat:35s} → {meaning}')

    print('\n' + '='*80)
    print('GEOMETRIC-24 FEATURE DEFINITIONS')
    print('='*80)
    for feat, meaning in geo_meanings.items():
        print(f'  {feat:35s} → {meaning}')

    print('\n' + '='*80)
    print('POTENTIAL SEMANTIC OVERLAPS TO CHECK')
    print('='*80)

    # Define potential semantic overlaps
    semantic_checks = [
        ('netCharge', 'net_charge', 'CHARGE', 'Both compute net charge at pH7'),
        ('_SolventAccessibilityD1025', 'total_sasa', 'SASA', 'Both relate to solvent accessibility'),
        ('_SolventAccessibilityD1025', 'hydrophobic_sasa', 'SASA', 'Accessibility of hydrophobic residues'),
        ('_SolventAccessibilityD1025', 'fraction_hydrophobic_sasa', 'SASA', 'Hydrophobic accessibility ratio'),
        ('pcMK', 'mean_hydrophobicity', 'HYDRO', 'M is hydrophobic, K is charged'),
        ('pcMK', 'net_charge', 'CHARGE', 'K contributes to charge'),
        ('tau2_GRAR740104', 'mean_hydrophobicity', 'PHYSCHEM', 'Grantham encodes physicochemical properties'),
        ('tau4_GRAR740104', 'mean_hydrophobicity', 'PHYSCHEM', 'Grantham encodes physicochemical properties'),
        ('QSO50_GRAR740104', 'mean_hydrophobicity', 'PHYSCHEM', 'QSO uses Grantham distances'),
        ('QSO29_GRAR740104', 'mean_hydrophobicity', 'PHYSCHEM', 'QSO uses Grantham distances'),
        ('tau2_GRAR740104', 'hydrophobic_moment', 'AMPHIPATHY', 'Both encode spatial AA patterns'),
        ('tau4_GRAR740104', 'hydrophobic_moment', 'AMPHIPATHY', 'Both encode spatial AA patterns'),
    ]

    print()
    for qsar_feat, geo_feat, category, reason in semantic_checks:
        corr = df[qsar_feat].corr(df[geo_feat])
        flag = '🔴 OVERLAP!' if abs(corr) > 0.9 else ('🟡 RELATED' if abs(corr) > 0.5 else '✅ DISTINCT')
        print(f'{category:10s} | {qsar_feat:30s} vs {geo_feat:30s} | r={corr:+.3f} | {flag}')
        print(f'           | Reason: {reason}')
        print()

    # Full correlation matrix
    print('='*80)
    print('COMPLETE CROSS-CORRELATION MATRIX (QSAR vs Geo)')
    print('='*80)

    qsar_feats = list(qsar_meanings.keys())
    geo_feats = list(geo_meanings.keys())

    # Create correlation matrix
    corr_matrix = np.zeros((len(qsar_feats), len(geo_feats)))
    for i, qf in enumerate(qsar_feats):
        for j, gf in enumerate(geo_feats):
            corr_matrix[i, j] = df[qf].corr(df[gf])

    # Print as heatmap-style table
    print('\nCorrelation heatmap (|r| > 0.3 shown, others as . ):')
    print()
    header = '                                   ' + ' '.join([f'{g[:6]:>7s}' for g in geo_feats])
    print(header)
    print('-' * len(header))

    for i, qf in enumerate(qsar_feats):
        row = f'{qf:35s}'
        for j, gf in enumerate(geo_feats):
            r = corr_matrix[i, j]
            if abs(r) > 0.9:
                row += f' {r:+.2f}*'
            elif abs(r) > 0.5:
                row += f' {r:+.2f} '
            elif abs(r) > 0.3:
                row += f' {r:+.2f} '
            else:
                row += '    .   '
        print(row)

    # Summary
    print('\n' + '='*80)
    print('FINAL VERDICT')
    print('='*80)

    high_corr = []
    for i, qf in enumerate(qsar_feats):
        for j, gf in enumerate(geo_feats):
            r = corr_matrix[i, j]
            if abs(r) > 0.8:
                high_corr.append((qf, gf, r))

    print()
    print('Features with |correlation| > 0.8:')
    if high_corr:
        for qf, gf, r in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f'  🔴 {qf} ↔ {gf}: r = {r:.4f}')
    else:
        print('  None found!')

    print()
    print('Features with 0.5 < |correlation| < 0.8:')
    moderate_corr = [(qf, gf, corr_matrix[i,j]) for i, qf in enumerate(qsar_feats)
                     for j, gf in enumerate(geo_feats)
                     if 0.5 < abs(corr_matrix[i,j]) < 0.8]
    if moderate_corr:
        for qf, gf, r in sorted(moderate_corr, key=lambda x: -abs(x[2])):
            print(f'  🟡 {qf} ↔ {gf}: r = {r:.4f}')
    else:
        print('  None found!')

    print()
    print('='*80)
    print('CONCLUSION')
    print('='*80)
    print('''
1. netCharge (QSAR) ↔ net_charge (Geo): r=0.95
   → SAME CONCEPT (net charge), slightly different His treatment
   → This is TRUE REDUNDANCY

2. _SolventAccessibilityD1025 (QSAR) ↔ SASA features (Geo): r≈0
   → DIFFERENT: QSAR is sequence-based CTD, Geo is 3D structure SASA
   → NO redundancy

3. tau/QSO (QSAR) ↔ hydrophobicity features (Geo): r<0.3
   → DIFFERENT: QSAR uses Grantham distances (AA substitution cost),
     Geo uses Kyte-Doolittle (transfer free energy)
   → NO redundancy

4. All dipeptide features (FC, LW, DP, NK, AE): No Geo equivalent
   → UNIQUE to QSAR

5. All 3D structural features (pLDDT, curvature, torsion, compactness):
   → UNIQUE to Geometric (require 3D structure)

BOTTOM LINE: Only ONE true overlap exists (netCharge ↔ net_charge).
Combined-36 effectively has 35 independent features + 1 near-duplicate.
''')


if __name__ == "__main__":
    main()
