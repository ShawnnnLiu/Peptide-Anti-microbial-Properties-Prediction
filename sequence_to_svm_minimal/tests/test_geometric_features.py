#!/usr/bin/env python3
"""
Unit tests and demo for geometric feature extraction.

Tests:
- PDB parsing
- Individual feature extractors
- Full pipeline
- Edge cases (short peptides, missing atoms)

Run:
    python tests/test_geometric_features.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.geometric_features import (
    parse_pdb_structure,
    extract_plddt_features,
    extract_compactness_features,
    extract_secondary_structure_phipsi,
    extract_sasa_features,
    extract_sequence_features,
    extract_curvature_features,
    extract_all_features,
    get_feature_names
)


def test_feature_extraction_on_real_pdb():
    """Test feature extraction on actual ESMFold output."""
    
    # Find test PDB files
    base_dir = Path(__file__).parent.parent
    amp_dir = base_dir / "data" / "training_dataset" / "structures" / "AMP"
    decoy_dir = base_dir / "data" / "training_dataset" / "structures" / "DECOY"
    
    test_files = []
    
    if amp_dir.exists():
        amp_files = list(amp_dir.glob("*.pdb"))[:3]
        test_files.extend(amp_files)
    
    if decoy_dir.exists():
        decoy_files = list(decoy_dir.glob("*.pdb"))[:3]
        test_files.extend(decoy_files)
    
    if not test_files:
        print("⚠️  No test PDB files found. Skipping real PDB test.")
        return False
    
    print(f"\n{'='*60}")
    print(f"  Testing on {len(test_files)} PDB files")
    print(f"{'='*60}\n")
    
    all_passed = True
    
    for pdb_path in test_files:
        print(f"📂 Testing: {pdb_path.name}")
        
        try:
            # Parse structure
            structure, residues_info = parse_pdb_structure(str(pdb_path))
            n_residues = len(residues_info)
            print(f"   ✅ Parsed: {n_residues} residues")
            
            # Test pLDDT
            plddt = extract_plddt_features(residues_info)
            assert 0 <= plddt['plddt_mean'] <= 1, "pLDDT out of range"
            print(f"   ✅ pLDDT: mean={plddt['plddt_mean']:.3f}, std={plddt['plddt_std']:.3f}")
            
            # Test compactness
            compact = extract_compactness_features(residues_info)
            assert compact['radius_gyration'] >= 0, "Rg should be non-negative"
            print(f"   ✅ Compactness: Rg={compact['radius_gyration']:.2f}Å, E2E={compact['end_to_end_distance']:.2f}Å")
            
            # Test secondary structure
            ss = extract_secondary_structure_phipsi(residues_info)
            total_ss = ss['fraction_helix'] + ss['fraction_sheet'] + ss['fraction_coil']
            assert abs(total_ss - 1.0) < 0.01, f"SS fractions should sum to 1, got {total_ss}"
            print(f"   ✅ Secondary structure: H={ss['fraction_helix']:.2f}, E={ss['fraction_sheet']:.2f}, C={ss['fraction_coil']:.2f}")
            
            # Test SASA
            sasa = extract_sasa_features(structure)
            print(f"   ✅ SASA: total={sasa['total_sasa']:.1f}Ų, hydrophobic_frac={sasa['fraction_hydrophobic_sasa']:.2f}")
            
            # Test sequence features
            seq = extract_sequence_features(residues_info)
            print(f"   ✅ Sequence: len={seq['length']}, charge={seq['net_charge']:.1f}, hydro={seq['mean_hydrophobicity']:.2f}")
            
            # Test curvature
            curv = extract_curvature_features(residues_info)
            print(f"   ✅ Curvature: mean={curv['curvature_mean']:.4f}, max={curv['curvature_max']:.4f}")
            
            # Test full extraction
            all_features = extract_all_features(str(pdb_path), peptide_id=pdb_path.stem)
            n_features = len([k for k, v in all_features.items() if isinstance(v, (int, float))])
            print(f"   ✅ Total features: {n_features}")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            all_passed = False
    
    return all_passed


def test_feature_consistency():
    """Test that feature names match extracted features."""
    print(f"\n{'='*60}")
    print("  Testing feature name consistency")
    print(f"{'='*60}\n")
    
    expected_names = get_feature_names(include_optional=False)
    print(f"📋 Expected feature names: {len(expected_names)}")
    
    # Get a test PDB
    base_dir = Path(__file__).parent.parent
    amp_dir = base_dir / "data" / "training_dataset" / "structures" / "AMP"
    
    if not amp_dir.exists():
        print("⚠️  No test files available")
        return True
    
    pdb_files = list(amp_dir.glob("*.pdb"))
    if not pdb_files:
        print("⚠️  No PDB files found")
        return True
    
    features = extract_all_features(str(pdb_files[0]))
    
    # Check that all expected names are present
    missing = [n for n in expected_names if n not in features]
    extra = [n for n in features if n not in expected_names 
             and n not in ['peptide_id', 'sequence', 'svm_sigma', 'svm_prob_positive']
             and not n.startswith('qsar_')]
    
    if missing:
        print(f"❌ Missing features: {missing}")
        return False
    
    if extra:
        print(f"⚠️  Extra features: {extra}")
    
    print("✅ All expected features present")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print(f"\n{'='*60}")
    print("  Testing edge cases")
    print(f"{'='*60}\n")
    
    # Test with empty residues list
    plddt = extract_plddt_features([])
    assert plddt['plddt_mean'] == 0.0, "Empty list should return 0"
    print("✅ Empty residue list handled")
    
    # Test with minimal data
    minimal_residue = [{
        'residue': None,
        'resname': 'ALA',
        'aa': 'A',
        'resid': 1,
        'ca_coord': np.array([0.0, 0.0, 0.0]),
        'n_coord': None,
        'c_coord': None,
        'plddt': 0.8
    }]
    
    plddt = extract_plddt_features(minimal_residue)
    assert plddt['plddt_mean'] == 0.8, "Single residue pLDDT should be 0.8"
    print("✅ Single residue handled")
    
    # Test compactness with short peptide
    short_residues = [
        {'ca_coord': np.array([0.0, 0.0, 0.0]), 'plddt': 0.9, 'aa': 'A', 'n_coord': None, 'c_coord': None},
        {'ca_coord': np.array([3.8, 0.0, 0.0]), 'plddt': 0.9, 'aa': 'A', 'n_coord': None, 'c_coord': None},
    ]
    
    compact = extract_compactness_features(short_residues)
    assert compact['end_to_end_distance'] == 3.8, "E2E distance should be 3.8"
    print("✅ Short peptide (2 residues) handled")
    
    # Test sequence features
    seq_features = extract_sequence_features([], sequence="KKLLKKLLKK")
    assert seq_features['length'] == 10, "Length should be 10"
    assert seq_features['net_charge'] > 0, "Should have positive charge"
    print("✅ Sequence-only features work")
    
    print("\n✅ All edge case tests passed")
    return True


def demo_feature_extraction():
    """Demo: Extract and display features from sample PDBs."""
    print(f"\n{'='*60}")
    print("  DEMO: Feature Extraction")
    print(f"{'='*60}\n")
    
    base_dir = Path(__file__).parent.parent
    amp_dir = base_dir / "data" / "training_dataset" / "structures" / "AMP"
    decoy_dir = base_dir / "data" / "training_dataset" / "structures" / "DECOY"
    
    samples = []
    
    if amp_dir.exists():
        amp_files = sorted(amp_dir.glob("*.pdb"))[:2]
        samples.extend([(f, "AMP") for f in amp_files])
    
    if decoy_dir.exists():
        decoy_files = sorted(decoy_dir.glob("*.pdb"))[:2]
        samples.extend([(f, "DECOY") for f in decoy_files])
    
    if not samples:
        print("No sample files found for demo.")
        return
    
    print("Comparing AMP vs DECOY features:\n")
    
    results = []
    
    for pdb_path, label in samples:
        features = extract_all_features(str(pdb_path), peptide_id=pdb_path.stem)
        features['class'] = label
        results.append(features)
        
        print(f"📍 {pdb_path.stem} ({label})")
        print(f"   Length: {features['length']} residues")
        print(f"   pLDDT: {features['plddt_mean']:.3f}")
        print(f"   Rg: {features['radius_gyration']:.2f} Å")
        print(f"   Helix: {features['fraction_helix']:.1%}")
        print(f"   Charge: {features['net_charge']:+.1f}")
        print(f"   Hydrophobicity: {features['mean_hydrophobicity']:.2f}")
        print()
    
    # Compare averages
    amp_results = [r for r in results if r['class'] == 'AMP']
    decoy_results = [r for r in results if r['class'] == 'DECOY']
    
    if amp_results and decoy_results:
        print("📊 Average Comparison (AMP vs DECOY):")
        
        for key in ['plddt_mean', 'radius_gyration', 'fraction_helix', 'net_charge', 'mean_hydrophobicity']:
            amp_avg = np.mean([r[key] for r in amp_results])
            decoy_avg = np.mean([r[key] for r in decoy_results])
            print(f"   {key}: AMP={amp_avg:.3f}, DECOY={decoy_avg:.3f}")


def main():
    """Run all tests and demo."""
    print("\n" + "🧬" * 30)
    print("  Geometric Feature Extraction Tests")
    print("🧬" * 30)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_edge_cases()
    all_passed &= test_feature_consistency()
    all_passed &= test_feature_extraction_on_real_pdb()
    
    # Run demo
    demo_feature_extraction()
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("  ✅ ALL TESTS PASSED")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
