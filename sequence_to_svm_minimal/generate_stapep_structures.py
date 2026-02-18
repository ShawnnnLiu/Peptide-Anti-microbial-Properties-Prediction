#!/usr/bin/env python3
"""
Generate PDB structures for StaPep dataset using ESMFold.

Uses the same approach as batch_esmfold.py which works successfully.
Reads from txt files in the same format as seqs_AMP.txt.

Usage:
    python generate_stapep_structures.py [--amp-only | --decoy-only]
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import torch
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'amp_seqs': 'data/training_dataset/StaPep/seqs_AMP_stapep.txt',
    'decoy_seqs': 'data/training_dataset/StaPep/seqs_DECOY_stapep.txt',
    'amp_output': 'data/training_dataset/StaPep/structures/AMP',
    'decoy_output': 'data/training_dataset/StaPep/structures/DECOY',
}


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
                sequences.append((idx, seq.strip()))
            elif len(parts) == 1:
                seq = parts[0]
                idx = len(sequences) + 1
                sequences.append((str(idx), seq.strip()))
    
    return sequences


def load_esmfold_model(device="cuda"):
    """Load ESMFold model with memory optimizations (same as batch_esmfold.py)"""
    from transformers import EsmForProteinFolding
    
    print(f"\n{'='*60}")
    print(f"  Loading ESMFold Model")
    print(f"{'='*60}")
    
    # Try local model first
    local_model_path = Path(__file__).parent / "models" / "esmfold_v1_local"
    
    # Load in FP16 for memory efficiency
    load_dtype = torch.float16 if device == "cuda" else torch.float32
    
    if local_model_path.exists():
        print(f"✅ Loading from local: {local_model_path}")
        model = EsmForProteinFolding.from_pretrained(
            str(local_model_path),
            local_files_only=True,
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )
    else:
        print("⚠️  Local model not found, downloading from HuggingFace...")
        model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1",
            torch_dtype=load_dtype,
            low_cpu_mem_usage=True
        )
    
    model = model.to(device)
    model.eval()
    
    if device == "cuda":
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Loaded in FP16 | GPU: {mem_used:.1f}/{mem_total:.1f} GB")
    
    return model


def predict_single_structure(model, sequence, device="cuda"):
    """Predict structure for a single sequence"""
    if device == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        pdb_string = model.infer_pdb(sequence)
    
    return pdb_string


def generate_structures(sequences, output_dir, model, device, label_name):
    """Generate structures for a list of sequences."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check what's already done
    existing = set(p.stem.replace('structure_', '') for p in output_dir.glob("*.pdb"))
    to_generate = [(idx, seq) for idx, seq in sequences if idx not in existing]
    
    print(f"\n📦 {label_name}: {len(existing)} existing, {len(to_generate)} to generate")
    
    if len(to_generate) == 0:
        print(f"✅ All {label_name} structures already exist!")
        return 0, 0
    
    success = 0
    failed = 0
    start_time = time.time()
    
    pbar = tqdm(to_generate, desc=f"{label_name}", unit="seq")
    
    for idx, seq in pbar:
        pdb_path = output_dir / f"structure_{idx}.pdb"
        
        try:
            pdb_string = predict_single_structure(model, seq, device)
            
            with open(pdb_path, 'w') as f:
                f.write(pdb_string)
            
            success += 1
            
        except torch.cuda.OutOfMemoryError:
            tqdm.write(f"⚠️  OOM for {idx} (len={len(seq)}), skipping...")
            torch.cuda.empty_cache()
            failed += 1
            
        except Exception as e:
            tqdm.write(f"❌ Error for {idx}: {e}")
            failed += 1
        
        # Update progress
        elapsed = time.time() - start_time
        if success > 0:
            avg_time = elapsed / success
            remaining = (len(to_generate) - success - failed) * avg_time
            pbar.set_postfix({
                'done': f"{success}✓ {failed}✗",
                'eta': str(timedelta(seconds=int(remaining)))
            })
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Generate StaPep structures using ESMFold")
    parser.add_argument('--amp-only', action='store_true', help='Only generate AMP structures')
    parser.add_argument('--decoy-only', action='store_true', help='Only generate Decoy structures')
    args = parser.parse_args()
    
    print("🧬" * 30)
    print("   StaPep Structure Generation")
    print("🧬" * 30)
    
    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU available, using CPU (will be slow)")
    
    # Determine what to generate
    do_amp = not args.decoy_only
    do_decoy = not args.amp_only
    
    # Load sequences
    print(f"\n{'='*60}")
    print("  Loading Sequences")
    print(f"{'='*60}")
    
    amp_seqs = []
    decoy_seqs = []
    
    if do_amp:
        amp_path = Path(CONFIG['amp_seqs'])
        if amp_path.exists():
            amp_seqs = parse_sequence_file(amp_path)
            print(f"✅ AMPs: {len(amp_seqs)} sequences from {amp_path}")
        else:
            print(f"❌ AMP file not found: {amp_path}")
    
    if do_decoy:
        decoy_path = Path(CONFIG['decoy_seqs'])
        if decoy_path.exists():
            decoy_seqs = parse_sequence_file(decoy_path)
            print(f"✅ Decoys: {len(decoy_seqs)} sequences from {decoy_path}")
        else:
            print(f"❌ Decoy file not found: {decoy_path}")
    
    if not amp_seqs and not decoy_seqs:
        print("\n❌ No sequences to process!")
        sys.exit(1)
    
    # Load model
    model = load_esmfold_model(device)
    
    # Generate structures
    print(f"\n{'='*60}")
    print("  Generating Structures")
    print(f"{'='*60}")
    
    total_success = 0
    total_failed = 0
    
    if amp_seqs:
        success, failed = generate_structures(
            amp_seqs, CONFIG['amp_output'], model, device, "AMP"
        )
        total_success += success
        total_failed += failed
    
    if decoy_seqs:
        success, failed = generate_structures(
            decoy_seqs, CONFIG['decoy_output'], model, device, "DECOY"
        )
        total_success += success
        total_failed += failed
    
    # Summary
    print(f"\n{'='*60}")
    print("  ✅ Generation Complete!")
    print(f"{'='*60}")
    print(f"   Successful: {total_success}")
    print(f"   Failed:     {total_failed}")
    
    # Count actual files
    amp_count = len(list(Path(CONFIG['amp_output']).glob("*.pdb"))) if do_amp else 0
    decoy_count = len(list(Path(CONFIG['decoy_output']).glob("*.pdb"))) if do_decoy else 0
    
    print(f"\n📁 Structures:")
    if do_amp:
        print(f"   AMP:   {amp_count} in {CONFIG['amp_output']}")
    if do_decoy:
        print(f"   DECOY: {decoy_count} in {CONFIG['decoy_output']}")
    
    print("\n✅ Ready for GNN training!")


if __name__ == '__main__':
    main()
