"""
Diagnose ESMFold model file corruption
"""

import torch
from pathlib import Path
import hashlib

def check_file_integrity(filepath):
    """Check if the model file is corrupted"""
    filepath = Path(filepath)
    
    print("=" * 70)
    print("  ESMFold Model Diagnostics")
    print("=" * 70)
    print()
    
    # Check file exists
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    # Check file size
    size_bytes = filepath.stat().st_size
    size_gb = size_bytes / (1024**3)
    print(f"📊 File Information:")
    print(f"   Path: {filepath}")
    print(f"   Size: {size_gb:.3f} GB ({size_bytes:,} bytes)")
    print(f"   Expected: ~2.58 GB (2,771,653,574 bytes)")
    
    if abs(size_bytes - 2771653574) > 1000:
        print(f"   ⚠️  Size mismatch!")
    else:
        print(f"   ✅ Size matches expected")
    
    print()
    
    # Calculate MD5 hash (for verification)
    print("🔐 Calculating MD5 hash (this may take a minute)...")
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096*1024), b""):
            md5_hash.update(chunk)
    
    file_md5 = md5_hash.hexdigest()
    print(f"   MD5: {file_md5}")
    print()
    
    # Try to load with PyTorch
    print("🔍 Attempting to load with PyTorch...")
    try:
        checkpoint = torch.load(filepath, map_location='cpu')
        print("✅ PyTorch can read the file")
        print()
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            print("📦 Checkpoint contents:")
            for key in list(checkpoint.keys())[:10]:
                print(f"   - {key}")
            if len(checkpoint.keys()) > 10:
                print(f"   ... and {len(checkpoint.keys()) - 10} more keys")
            print()
            
            # Check for the missing keys
            missing_keys = [
                'trunk.structure_module.ipa.linear_q_points.linear.weight',
                'trunk.structure_module.ipa.linear_q_points.linear.bias',
                'trunk.structure_module.ipa.linear_kv_points.linear.weight',
                'trunk.structure_module.ipa.linear_kv_points.linear.bias'
            ]
            
            print("🔎 Checking for reported missing keys:")
            for key in missing_keys:
                if key in checkpoint:
                    print(f"   ✅ Found: {key}")
                else:
                    print(f"   ❌ Missing: {key}")
            print()
            
            # List all keys with 'ipa' in them
            ipa_keys = [k for k in checkpoint.keys() if 'ipa' in k.lower()]
            if ipa_keys:
                print(f"🔍 Found {len(ipa_keys)} keys containing 'ipa':")
                for key in ipa_keys[:20]:
                    print(f"   - {key}")
                if len(ipa_keys) > 20:
                    print(f"   ... and {len(ipa_keys) - 20} more")
        else:
            print(f"   ⚠️  Checkpoint is not a dict, it's: {type(checkpoint)}")
        
    except Exception as e:
        print(f"❌ PyTorch load failed: {e}")
        return False
    
    print()
    print("=" * 70)
    return True


if __name__ == "__main__":
    meta_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "esmfold_3B_v1.pt"
    check_file_integrity(meta_path)
