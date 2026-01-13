"""
Fix HuggingFace cache structure for aria2c downloaded model
"""

import os
from pathlib import Path
import shutil

def fix_huggingface_cache():
    """Reorganize aria2c downloaded file into proper HuggingFace cache structure"""
    
    cache_base = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--esmfold_v1"
    
    # Find the downloaded file (it's named with hash)
    snapshots_dir = cache_base / "snapshots" / "main"
    
    if not snapshots_dir.exists():
        print(f"❌ Error: {snapshots_dir} doesn't exist")
        return False
    
    print(f"📁 Looking in: {snapshots_dir}")
    files = list(snapshots_dir.iterdir())
    
    print(f"Found {len(files)} files:")
    for f in files:
        size_gb = f.stat().st_size / (1024**3)
        print(f"  - {f.name}: {size_gb:.2f} GB")
    
    # Find the large model file (should be ~8.4GB)
    model_file = None
    for f in files:
        if f.is_file() and f.stat().st_size > 5e9:  # > 5GB
            model_file = f
            break
    
    if not model_file:
        print("❌ Could not find the model file (should be ~8.4GB)")
        return False
    
    print(f"\n✅ Found model file: {model_file.name}")
    print(f"   Size: {model_file.stat().st_size / (1024**3):.2f} GB")
    
    # Check if it needs to be renamed
    target_name = "pytorch_model.bin"
    if model_file.name != target_name:
        print(f"\n🔄 Renaming to {target_name}...")
        target_path = snapshots_dir / target_name
        model_file.rename(target_path)
        print(f"✅ Renamed: {target_path}")
    else:
        print(f"\n✅ Already named correctly: {target_name}")
    
    # Create blobs directory with symlink (HuggingFace cache structure)
    blobs_dir = cache_base / "blobs"
    blobs_dir.mkdir(exist_ok=True)
    
    # Also download config.json if not present
    config_path = snapshots_dir / "config.json"
    if not config_path.exists():
        print("\n📥 Downloading config.json...")
        import requests
        config_url = "https://huggingface.co/facebook/esmfold_v1/resolve/main/config.json"
        response = requests.get(config_url)
        config_path.write_bytes(response.content)
        print("✅ Downloaded config.json")
    
    print("\n" + "="*60)
    print("✅ Cache structure fixed!")
    print("="*60)
    print("\nNow you can load the model with:")
    print("  from transformers import EsmForProteinFolding")
    print("  model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')")
    print()
    
    return True


if __name__ == "__main__":
    fix_huggingface_cache()
