"""
Check what ESMFold files have been downloaded so far
"""

import sys
from pathlib import Path

# Make utils/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import TORCH_HUB_CHECKPOINTS


def check_cache():
    """Check ESMFold cache status"""
    cache_dir = TORCH_HUB_CHECKPOINTS
    
    print("=" * 60)
    print("  ESMFold Cache Status")
    print("=" * 60)
    print()
    print(f"📁 Cache location: {cache_dir}")
    print()
    
    if not cache_dir.exists():
        print("❌ No cache directory found")
        print("   ESMFold has not been downloaded yet")
        return
    
    files = list(cache_dir.glob("*"))
    
    if not files:
        print("❌ Cache directory is empty")
        print("   ESMFold has not been downloaded yet")
        return
    
    total_size = 0
    print("📦 Downloaded files:")
    print()
    
    for f in sorted(files):
        if f.is_file():
            size_bytes = f.stat().st_size
            size_gb = size_bytes / (1024**3)
            size_mb = size_bytes / (1024**2)
            total_size += size_bytes
            
            # Show in MB if < 1GB, otherwise GB
            if size_gb < 1:
                print(f"  ✓ {f.name}")
                print(f"    Size: {size_mb:.1f} MB")
            else:
                print(f"  ✓ {f.name}")
                print(f"    Size: {size_gb:.2f} GB")
            print()
    
    total_gb = total_size / (1024**3)
    
    print("=" * 60)
    print(f"📊 Total downloaded: {total_gb:.2f} GB")
    print()
    
    # ESMFold expected size
    expected_size = 15.0
    progress = (total_gb / expected_size) * 100
    
    if total_gb < 1:
        print("⏳ Status: Just started (< 1 GB)")
        print(f"   Progress: {progress:.1f}%")
        print(f"   Remaining: ~{expected_size - total_gb:.1f} GB")
    elif total_gb < 10:
        print("⏳ Status: Downloading... (partial)")
        print(f"   Progress: {progress:.1f}%")
        print(f"   Remaining: ~{expected_size - total_gb:.1f} GB")
    elif total_gb < 14:
        print("⏳ Status: Almost done!")
        print(f"   Progress: {progress:.1f}%")
        print(f"   Remaining: ~{expected_size - total_gb:.1f} GB")
    else:
        print("✅ Status: Complete!")
        print("   ESMFold is ready to use")
    
    print("=" * 60)


if __name__ == "__main__":
    check_cache()
