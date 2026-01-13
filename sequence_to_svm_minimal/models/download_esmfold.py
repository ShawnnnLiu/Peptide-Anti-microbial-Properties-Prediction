"""
Download ESMFold model with progress tracking
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm


def check_existing_cache():
    """Check what's already in the cache"""
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    
    print("🔍 Checking existing cache...")
    print(f"Cache directory: {cache_dir}")
    print()
    
    if not cache_dir.exists():
        print("❌ Cache directory doesn't exist yet")
        return 0
    
    total_size = 0
    files = list(cache_dir.glob("*"))
    
    if not files:
        print("❌ No files in cache yet")
        return 0
    
    print("📦 Files in cache:")
    for f in files:
        if f.is_file():
            size_gb = f.stat().st_size / (1024**3)
            total_size += f.stat().st_size
            print(f"  - {f.name}: {size_gb:.2f} GB")
    
    total_gb = total_size / (1024**3)
    print(f"\n✅ Total cached: {total_gb:.2f} GB")
    
    return total_size


def download_with_progress(url, dest_path):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)


def download_esmfold():
    """Download ESMFold with progress tracking"""
    
    print("=" * 60)
    print("  ESMFold Model Downloader")
    print("=" * 60)
    print()
    
    # Check existing cache
    existing_size = check_existing_cache()
    
    print("\n" + "=" * 60)
    print("  Starting ESMFold Download")
    print("=" * 60)
    print()
    
    # Expected total size
    expected_total_gb = 15.0
    existing_gb = existing_size / (1024**3)
    remaining_gb = expected_total_gb - existing_gb
    
    print(f"📊 Download Status:")
    print(f"   Already downloaded: {existing_gb:.2f} GB")
    print(f"   Expected total: {expected_total_gb:.2f} GB")
    print(f"   Remaining: {remaining_gb:.2f} GB")
    print()
    
    if existing_gb > 14:
        print("✅ Looks like ESMFold is already downloaded!")
        test = input("Test loading it? (y/n): ")
        if test.lower() == 'y':
            print("\n🔄 Testing ESMFold...")
            import esm
            model = esm.pretrained.esmfold_v1()
            print("✅ ESMFold loaded successfully!")
            return
    
    print("🚀 Downloading ESMFold...")
    print("⚠️  This will take 10-30 minutes depending on your internet speed")
    print()
    
    # Import and download
    try:
        import esm
        from tqdm.auto import tqdm as auto_tqdm
        
        # Monkey-patch torch.hub to show progress
        import torch.hub
        original_download = torch.hub.download_url_to_file
        
        def download_with_tqdm(url, dst, *args, **kwargs):
            """Wrapper to add progress bar to torch.hub downloads"""
            kwargs['progress'] = True
            return original_download(url, dst, *args, **kwargs)
        
        torch.hub.download_url_to_file = download_with_tqdm
        
        print("🔄 Loading ESMFold (downloading if needed)...")
        model = esm.pretrained.esmfold_v1()
        
        print("\n✅ ESMFold download complete!")
        print(f"   Model cached for future use")
        
        # Check final cache size
        print("\n" + "=" * 60)
        check_existing_cache()
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted!")
        print("Run this script again to resume from where you left off")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Run this script again to retry")
        print("3. If it keeps failing, the download will resume from where it stopped")
        sys.exit(1)


def main():
    """Main function"""
    download_esmfold()
    
    print("\n" + "=" * 60)
    print("  ✅ All Done!")
    print("=" * 60)
    print()
    print("Now you can run:")
    print("  python models/esm_sequence_processor.py --input seqs.txt --output structures/ --mode fold")
    print()


if __name__ == "__main__":
    main()
