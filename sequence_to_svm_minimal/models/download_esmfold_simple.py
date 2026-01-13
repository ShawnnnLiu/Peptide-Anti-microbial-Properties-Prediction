"""
Simplest way to download ESMFold from HuggingFace
Uses transformers library (most reliable)
"""

import sys


def main():
    print("=" * 70)
    print("  ESMFold Download - Simple Method")
    print("=" * 70)
    print()
    
    # Install transformers if needed
    print("🔧 Installing/checking transformers library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
    
    print()
    print("🚀 Downloading ESMFold from HuggingFace...")
    print("   This will download ~8.44 GB")
    print("   Progress bar will show below...")
    print()
    
    try:
        from transformers import EsmForProteinFolding
        
        # This automatically downloads from HuggingFace with progress bar
        print("📥 Loading model (will download if not cached)...")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        
        print()
        print("=" * 70)
        print("  ✅ SUCCESS!")
        print("=" * 70)
        print()
        print("✅ ESMFold downloaded and ready to use!")
        print()
        print("To use it in your code:")
        print()
        print("  from transformers import EsmForProteinFolding")
        print("  model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis script will resume if you run it again!")
        sys.exit(1)


if __name__ == "__main__":
    main()
