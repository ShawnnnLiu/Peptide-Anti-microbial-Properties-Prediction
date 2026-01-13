"""
Download ESMFold from HuggingFace
More reliable than direct Meta downloads!
"""

import os
from pathlib import Path
import subprocess
import sys


def main():
    print("=" * 70)
    print("  ESMFold Download from HuggingFace")
    print("=" * 70)
    print()
    print("📦 Source: https://huggingface.co/facebook/esmfold_v1")
    print("📊 Size: ~8.44 GB")
    print("⏱️  Estimated time: 10-30 minutes")
    print()
    
    # Install huggingface_hub if needed
    print("🔧 Checking dependencies...")
    try:
        import huggingface_hub
        print("✅ huggingface_hub already installed")
    except ImportError:
        print("📥 Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        import huggingface_hub
        print("✅ huggingface_hub installed")
    
    print()
    print("=" * 70)
    print("  Starting Download")
    print("=" * 70)
    print()
    
    # Download using HuggingFace Hub
    from huggingface_hub import hf_hub_download
    
    try:
        print("🚀 Downloading ESMFold model...")
        print("   This will show a progress bar...")
        print()
        
        # Download the main model file
        model_path = hf_hub_download(
            repo_id="facebook/esmfold_v1",
            filename="pytorch_model.bin",
            cache_dir=str(Path.home() / ".cache" / "huggingface"),
            resume_download=True  # Allows resuming if interrupted
        )
        
        print()
        print("=" * 70)
        print("  ✅ Download Complete!")
        print("=" * 70)
        print()
        print(f"📁 Model saved to: {model_path}")
        print()
        print("🔄 Now you need to convert this to work with ESM library...")
        print()
        
        # Also download config files
        print("📥 Downloading config files...")
        config_path = hf_hub_download(
            repo_id="facebook/esmfold_v1",
            filename="config.json",
            cache_dir=str(Path.home() / ".cache" / "huggingface"),
            resume_download=True
        )
        print(f"✅ Config saved to: {config_path}")
        
        print()
        print("=" * 70)
        print("  All Done!")
        print("=" * 70)
        print()
        print("⚠️  NOTE: This HuggingFace model uses 'transformers' library")
        print("You can load it with:")
        print()
        print("  from transformers import EsmForProteinFolding")
        print("  model = EsmForProteinFolding.from_pretrained('facebook/esmfold_v1')")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted!")
        print("✅ Progress saved - run this script again to resume")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Run this script again - it will resume automatically")
        print("3. Make sure you have ~10 GB free disk space")
        sys.exit(1)


if __name__ == "__main__":
    main()
