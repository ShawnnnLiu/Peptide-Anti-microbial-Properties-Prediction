"""
Test script to verify RTX 5070 GPU works with ESMFold
Run this with esm_env activated to test your GPU setup
"""

import sys
import torch
import time

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_pytorch_cuda():
    """Test basic PyTorch CUDA functionality"""
    print_section("1. PyTorch & CUDA Setup")
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\n❌ ERROR: CUDA is not available!")
        print("   Make sure you installed PyTorch with CUDA support.")
        return False
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU 0 Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU 0 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return True

def test_gpu_computation():
    """Test basic GPU computation"""
    print_section("2. GPU Computation Test")
    
    # Create a test tensor
    size = 10000
    print(f"Creating {size}x{size} matrix multiplication test...")
    
    # CPU test
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start = time.time()
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"CPU Time: {cpu_time:.4f} seconds")
    
    # GPU test
    device = torch.device("cuda:0")
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    
    # Warm up GPU
    _ = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start = time.time()
    c_gpu = torch.matmul(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU Time: {gpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x faster on GPU")
    
    print(f"\n✅ GPU computation working! Your RTX 5070 is {cpu_time/gpu_time:.1f}x faster than CPU.")
    return True

def test_esm_model():
    """Test ESM-2 model (lighter than ESMFold for quick test)"""
    print_section("3. ESM-2 Model Test")
    
    try:
        import esm
        print("✅ ESM library imported successfully")
    except ImportError:
        print("❌ ERROR: ESM library not found!")
        print("   Install with: pip install fair-esm")
        return False
    
    # Load a smaller ESM-2 model first
    print("\nLoading ESM-2 650M model (quick test)...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    
    # Move to GPU
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    
    # Test with a sample peptide sequence
    test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN"
    data = [("test_peptide", test_sequence)]
    
    print(f"\nTest sequence: {test_sequence}")
    print(f"Length: {len(test_sequence)} amino acids")
    
    # Convert sequence
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    print("\n🚀 Running inference on GPU...")
    start = time.time()
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
        embeddings = results["representations"][33]
    
    inference_time = time.time() - start
    
    print(f"✅ Inference complete in {inference_time:.4f} seconds")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    return True

def test_esmfold():
    """Test ESMFold model (structure prediction)"""
    print_section("4. ESMFold Structure Prediction Test")
    
    try:
        import esm
        print("Loading ESMFold model...")
        print("⚠️  WARNING: This will download ~15GB if not cached!")
        print("   First run will take several minutes...")
        
        user_input = input("\nContinue with ESMFold test? (y/n): ")
        if user_input.lower() != 'y':
            print("Skipping ESMFold test.")
            return True
        
        # Load ESMFold
        model = esm.pretrained.esmfold_v1()
        device = torch.device("cuda:0")
        model = model.to(device)
        model.eval()
        
        print(f"✅ ESMFold loaded on GPU: {torch.cuda.get_device_name(0)}")
        
        # Test with a short peptide
        test_sequence = "MKTAYIAKQRQISFVKSHFSRQL"
        print(f"\nPredicting structure for: {test_sequence}")
        print(f"Length: {len(test_sequence)} amino acids")
        
        print("\n🚀 Running structure prediction on GPU...")
        start = time.time()
        
        with torch.no_grad():
            output = model.infer_pdb(test_sequence)
        
        inference_time = time.time() - start
        
        print(f"✅ Structure prediction complete in {inference_time:.4f} seconds")
        print(f"PDB output generated ({len(output)} characters)")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        # Save PDB output
        output_file = "test_structure.pdb"
        with open(output_file, "w") as f:
            f.write(output)
        print(f"\n💾 Structure saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ESMFold test failed: {str(e)}")
        return False

def main():
    """Run all GPU tests"""
    print("\n" + "🔬" * 30)
    print("   RTX 5070 + ESM GPU Test Suite")
    print("🔬" * 30)
    
    # Test 1: PyTorch CUDA
    if not test_pytorch_cuda():
        print("\n❌ CUDA not available. Cannot proceed with GPU tests.")
        sys.exit(1)
    
    # Test 2: GPU computation
    try:
        test_gpu_computation()
    except Exception as e:
        print(f"\n❌ GPU computation test failed: {str(e)}")
        sys.exit(1)
    
    # Test 3: ESM-2 model
    try:
        test_esm_model()
    except Exception as e:
        print(f"\n❌ ESM-2 test failed: {str(e)}")
        print("   This might be due to missing fair-esm package.")
        sys.exit(1)
    
    # Test 4: ESMFold (optional, large download)
    try:
        test_esmfold()
    except Exception as e:
        print(f"\n⚠️  ESMFold test encountered an issue: {str(e)}")
        print("   ESM-2 embeddings still work fine for feature extraction!")
    
    # Final summary
    print_section("✅ TEST SUMMARY")
    print("Your RTX 5070 is properly configured for ESM!")
    print("\nNext steps:")
    print("  1. Use ESM-2 for generating sequence embeddings")
    print("  2. Use ESMFold for structure prediction (optional)")
    print("  3. Feed embeddings to your SVM model (skl_legacy env)")
    print("\nEnvironment ready! 🚀")

if __name__ == "__main__":
    main()
