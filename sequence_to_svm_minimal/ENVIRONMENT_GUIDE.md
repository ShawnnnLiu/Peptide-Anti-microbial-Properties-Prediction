# Environment Setup Guide

This project uses **two separate conda environments** for different tasks:

## 1. skl_legacy (Python 3.7)
**Purpose:** Run the legacy SVM predictions with scikit-learn 0.19.2

**Activate:**
```cmd
conda activate skl_legacy
# or
.\activate_env.bat
```

**Use for:**
- Running `scripts/run_sequence_svm.py`
- Using the pre-trained SVM model (`svc.pkl`)
- Classic descriptor-based predictions

---

## 2. esm_env (Python 3.10)
**Purpose:** Extract ESM embeddings using Meta's ESM models with PyTorch 2.x and GPU support

**Activate:**
```cmd
conda activate esm_env
# or
.\activate_esm.bat
```

**Use for:**
- ESM-2 protein language model embeddings
- ESMFold structure predictions
- Modern PyTorch workflows
- GPU-accelerated feature extraction

---

## Setup Instructions

### First-time Setup:

**1. Create the legacy SVM environment:**
```cmd
conda env create -f skl_legacy_env.yml
```

**2. Create the ESM environment:**
```cmd
conda env create -f esm_env.yml
```

**3. Install PyTorch with CUDA support:**

After creating the environment, install PyTorch with GPU support:

For **RTX 5070 (Blackwell)** or newer GPUs:
```cmd
conda activate esm_env
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For **RTX 30/40 series** GPUs:
```cmd
conda activate esm_env
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**4. Verify GPU is working:**
```cmd
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## Typical Workflow

### Option A: SVM Predictions Only
```cmd
conda activate skl_legacy
python scripts/run_sequence_svm.py --seqs ... --model-pkl ...
```

### Option B: ESM + SVM Pipeline
```cmd
# Step 1: Extract ESM embeddings
conda activate esm_env
python extract_esm_features.py --input sequences.fasta --output esm_embeddings.csv

# Step 2: Run SVM predictions
conda activate skl_legacy
python scripts/run_sequence_svm.py --features esm_embeddings.csv ...
```

---

## Environment Details

| Feature | skl_legacy | esm_env |
|---------|-----------|---------|
| Python | 3.7.16 | 3.10+ |
| PyTorch | ❌ Not installed | ✅ 2.9+ with CUDA 12.8 (Blackwell) |
| scikit-learn | 0.19.2 (legacy) | Latest |
| ESM models | ❌ Not compatible | ✅ fair-esm |
| GPU Support | ❌ N/A | ✅ RTX 5070/Blackwell ready |

---

## Troubleshooting

**"conda not recognized":**
- Use CMD instead of PowerShell, or
- Run: `(& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression`

**"CUDA available: False" after installing PyTorch:**
- You likely have the CPU-only version installed
- Run: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- The `--force-reinstall` flag is critical to replace CPU version with CUDA version
- Verify with: `python -c "import torch; print(torch.__version__)"`
- Should show `2.9.1+cu128`, NOT `2.9.1+cpu`

**"CUDA out of memory":**
- Reduce batch size in ESM extraction
- Use smaller ESM models (esm2_t12 instead of esm2_t33)

**"Module not found":**
- Make sure you're in the correct environment
- Run `conda list` to verify installed packages

**PowerShell conda activation issues:**
- Strongly recommend using **CMD (Command Prompt)** instead of PowerShell on Windows
- Conda works more reliably with CMD
