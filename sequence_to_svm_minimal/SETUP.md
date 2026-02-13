# Setup Guide for SVM_ESM_Peptides

This guide covers setting up the project on a **new Windows machine** with WSL (Windows Subsystem for Linux) for GPU-accelerated training.

---

## Quick Start (Existing Setup)

If you already have WSL and CUDA configured:

```bash
# Activate WSL
wsl

# Navigate to project
cd /mnt/c/Users/YOUR_USERNAME/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal

# Activate virtual environment
source venv/bin/activate

# Run training
python run_gnn_comparison.py
```

---

## Full Setup on a New Machine

### Step 1: Install WSL2

Open PowerShell as Administrator:

```powershell
# Install WSL with Ubuntu
wsl --install -d Ubuntu

# Restart your computer when prompted
```

After restart, set up your Ubuntu username and password.

### Step 2: Install NVIDIA Drivers (Windows Side)

1. Download latest NVIDIA Game Ready drivers from: https://www.nvidia.com/Download/index.aspx
2. Install on Windows (NOT inside WSL)
3. Verify in PowerShell:
   ```powershell
   nvidia-smi
   ```

### Step 3: Install CUDA Toolkit in WSL

```bash
# Open WSL
wsl

# Update packages
sudo apt update && sudo apt upgrade -y

# Install CUDA toolkit (WSL version)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Add to PATH (add to ~/.bashrc)
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA
nvcc --version
nvidia-smi
```

### Step 4: Install Python and Dependencies

```bash
# Install Python 3.10+ and venv
sudo apt install -y python3.10 python3.10-venv python3-pip

# Navigate to project
cd /mnt/c/Users/YOUR_USERNAME/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 5: Install PyTorch with CUDA

```bash
# Install PyTorch for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Step 6: Install PyTorch Geometric

```bash
# Install PyG
pip install torch-geometric

# Install PyG extensions (for faster operations)
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```

### Step 7: Install Remaining Dependencies

```bash
# Install all other requirements
pip install numpy pandas scipy scikit-learn
pip install biopython propy3
pip install matplotlib seaborn tqdm joblib requests
pip install jupyter ipywidgets py3Dmol

# Optional: ESMFold inference (large download)
pip install transformers fair-esm
```

### Step 8: Install DSSP (Optional, for secondary structure)

```bash
sudo apt install -y dssp

# Verify
mkdssp --version
```

---

## Project Structure

```
sequence_to_svm_minimal/
├── data/
│   └── training_dataset/
│       ├── AMP/                  # ESMFold PDB structures
│       ├── DECOY/
│       ├── geometric_features_clustered.csv
│       └── seqs_AMP.txt, seqs_decoy_subsample.txt
├── features/
│   └── geometric_features.py     # Extract 24 geometric features
├── gnn/
│   ├── data_utils.py             # PDB → Graph conversion
│   ├── models.py                 # GCN, GAT, EGNN architectures
│   └── train.py                  # Training loop
├── nn_pipeline/
│   ├── feature_dataset.py        # Feature loading
│   ├── models.py                 # MLP architecture
│   └── train.py                  # NN training
├── results/
│   └── gnn/
│       ├── curves/               # Training curves per run
│       └── *.json                # Results
├── run_gnn_comparison.py         # Main GNN comparison script
├── run_nn_training.py            # MLP training
├── run_feature_fusion_experiments.py
├── plot_training_curves.ipynb    # Visualization notebook
└── requirements.txt
```

---

## Running Experiments

### 1. GNN Architecture Comparison

```bash
# Full run (500 epochs, ~1-2 hours on RTX 3090)
python run_gnn_comparison.py

# Quick test (10 epochs)
python run_gnn_comparison.py  # (edit CONFIG in script)
```

### 2. MLP Feature Fusion

```bash
python run_feature_fusion_experiments.py
```

### 3. Visualize Training Curves

```bash
# Open Jupyter in WSL
jupyter notebook --no-browser

# Open plot_training_curves.ipynb
```

---

## Troubleshooting

### CUDA not detected in WSL

```bash
# Check if nvidia-smi works
nvidia-smi

# If not, ensure Windows NVIDIA drivers are installed (not WSL drivers)
# WSL uses Windows GPU drivers through virtualization
```

### PyTorch Geometric installation issues

```bash
# Install from source if wheels fail
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Or use conda
conda install pyg -c pyg
```

### Out of GPU memory

```bash
# Reduce batch size in the training script
# Edit CONFIG['batch_size'] = 16  # or smaller
```

### Import errors

```bash
# Ensure you're in the correct directory
cd /mnt/c/Users/.../sequence_to_svm_minimal

# Ensure venv is activated
source venv/bin/activate
```

---

## Version Reference

Tested with:
- Python 3.10.12
- PyTorch 2.1.0+cu121
- PyTorch Geometric 2.4.0
- CUDA 12.1
- Ubuntu 22.04 (WSL2)
- Windows 11

---

## Contact

For issues, check the project's GitHub or contact the maintainer.
