# Models Directory

This directory contains scripts for ESM model processing and testing.

---

## Scripts

### 1. `esm_sequence_processor.py`
**Main script for ESM processing**

Process peptide sequences with Meta's ESM models:
- Extract ESM-2 embeddings (1280D feature vectors)
- Predict 3D structures with ESMFold
- GPU-accelerated

**Usage:**
```cmd
# Extract embeddings (fast, for ML features)
python esm_sequence_processor.py --input seqs.txt --output embeddings.csv --mode embeddings

# Predict structures (slow, for structural analysis)
python esm_sequence_processor.py --input seqs.txt --output structures/ --mode fold

# Do both
python esm_sequence_processor.py --input seqs.txt --output results/ --mode both
```

**Options:**
- `--input`: Sequence file (SVM format: `index sequence`)
- `--output`: Output path (CSV for embeddings, dir for structures)
- `--mode`: `embeddings`, `fold`, or `both`
- `--device`: `cuda` or `cpu` (auto-detected)
- `--esm-model`: Model size (`esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D`, `esm2_t36_3B_UR50D`)
- `--max-length`: Max sequence length for folding (default: 400)

**Requirements:**
- `esm_env` conda environment
- PyTorch with CUDA 12.8+ (for RTX 5070)

---

### 2. `test_gpu_esmfold.py`
**GPU testing and validation**

Test your GPU setup with ESM models:
- Verify CUDA is working
- Benchmark GPU vs CPU performance
- Test ESM-2 model loading
- Test ESMFold (optional, large download)

**Usage:**
```cmd
conda activate esm_env
python test_gpu_esmfold.py
```

**What it tests:**
1. ✅ PyTorch & CUDA detection
2. ✅ GPU computation speed
3. ✅ ESM-2 model inference
4. ✅ ESMFold structure prediction (optional)

**Expected output:**
- GPU detected (NVIDIA GeForce RTX 5070)
- CUDA available: True
- GPU 10-50x faster than CPU
- ESM models load and run successfully

---

## Quick Start

### Test GPU First:
```cmd
conda activate esm_env
cd models
python test_gpu_esmfold.py
```

### Process Your Sequences:
```cmd
# From project root
conda activate esm_env
python models\esm_sequence_processor.py --input experiments\exp1\raw.txt --output experiments\exp1\esm_results --mode embeddings
```

---

## Input Format

Both scripts use the same input format as the SVM pipeline:

```
1 MKTAYIAKQRQISFVKSHFSRQL
2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG
3 MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVF
```

Format: `<index> <sequence>` (space-separated)

---

## Output Formats

### ESM-2 Embeddings (CSV):
```csv
seqIndex,esm2_dim_0,esm2_dim_1,...,esm2_dim_1279
1,0.123,-0.456,...,0.789
2,-0.234,0.567,...,-0.890
```

- 1280 dimensions (for 650M model)
- Ready for ML/SVM input
- Mean-pooled per-sequence representations

### ESMFold Structures:
```
structures/
├── structure_1.pdb
├── structure_2.pdb
├── structure_3.pdb
└── folding_summary.csv
```

- Standard PDB format
- Viewable in PyMOL, ChimeraX, etc.
- Summary CSV with metadata

---

## Model Comparison

| Model | Parameters | Speed | Embedding Dim | Best For |
|-------|-----------|-------|---------------|----------|
| esm2_t30_150M | 150M | Fast | 640 | Quick screening |
| esm2_t33_650M | 650M | Medium | 1280 | **Recommended balance** |
| esm2_t36_3B | 3B | Slow | 2560 | Highest accuracy |
| esmfold_v1 | ~15GB | Very slow | N/A | Structure prediction |

**Recommendation:** Start with `esm2_t33_650M` (default)

---

## Performance Notes

### ESM-2 Embeddings:
- **Speed:** ~0.1-1 second per sequence
- **GPU Memory:** ~4GB
- **Batch processing:** Supported (processes one at a time currently)

### ESMFold:
- **Speed:** ~5-30 seconds per sequence (depends on length)
- **GPU Memory:** ~6-8GB
- **Max length:** 400 aa recommended (longer = more memory)
- **First run:** Downloads ~15GB model (one-time)

### Tips:
- Process embeddings for all sequences (fast)
- Only fold top candidates from SVM predictions (slow)
- Use smaller sequences (<100 aa) for faster folding

---

## Integration with SVM Pipeline

### Workflow 1: Parallel Features
```
Sequences → ESM-2 embeddings → New ML model
         → Traditional descriptors → Existing SVM
```

### Workflow 2: Sequential (Recommended)
```
Sequences → ESM-2 embeddings → Feature vector
         → SVM predictions → Top hits
         → ESMFold structures → Structural analysis
```

See `WORKFLOW_EXAMPLE.md` in the parent directory for detailed examples.

---

## Troubleshooting

**"CUDA out of memory":**
- Use smaller model: `--esm-model esm2_t30_150M_UR50D`
- Reduce batch size (future enhancement)
- Use CPU: `--device cpu` (much slower)

**"ESM import error":**
- Install: `pip install fair-esm`
- Make sure you're in `esm_env`

**Slow on CPU:**
- Install CUDA PyTorch: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- Verify: `python -c "import torch; print(torch.cuda.is_available())"`

**ESMFold download fails:**
- Check internet connection
- Model downloads to `~/.cache/torch/hub/`
- ~15GB download on first run

---

## Future Enhancements

- [ ] Batch processing for multiple sequences
- [ ] Integration with SVM feature vectors
- [ ] Structure quality metrics (pLDDT scores)
- [ ] Multi-GPU support
- [ ] Progress saving/resuming for large datasets

---

## References

- **ESM-2:** [Meta AI Research](https://github.com/facebookresearch/esm)
- **ESMFold:** [Lin et al. 2023](https://www.science.org/doi/10.1126/science.ade2574)
- **Paper:** "Language models of protein sequences at the scale of evolution enable accurate structure prediction"
