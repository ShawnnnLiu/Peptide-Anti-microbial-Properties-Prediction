# Complete Workflow: ESM + SVM Pipeline

This guide shows how to use ESM models for feature extraction and then run SVM predictions.

---

## Quick Start

### Option 1: ESM Embeddings → SVM (Recommended)

```cmd
REM Step 1: Activate ESM environment and extract embeddings
conda activate esm_env
python models\esm_sequence_processor.py --input experiments\exp1\raw.txt --output experiments\exp1\esm_embeddings.csv --mode embeddings

REM Step 2: Switch to SVM environment and run predictions
conda activate skl_legacy
python scripts\run_sequence_svm.py --seqs experiments\exp1\predictionsParameters\seqs_1.txt --aaindex descriptors\aaindex --output-dir experiments\exp1\predictionsParameters --model-pkl predictionsParameters\svc.pkl --scaler-csv predictionsParameters\Z_score_mean_std__intersect_noflip.csv
```

### Option 2: ESMFold Structures → Extract Features → SVM

```cmd
REM Step 1: Predict structures
conda activate esm_env
python models\esm_sequence_processor.py --input experiments\exp1\raw.txt --output experiments\exp1\structures --mode fold

REM Step 2: Extract structural features (if you have structure feature extraction)
REM [Your structure feature extraction script here]

REM Step 3: Run SVM
conda activate skl_legacy
python scripts\run_sequence_svm.py [...]
```

---

## Detailed Workflow

### Step 1: Prepare Input Sequences

Create a sequence file in the format:
```
experiments/exp1/raw.txt
```

Format (index + sequence, space-separated):
```
1 MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPN
2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG
3 MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVF
```

### Step 2: Extract ESM Features

**Activate ESM environment:**
```cmd
conda activate esm_env
```

**Extract embeddings:**
```cmd
python models\esm_sequence_processor.py ^
    --input experiments\exp1\raw.txt ^
    --output experiments\exp1\esm_embeddings.csv ^
    --mode embeddings ^
    --esm-model esm2_t33_650M_UR50D ^
    --device cuda
```

**Options:**
- `--mode embeddings` - Extract ESM-2 embeddings (for ML features)
- `--mode fold` - Predict 3D structures with ESMFold
- `--mode both` - Do both
- `--esm-model` - Choose model size:
  - `esm2_t30_150M_UR50D` (fastest, smaller)
  - `esm2_t33_650M_UR50D` (balanced, recommended)
  - `esm2_t36_3B_UR50D` (best quality, slower)

**Output:**
- `esm_embeddings.csv` - Feature vectors (1280 dimensions for 650M model)
- Each row corresponds to one sequence

### Step 3: Generate Sequence Windows (if needed)

```cmd
conda activate skl_legacy
python scripts\make_seqs_windows.py ^
    --in experiments\exp1\raw.txt ^
    --out experiments\exp1\predictionsParameters\seqs_1.txt ^
    --min-len 10 ^
    --max-len 35 ^
    --stride 1
```

### Step 4: Run SVM Predictions

**Activate SVM environment:**
```cmd
conda activate skl_legacy
```

**Run predictions:**
```cmd
python scripts\run_sequence_svm.py ^
    --seqs experiments\exp1\predictionsParameters\seqs_1.txt ^
    --aaindex descriptors\aaindex ^
    --output-dir experiments\exp1\predictionsParameters ^
    --model-pkl predictionsParameters\svc.pkl ^
    --scaler-csv predictionsParameters\Z_score_mean_std__intersect_noflip.csv
```

**Output:**
- `descriptors.csv` - Generated descriptors
- `descriptors_PREDICTIONS.csv` - Sorted predictions
- `descriptors_PREDICTIONS_unsorted.csv` - Original order predictions

---

## Complete Example Script

Save as `run_full_pipeline.bat`:

```cmd
@echo off
REM Complete ESM + SVM Pipeline

set EXPERIMENT=experiments\my_experiment
set RAW_SEQS=%EXPERIMENT%\raw.txt

REM Create directories
mkdir %EXPERIMENT%
mkdir %EXPERIMENT%\predictionsParameters
mkdir %EXPERIMENT%\esm_output

echo ========================================
echo Step 1: Extract ESM Embeddings
echo ========================================
call %USERPROFILE%\miniconda3\condabin\conda.bat activate esm_env
python models\esm_sequence_processor.py --input %RAW_SEQS% --output %EXPERIMENT%\esm_output --mode both

echo.
echo ========================================
echo Step 2: Generate Sequence Windows
echo ========================================
call %USERPROFILE%\miniconda3\condabin\conda.bat activate skl_legacy
python scripts\make_seqs_windows.py --in %RAW_SEQS% --out %EXPERIMENT%\predictionsParameters\seqs_1.txt --min-len 10 --max-len 35 --stride 1

echo.
echo ========================================
echo Step 3: Run SVM Predictions
echo ========================================
python scripts\run_sequence_svm.py --seqs %EXPERIMENT%\predictionsParameters\seqs_1.txt --aaindex descriptors\aaindex --output-dir %EXPERIMENT%\predictionsParameters --model-pkl predictionsParameters\svc.pkl --scaler-csv predictionsParameters\Z_score_mean_std__intersect_noflip.csv

echo.
echo ========================================
echo Pipeline Complete!
echo ========================================
echo Results:
echo   ESM embeddings: %EXPERIMENT%\esm_output\esm2_embeddings.csv
echo   Structures:     %EXPERIMENT%\esm_output\structures\
echo   SVM predictions: %EXPERIMENT%\predictionsParameters\descriptors_PREDICTIONS.csv
echo.

pause
```

---

## Performance Tips

### For Large Datasets:

1. **Batch processing ESM:**
   - Split large sequence files into chunks
   - Process each chunk separately
   - Combine results afterwards

2. **GPU Memory:**
   - ESM-2 650M: ~4GB GPU memory per sequence
   - ESMFold: ~6-8GB GPU memory (depends on sequence length)
   - Use smaller model if running out of memory

3. **Speed optimization:**
   - Use `esm2_t30_150M_UR50D` for faster processing
   - ESMFold is slower (~5-30 seconds per sequence)
   - ESM-2 embeddings are fast (~0.1-1 second per sequence)

### Recommended Workflow:

```
Raw Sequences
     ↓
ESM-2 Embeddings (fast, GPU) ← For ML features
     ↓
Combine with traditional descriptors
     ↓
SVM Predictions
     ↓
Top candidates
     ↓
ESMFold Structure Prediction (slow, GPU) ← Only for interesting hits
```

---

## Troubleshooting

**GPU out of memory:**
- Use smaller ESM model: `--esm-model esm2_t30_150M_UR50D`
- Process sequences one at a time
- Reduce max length for folding: `--max-length 200`

**ESMFold download:**
- First run downloads ~15GB
- Be patient, it's a one-time download
- Model cached in: `~/.cache/torch/hub/`

**CUDA errors:**
- Make sure PyTorch CUDA is installed: `pip list | grep torch`
- Should show `torch 2.9.1+cu128`, NOT `+cpu`
- Reinstall if needed: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128`

---

## Integration with Existing SVM Model

The SVM model expects specific descriptors. You have two options:

1. **Keep using original descriptors** (from `descripGen_12_py3.py`)
   - Continue using the existing pipeline
   - Use ESM features for separate analysis

2. **Train new SVM with ESM features**
   - Use ESM embeddings as input features
   - Requires retraining the SVM model
   - Potentially better performance with deep learning features

The current script supports option 1 (parallel workflows) and can be adapted for option 2.
