# Quick Start Guide - ESMFold Integration Project

## 🎯 Project Goal in One Sentence
Add ESMFold structural features to existing SVM system and train a neural network to predict actual MIC (antimicrobial potency) instead of just membrane activity.

---

## 📋 What You Need

### ✅ What You Have:
- Working SVM (484 samples, 46.5% SV ratio)
- 12 biochemical descriptors
- RTX 5070ti GPU (sufficient!)

### ❌ What You Need:
- **MIC-labeled dataset** → Download from DBAASP: https://dbaasp.org/
- ESM library → `pip install fair-esm biotite`

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Install ESM
pip install fair-esm biotite biopython

# 2. Test ESMFold
cat > test_esmfold.py << 'EOF'
import torch
import esm

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

sequence = "KFLKKAKKFGKAFVKILKK"
with torch.no_grad():
    output = model.infer_pdb(sequence)

with open("test.pdb", "w") as f:
    f.write(output)
print("✓ ESMFold works! Check test.pdb")
EOF

python test_esmfold.py

# 3. Download MIC data
# Go to https://dbaasp.org/ → Download → Save as data/dbaasp_mic_data.csv
```

---

## 🏗️ Architecture Overview

```
INPUT: Peptide Sequence (18-30 AA)
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
    v                     v                     v
Branch A:           Branch B:           Metadata:
Traditional         ESMFold             Length
Descriptors (12)    Structure (9)       (1)
    │                     │                     │
    v                     v                     v
SVM → σ, p(+1)     Geometry, Helix,    
    (2)                Confidence           
    │                     │                     │
    └─────────────────────┴─────────────────────┘
                          │
                          v
              Concatenate (~24 features)
                          │
                          v
            Neural Network Regressor
                          │
                          v
                   log(MIC) → MIC
```

---

## 📊 Feature Vector (24 dimensions)

### Traditional (12):
1-8. netCharge, FC, LW, DP, NK, AE, pcMK, SolventAccessibility  
9-12. tau2, tau4, QSO50, QSO29

### SVM Outputs (2):
13. σ (signed margin)  
14. p(+1) (antimicrobial probability)

### Structure (9):
15-17. pLDDT (mean, min, std)  
18-21. Geometry (radius_gyration, end_to_end, compactness, asphericity)  
22. helix_fraction  
23. structure_available (flag)

### Metadata (1):
24. sequence_length

---

## 📂 Files to Create

### Priority 1 (Core):
1. `esmfold_features.py` - Structure feature extractor with caching
2. `feature_pipeline.py` - Unified feature extraction
3. `mic_predictor.py` - Neural network model
4. `train_mic_model.py` - Training script
5. `predict_mic.py` - Inference CLI

### Priority 2 (Data):
6. `prepare_dataset.py` - MIC data preprocessing
7. `cluster_sequences.py` - CD-HIT for similarity-aware splits

---

## 🎯 Implementation Checklist

### Day 1: Setup
- [ ] Install fair-esm, biotite, biopython
- [ ] Test ESMFold on one sequence
- [ ] Download DBAASP MIC dataset
- [ ] Verify GPU works with ESMFold

### Day 2-3: Feature Extraction
- [ ] Create esmfold_features.py
- [ ] Create feature_pipeline.py
- [ ] Test on 10 sequences
- [ ] Verify caching works

### Day 4-5: Model & Training
- [ ] Create mic_predictor.py
- [ ] Prepare MIC dataset (clean, log-transform, split)
- [ ] Create train_mic_model.py
- [ ] Train baseline models (σ only, descriptors only, structure only)

### Day 6-7: Fusion & Evaluation
- [ ] Train fusion model (all features)
- [ ] Compare baselines vs fusion
- [ ] Evaluate: Spearman, RMSE, MAE
- [ ] Generate plots (predicted vs actual)

### Day 8: Inference
- [ ] Create predict_mic.py CLI
- [ ] Test on exp1/exp2 sequences
- [ ] Document usage

---

## 💡 Key Design Decisions

### Why ESMFold?
- Predicts 3D structure from sequence alone (no templates)
- Includes confidence scores (pLDDT)
- Works well on short peptides (18-30 AA)

### Why Two-Stage?
- Stage 1 (SVM): Fast membrane activity screening
- Stage 2 (NN): Accurate MIC prediction
- Can use Stage 1 alone if Stage 2 unavailable

### Why Caching?
- ESMFold is slow (~5-10s per peptide)
- Cache by sequence hash to avoid recomputation
- Saves hours on large datasets

### Why Log(MIC)?
- MIC spans orders of magnitude (0.1 to 1000 μg/mL)
- Log scale normalizes range
- Better for regression

---

## ⚠️ Common Pitfalls

1. **Forgetting to normalize MIC** → Always use log10(MIC)
2. **Data leakage** → Use CD-HIT clustering for splits
3. **Not caching ESMFold** → Will be painfully slow
4. **Training on windows with wrong labels** → Only use MIC if peptide matches exactly
5. **Ignoring structure failures** → Always have fallback mode

---

## 📈 Expected Performance

| Model | Spearman | RMSE |
|-------|----------|------|
| σ alone | 0.3-0.5 | 1.0-1.2 |
| Descriptors only | 0.4-0.6 | 0.9-1.1 |
| Structure only | 0.5-0.7 | 0.8-1.0 |
| **Fusion (target)** | **0.7-0.8** | **0.6-0.8** |

---

## 🔧 Training Command

```bash
python train_mic_model.py \
    --data data/dbaasp_mic_data.csv \
    --use-structure \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-3 \
    --hidden-dims 128 64 32 \
    --output models/mic_predictor
```

---

## 🔮 Inference Command

```bash
# Single sequence
python predict_mic.py \
    --sequence "KFLKKAKKFGKAFVKILKK" \
    --model models/mic_predictor/best_model.pt \
    --output result.csv

# Batch
python predict_mic.py \
    --input sequences.txt \
    --model models/mic_predictor/best_model.pt \
    --output predictions.csv
```

---

## 🐛 Troubleshooting

### ESMFold Out of Memory
```python
# Reduce batch size or process one at a time
# For very long sequences (>100 AA), consider skipping structure
```

### Structure Prediction Fails
```python
# System automatically uses fallback with null features
# structure_available flag = 0 tells NN to ignore structure branch
```

### Slow Training
```python
# Pre-compute all features first:
features = feature_extractor.batch_extract(sequences, include_structure=True)
np.save('features_cache.npy', features)

# Then load in training:
train_dataset = MICDataset(seqs, mics, feature_extractor, 
                           cache_features=np.load('features_cache.npy'))
```

---

## 📚 Full Documentation

See `ESMFOLD_INTEGRATION_PROJECT_PLAN.md` for:
- Complete code templates
- Detailed scientific rationale
- Data preprocessing pipeline
- Model architecture details
- Evaluation metrics
- References

---

## ✅ Success Criteria

You'll know it's working when:
1. ✅ ESMFold predicts structures with pLDDT > 70
2. ✅ Feature extraction completes without errors
3. ✅ Training loss decreases steadily
4. ✅ Validation Spearman > 0.7
5. ✅ Predictions correlate with known MIC values
6. ✅ Fusion model outperforms all baselines

---

## 🚨 Critical Path

**BLOCKER**: Need MIC-labeled dataset  
**SOLUTION**: Download from DBAASP (https://dbaasp.org/)

**If DBAASP unavailable**:
- DRAMP: http://dramp.cpu-bioinfor.org/
- APD3: https://aps.unmc.edu/
- Literature mining (slow)

---

## 🎓 Learning Resources

- ESMFold paper: https://www.biorxiv.org/content/10.1101/2022.07.20.500902
- ESM GitHub: https://github.com/facebookresearch/esm
- DBAASP: https://dbaasp.org/
- Your current analysis: `CODEBASE_ANALYSIS_SUMMARY.md`

---

**Next Step**: Install ESM and test → `pip install fair-esm && python test_esmfold.py`

Good luck! 🚀


