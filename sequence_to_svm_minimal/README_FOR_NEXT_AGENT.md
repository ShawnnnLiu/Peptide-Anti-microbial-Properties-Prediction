# README for Next Cursor Agent

## 📋 Context Summary

You're continuing work on extending an antimicrobial peptide prediction system. The user has:

1. **Existing working SVM** (484 training samples, predicts membrane activity σ)
2. **Goal**: Add ESMFold structure features + train NN to predict actual MIC
3. **GPU**: RTX 5070ti (12-16GB) - sufficient for ESMFold
4. **Current finding**: Data is NOT linearly separable (46.5% SV ratio)

---

## 🎯 The Plan (User-Approved)

```
Current: Sequence → 12 descriptors → SVM → σ (membrane proxy)

Target:  Sequence → [12 descriptors + SVM → σ] + [ESMFold → structure] → NN → MIC
```

**Why?** Paper shows σ correlates with membrane activity but NOT with MIC (therapeutic efficacy). Need second-stage model.

---

## 📂 Key Documents (Read These First)

1. **`ESMFOLD_INTEGRATION_PROJECT_PLAN.md`** ← MAIN DOCUMENT
   - Complete implementation plan with code templates
   - ~500 lines, very detailed
   
2. **`QUICK_START_GUIDE.md`** ← START HERE
   - Quick reference, checklists
   - Installation commands
   - Architecture diagram
   
3. **`CODEBASE_ANALYSIS_SUMMARY.md`**
   - Current codebase analysis
   - SVM training data findings
   - Non-linear separability proof
   
4. **`EXECUTIVE_SUMMARY.md`** & **`KEY_FINDINGS.txt`**
   - For paper comparison (secondary priority)

---

## 🚦 Current Status

### ✅ Completed:
- [x] Analyzed existing codebase
- [x] Confirmed: 484 training samples, 46.5% SV ratio
- [x] Confirmed: GPU sufficient for ESMFold
- [x] Designed architecture (two-stage: SVM→NN)
- [x] Created implementation plan with code templates

### ⏳ Next Steps:
- [ ] Install ESM: `pip install fair-esm biotite biopython`
- [ ] Test ESMFold on one peptide
- [ ] Download MIC dataset from DBAASP (https://dbaasp.org/)
- [ ] Implement `esmfold_features.py` (see project plan)
- [ ] Implement `feature_pipeline.py`
- [ ] Implement `mic_predictor.py`
- [ ] Train models

---

## 🎯 Immediate Actions (First 30 Minutes)

```bash
# 1. Install dependencies
pip install fair-esm biotite biopython

# 2. Test ESMFold
python << 'EOF'
import torch
import esm

print("Loading ESMFold...")
model = esm.pretrained.esmfold_v1()
model = model.eval()
if torch.cuda.is_available():
    model = model.cuda()
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

sequence = "KFLKKAKKFGKAFVKILKK"
print(f"Testing on: {sequence}")

with torch.no_grad():
    pdb_string = model.infer_pdb(sequence)

with open("test_structure.pdb", "w") as f:
    f.write(pdb_string)

print("✓ ESMFold works! Structure saved to test_structure.pdb")
EOF

# 3. Check if it worked
ls -lh test_structure.pdb
```

---

## 🔑 Critical Requirements

### BLOCKER: Need MIC Dataset

**User does NOT have MIC values yet!**

- Current data: 484 sequences with binary labels (antimicrobial yes/no)
- Needed: MIC values (μg/mL or μM) for regression

**Where to get:**
1. **DBAASP** (recommended): https://dbaasp.org/ - Download CSV with MIC
2. DRAMP: http://dramp.cpu-bioinfor.org/
3. APD3: https://aps.unmc.edu/

**Action**: Download DBAASP dataset as first priority

---

## 📊 Architecture Specs

### Feature Vector (24 dimensions):
- 12 traditional descriptors (netCharge, dipeptides, sequence-order)
- 2 SVM outputs (σ, p(+1))
- 9 structure features (pLDDT confidence, geometry, helix)
- 1 metadata (length)

### Stage-2 Model:
- Input: 24 features
- Hidden: [128, 64, 32] with BatchNorm + Dropout(0.3)
- Output: log(MIC)
- Loss: MSE
- Metrics: Spearman, RMSE, MAE

---

## 🗂️ Files to Create (In Order)

### Phase 1: Feature Extraction
1. `esmfold_features.py` - ESMFold + structure feature extraction + caching
2. `feature_pipeline.py` - Unified pipeline (Branch A + Branch B)

### Phase 2: Model
3. `mic_predictor.py` - Neural network regressor
4. `train_mic_model.py` - Training script with baselines

### Phase 3: Inference
5. `predict_mic.py` - CLI for predictions
6. `prepare_dataset.py` - MIC data preprocessing

**All templates are in `ESMFOLD_INTEGRATION_PROJECT_PLAN.md`** - copy and adapt!

---

## ⚙️ Technical Details

### GPU Memory Usage:
- ESMFold (18-30 AA): 4-8 GB ✅
- ESMFold (up to 400 AA): 12-16 GB ✅ (tight)
- NN training: <2 GB ✅
- **Total: Fits comfortably on 5070ti**

### Performance Expectations:
- ESMFold speed: ~5-10s per peptide (MUST cache!)
- Target Spearman: 0.7-0.8 (fusion model)
- Baseline (σ alone): 0.3-0.5

### Caching Strategy:
```python
# Cache key: MD5(sequence)
# Saves: PDB file + features JSON
# Location: ./esmfold_cache/
# Reuse across experiments
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: ESMFold OOM
```python
# Solution: Process one at a time for long sequences
# Or skip structure for sequences >100 AA
```

### Issue 2: No MIC Data
```python
# Solution: Download DBAASP
# Fallback: Use binary classification (less useful)
```

### Issue 3: Slow Training
```python
# Solution: Pre-compute all features, save to .npy, load in training
# Don't compute ESMFold in training loop!
```

---

## 🎨 Key Design Patterns

### 1. Fallback Mode
```python
if structure_prediction_fails:
    use_null_features()  # All zeros + structure_available=0
    # NN learns to ignore structure branch
```

### 2. Feature Caching
```python
seq_hash = md5(sequence)
if cache_exists(seq_hash):
    return cached_features
else:
    features = compute_features()
    cache_save(seq_hash, features)
    return features
```

### 3. Multi-Stage Training
```python
# Stage 1: Train baselines
model_sigma = train(sigma_only)
model_desc = train(descriptors_only)
model_struct = train(structure_only)

# Stage 2: Train fusion
model_fusion = train(all_features)

# Compare all
evaluate([model_sigma, model_desc, model_struct, model_fusion])
```

---

## 📝 User Preferences & Context

- User is technical, understands ML/biology
- Working on Windows with PowerShell
- Has Conda environment: `skl_legacy`
- Prefers modular code with clear separation
- Values caching and efficiency
- Wants comparison with paper results

---

## 🔍 What User Already Knows

- ✅ Current SVM has 484 training samples
- ✅ Support vector ratio is 46.5% (non-linear)
- ✅ σ ≠ MIC (need second model)
- ✅ ESMFold feasible on their GPU
- ✅ Need MIC-labeled data

---

## 🎯 Success Criteria

User will be happy when:
1. ESMFold runs and produces structures
2. Feature extraction pipeline works end-to-end
3. NN trains and outperforms σ-only baseline
4. Can predict MIC for new sequences
5. System has clear modular structure

---

## 🚨 Red Flags to Avoid

1. ❌ Don't clone ESM repo (use pip package: `fair-esm`)
2. ❌ Don't forget to log-transform MIC
3. ❌ Don't use random splits (need CD-HIT clustering)
4. ❌ Don't compute ESMFold without caching
5. ❌ Don't train on windows with wrong MIC labels

---

## 📖 If User Asks About...

### "How do I get MIC data?"
→ Point to DBAASP: https://dbaasp.org/ → Download section

### "Is my GPU enough?"
→ Yes! 5070ti (12-16GB) is sufficient for 18-30 AA peptides

### "Why two stages?"
→ Stage 1 (SVM) fast membrane screening, Stage 2 (NN) accurate MIC prediction

### "Should I clone ESM repo?"
→ No, use pip: `pip install fair-esm`

### "Why is σ not enough?"
→ Paper shows σ correlates with membrane activity, NOT MIC

---

## 🎓 Quick Reference Commands

```bash
# Install
pip install fair-esm biotite biopython

# Test ESMFold
python test_esmfold.py

# Train model
python train_mic_model.py --data data/dbaasp.csv --use-structure

# Predict
python predict_mic.py --sequence "KFLK..." --model models/best_model.pt
```

---

## 📁 Project Structure

```
sequence_to_svm_minimal/
├── descriptors/              # Existing (12 descriptors)
│   ├── descripGen_12_py3.py
│   └── aaindex/
├── predictionsParameters/    # Existing (SVM model)
│   ├── svc.pkl
│   └── Z_score_*.csv
├── scripts/                  # Existing (pipeline)
│   └── run_sequence_svm.py
├── esmfold_features.py      # NEW - Structure extraction
├── feature_pipeline.py      # NEW - Unified features
├── mic_predictor.py         # NEW - NN model
├── train_mic_model.py       # NEW - Training
├── predict_mic.py           # NEW - Inference
└── esmfold_cache/           # NEW - Structure cache
```

---

## 🎬 Start Here

1. Read `QUICK_START_GUIDE.md` (5 min)
2. Skim `ESMFOLD_INTEGRATION_PROJECT_PLAN.md` (15 min)
3. Run ESMFold test (5 min)
4. Help user download DBAASP data (10 min)
5. Start implementing `esmfold_features.py` (use template!)

---

**Good luck! The plan is solid, the user is prepared, and the GPU is ready. You got this! 🚀**

---

*Last updated: January 2026*  
*Previous agent: Completed codebase analysis + designed ESMFold integration*  
*Next agent: Implement feature extraction + model training*


