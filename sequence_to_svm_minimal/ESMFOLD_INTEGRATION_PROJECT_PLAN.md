# ESMFold Integration Project Plan
## Extending SVM Antimicrobial Predictor with Structural Features & MIC Prediction

**Created**: January 2026  
**Status**: Ready to implement  
**For**: Next Cursor agent session

---

## 🎯 PROJECT GOAL

**Extend the current 12-descriptor SVM system to predict actual therapeutic efficacy (MIC) instead of just membrane activity proxy (σ).**

### Current System:
```
Sequence → 12 Descriptors → SVM → σ (membrane activity proxy)
```

### Target System:
```
                    ┌─ Branch A: 12 Descriptors → SVM → σ, p(+1) ────┐
Sequence ──────────┤                                                  ├→ Stage-2 NN → MIC Prediction
                    └─ Branch B: ESMFold → Structure Features ────────┘
```

---

## 📊 CURRENT CODEBASE STATE

### What We Have:
- ✅ Working SVM trained on **484 peptide sequences**
- ✅ 12 biochemical descriptor extractor (`descripGen_12_py3.py`)
- ✅ Pre-trained model: `svc.pkl` (225 support vectors, 46.5% ratio)
- ✅ Inference pipeline for sliding window analysis
- ✅ GPU: RTX 5070ti (12-16GB VRAM) - **sufficient for ESMFold**

### Key Finding:
- **Data is NOT linearly separable** (46.5% SV ratio indicates complex boundary)
- σ (SVM margin) correlates with membrane activity but NOT with MIC
- Need second-stage model to predict actual therapeutic efficacy

### File Structure:
```
sequence_to_svm_minimal/
├── descriptors/
│   ├── descripGen_12_py3.py          # 12 feature extractor
│   └── aaindex/                       # Amino acid property database
├── predictionsParameters/
│   ├── svc.pkl + svc.pkl_*.npy       # Pre-trained SVM
│   ├── predictSVC.py                  # Inference script
│   └── Z_score_mean_std_*.csv        # Normalization params
├── scripts/
│   ├── make_seqs_windows.py          # Sliding window generator
│   └── run_sequence_svm.py           # End-to-end pipeline
└── experiments/                       # Test runs (exp1, exp2)
```

---

## 🔬 SCIENTIFIC RATIONALE

### Why This Approach?

**Problem**: Current SVM predicts σ (membrane curvature proxy) which:
- ✅ Correlates with membrane interaction
- ❌ Does NOT correlate strongly with MIC (actual antimicrobial potency)

**Solution**: Two-stage architecture
1. **Stage 1 (Existing SVM)**: Predict σ as membrane activity feature
2. **Stage 2 (New NN)**: Predict MIC from (σ + descriptors + structure)

**Why Add Structure?**
- 3D geometry affects membrane insertion
- Helix formation, amphipathicity, compactness matter
- ESMFold provides these features from sequence alone

---

## 🏗️ PROPOSED ARCHITECTURE

### Stage 1: Feature Extraction (Parallel Branches)

#### Branch A: Traditional Features + SVM
```python
Input: Peptide sequence (18-30 AA recommended)
    ↓
Compute 12 descriptors:
    1. netCharge
    2-6. Dipeptide compositions (FC, LW, DP, NK, AE)
    7. pcMK (M/K ratio)
    8. Solvent accessibility
    9-12. Sequence-order features (tau2, tau4, QSO50, QSO29)
    ↓
Z-score normalize
    ↓
SVM predict:
    - σ (signed margin distance)
    - p(+1) (probability of antimicrobial class)
    - predicted class
```

#### Branch B: Structural Features (NEW)
```python
Input: Same peptide sequence
    ↓
ESMFold prediction → PDB structure + pLDDT confidence
    ↓
Extract features:
    - Confidence: pLDDT mean, min, std
    - Geometry: radius of gyration, end-to-end distance, compactness
    - Secondary structure: helix fraction, helix segments
    - Charge/hydrophobic mapping: polar surface area, hydrophobic moment
    - Contact density: inter-residue contacts
    ↓
Output: Structure feature vector (~8-15 features)
```

### Stage 2: MIC Prediction

```python
Feature Fusion:
    x_all = concat([
        12 descriptors,           # Traditional biochemical
        σ, p(+1),                # SVM outputs
        structure_features,       # ESMFold-derived
        sequence_length          # Metadata
    ])
    # Total: ~25-30 features
    
    ↓
    
Neural Network Regressor:
    Input: 25-30 features
    Hidden: [128, 64, 32]
    Output: log(MIC) in μg/mL or μM
    
    Optional: Multi-task heads for different organisms
```

---

## 💾 DATA REQUIREMENTS

### Critical: Need MIC-Labeled Dataset

**Current situation:**
- ✅ Have: 484 sequences with binary labels (antimicrobial yes/no)
- ❌ Missing: MIC values (Minimum Inhibitory Concentration)

### Where to Get MIC Data:

#### **RECOMMENDED: DBAASP**
- **URL**: https://dbaasp.org/
- **Size**: ~20,000 peptides with activity data
- **Contains**: Sequence, MIC, organism, assay conditions
- **Quality**: Curated from peer-reviewed literature
- **Format**: CSV download available
- **Action**: Download dataset with MIC annotations

#### Alternative Sources:
1. **DRAMP**: http://dramp.cpu-bioinfor.org/ (~5,000 peptides)
2. **APD3**: https://aps.unmc.edu/ (~3,000 peptides)
3. **DAMPD**: Database of Antimicrobial Peptides

### Data Preparation Checklist:
```python
Required columns:
- sequence: Amino acid sequence (string)
- MIC: Minimum Inhibitory Concentration (numeric)
- MIC_unit: μg/mL or μM
- organism: Target organism (E. coli, S. aureus, etc.)
- assay_type: Broth dilution, agar diffusion, etc.

Preprocessing:
1. Filter sequences: 5-50 amino acids (ESMFold stable range)
2. Remove sequences with missing MIC
3. Handle censored values (">", "<"):
   - Initial: discard
   - Advanced: interval regression
4. Convert MIC to log scale: log10(MIC)
5. Normalize by organism if needed
6. Similarity-aware splitting (CD-HIT at 0.8 threshold)
```

---

## 🛠️ IMPLEMENTATION PLAN

### Phase 1: Setup & Dependencies (Day 1)

#### Install ESM
```bash
# Primary method (RECOMMENDED)
pip install fair-esm
pip install biotite           # For structure parsing
pip install biopython         # For PDB handling
pip install torch torchvision # If not already installed

# Verify installation
python -c "import esm; print('✓ ESM installed')"
```

#### Test ESMFold
```bash
# Create test_esmfold.py
python test_esmfold.py
```

**Test Script**: `test_esmfold.py`
```python
#!/usr/bin/env python3
"""Test ESMFold installation and basic functionality"""
import torch
import esm

print("Loading ESMFold...")
model = esm.pretrained.esmfold_v1()
model = model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ Using CPU (will be slow)")

# Test on antimicrobial peptide
sequence = "KFLKKAKKFGKAFVKILKK"  # Cathelicidin-derived AMP
print(f"Testing on sequence: {sequence} ({len(sequence)} AA)")

with torch.no_grad():
    output = model.infer_pdb(sequence)

# Save structure
with open("test_structure.pdb", "w") as f:
    f.write(output)

print("✓ Structure prediction successful!")
print("✓ Output saved: test_structure.pdb")
print("\nNext: Visualize with PyMOL or ChimeraX")
```

#### Download MIC Dataset
```bash
# Manual download from DBAASP:
# 1. Go to https://dbaasp.org/
# 2. Navigate to "Download" section
# 3. Download full database or filtered dataset
# 4. Save as: data/dbaasp_mic_data.csv

# Or use wget if direct link available
mkdir -p data
# wget [DBAASP_CSV_URL] -O data/dbaasp_mic_data.csv
```

---

### Phase 2: Feature Extraction Module (Days 2-3)

Create `sequence_to_svm_minimal/esmfold_features.py`:

```python
#!/usr/bin/env python3
"""
ESMFold-based structural feature extraction with caching
"""
import torch
import esm
import numpy as np
from pathlib import Path
import hashlib
import json
import warnings
from biotite.structure.io import pdb
from biotite.structure import get_chains, apply_residue_wise

class ESMFoldFeatureExtractor:
    """Extract structural features from ESMFold predictions"""
    
    def __init__(self, cache_dir="./esmfold_cache", device="auto"):
        """
        Args:
            cache_dir: Directory to cache structure predictions
            device: 'cuda', 'cpu', or 'auto'
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Lazy loading - only load when needed
        self._model = None
        self._device = self._setup_device(device)
        
    def _setup_device(self, device):
        """Determine compute device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Lazy load ESMFold model"""
        if self._model is None:
            print("Loading ESMFold model (this takes ~1 minute)...")
            self._model = esm.pretrained.esmfold_v1()
            self._model = self._model.eval().to(self._device)
            print(f"✓ ESMFold loaded on {self._device}")
        return self._model
    
    def _seq_hash(self, sequence):
        """Generate cache key for sequence"""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def predict_structure(self, sequence, force_recompute=False):
        """
        Predict structure with caching
        
        Args:
            sequence: Amino acid sequence
            force_recompute: Ignore cache and recompute
            
        Returns:
            dict with structure info and confidence scores
        """
        seq_hash = self._seq_hash(sequence)
        cache_file = self.cache_dir / f"{seq_hash}.json"
        pdb_file = self.cache_dir / f"{seq_hash}.pdb"
        
        # Check cache
        if not force_recompute and cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                cached['from_cache'] = True
                return cached
        
        # Run ESMFold
        try:
            model = self._load_model()
            
            # Validate sequence
            if len(sequence) < 5:
                raise ValueError("Sequence too short (min 5 AA)")
            if len(sequence) > 400:
                warnings.warn(f"Long sequence ({len(sequence)} AA) may be slow/fail")
            
            with torch.no_grad():
                pdb_string = model.infer_pdb(sequence)
            
            # Save PDB
            with open(pdb_file, 'w') as f:
                f.write(pdb_string)
            
            # Extract pLDDT from PDB (stored in B-factor column)
            plddt_scores = self._extract_plddt(pdb_file)
            
            result = {
                'sequence': sequence,
                'length': len(sequence),
                'pdb_path': str(pdb_file),
                'success': True,
                'plddt_mean': float(np.mean(plddt_scores)),
                'plddt_min': float(np.min(plddt_scores)),
                'plddt_std': float(np.std(plddt_scores)),
                'from_cache': False,
                'error': None
            }
            
            # Cache result
            with open(cache_file, 'w') as f:
                cache_data = result.copy()
                cache_data['pdb_path'] = str(pdb_file)
                json.dump(cache_data, f, indent=2)
            
            return result
            
        except Exception as e:
            return {
                'sequence': sequence,
                'length': len(sequence),
                'success': False,
                'error': str(e),
                'from_cache': False
            }
    
    def _extract_plddt(self, pdb_file):
        """Extract pLDDT scores from PDB B-factor column"""
        try:
            structure = pdb.PDBFile.read(str(pdb_file)).get_structure()[0]
            return structure.b_factor
        except Exception as e:
            warnings.warn(f"Could not extract pLDDT: {e}")
            return np.array([0.0])
    
    def extract_geometric_features(self, pdb_path):
        """Extract geometric features from structure"""
        try:
            structure = pdb.PDBFile.read(str(pdb_path)).get_structure()[0]
            coords = structure.coord
            
            features = {}
            
            # Radius of gyration
            center = np.mean(coords, axis=0)
            rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=1)))
            features['radius_gyration'] = float(rg)
            
            # End-to-end distance
            end_to_end = np.linalg.norm(coords[0] - coords[-1])
            features['end_to_end'] = float(end_to_end)
            
            # Compactness (end-to-end / length)
            features['compactness'] = float(end_to_end / len(coords))
            
            # Asphericity (measure of deviation from sphere)
            features['asphericity'] = self._calc_asphericity(coords)
            
            return features
            
        except Exception as e:
            warnings.warn(f"Geometric feature extraction failed: {e}")
            return self._null_geometric_features()
    
    def _calc_asphericity(self, coords):
        """Calculate asphericity (shape parameter)"""
        try:
            center = np.mean(coords, axis=0)
            centered = coords - center
            
            # Gyration tensor
            gyration_tensor = np.dot(centered.T, centered) / len(coords)
            eigenvalues = np.linalg.eigvalsh(gyration_tensor)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Asphericity
            lambda1, lambda2, lambda3 = eigenvalues
            asphericity = lambda1 - 0.5 * (lambda2 + lambda3)
            
            return float(asphericity)
        except:
            return 0.0
    
    def extract_secondary_structure(self, pdb_path):
        """Estimate secondary structure content"""
        try:
            structure = pdb.PDBFile.read(str(pdb_path)).get_structure()[0]
            coords = structure.coord
            
            # Simple helix detector based on C-alpha distances
            # Real implementation would use DSSP
            helix_count = 0
            total_residues = len(coords)
            
            for i in range(total_residues - 4):
                # Check if residues i to i+4 form helix-like pattern
                # Alpha helix: C-alpha(i) to C-alpha(i+4) ~5.4 Å
                dist = np.linalg.norm(coords[i] - coords[i+4])
                if 4.5 < dist < 6.5:  # Helix-like
                    helix_count += 1
            
            helix_fraction = helix_count / max(total_residues - 4, 1)
            
            return {
                'helix_fraction': float(helix_fraction),
                'n_residues': int(total_residues)
            }
            
        except Exception as e:
            warnings.warn(f"Secondary structure estimation failed: {e}")
            return {'helix_fraction': 0.0, 'n_residues': 0}
    
    def extract_all_features(self, sequence):
        """
        Complete feature extraction pipeline
        
        Returns:
            dict with all structural features + confidence
        """
        # Predict structure
        structure_result = self.predict_structure(sequence)
        
        if not structure_result['success']:
            # Return null features with error flag
            return self._null_all_features(structure_result['error'])
        
        pdb_path = structure_result['pdb_path']
        
        # Extract feature groups
        geometric = self.extract_geometric_features(pdb_path)
        secondary = self.extract_secondary_structure(pdb_path)
        
        # Combine all features
        features = {
            # Confidence
            'plddt_mean': structure_result['plddt_mean'],
            'plddt_min': structure_result['plddt_min'],
            'plddt_std': structure_result['plddt_std'],
            
            # Geometric
            'radius_gyration': geometric['radius_gyration'],
            'end_to_end': geometric['end_to_end'],
            'compactness': geometric['compactness'],
            'asphericity': geometric['asphericity'],
            
            # Secondary structure
            'helix_fraction': secondary['helix_fraction'],
            
            # Metadata
            'structure_success': 1,
            'sequence_length': len(sequence)
        }
        
        return features
    
    def _null_geometric_features(self):
        """Return when geometric extraction fails"""
        return {
            'radius_gyration': 0.0,
            'end_to_end': 0.0,
            'compactness': 0.0,
            'asphericity': 0.0
        }
    
    def _null_all_features(self, error_msg):
        """Return when structure prediction fails"""
        return {
            'plddt_mean': 0.0,
            'plddt_min': 0.0,
            'plddt_std': 0.0,
            'radius_gyration': 0.0,
            'end_to_end': 0.0,
            'compactness': 0.0,
            'asphericity': 0.0,
            'helix_fraction': 0.0,
            'structure_success': 0,  # Flag: structure unavailable
            'sequence_length': 0,
            'error': error_msg
        }
```

---

### Phase 3: Unified Feature Extractor (Day 4)

Create `sequence_to_svm_minimal/feature_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Unified feature extraction combining traditional + structural features
"""
import numpy as np
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from descriptors.descripGen_12_py3 import descripGen_bespoke
from propy import AAIndex
from esmfold_features import ESMFoldFeatureExtractor
import joblib

class UnifiedFeatureExtractor:
    """Combines Branch A (traditional+SVM) and Branch B (structure)"""
    
    def __init__(self, 
                 aaindex_path="descriptors/aaindex",
                 svm_path="predictionsParameters/svc.pkl",
                 scaler_path="predictionsParameters/Z_score_mean_std__intersect_noflip.csv",
                 cache_dir="./esmfold_cache"):
        """
        Initialize all extractors
        
        Args:
            aaindex_path: Path to AAIndex database
            svm_path: Path to trained SVM model
            scaler_path: Path to Z-score normalization parameters
            cache_dir: ESMFold cache directory
        """
        # Load AAIndex for traditional descriptors
        print("Loading AAIndex database...")
        self.aap_dict = AAIndex.GetAAIndex23('GRAR740104', path=aaindex_path)
        
        # Load SVM model
        print("Loading SVM model...")
        self.svm = joblib.load(svm_path)
        
        # Load scaler
        print("Loading normalization parameters...")
        self.scaler = self._load_scaler(scaler_path)
        
        # Initialize ESMFold extractor (lazy loaded)
        print("Initializing ESMFold extractor...")
        self.esmfold_extractor = ESMFoldFeatureExtractor(cache_dir=cache_dir)
        
        print("✓ All extractors initialized")
    
    def _load_scaler(self, scaler_path):
        """Load Z-score normalization parameters"""
        import pandas as pd
        df = pd.read_csv(scaler_path)
        return {
            'mean': df.iloc[0].values,
            'std': df.iloc[1].values
        }
    
    def extract_traditional_features(self, sequence):
        """
        Branch A: Compute 12 traditional descriptors
        
        Returns:
            numpy array of 12 descriptors
        """
        desc_names, desc_values = descripGen_bespoke(sequence, self.aap_dict)
        return desc_values
    
    def compute_svm_scores(self, descriptors):
        """
        Run SVM on normalized descriptors
        
        Args:
            descriptors: 12 raw descriptor values
            
        Returns:
            dict with sigma, probabilities, class
        """
        # Normalize
        descriptors_norm = (descriptors - self.scaler['mean']) / self.scaler['std']
        descriptors_norm = descriptors_norm.reshape(1, -1)
        
        # SVM prediction
        sigma = self.svm.decision_function(descriptors_norm)[0]
        probs = self.svm.predict_proba(descriptors_norm)[0]
        pred_class = self.svm.predict(descriptors_norm)[0]
        
        # Map probabilities to classes correctly
        neg_idx = np.where(self.svm.classes_ == -1)[0][0]
        pos_idx = np.where(self.svm.classes_ == 1)[0][0]
        
        return {
            'sigma': float(sigma),
            'prob_neg': float(probs[neg_idx]),
            'prob_pos': float(probs[pos_idx]),
            'predicted_class': int(pred_class)
        }
    
    def extract_all_features(self, sequence, include_structure=True):
        """
        Complete unified feature extraction
        
        Args:
            sequence: Amino acid sequence
            include_structure: Whether to compute structure features (slow)
            
        Returns:
            dict with feature_vector and all intermediate results
        """
        # Branch A: Traditional descriptors + SVM
        descriptors = self.extract_traditional_features(sequence)
        svm_output = self.compute_svm_scores(descriptors)
        
        # Branch B: Structure features (optional, can be slow)
        if include_structure:
            structure_features = self.esmfold_extractor.extract_all_features(sequence)
        else:
            structure_features = self.esmfold_extractor._null_all_features("skipped")
        
        # Build feature vector
        feature_vector = self._build_feature_vector(
            descriptors, svm_output, structure_features, sequence
        )
        
        return {
            'feature_vector': feature_vector,
            'feature_names': self.get_feature_names(),
            'descriptors': descriptors,
            'svm': svm_output,
            'structure': structure_features,
            'sequence_length': len(sequence)
        }
    
    def _build_feature_vector(self, descriptors, svm_output, structure_features, sequence):
        """Concatenate all features into single vector"""
        feature_list = [
            # Traditional descriptors (12)
            *descriptors,
            
            # SVM outputs (2)
            svm_output['sigma'],
            svm_output['prob_pos'],
            
            # Structure features (9)
            structure_features['plddt_mean'],
            structure_features['plddt_min'],
            structure_features['plddt_std'],
            structure_features['radius_gyration'],
            structure_features['end_to_end'],
            structure_features['compactness'],
            structure_features['asphericity'],
            structure_features['helix_fraction'],
            structure_features['structure_success'],
            
            # Metadata (1)
            len(sequence)
        ]
        
        return np.array(feature_list, dtype=np.float32)
    
    def get_feature_names(self):
        """Return list of feature names in order"""
        return [
            # Traditional (12)
            'netCharge', 'FC', 'LW', 'DP', 'NK', 'AE', 'pcMK',
            '_SolventAccessibilityD1025',
            'tau2_GRAR740104', 'tau4_GRAR740104',
            'QSO50_GRAR740104', 'QSO29_GRAR740104',
            
            # SVM (2)
            'sigma', 'prob_antimicrobial',
            
            # Structure (9)
            'plddt_mean', 'plddt_min', 'plddt_std',
            'radius_gyration', 'end_to_end', 'compactness',
            'asphericity', 'helix_fraction', 'structure_available',
            
            # Metadata (1)
            'sequence_length'
        ]
    
    def batch_extract(self, sequences, include_structure=True, verbose=True):
        """
        Extract features for multiple sequences
        
        Args:
            sequences: List of amino acid sequences
            include_structure: Whether to compute structure (slow)
            verbose: Print progress
            
        Returns:
            numpy array of shape (n_sequences, n_features)
        """
        features_list = []
        
        for i, seq in enumerate(sequences):
            if verbose and i % 10 == 0:
                print(f"Processing sequence {i+1}/{len(sequences)}")
            
            try:
                result = self.extract_all_features(seq, include_structure=include_structure)
                features_list.append(result['feature_vector'])
            except Exception as e:
                print(f"Error on sequence {i}: {e}")
                # Append null features
                features_list.append(np.zeros(24, dtype=np.float32))
        
        return np.array(features_list)
```

---

### Phase 4: MIC Prediction Model (Day 5)

Create `sequence_to_svm_minimal/mic_predictor.py`:

```python
#!/usr/bin/env python3
"""
Stage-2 Neural Network for MIC prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MICPredictor(nn.Module):
    """Neural network regressor for log(MIC) prediction"""
    
    def __init__(self, n_features=24, hidden_dims=[128, 64, 32], dropout=0.3):
        """
        Args:
            n_features: Number of input features (24 for unified pipeline)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_features = n_features
        
        # Build network layers
        layers = []
        prev_dim = n_features
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer (log MIC prediction)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, n_features)
            
        Returns:
            log(MIC) predictions (batch_size,)
        """
        return self.network(x).squeeze(-1)


class MultiOrganismMICPredictor(nn.Module):
    """Multi-task MIC predictor for different organisms"""
    
    def __init__(self, n_features=24, n_organisms=4, 
                 shared_dims=[128, 64], head_dim=32, dropout=0.3):
        """
        Args:
            n_features: Number of input features
            n_organisms: Number of target organisms
            shared_dims: Shared feature extractor dimensions
            head_dim: Organism-specific head dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Shared feature extractor
        shared_layers = []
        prev_dim = n_features
        
        for hidden_dim in shared_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Organism-specific heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1)
            )
            for _ in range(n_organisms)
        ])
        
    def forward(self, x, organism_idx):
        """
        Forward pass with organism selection
        
        Args:
            x: Input features (batch_size, n_features)
            organism_idx: Organism indices (batch_size,) in range [0, n_organisms)
            
        Returns:
            log(MIC) predictions (batch_size,)
        """
        shared_features = self.shared(x)
        
        # Select appropriate head for each sample
        predictions = []
        for i, org_idx in enumerate(organism_idx):
            pred = self.heads[org_idx](shared_features[i:i+1])
            predictions.append(pred)
        
        return torch.cat(predictions).squeeze(-1)


class EnsembleMICPredictor:
    """Ensemble of multiple MIC predictors for uncertainty estimation"""
    
    def __init__(self, models):
        """
        Args:
            models: List of trained MICPredictor models
        """
        self.models = models
        
    def predict(self, x, return_std=True):
        """
        Ensemble prediction with uncertainty
        
        Args:
            x: Input features
            return_std: Whether to return standard deviation
            
        Returns:
            mean prediction, optional std
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        
        if return_std:
            std = predictions.std(dim=0)
            return mean, std
        
        return mean
```

---

### Phase 5: Training Script (Days 6-7)

Create `sequence_to_svm_minimal/train_mic_model.py`:

```python
#!/usr/bin/env python3
"""
Training script for MIC prediction model
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import argparse
from pathlib import Path

from feature_pipeline import UnifiedFeatureExtractor
from mic_predictor import MICPredictor

class MICDataset(Dataset):
    """Dataset for MIC prediction"""
    
    def __init__(self, sequences, mic_values, feature_extractor, 
                 use_structure=True, cache_features=None):
        """
        Args:
            sequences: List of amino acid sequences
            mic_values: List of log(MIC) values
            feature_extractor: UnifiedFeatureExtractor instance
            use_structure: Whether to compute structure features
            cache_features: Pre-computed features (optional)
        """
        self.sequences = sequences
        self.mic_values = np.array(mic_values, dtype=np.float32)
        self.feature_extractor = feature_extractor
        self.use_structure = use_structure
        
        # Pre-compute or load features
        if cache_features is not None:
            self.features = cache_features
        else:
            print(f"Computing features for {len(sequences)} sequences...")
            self.features = feature_extractor.batch_extract(
                sequences, 
                include_structure=use_structure,
                verbose=True
            )
            print("✓ Feature extraction complete")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        mic = torch.FloatTensor([self.mic_values[idx]])
        return features, mic


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device).squeeze()
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for features, target in dataloader:
            features = features.to(device)
            pred = model(features)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    spearman, _ = spearmanr(predictions, targets)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'spearman': spearman,
        'predictions': predictions,
        'targets': targets
    }


def main(args):
    """Main training function"""
    
    # Load dataset
    print(f"Loading dataset from {args.data}")
    df = pd.read_csv(args.data)
    
    # Prepare data
    sequences = df['sequence'].tolist()
    mic_values = df['log_MIC'].tolist()
    
    print(f"Dataset: {len(sequences)} sequences")
    print(f"MIC range: {np.min(mic_values):.2f} to {np.max(mic_values):.2f}")
    
    # Split data (should use CD-HIT clustering in production)
    train_seqs, test_seqs, train_mic, test_mic = train_test_split(
        sequences, mic_values, test_size=0.2, random_state=42
    )
    
    train_seqs, val_seqs, train_mic, val_mic = train_test_split(
        train_seqs, train_mic, test_size=0.15, random_state=42
    )
    
    print(f"Split: {len(train_seqs)} train, {len(val_seqs)} val, {len(test_seqs)} test")
    
    # Initialize feature extractor
    feature_extractor = UnifiedFeatureExtractor(
        aaindex_path=args.aaindex_path,
        svm_path=args.svm_path,
        scaler_path=args.scaler_path,
        cache_dir=args.cache_dir
    )
    
    # Create datasets
    train_dataset = MICDataset(train_seqs, train_mic, feature_extractor, 
                                use_structure=args.use_structure)
    val_dataset = MICDataset(val_seqs, val_mic, feature_extractor,
                              use_structure=args.use_structure,
                              cache_features=None)  # Will compute
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                               shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    n_features = train_dataset.features.shape[1]
    model = MICPredictor(n_features=n_features,
                         hidden_dims=args.hidden_dims,
                         dropout=args.dropout).to(device)
    
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Training loop
    best_val_rmse = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step(val_metrics['rmse'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val MAE: {val_metrics['mae']:.4f}")
        print(f"  Val Spearman: {val_metrics['spearman']:.4f}")
        
        # Save best model
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_rmse': best_val_rmse,
                'feature_names': feature_extractor.get_feature_names()
            }, args.output / 'best_model.pt')
            print(f"  ✓ Saved best model (RMSE: {best_val_rmse:.4f})")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIC prediction model")
    
    # Data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to MIC dataset CSV")
    parser.add_argument("--use-structure", action="store_true",
                        help="Use structure features (slow)")
    
    # Paths
    parser.add_argument("--aaindex-path", type=str, 
                        default="descriptors/aaindex")
    parser.add_argument("--svm-path", type=str,
                        default="predictionsParameters/svc.pkl")
    parser.add_argument("--scaler-path", type=str,
                        default="predictionsParameters/Z_score_mean_std__intersect_noflip.csv")
    parser.add_argument("--cache-dir", type=str,
                        default="./esmfold_cache")
    parser.add_argument("--output", type=Path, default=Path("./models"),
                        help="Output directory for models")
    
    # Model
    parser.add_argument("--hidden-dims", type=int, nargs="+",
                        default=[128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    
    main(args)
```

---

### Phase 6: Inference CLI (Day 8)

Create `sequence_to_svm_minimal/predict_mic.py`:

```python
#!/usr/bin/env python3
"""
Command-line interface for MIC prediction
"""
import torch
import argparse
import pandas as pd
from pathlib import Path

from feature_pipeline import UnifiedFeatureExtractor
from mic_predictor import MICPredictor

def predict_single(sequence, model, feature_extractor, device):
    """Predict MIC for single sequence"""
    
    # Extract features
    result = feature_extractor.extract_all_features(sequence, include_structure=True)
    features = torch.FloatTensor(result['feature_vector']).unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        log_mic = model(features).item()
    
    mic = 10 ** log_mic
    
    return {
        'sequence': sequence,
        'log_MIC': log_mic,
        'MIC': mic,
        'sigma': result['svm']['sigma'],
        'prob_antimicrobial': result['svm']['prob_pos'],
        'structure_confidence': result['structure']['plddt_mean'],
        'structure_available': bool(result['structure']['structure_success'])
    }


def main():
    parser = argparse.ArgumentParser(description="Predict MIC for peptide sequences")
    
    parser.add_argument("--sequence", type=str, help="Single sequence to predict")
    parser.add_argument("--input", type=str, help="Input file with sequences (one per line or CSV)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output file")
    
    # Feature extractor paths
    parser.add_argument("--aaindex-path", type=str, default="descriptors/aaindex")
    parser.add_argument("--svm-path", type=str, default="predictionsParameters/svc.pkl")
    parser.add_argument("--scaler-path", type=str, 
                        default="predictionsParameters/Z_score_mean_std__intersect_noflip.csv")
    parser.add_argument("--cache-dir", type=str, default="./esmfold_cache")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load feature extractor
    print("Loading feature extractor...")
    feature_extractor = UnifiedFeatureExtractor(
        aaindex_path=args.aaindex_path,
        svm_path=args.svm_path,
        scaler_path=args.scaler_path,
        cache_dir=args.cache_dir
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device)
    
    n_features = len(checkpoint['feature_names'])
    model = MICPredictor(n_features=n_features).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")
    
    # Get sequences
    sequences = []
    if args.sequence:
        sequences = [args.sequence]
    elif args.input:
        if args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            sequences = df['sequence'].tolist()
        else:
            with open(args.input, 'r') as f:
                sequences = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Must provide --sequence or --input")
    
    print(f"\nPredicting for {len(sequences)} sequence(s)...")
    
    # Predict
    results = []
    for i, seq in enumerate(sequences):
        print(f"[{i+1}/{len(sequences)}] {seq[:30]}{'...' if len(seq) > 30 else ''}")
        try:
            result = predict_single(seq, model, feature_extractor, device)
            results.append(result)
            print(f"  MIC = {result['MIC']:.2f} μg/mL")
            print(f"  σ = {result['sigma']:.3f}, p(AMP) = {result['prob_antimicrobial']:.3f}")
            print(f"  Structure confidence = {result['structure_confidence']:.1f}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'sequence': seq, 'error': str(e)})
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 📝 IMMEDIATE ACTION ITEMS

### Today (Setup):
```bash
# 1. Install dependencies
pip install fair-esm biotite biopython

# 2. Test ESMFold
python test_esmfold.py

# 3. Download MIC data
# Go to https://dbaasp.org/ and download dataset
```

### Tomorrow (Implementation):
```bash
# 1. Create feature extraction modules
#    - esmfold_features.py
#    - feature_pipeline.py

# 2. Process MIC dataset
#    - Filter, clean, log-transform
#    - CD-HIT clustering for splits

# 3. Start training
python train_mic_model.py \
    --data data/dbaasp_mic_data.csv \
    --use-structure \
    --epochs 100 \
    --output models/mic_predictor
```

---

## ⚠️ CRITICAL NOTES

1. **Data is Key**: Must obtain MIC-labeled dataset (DBAASP recommended)
2. **Caching ESMFold**: Structure prediction is slow (~5-10s per peptide), cache everything
3. **Fallback Mode**: Always allow prediction without structure if ESMFold fails
4. **Similarity-Aware Splits**: Use CD-HIT to avoid data leakage
5. **Log-Scale MIC**: Always work with log10(MIC) for regression
6. **GPU Memory**: 5070ti is sufficient but watch batch sizes with ESMFold

---

## 📊 EXPECTED OUTCOMES

### Baseline Performance:
- σ alone → MIC: Spearman ~0.3-0.5 (weak correlation per paper)
- Descriptors only → MIC: Spearman ~0.4-0.6
- Structure only → MIC: Spearman ~0.5-0.7

### Target Performance:
- **Fusion model → MIC: Spearman ~0.7-0.8, RMSE <0.8 on log(MIC)**

### Deliverables:
1. ✅ Working ESMFold integration with caching
2. ✅ Unified feature extraction pipeline
3. ✅ Trained MIC prediction model
4. ✅ Inference CLI tool
5. ✅ Evaluation report comparing baselines

---

## 🔄 FALLBACK PLAN

If ESMFold is too slow/problematic:
1. Use ESM-2 embeddings only (much faster)
2. Train small adapter on embeddings → MIC
3. Ensemble with existing SVM

---

## 📚 REFERENCES

- **ESMFold**: https://github.com/facebookresearch/esm
- **DBAASP**: https://dbaasp.org/
- **Original Paper**: (assumed context about σ and MIC correlation)
- **Current SVM**: 484 samples, 46.5% SV ratio, non-linear boundary

---

**STATUS**: Ready to implement  
**ESTIMATED TIME**: 1-2 weeks for MVP  
**NEXT SESSION**: Start with Phase 1 (Setup & Dependencies)

---

*End of project plan. Good luck! 🚀*


