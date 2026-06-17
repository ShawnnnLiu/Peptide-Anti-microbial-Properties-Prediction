# Project Handoff — Peptide Antimicrobial Properties Prediction

> Onboarding guide for the student taking over this project.
> Written 2026-06-17. It consolidates the whole project, **including the StaPep / buforin
> regression work that the older markdown docs in `sequence_to_svm_minimal/` predate.**
> Where this doc and those older docs disagree, **this doc is newer.**

---

## 0. TL;DR — what is this project?

Machine-learning prediction of **antimicrobial peptide (AMP) properties from sequence (and
predicted 3D structure).** It grew through three phases:

1. **Phase 1 — Binary AMP classification (legacy).** A pretrained 2016-era RBF-SVM on 12
   hand-crafted QSAR descriptors that outputs "is this peptide antimicrobial, ±1?" Used
   for sliding-window genome/proteome mining. Inference only — never retrained.
2. **Phase 2 — Structure-aware classification.** Predict ESMFold 3D structures from
   sequence, derive geometric features, and train MLPs and GNNs (GCN/GAT/EGNN) for
   AMP-vs-decoy classification. This is where the strong classifiers live.
3. **Phase 3 — Stapled-peptide regression (the current frontier).** For **hydrocarbon-stapled
   AMPs** and specifically **buforin II / magainin 2 variants**, predict *quantitative*
   targets: **pMIC** (potency, per bacterial organism), **% hemolysis / lyticity** (toxicity),
   and correlate model scores against experimental **NGC** (membrane negative Gaussian
   curvature). RandomForest regression on StaPep MD-derived features.

Everything lives in `sequence_to_svm_minimal/`. (`pretrained_svm/` is just a copy of the
legacy SVM bundle.)

Key biological premise repeated throughout: the SVM margin score σ correlates with **membrane
activity but NOT with MIC**, which is *why* the project moved to structure features and then to
direct potency/toxicity regression.

---

## 1. Timeline (from git history)

| Date | Milestone |
|---|---|
| 2025-12-27 | First commit — legacy SVM descriptors |
| 2026-01-07 | ESM / ESMFold setup, "next step directions" |
| 2026-01-13 | First ESMFold run + visualization |
| 2026-01-22 | MLP with geometric features; feature-fusion experiments |
| 2026-02-02 | First GNN runs (GCN/GAT/EGNN) + visualization |
| 2026-02-17 | StaPep stapled-peptide work begins; new peptide tests |
| 2026-02-23 → 03-05 | StaPep MLP/SVM results + ablation studies |
| 2026-03-13/16 | MIC-tier "group" classification (3-/4-class) |
| 2026-03-30 | Feature-correlation visualization incl. PNAS paper dataset |
| 2026-03-31 → 04-06 | **Regression phase**: sanity checks, RF regression on hemolysis %, buforin-specific features |
| 2026-05-06 | "good run" + "comparisons done" (NGC comparison study) — most recent |

So: **SVM → ESM/ESMFold → geometric MLP → GNN → StaPep classification → StaPep regression → NGC comparison.**

---

## 2. Datasets

All under `sequence_to_svm_minimal/data/training_dataset/`.

- **Legacy SVM training set (Phase 1):** ~484 peptides (≈242 AMP / 242 non-AMP), 12 descriptors.
  Deduced from `svc.pkl` (225 support vectors, 46.5% SV ratio). Original sklearn ≤0.18.
- **Structure-aware classification set (Phase 2):** **572 peptides = 286 AMP + 286 decoy**, each
  with an ESMFold PDB. Files: `seqs_AMP.txt`, `seqs_decoy_subsample.txt`,
  `geometric_features_clustered.csv` (24 geometric features), `qsar12_descriptors.csv`.
- **StaPep stapled-peptide set (Phase 3):** under `data/training_dataset/StaPep/`.
  - `stapled_amps_features_training_XZ_md50ns.csv` — **~172 stapled AMPs** (DRAMP + PMID 31427820
    supplement), each run through the **StaPep MD pipeline** (AmberTools tleap → OpenMM
    implicit-GB MD, 50 ns → pytraj) to give a **17-feature** descriptor (9 sequence-only +
    8 MD-derived). K-stapled peptides and one MD failure (DRAMP21556) were dropped.
  - `stapled_decoys.csv` — **~354 stapled non-AMPs** (MDM2 inhibitors, AKAP disruptors, etc.):
    a strong negative class with identical staple chemistry but no antimicrobial function.
  - Test sets: `test_stapled_features.csv` (8 buforin/magainin stapled variants),
    `test_buf_specific_stapep_features.csv` (7 F10W buforin variants).
  - `qsar_stapled_{amps,decoys,test}.csv` — the 12 QSAR descriptors for the same peptides.

**The 17 StaPep features:** length, weight, hydrophobic_index, charge, aromaticity,
isoelectric_point, fraction_arginine, fraction_lysine, lyticity_index, helix_percent,
sheet_percent, loop_percent, mean_bfactor, mean_gyrate, num_hbonds, psa, sasa. (Some scripts
use a 14-feature subset.)

> **Data caveats (from `StaPep/HDC_HANDOFF_BRIEF.md` — read it):**
> MD features have a noise floor; implicit-GB over-stabilizes helix ~10–20%; **~57 of 172
> AMPs are magainin variants** (high redundancy → **use sequence/family-clustered splits, not
> random**, to avoid leakage); regression labels (MIC, hemolysis) are buried in free-text
> `Activity` / `Hemolytic_Activity` columns and must be **regex-parsed** — label engineering
> is the explicitly unfinished step.

**Note on what's NOT in git:** `.gitignore` excludes model checkpoints (`*.pt/*.pth`), the
ESMFold cache (~15 GB), MIC datasets, and all notebooks. A fresh clone must regenerate/obtain
these. The DBAASP large MIC dataset (~20k peptides) was planned but never obtained — a
long-standing blocker for general MIC regression.

---

## 3. Environment setup

**Two (really three) conda environments** — this is the most important operational fact:

| Env | Python | For | Key deps |
|---|---|---|---|
| `skl_legacy` | 3.7 | Legacy SVM (`svc.pkl`), QSAR/propy descriptors | scikit-learn==0.19.2, propy3 |
| `esm_env` | 3.10 | ESM-2 embeddings, ESMFold, GNN/MLP training | torch 2.x, torch-geometric, fair-esm, transformers, biotite |
| `stap` (WSL) | — | StaPep MD feature extraction | OpenMM, AmberTools, pytraj |

Env files in repo: `skl_legacy_env.yml`, `esm_env.yml`, `requirements.txt` (note: the single
`requirements.txt` mixes both envs and conflicts with the legacy pinning — prefer the `.yml`s).

**Hardware:** developed on Windows 10/11 + WSL2 (Ubuntu 22.04) with an NVIDIA RTX 5070ti
(Blackwell). GPU needed for ESM/ESMFold/MD; SVM and RF regression are CPU-only.
- ESMFold: ~6–8 GB VRAM, ~5–30 s/peptide, first run downloads ~15 GB to `~/.cache/torch/hub/`.
  Caching by MD5(sequence) is mandatory for throughput.
- **CUDA trap:** Blackwell GPUs need the **cu128** PyTorch wheel
  (`pip install --force-reinstall torch torchvision --index-url .../cu128`). cu121 is for
  RTX 30/40. The `--force-reinstall` matters or you get a silent CPU-only install.

Sanity-check the install with `python check_pyg.py` (verifies torch + torch_geometric + CUDA).

---

## 4. Training & prediction scripts (what was run, how to run it)

All paths relative to `sequence_to_svm_minimal/`. **Important gotcha: almost every `run_*`
and `predict_*` script trains its model from scratch on each invocation — there are no saved
sklearn artifacts.** Only ESMFold weights, the legacy `svc.pkl`, and GNN `.pt` checkpoints
are persisted.

### 4a. Feature building (run these first to (re)generate features)
| Script | Does | Run |
|---|---|---|
| `generate_stapep_structures.py` | ESMFold → PDBs for AMP/decoy seqs | `python generate_stapep_structures.py [--amp-only|--decoy-only]` |
| `build_stapep_geo.py` | Geometric features for AMP/DECOY/TEST splits (preferred builder) | `python build_stapep_geo.py` |
| `build_geometric_features.py` | General PDB-dir → geometric CSV (argparse) | `python build_geometric_features.py --pdb-dir ... --output ...` |
| `extract_stapep_qsar.py` | 12 QSAR descriptors (run under `skl_legacy`) | `python extract_stapep_qsar.py` |
| `features/geometric_features.py` | **Library** computing all geometric features from a PDB (the engine behind the builders) | imported |

### 4b. Classification training
**Legacy SVM (inference only):**
- `run_pretrained_svm_inference.py` → runs the 2016 SVM on the 8 test peptides →
  `pretrained_svm_test_predictions.csv`. (Loads numpy sidecars to bypass the broken old pickle.)
- `scripts/run_sequence_svm.py` → full legacy pipeline (descriptors → predictSVC) for
  sliding-window mining.

**StaPep AMP-vs-decoy classifiers** (each compares StaPep-only / QSAR-only / QSAR+StaPep):
- `run_stapep_svm.py` (RBF-SVM), `run_stapep_mlp.py` (MLP), `run_combined_svm.py` (3-way),
  all `[--cv-folds N]`.

**MIC-tier classifiers** (4-class potency tiers from E. coli MIC, AMPs only):
- `run_mic_rf.py`, `run_mic_svm.py`, `run_mic_mlp.py`, `run_mic_classifier.py`.

**Structure MLP track (`nn_pipeline/`):**
- `run_nn_training.py` — cluster-CV MLP on geometric features (saves `.pt` + scaler).
- `run_nn_training_pnas_style.py` — strict PNAS (Lee 2016) protocol →
  `results/pnas_style_eval.{json,csv}`.
- `run_feature_fusion_experiments.py` — SVM/MLP × QSAR-12 / Geo-24 / Combined-36 →
  `results/fusion_experiments_<ts>.{json,csv}`.

**GNN track (`gnn/`):**
- `run_gnn_training.py --architecture {gcn,gat,egnn} [...]` — one architecture →
  `results/gnn/gnn_<arch>_<ts>.json` + `.pt`.
- `run_gnn_comparison.py` — all 3 archs × 3 feature configs → `results/gnn/gnn_comparison_<ts>.json`.
- `run_stapep_gnn_comparison.py` — same on the StaPep dataset → `results/stapep_gnn/...json`.

### 4c. Regression & prediction (Phase 3, the current work)
> v2 scripts supersede v1: 18 features instead of 14, plus better free-text MIC/hemolysis parsing.

| Script | Target / model | Output |
|---|---|---|
| `predict_buf_pmic_v2.py` | pMIC (E. coli), RF, 18 feat — 12 buforin variants | `buf_pmic_regression_v2.png` |
| `predict_pmic_all_organisms.py` | per-organism pMIC RF (7 organisms) | `pmic_by_organism/*.png` |
| `predict_pmic_stapled_variants.py` | pMIC for native vs stapled variants | `stapled_pmic.png` |
| `predict_buf_hemolysis_v2.py` | % hemolysis RF, 18 feat | `buf_hemolysis_regression_v2.png` |
| `predict_buf_hemolysis_lyticity.py` | hemolysis from lyticity_index alone | `buf_hemolysis_lyticity_only.png` |
| `predict_hemolysis_regression.py` | % hemolysis RF (buforin + magainin) | `hemolysis_regression.png` |
| `predict_mic_single.py SEQ` | MIC tier for one sequence (RF+SVM+MLP) | console |
| `predict_gcn_single.py` / `predict_stapep_candidates.py` | GNN AMP prediction for 8 candidates | `results/stapep_predictions/*.json` |
| `compare_ngc_scores.py` | model scores vs experimental NGC (4 buforin variants) | `ngc_*.png` |
| `compare_buf_pmic.py` | literature vs predicted pMIC (7 F10W variants) | `buforin_pmic_comparison.png` |

Plotting/diagnostics: `plot_mic_distribution.py`, `plot_lyticity_vs_hemolysis.py`,
`plot_hydrophobic_vs_mic.py`, `check_feature_overlap*.py`, `check_features*.py`,
`debug_secondary_structure.py`, `test_esmfold_quick.py`.

### 4d. Supporting modules
- `nn_pipeline/` — `models.py` (`AMPClassifier` MLP + attention/focal-loss variants),
  `train.py` (Trainer/CV/early-stopping), `feature_dataset.py`, `prepare_clusters.py`
  (CD-HIT or difflib clustering for leakage-free splits).
- `gnn/` — `data_utils.py` (PDB → PyG graph, 26-dim nodes, 8 Å edges), `models.py`
  (GCN/GAT/EGNN + `PeptideGNN` wrapper), `train.py`.
- `models/` — ESM-2 embeddings (`esm_sequence_processor.py`), batch ESMFold
  (`batch_esmfold.py`, `run_esmfold_peptides.py`), `extract_structure_features.py`, plus
  weight-download/diagnostic utils.

> **Stubs (documented intent, no working code):** all of `structure/` and `utils/`, most of
> `cli/`, plus `models/{mic_predictor,train_mic}.py` and `features/{feature_extractor,feature_utils}.py`.
> Don't waste time looking for logic there.

---

## 5. Results

### 5a. AMP-vs-decoy classification (572 peptides) — these ARE saved to JSON/CSV

**PNAS-style MLP** (24 geometric features) — `results/pnas_style_eval.json`:

| Metric | CV (15 rounds) | Blind test |
|---|---|---|
| Accuracy | 0.949 ± 0.021 | 0.919 |
| F1 | 0.949 ± 0.020 | 0.918 |
| AUC-ROC | 0.986 ± 0.007 | 0.976 |
| MCC | 0.900 ± 0.039 | 0.837 |

**Feature-fusion experiments** — `results/fusion_experiments_*`. Winner: **MLP on Combined-36
(QSAR-12 + Geo-24)**, cluster-CV AUC 0.991 / F1 0.945 / MCC 0.895; PNAS blind AUC 0.995 /
F1 0.977 / MCC 0.953. **Clear conclusion: QSAR-12 alone is weakest; adding geometric features
lifts every model** (MLP cluster MCC 0.723 → 0.895). The Two-Tower fusion net ties on blind
AUC (0.997) but doesn't beat plain MLP-Combined-36 overall.

**GNN comparison** — `results/gnn/gnn_comparison_20260202_142709.json` (the canonical full run):
best is **GAT (Graph+Combined36), AUC 0.982**. GAT consistently edges GCN; EGNN is weakest and
highest-variance. Adding geometric features to the graph gives marginal AUC gains at best.

### 5b. StaPep stapled-peptide classification

**StaPep GNN comparison** (541 peptides, 187 AMP / 354 decoy) —
`results/stapep_gnn/stapep_gnn_comparison_20260212_181741.json`:

| Arch | AUC-ROC | F1 | MCC |
|---|---|---|---|
| **GCN** | **0.997 ± 0.003** | 0.971 | 0.956 |
| GAT | 0.996 ± 0.004 | 0.960 | 0.940 |
| EGNN | 0.988 ± 0.018 | 0.904 | 0.875 |

(Note GCN wins here, opposite of the main dataset where GAT led.)

**Candidate screening** (8 stapled variants, GNN meta-ensemble) —
`results/stapep_predictions/predictions_*.json`: the **magainin-derived staples (Mag 31/36/25)
score as promising AMPs** (Mag31 P≈0.88), while **all four stapled buforin variants score as
non-AMP** (P≈0.007–0.024).

### 5c. Regression (Phase 3) — ⚠️ NOT saved to files

The hemolysis and pMIC regression metrics are computed at runtime and **only rendered into the
PNGs** — there is no JSON/CSV dump. The error metric used is **MAE** (RMSE is never computed).
What's recoverable from hardcoded baselines in the source:

- **Hemolysis v1 baseline:** 14 feat, n=98, **CV Pearson r = 0.844**, test Pearson r = 0.596,
  test Spearman 0.317 (a real generalization gap). `lyticity_index` is the dominant single
  predictor (~40% RF importance). Buf_i7_9_F10W is the most hemolytic variant (~57% @ 50 µg/mL).
- **pMIC baseline:** 14 feat, n≈147, **CV Pearson r ≈ 0.83** (vs StaPep paper R=0.84 for E. coli).
- **Per-organism pMIC:** built for E. coli, P. aeruginosa, S. aureus, B. subtilis, M. luteus,
  Klebsiella, Salmonella (organisms with ≥15 parseable MIC samples). Numeric Pearson r's exist
  only inside `pmic_by_organism/_summary_pearson_r.png`.

> **Action item for the new student:** add CSV/JSON dumps to the regression scripts so these
> numbers are recorded, not just plotted.

### 5d. NGC comparison study (4 buforin variants) — `compare_ngc_scores.py`

Key finding: the **legacy QSAR/SVM model is "blind"** — outputs ~identical scores (~0.5) for
all variants, whereas **StaPep-feature models differentiate them and track NGC direction.**
Caveat: **N=4, no significant p-values** — directional/qualitative only. Measured NGC: WT
0.0133, Buf12 0.0109, Buf13 0.0145, Q9K 0.0130 nm⁻².

### 5e. Figures
~30 PNGs. Grouped: MIC distribution (`mic_distribution.png`); hemolysis regression
(`buf_hemolysis_regression_v2.png` etc.); pMIC (`buf_pmic_regression_v2.png`,
`pmic_by_organism/`); lyticity (`lyticity_vs_hemolysis.png`); hydrophobicity vs MIC; NGC
(`ngc_*.png`); GNN training curves (`results/gnn/*.png`). Note: `hemolysis_feature_bias_analysis.png`
and `hemolysis_model_optimization.png` have **no generating script in the repo** — provenance
unverified.

---

## 6. Key findings (the story so far)

1. AMP-vs-non-AMP is **not linearly separable** (46.5% SV ratio in the legacy SVM) — simple
   rules fail; non-linear models are needed.
2. **Structure-aware features work very well for classification** — geometric MLP and GNNs
   reach ~0.95–0.99 AUC. **Combining QSAR + geometric features beats either alone.**
3. On stapled peptides, GNN classification is near-saturated (GCN AUC 0.997) — classification
   is largely "solved" for this data; the **open problem is quantitative regression.**
4. **Regression is harder and still preliminary**: good CV (r~0.83–0.84 for both pMIC and
   hemolysis) but a real generalization gap to held-out / native peptides. Models trained on
   *stapled* peptides extrapolate poorly to *native/unstapled* buforin.
5. `lyticity_index` dominates hemolysis prediction.
6. The legacy QSAR model can't distinguish closely related buforin variants; StaPep MD features
   can (NGC study) — but on N=4.

---

## 7. Known gotchas

- Models retrain from scratch each run (no saved sklearn artifacts).
- v2 scripts > v1 (18 vs 14 features, better label parsing). Prefer `*_v2.py`.
- `build_stapep_features.py` and `build_stapep_geo.py` both write the same
  `stapep_*_geometric.csv` — prefer `build_stapep_geo.py`.
- `compare_anomalous_features.py` has hardcoded Windows/WSL paths — won't run unmodified.
- Regression metrics live only in PNGs (see action item above).
- Use **clustered/family-aware splits**, never random — ~1/3 of AMPs are magainin variants.
- Regression labels need regex parsing out of free-text columns; ">100" MIC values are
  currently dropped (censoring not handled).

---

## 8. Next steps / open questions

From `StaPep/HDC_HANDOFF_BRIEF.md` (most current) and the older planning docs:

1. **Label engineering first** — regex-parse `Activity`/`Hemolytic_Activity` into numeric
   `MIC_uM` (per organism), `MHC_uM`, `pct_hemolysis`; report parse-success rate. This
   determines whether regression is even viable on the small labeled subset.
2. **Family/sequence-clustered train-test splits** to kill leakage.
3. Decide feature-only vs sequence-aware encoding; how to represent paired staples (i,i+4 vs
   i,i+7); multi-output vs separate hemolysis/MIC regressors.
4. The brief is scoped toward a **hyperdimensional-computing (HDC)** regression model as the
   next modeling direction.
5. Persist regression metrics to disk.
6. Longer-standing: obtain a large MIC dataset (DBAASP) for general MIC regression; handle MIC
   censoring; address small-N statistics in the NGC study.

---

## 9. Suggested reading order

1. This file.
2. `sequence_to_svm_minimal/README.md` + `STRUCTURE.md` (project shape).
3. `EXECUTIVE_SUMMARY.md` / `KEY_FINDINGS.txt` (the SVM non-linearity story).
4. `ENVIRONMENT_GUIDE.md` + `SETUP.md` (two-env setup, CUDA trap).
5. **`data/training_dataset/StaPep/HDC_HANDOFF_BRIEF.md`** — the authoritative current-state +
   next-tasks doc.
6. Then skim `run_stapep_*`, `predict_buf_*`, `predict_pmic_*`, `compare_ngc_scores.py` and the
   PNG outputs to see live results.

> The other top-level markdown docs (`ESMFOLD_INTEGRATION_PROJECT_PLAN.md`, `WORKFLOW_EXAMPLE.md`,
> `README_FOR_NEXT_AGENT.md`, `CODEBASE_ANALYSIS_SUMMARY.md`, `QUICK_START_GUIDE.md`) are
> Phase-1/2 history — useful background, but they predate the StaPep/regression work.
