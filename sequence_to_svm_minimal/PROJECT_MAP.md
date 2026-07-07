# PROJECT_MAP.md

Map of the `SVM_ESM_Peptides\` workspace. Read this before touching code so you know where canonical data lives, which artifacts are research records that must not be regenerated casually, and where the high-risk swap points are. Read `RUNNABLE_FILES.md` for the per-file inventory and `SMOKE_TESTS.md` for the test plan.

## 1. Workspace layout

```
SVM_ESM_Peptides\
├── Miniconda3-latest-Linux-x86_64.sh        # installer binary; ignore
├── check_stapep.sh / find_stapep.sh / find_stapep2.sh   # one-shot WSL helpers
├── run_buforin_stapep.py                    # CLI: 1 peptide → 5 ns MD; WSL+stap
├── run_buf_variants_stapep.py               # batch: 7 Buforin variants → 10 ns MD; WSL+stap
├── test_buf_variant_single.py               # 100 ps smoke test for the MD pipeline
├── test_wsl_stapep.py                       # WSL+stap env import smoke test
├── stapep_package\                          # vendored library: AmberTools/OpenMM/pytraj MD pipeline
│   ├── setup.py
│   ├── stapep\                              # live source
│   │   ├── molecular_dynamics.py            # PrepareProt + Simulation classes
│   │   ├── utils.py                         # ProtParamsSeq, PhysicochemicalPredictor
│   │   ├── structure.py, esmfold.py, filter.py, params.py, generate_*.py, run_pipeline.py
│   │   ├── example\                         # 7 notebooks + 1 outdated demo
│   │   ├── img\                             # reference figures
│   │   └── models\rf_cls_model.pkl          # bundled RF for permeability
│   ├── build\lib\stapep\*.py                # build-time mirror of stapep\* (legacy/reference)
│   └── stapep.egg-info\
└── Peptide-Anti-microbial-Properties-Prediction\
    ├── pretrained_svm\sequence_to_svm_minimal\   # FROZEN snapshot — see §6
    └── sequence_to_svm_minimal\                  # ★ active codebase ★ (everything below)
        ├── README.md, SETUP.md, STRUCTURE.md, REFACTOR_PLAN.md,
        │   ENVIRONMENT_GUIDE.md, QUICK_START_GUIDE.md, WORKFLOW_EXAMPLE.md,
        │   ESMFOLD_INTEGRATION_PROJECT_PLAN.md, CODEBASE_ANALYSIS_SUMMARY.md,
        │   EXECUTIVE_SUMMARY.md, KEY_FINDINGS.txt, CLAUDE_REFACTOR_PROMPT.md,
        │   README_FOR_NEXT_AGENT.md, requirements.txt,
        │   esm_env.yml, skl_legacy_env.yml,
        │   run_esm_processor.bat, predict_mic_single.py
        ├── descriptors\          # legacy QSAR-12 descriptor generator (Py3 port) + propy3 vendored
        ├── predictionsParameters\ # pre-trained SVM bundle (svc.pkl + .npy + Z-score CSV)
        ├── scripts\              # 2 CLIs: SVM pipeline orchestrator + seqs.txt window maker
        ├── cli\                  # empty TODO stubs (3 files)
        ├── features\             # geometric_features.py (live) + 2 TODO stubs
        ├── feature_extraction\   # CSV-builders: geometric, QSAR, StaPep geo, ESMFold-driver
        ├── structure\            # 3 TODO stubs (scaffolding for refactor)
        ├── utils\                # 3 TODO stubs
        ├── models\               # ESMFold scripts + ESM-2 embedder + 2 TODO stubs
        ├── nn_pipeline\          # PyTorch MLP (FeaturePipeline + AMPClassifier + trainer)
        ├── gnn\                  # PyG GCN/GAT/EGNN + trainer + StaPep candidate predictor
        ├── svm\                  # 7 sklearn SVM scripts (training + inference + ablations)
        ├── mlp\                  # 5 sklearn/PyTorch MLP scripts
        ├── regression\           # 6 RF regression scripts (pMIC, hemolysis, organism-by-organism)
        ├── comparison\           # cross-script benchmarking — see §5 LANDMINE
        ├── figures\              # standalone figure scripts + 3 notebooks
        ├── debug_checks\         # one-off data sanity scripts (mostly data-dependent)
        ├── archive\              # 6 superseded/un-migrated Buforin scripts — FROZEN, see §10 + archive\README.md
        ├── experiments\          # exp1\, exp2\ — recorded inference runs (do not overwrite)
        ├── results\              # fusion experiments, PNAS eval, GNN runs, stapep predictions
        ├── tests\                # one ad-hoc test file (test_geometric_features.py)
        ├── data\training_dataset\
        │   ├── structures\AMP\      # 286 ESMFold PDBs
        │   ├── structures\DECOY\    # 286 ESMFold PDBs
        │   ├── structures\          # checkpoint.json, results_log.csv, results_with_plddt.csv
        │   ├── geometric_features.csv              # canonical, 286+286
        │   ├── geometric_features_clustered.csv    # canonical + cluster IDs
        │   ├── geometric_features_broken.csv       # REFERENCE only (kept for provenance)
        │   ├── qsar12_descriptors.csv              # canonical QSAR-12
        │   ├── seqs_AMP.txt, seqs_decoy_subsample.txt
        │   └── StaPep\           # see §3 — single most data-dense subfolder
        ├── pmic_by_organism\     # 8 PNG outputs from regression\predict_pmic_all_organisms.py
        ├── 10ns vs 50ns comparison\
        │   ├── README.md                       # frozen research log narrative
        │   ├── legacy_147rows\                  # logs + 4 PNGs — FROZEN
        │   └── md50ns_130rows\                  # logs + 4 PNGs — FROZEN
        ├── comparison_legacy_vs_md50ns.zip     # zipped backup of the above
        ├── pretrained_svm_test_predictions.csv # output artifact
        ├── visualize_peptides.html, visualize_structure.html
        └── ~18 standalone PNG figures at root  # research outputs; do not delete
```

## 2. The four environment islands

| Island | Conda env | Purpose | Where to find the env spec |
|---|---|---|---|
| Pre-trained SVM (PNAS 2016) | `skl_legacy` (sklearn 0.19.2, Py 3.7) | `descripGen_12_py3.py` → `predictSVC.py`; reads `svc.pkl` + `.npy` sidecars | `seq2svm\skl_legacy_env.yml` |
| Modern sklearn / ESMFold | `esm_env` (sklearn modern, fair-esm, transformers, CUDA torch, Py 3.10) | Most svm/mlp/regression/comparison scripts; all ESMFold inference | `seq2svm\esm_env.yml` |
| StaPep MD (WSL only) | `stap` (OpenMM, AmberTools `tleap`, pytraj, parmed; install via `stapep_package\README.md`) | All MD-feature extraction (50 ns implicit-GB MD) | `stapep_package\setup.py` + manual conda recipe in `stapep_package\README.md` |
| PyTorch + PyG | `venv` (torch 2.x, torch_geometric 2.3+, WSL recommended) | GNN pipeline + `nn_pipeline\` PyTorch MLP + feature fusion | `seq2svm\requirements.txt` |

Pickled-SVM caveat: `svc.pkl` was serialized with sklearn 0.19.2. Loading it in `esm_env` silently miscomputes or crashes. The workaround script `svm\run_pretrained_svm_inference.py` rebuilds the SVM manually from `.npy` sidecars.

## 3. The StaPep data folder — `data\training_dataset\StaPep\`

This is where the stapled-peptide work lives. The HDC handoff brief (`HDC_HANDOFF_BRIEF.md`) is authoritative here.

### Canonical inputs (must not overwrite)

| File | Rows | Role |
|---|---:|---|
| `stapled_amps_features_training_XZ_md50ns.csv` | 172 | **THE** training feature matrix — 50 ns implicit-GB MD, K-stapled removed, DRAMP21556 removed |
| `stapled_amps_combined_paper_dataset_XZ_only.csv` | 172 | Identity / metadata for the same 172 peptides (joins on `DRAMP_ID`) |
| `stapled_decoys.csv` | 355 | Decoy stapled peptides (label=0); same 17-feature schema |
| `stapled_amps.csv` | — | Raw AMP metadata (sequences, MIC values from literature) |
| `qsar_stapled_amps.csv`, `qsar_stapled_decoys.csv`, `qsar_stapled_test.csv` | — | QSAR-12 descriptors aligned to the same peptides |
| `stapep_amp_geometric.csv`, `stapep_decoy_geometric.csv` | — | Geometric-24 features for stapled set |
| `test_stapled_features.csv`, `test_buf_specific_stapep_features.csv` | — | Test sets |
| `structures\AMP\` (187 PDB), `structures\DECOY\` (354 PDB), `structures\` (8 PDB) | — | ESMFold structures |

### Reference / legacy (do not use as primary; keep for provenance)

| File | Rows | What it is |
|---|---:|---|
| `stapled_amps_features_training_XZ_md4ns.csv` | 172 | 4 ns MD predecessor — lower quality |
| `stapled_amps_features_training_XZ_md4ns_REP2.csv` | 3 | 4 ns replicate; defines per-peptide MD noise floor |
| `stapled_amps_features.csv` | 188 | Original DRAMP-only feature run — **also actively rewritten by the comparison runner (see §5)** |
| `stapled_amps_features_combined_paper_dataset.csv` | 195 | Includes the 22 K-stapled peptides later dropped |
| `stapled_amps_combined_paper_dataset.csv` | — | Raw combined paper dataset (pre-XZ-filter) |
| `stapled_amps_paper_supplement.csv`, `_features.csv` | — | Mag(i+4)1,15 multi-mutation paper supplement |
| `PNAS_paper_datasets\Stapled-peptide_permeability*.csv` | — | Permeability data from the PNAS source paper |

### Audit logs (small, must not overwrite)

- `dropped_k_stapled.csv` (22 rows removed)
- `dropped_md_failures.csv` (DRAMP21556 only)
- `paper_vs_dramp_side_by_side.csv`
- `advisor_peptide_coverage_side_by_side.csv`, `advisor_staPep_coverage_summary.txt`
- `sanity_legacy_vs_md4ns.csv`, `sanity_outliers_md4ns.csv`, `sanity_report_md4ns.txt`
- `md_replicate_comparison.csv`, `.txt`
- `replicate_subset.csv`
- `stapep_md50ns.log` (114 KB — full 50 ns MD run log; load-bearing provenance)

### Diagnostic scripts (runnable, safe — see RUNNABLE_FILES.md §4)

`_verify_training_md4ns.py`, `sanity_check_md_features.py`, `_inspect_md50ns_in_progress.py`, `_compare_replicate_md.py`, `_probe_dssp_staples.py` (WSL+pytraj), `_check_*.py`.

## 4. Model artifacts

| Path | Purpose | Status |
|---|---|---|
| `seq2svm\predictionsParameters\svc.pkl` + `svc.pkl_01.npy`…`_11.npy` + `svc_model_bundle.zip` | Pre-trained 2016 PNAS SVM | **MUST NOT OVERWRITE** (no easy retraining path) |
| `seq2svm\predictionsParameters\Z_score_mean_std__intersect_noflip.csv` | Z-score normalization for the pre-trained SVM | MUST NOT OVERWRITE |
| `pretrained_svm\predictionsParameters\svc.pkl` etc. | Snapshot duplicate | MUST NOT OVERWRITE (frozen) |
| `stapep_pkg\models\rf_cls_model.pkl` | Bundled RF for permeability | MUST NOT OVERWRITE |
| `seq2svm\nn_pipeline\checkpoints\amp_classifier_mlp_*.pt` + `scaler_*.joblib` (6 each, Jan 2026) | MLP training outputs | Regenerable (minutes–hours) |
| `seq2svm\results\checkpoints\amp_classifier_pnas_*.pt` + `scaler_pnas_*.joblib` (2 each) | PNAS-style MLP outputs | Regenerable |
| ESMFold weights | NOT stored locally; downloaded on demand (~15 GB) from HuggingFace | — |

## 5. Result artifacts (must not overwrite)

Treat everything under `results\`, `experiments\`, `pmic_by_organism\`, and `10ns vs 50ns comparison\` as **research records**. They are snapshots produced at specific timestamps and are referenced by name in the comparison README and KEY_FINDINGS.txt.

- `seq2svm\results\fusion_experiments_2026*.json/csv` — 6 fusion runs, Jan 2026
- `seq2svm\results\pnas_style_eval.csv/json`
- `seq2svm\results\gnn\gnn_*.json` + 6 summary PNGs + per-fold curves under `curves\run_2026*\`
- `seq2svm\results\stapep_gnn\` — Feb 2026 stapled-peptide GNN runs
- `seq2svm\results\stapep_predictions\predictions_2026*.json` (4 files)
- `seq2svm\experiments\exp1\` and `exp2\` — recorded SVM inference runs
- `seq2svm\10ns vs 50ns comparison\legacy_147rows\` and `md50ns_130rows\` — the FROZEN dataset-comparison record (also zipped at root as `comparison_legacy_vs_md50ns.zip`)
- 18 root-level PNGs in `seq2svm\` (`buf_*`, `hemolysis_*`, `lyticity_*`, `ngc_*`, `mic_distribution.png`, `pmic_regression.png`, `stapled_pmic.png`, `hydrophobic_vs_mic.png`)

## 6. The frozen snapshot tree

`Peptide-Anti-microbial-Properties-Prediction\pretrained_svm\sequence_to_svm_minimal\` is a byte-for-byte snapshot of an earlier state of the active project. It contains:

- `descriptors\descripGen_12_py3.py` (identical to live)
- `predictionsParameters\predictSVC.py`, `seqWindowConstructor.py` (Python 2; syntax-error)
- `scripts\make_seqs_windows.py`, `run_sequence_svm.py` (identical to live)
- Its own `svc.pkl` + `.npy` sidecars + `svc_model_bundle.zip`
- Its own `skl_legacy_env.yml` (with a hardcoded `prefix:` path — not portable)
- Its own `README.md`, `WORKFLOW_EXAMPLE.md`, `experiments\exp1\`, `predictionsParameters\seqs.txt`

**Treat this tree as immutable.** Any change should land only in the live tree. Smoke tests should `--ignore` this tree (see `SMOKE_TESTS.md`).

## 7. High-risk workflows

### 7a. StaPep MD (50 ns, WSL `stap` env)
- Entry points: `data\training_dataset\StaPep\run_amp_md_features.py`, `run_test_stapep_md.py`, `run_buf_variants_stapep.py`, `run_buforin_stapep.py`.
- Cost: hours–days per dataset. Implicit-GB OpenMM via `stapep.molecular_dynamics.Simulation`.
- Caveats per HDC brief: helix_percent has ~10% noise even at 50 ns; long peptides (≥30 aa) under-converged; implicit-GB over-stabilises α-helix ~10–20% vs CD.
- Never run as part of refactor verification. Treat existing CSV outputs as ground truth.

### 7b. ESMFold inference
- Entry points: `models\batch_esmfold.py`, `run_esmfold_peptides.py`, `esm_sequence_processor.py`, `feature_extraction\generate_stapep_structures.py`.
- Cost: hours–days of GPU time. ~15 GB first-run download.
- Existing PDBs (286+286+187+354+8) must be preserved.

### 7c. GNN training (`gnn\run_gnn_*.py`)
- Cost: 1–2 hours on RTX 3090 per `run_gnn_comparison.py` invocation.
- Existing JSON metrics + curves under `results\gnn\curves\` are research outputs.

### 7d. PyTorch MLP training (`mlp\run_nn_training*.py`, `nn_pipeline\train.py`)
- Cost: minutes–hours.
- Existing checkpoints under `nn_pipeline\checkpoints\` and `results\checkpoints\` are research outputs.

### 7e. MIC / hemolysis text parsing
- Free-text columns `Activity` and `Hemolytic_Activity` in `stapled_amps_combined_paper_dataset_XZ_only.csv` are parsed with regex in at least 6 scripts (see RUNNABLE_FILES.md §9). Each implementation has slightly different fallback behavior for organisms/units/multi-organism rows. Centralizing this is high-value but high-risk — silent label changes will reshape every downstream metric.
- For `PAPER_*` peptides the source paper (PMID 31427820) provides clean MIC tables but a different parser is needed.

### 7f. Dataset swap landmine — `comparison\run_dataset_comparison.py`
- **Sequence:** `shutil.copy2(LEGACY_CSV, BACKUP_CSV)` → `shutil.copy2(MD50_CSV, LEGACY_CSV)` → run 6 sub-scripts → restore from backup.
- **Crash window:** ~15 lines (156→171). If interrupted, `stapled_amps_features.csv` holds 50 ns content with no automatic rollback; `.legacy_backup` file is the only correct copy.
- **Blast radius:** 8 scripts read `stapled_amps_features.csv` directly: `svm\run_stapep_svm_no_loop.py`, `comparison\compare_ngc_scores.py`, `comparison\compare_anomalous_features.py`, `comparison\compare_buf_pmic.py`, `svm\predict_mic_svm.py`, `mlp\predict_mic_mlp.py`, `figures\lyticity_vs_mic.py`, `figures\plot_mic_distribution.py` — each silently sees whichever version is on disk.
- Mitigation candidates for Phase 2/3: parametrize the canonical CSV path, or rename outputs to indicate provenance.

### 7g. Family leakage in evaluation
- ~57 of the 172 stapled AMPs are magainin variants (single point mutations of one parent). Naive random splits will leak; cluster-based / family-aware splits are required for honest metrics. The GNN/MLP pipelines already use `GroupKFold` against `cluster` IDs — preserve this when refactoring.

### 7h. Windows ↔ WSL path hardcodes
- Live offenders: `StaPep\run_test_stapep_md.py:21` (hardcoded `/mnt/c/Users/bioin/...`), `StaPep\_paper_vs_dramp_compare.py` (hardcoded `C:\Users\bioin\Downloads\...`).
- The only file that handles both cleanly is `comparison\compare_anomalous_features.py` via `platform.system()`.
- The `pretrained_svm\skl_legacy_env.yml` hardcodes another user's home (`/home/salaars/...`) — env spec only, not runtime.

## 8. Canonical entry points by purpose

| If you want to… | Run this |
|---|---|
| Score sequences with the pre-trained 2016 SVM | `seq2svm\scripts\run_sequence_svm.py` (env: `skl_legacy`) |
| Predict MIC tier for a single sequence | `seq2svm\predict_mic_single.py` (env: `esm_env`) |
| Build geometric features from existing PDBs | `seq2svm\feature_extraction\build_geometric_features.py` |
| Train a fresh SVM on stapled features | `seq2svm\svm\run_stapep_svm.py` (env: `esm_env`) |
| Train MLP on stapled features (sklearn) | `seq2svm\mlp\run_stapep_mlp.py` |
| Train MLP on geometric features (PyTorch) | `seq2svm\mlp\run_nn_training.py` (env: `venv`) |
| Train GNN comparison | `seq2svm\gnn\run_gnn_comparison.py` (env: `venv`, WSL+GPU) |
| Run feature-fusion benchmark | `seq2svm\comparison\run_feature_fusion_experiments.py` |
| ESMFold a sequence list | `seq2svm\models\batch_esmfold.py` (env: `esm_env`, GPU, hours) |
| Run StaPep MD on a stapled sequence | `run_buforin_stapep.py` (env: `stap`, WSL) |
| Build the StaPep 17-feature CSV | `seq2svm\data\training_dataset\StaPep\run_amp_md_features.py` (env: `stap`, WSL, hours–days) |
| Reproduce the 10ns/50ns comparison report | `seq2svm\comparison\run_dataset_comparison.py` **(rewrites canonical CSV — see §7f)** |

## 9. Where the docs live

| Document | Purpose |
|---|---|
| `seq2svm\REFACTOR_PLAN.md` | The phase plan this work follows |
| `seq2svm\CLAUDE_REFACTOR_PROMPT.md` | The agent prompt template |
| `seq2svm\README.md` | Public-facing pipeline overview (GNN-centric) |
| `seq2svm\README_FOR_NEXT_AGENT.md` | Historic handoff — ESMFold integration plan |
| `seq2svm\SETUP.md` | WSL2+CUDA+venv setup for GNN pipeline |
| `seq2svm\ENVIRONMENT_GUIDE.md` | Two-Conda-env split (`skl_legacy` vs `esm_env`) |
| `seq2svm\STRUCTURE.md` | Module-by-module layout (predates this map) |
| `seq2svm\QUICK_START_GUIDE.md` | 5-minute ESMFold quickstart |
| `seq2svm\WORKFLOW_EXAMPLE.md` | End-to-end ESM-2→SVM workflow on Windows |
| `seq2svm\ESMFOLD_INTEGRATION_PROJECT_PLAN.md` | Stage-2 NN predictor plan |
| `seq2svm\EXECUTIVE_SUMMARY.md`, `CODEBASE_ANALYSIS_SUMMARY.md`, `KEY_FINDINGS.txt` | Research summaries — paper-facing |
| `seq2svm\data\training_dataset\StaPep\HDC_HANDOFF_BRIEF.md` | **Authoritative for the StaPep dataset** |
| `seq2svm\10ns vs 50ns comparison\README.md` | Frozen narrative for the dataset-swap benchmark |
| `seq2svm\models\README.md` | ESMFold script usage |
| `seq2svm\archive\README.md` | Provenance for the 6 archived Buforin scripts (see §10) |
| `stapep_package\README.md` | StaPep library install + API |

## 10. Archived scripts (Phase 4, 2026-07-06)

Six top-level Buforin analysis scripts were removed from the working tree during the
Phase-3 folder reorganization **without a subfolder home**. Rather than delete them
(REFACTOR_PLAN.md Phase 4: "archive after approval, never delete"), they were restored
from git `HEAD` (`b12974a`) into `seq2svm\archive\` and frozen. They are **not** on the
live code path and **not** covered by smoke tests. Full per-file provenance and revival
instructions are in `seq2svm\archive\README.md`.

| Archived file | Status |
|---|---|
| `predict_buf_hemolysis.py` | v1 hemolysis RF — superseded by `_v2` |
| `predict_buf_hemolysis_v2.py` | 18-feature hemolysis RF — un-migrated (distinct from `regression\predict_hemolysis_regression.py`, 14-feat) |
| `predict_buf_hemolysis_lyticity.py` | lyticity-only parametric hemolysis model — distinct experiment |
| `predict_buf_pmic_v2.py` | 18-feature pMIC RF — un-migrated (original: `regression\predict_pmic_regression.py`) |
| `plot_hydrophobic_vs_mic.py` | figure (hydrophobic SASA/KD index vs MIC); generated `hydrophobic_vs_mic.png` (§5) |
| `plot_lyticity_vs_hemolysis.py` | figure (lyticity vs hemolysis); distinct from `figures\lyticity_vs_mic.py` |
