# RUNNABLE_FILES.md

Inventory of every `.py` file under `C:\Users\bioin\Documents\SVM_ESM_Peptides\`. Started by Phase-0 read-only discovery; Phase-2 + first Phase-3 slice in progress (see §10). Paths are relative to `SVM_ESM_Peptides\`. Abbreviations:

- `seq2svm\` = `Peptide-Anti-microbial-Properties-Prediction\sequence_to_svm_minimal\`
- `pretrained_svm\` = `Peptide-Anti-microbial-Properties-Prediction\pretrained_svm\sequence_to_svm_minimal\` (frozen legacy snapshot)
- `stapep_pkg\` = `stapep_package\stapep\`

## 1. Summary counts (137 `.py` files total)

| Classification | Count | Notes |
|---|---:|---|
| core library | 28 | Importable, no runnable `__main__`. Heaviest concentration in `stapep_pkg\` and `seq2svm\gnn\nn_pipeline\features\`. |
| CLI script | 9 | argparse-driven; smoke-testable via `--help`. |
| experiment | 36 | Hardcoded analysis or training runs; usually no argparse. |
| data generation | 11 | Writes a CSV. Several overwrite canonical inputs (flagged below). |
| expensive/manual | 8 | MD, ESMFold batch, full GNN training. Hours–days. |
| legacy/reference | 22 | `pretrained_svm\` duplicate tree + `stapep_package\build\lib\` mirror. |
| broken/unknown | 23 | 15 empty TODO stubs, 2 Python-2 syntax-error files, 5 data-dependent debug scripts, 1 outdated example. |

## 2. Environments at a glance

| Env name | Where defined | Used by |
|---|---|---|
| `skl_legacy` (Conda, sklearn 0.19.2, Py 3.7) | `seq2svm\skl_legacy_env.yml` | Pre-trained SVM only (`scripts\run_sequence_svm.py`, `predictionsParameters\predictSVC.py`, `descriptors\descripGen_12_py3.py`) |
| `esm_env` (Conda, sklearn modern + fair-esm + CUDA torch, Py 3.10) | `seq2svm\esm_env.yml` | ESMFold scripts + most modern sklearn/RF/MLP/SVM scripts (svm/, mlp/, regression/, comparison/) |
| `stap` (Conda, WSL: OpenMM, AmberTools, pytraj, parmed) | `stapep_package\setup.py` + WSL conda manual setup | All MD scripts (`run_buforin_stapep.py`, `run_buf_variants_stapep.py`, `StaPep\run_amp_md_features.py`, `StaPep\run_test_stapep_md.py`, `feature_extraction\generate_stapep_structures.py`, anything importing `stapep`) |
| `venv` (WSL, torch + torch_geometric) | `seq2svm\requirements.txt` | GNN pipeline (`gnn\*`), PyTorch MLP (`nn_pipeline\*`, `mlp\run_nn_training*.py`), `comparison\run_feature_fusion_experiments.py` |

**Env conflict:** `pretrained_svm\sequence_to_svm_minimal\skl_legacy_env.yml` hardcodes `prefix: /home/salaars/miniconda3/...` — only the `seq2svm\skl_legacy_env.yml` copy is portable.

## 3. CLI scripts — smoke-testable via `--help`

| File | Env | Inputs | Outputs | Runtime | Smoke safety |
|---|---|---|---|---|---|
| `seq2svm\scripts\run_sequence_svm.py` | `skl_legacy` | `seqs.txt`, `aaindex/`, `svc.pkl`, `Z_score_...csv` | `descriptors.csv`, `*_PREDICTIONS.csv` | sec–min | safe: `--help` only |
| `seq2svm\scripts\make_seqs_windows.py` | system python | raw AA text | `seqs.txt` (2-col) | sec | safe: `--help` only |
| `seq2svm\descriptors\descripGen_12_py3.py` | `skl_legacy` | aaindex dir, seqs.txt | `descriptors.csv` | sec–min/100 seqs | safe: `--help` only (positional args; prints usage when bare) |
| `seq2svm\predict_mic_single.py` | `esm_env` | StaPep features CSV | stdout | sec | safe: `--help` only |
| `seq2svm\feature_extraction\build_geometric_features.py` | `esm_env` | PDB dir | geometric features CSV | minutes | safe: `--help` only |
| `seq2svm\gnn\run_gnn_training.py` | `venv` (WSL+GPU) | geometric features CSV, PDB dir | `.pt` checkpoints + JSON | minutes–hours | safe: `--help` only |
| `seq2svm\nn_pipeline\prepare_clusters.py` | `esm_env` | feature CSV | FASTA + clustered CSV | sec | safe: `--help` only |
| `seq2svm\nn_pipeline\train.py` | `venv` | feature CSV | `.pt` checkpoint | minutes–hours | safe: `--help` only (argparse per docstring; verify on first run) |
| `seq2svm\models\esm_sequence_processor.py` | `esm_env` (GPU) | seqs.txt | PDB dir + embeddings CSV | hours | safe: `--help` only — **never invoke `main()` in tests** |
| `seq2svm\models\batch_esmfold.py` | `esm_env` (GPU) | seqs.txt | PDB dir + checkpoint JSON | hours–days | safe: `--help` only — **never invoke `main()`** |
| `seq2svm\models\run_esmfold_peptides.py` | `esm_env` (GPU) | AMP/Decoy seq files | PDB dir + results CSV | hours | safe: `--help` only — **never invoke `main()`** |
| `seq2svm\models\extract_structure_features.py` | `esm_env` | PDB dir | features CSV | minutes | safe: `--help` only |
| `seq2svm\mlp\run_stapep_mlp.py` | `esm_env` | StaPep features CSVs | stdout + CV metrics | minutes | safe: `--help` only |
| `seq2svm\mlp\run_nn_training_pnas_style.py` | `venv` | feature CSVs | `.pt` checkpoints + metrics | minutes–hours | safe: `--help` only — **don't run `main()`** |
| `seq2svm\svm\run_stapep_svm.py` | `esm_env` | StaPep features CSVs | stdout + CV metrics | minutes | safe: `--help` only |
| `seq2svm\svm\run_combined_svm.py` | `esm_env` | StaPep + QSAR CSVs | stdout | minutes | safe: `--help` only |
| `seq2svm\figures\plot_mic_distribution.py` | `esm_env` | `stapled_amps.csv` | PNG | sec | safe: `--help` only (uses `--save`) |
| `seq2svm\comparison\run_feature_fusion_experiments.py` | `venv` | geometric + qsar CSVs | `results\fusion_experiments_*.json/csv` | min–hr | safe: `--help` only |
| `seq2svm\regression\predict_pmic_regression.py` | `esm_env` | StaPep features CSV | PNG | sec–min | safe: `--help` only |
| `seq2svm\regression\predict_pmic_all_organisms.py` | `esm_env` | StaPep features CSV | PNGs in `pmic_by_organism\` | minutes | safe: `--help` only |
| `run_buforin_stapep.py` (root) | `stap` (WSL) | single sequence on CLI | stdout features; `/tmp/stapep_<id>/` | 15–30 min | unsafe: needs WSL+OpenMM |

## 4. Experiment scripts — runnable but no argparse (import-only or invoke directly)

These scripts have hardcoded paths/sequences and usually **do not** have `--help`. Many read CSVs at module top level — flagged where verified.

| File | Env | What it does | Smoke safety |
|---|---|---|---|
| `seq2svm\svm\predict_mic_svm.py` | `esm_env` | RBF-SVC training + Buforin variant predictions + figures | **unsafe to import** (sets `sys.stdout` at module level); needs CSVs |
| `seq2svm\svm\run_mic_svm.py` | `esm_env` | 3-tier MIC GridSearchCV-SVM | unknown — module-level reads possible |
| `seq2svm\svm\run_pretrained_svm_inference.py` | `esm_env` (uses numpy sidecars to bypass legacy sklearn) | Pretrained SVM inference on 8 test peptides | unknown — module-level numpy loads |
| `seq2svm\svm\run_pretrained_svm_low_loop.py` | `esm_env` | Ablation of pretrained SVM with low-loop subset | unknown |
| `seq2svm\svm\run_stapep_svm_no_loop.py` | `esm_env` | Retrained StaPep SVM without `loop_percent` | unknown |
| `seq2svm\mlp\predict_mic_mlp.py` | `esm_env` | 20-MLP ensemble, isotonic calibration | unknown |
| `seq2svm\mlp\run_mic_mlp.py` | `esm_env` | 3-tier MIC GridSearchCV-MLP | unknown |
| `seq2svm\mlp\run_nn_training.py` | `venv` | Cluster-CV PyTorch MLP | **don't run; long** |
| `seq2svm\regression\predict_hemolysis_regression.py` | `esm_env` | RF hemolysis regression + Buf/Mag variant preds | unknown |
| `seq2svm\regression\predict_pmic_stapled_variants.py` | `esm_env` | RF pMIC regression with 18 features on Buforin variants | unknown |
| `seq2svm\regression\run_mic_classifier.py` | `esm_env` | 4-class MIC RF | unknown |
| `seq2svm\regression\run_mic_rf.py` | `esm_env` | 4-class MIC RF comparing 3 feature sets | unknown |
| `seq2svm\figures\lyticity_vs_mic.py` | `esm_env` | 3-panel lyticity vs MIC figure | unknown |
| `seq2svm\comparison\compare_anomalous_features.py` | `esm_env` | Anomalous-feature deviation analysis | **unsafe to import** — reads CSV at module level; will crash without data |
| `seq2svm\comparison\compare_buf_pmic.py` | `esm_env` | RF pMIC literature vs predicted Buf variants | unknown |
| `seq2svm\comparison\compare_ngc_scores.py` | `esm_env` | SVM/MLP score vs NGC induction correlation | unknown |
| `seq2svm\gnn\run_gnn_comparison.py` | `venv` (WSL+GPU) | GCN/GAT/EGNN × 3 feature sets, 5-fold CV | **don't run; long** |
| `seq2svm\gnn\run_stapep_gnn_comparison.py` | `venv` (WSL+GPU) | GNN comparison on stapled dataset | **don't run; long** |
| `seq2svm\gnn\predict_gcn_single.py` | `venv` (WSL+GPU) | Single GCN trained on full StaPep | **don't run; long** |
| `seq2svm\gnn\predict_stapep_candidates.py` | `venv` (WSL+GPU) | GNN ensemble + candidate predictions | **don't run; long** |
| `seq2svm\debug_checks\check_pyg.py` | `venv` | torch_geometric importable? | safe: import-only |
| `seq2svm\debug_checks\check_feature_overlap.py` | `esm_env` | QSAR-12 vs Geo-24 name overlap | data-dependent |
| `seq2svm\debug_checks\check_feature_overlap_detailed.py` | `esm_env` | Deep semantic overlap analysis | data-dependent |
| `seq2svm\debug_checks\check_features.py` | `esm_env` | NaN/Inf check on `geometric_features.csv` | data-dependent |
| `seq2svm\debug_checks\check_features_fixed.py` | `esm_env` | NaN/Inf check on `geometric_features_fixed.csv` | data-dependent |
| `seq2svm\debug_checks\debug_secondary_structure.py` | `esm_env` | phi/psi inspection from PDB | data-dependent |
| `seq2svm\debug_checks\test_esmfold_quick.py` | `esm_env` (GPU) | ESMFold one-shot test | **unsafe: needs GPU** |
| `seq2svm\models\test_gpu_esmfold.py` | `esm_env` (GPU) | GPU benchmark + ESMFold validation | **never run in CI — uses `input()`; will hang** |
| `seq2svm\models\check_cache.py` | system python | Inspect `~/.cache/torch/hub` | safe |
| `seq2svm\models\diagnose_model.py` | `esm_env` | ESMFold file size + MD5 | safe |
| `seq2svm\models\fix_hf_cache.py` | system python | Move aria2 download into HF cache layout | safe |
| `seq2svm\StaPep diagnostic scripts (`_check_*.py`, `_inspect_md50ns_in_progress.py`, `sanity_check_md_features.py`, etc.)` | `esm_env` (pandas only) | Sanity / coverage reports | safe if CSVs present; never modify data |

## 5. Data-generation scripts (write CSVs)

| File | Env | Writes | Overwrites canonical? |
|---|---|---|---|
| `seq2svm\data\training_dataset\StaPep\extract_amp_features.py` | `esm_env` | `stapled_amps_features.csv` | **YES — canonical** |
| `seq2svm\data\training_dataset\StaPep\_drop_dramp21556.py` | `esm_env` | overwrites `..._XZ_only.csv` + features + audit log | **YES — canonical** |
| `seq2svm\data\training_dataset\StaPep\combine_stapled_paper_dataset.py` | `esm_env` | `stapled_amps_combined_paper_dataset*.csv` | YES — canonical (reference dataset) |
| `seq2svm\data\training_dataset\StaPep\filter_drop_k_stapled.py` | `esm_env` | `..._XZ_only.csv` + `dropped_k_stapled.csv` | YES — canonical |
| `seq2svm\data\training_dataset\StaPep\build_paper_supplement_mag115.py` | `esm_env` | `stapled_amps_paper_supplement.csv` | YES — canonical |
| `seq2svm\data\training_dataset\StaPep\convert_to_txt.py` | `esm_env` | `seqs_AMP_stapep.txt`, `seqs_DECOY_stapep.txt` | regenerable txt |
| `seq2svm\data\training_dataset\StaPep\_build_replicate_subset.py` | `esm_env` | `replicate_subset.csv` | small derivative |
| `seq2svm\feature_extraction\build_stapep_features.py` | `esm_env` (biopython, propy) | `stapep_amp_geometric.csv`, `stapep_decoy_geometric.csv`, `stapep_*_qsar.csv` | YES — canonical |
| `seq2svm\feature_extraction\build_stapep_geo.py` | `esm_env` | `stapep_amp_geometric.csv`, `stapep_decoy_geometric.csv`, `test_stapled_geometric.csv` | YES — canonical |
| `seq2svm\feature_extraction\extract_stapep_qsar.py` | `esm_env` | `qsar_stapled_amps.csv`, `qsar_stapled_decoys.csv`, `qsar_stapled_test.csv` | YES — canonical |
| `seq2svm\comparison\run_dataset_comparison.py` | `esm_env` | logs/PNGs in `10ns vs 50ns comparison\` **AND swaps `stapled_amps_features.csv` in place** | **CRITICAL LANDMINE** (see §7) |

## 6. Expensive / manual (hours–days; never invoke in tests)

| File | Env | Why |
|---|---|---|
| `seq2svm\data\training_dataset\StaPep\run_amp_md_features.py` | `stap` (WSL) | 50 ns OpenMM MD × ~130–188 peptides → hours–days |
| `seq2svm\data\training_dataset\StaPep\run_test_stapep_md.py` | `stap` (WSL) | 50 ns MD × 8 peptides → hours |
| `seq2svm\feature_extraction\generate_stapep_structures.py` | `esm_env` (GPU) | ESMFold × 541 stapled sequences → hours |
| `seq2svm\models\batch_esmfold.py` | `esm_env` (GPU) | ESMFold × ~18K sequences per docstring → days |
| `seq2svm\models\run_esmfold_peptides.py` | `esm_env` (GPU) | ESMFold on 572-peptide dataset → hours |
| `seq2svm\models\download_esmfold.py` / `download_esmfold_simple.py` / `download_from_huggingface.py` | `esm_env` | ~15 GB download from HuggingFace |
| `run_buf_variants_stapep.py` (root) | `stap` (WSL) | 7 × 10 ns MD on Buforin variants → ~1.5–2 h |
| `seq2svm\data\training_dataset\StaPep\_probe_dssp_staples.py` | `stap` (WSL) | pytraj DSSP probe — minutes per peptide but WSL-only |

## 7. Broken / unknown files (do NOT auto-import)

### 7a. Empty `# TODO` stubs (15 files)
Safe to import (no body), but functionally inert. Will need to be implemented or deleted.
- `seq2svm\cli\extract_features.py`, `predict_mic.py`, `train_model.py`
- `seq2svm\features\feature_extractor.py`, `feature_utils.py`
- `seq2svm\utils\caching.py`, `data_prep.py`, `evaluation.py`
- `seq2svm\structure\esmfold_predictor.py`, `pdb_parser.py`, `structure_features.py`
- `seq2svm\models\mic_predictor.py`, `train_mic.py`

### 7b. Python 2 syntax (cannot import on Py3)
- `seq2svm\predictionsParameters\seqWindowConstructor.py` — `print` statements
- `pretrained_svm\predictionsParameters\seqWindowConstructor.py` — same file

### 7c. Outdated example
- `stapep_pkg\example\predictor.py` — uses deprecated `alphafold=True` kwarg of `PrepareProt()`; current API uses `method=`. Would throw `TypeError`.

### 7d. Hardcoded personal paths
- `seq2svm\data\training_dataset\StaPep\_paper_vs_dramp_compare.py` — refs `C:/Users/bioin/Downloads/Stapled Peptide Data Extraction - ....csv`. Cannot run elsewhere.
- `seq2svm\data\training_dataset\StaPep\run_test_stapep_md.py` line 21 — hardcoded `/mnt/c/Users/bioin/...` (live code, not docstring; will break if username/mount differs)

### 7e. Data-dependent debug scripts (need CSVs/PDBs the agent did not verify exist)
- All of `seq2svm\debug_checks\` except `check_pyg.py` (see §4).
- `seq2svm\comparison\compare_anomalous_features.py` — reads `stapled_amps_features.csv` at **module level**, not inside `if __name__`. Import-only tests would crash on missing data.

## 8. Critical landmines

1. **`comparison\run_dataset_comparison.py` swaps the canonical CSV in-place.** Sequence: backup `stapled_amps_features.csv` → swap `stapled_amps_features_training_XZ_md50ns.csv` into that filename → run 6 scripts → restore. A crash between lines 156–171 leaves the 50ns data living in the legacy filename. Eight other scripts hard-code that filename and will silently see whichever version is there.
2. **`pretrained_svm\sequence_to_svm_minimal\`** is a frozen snapshot duplicate. Any refactor touching shared filenames must be applied to both trees, or use `--ignore` to keep the snapshot frozen.
3. **`skl_legacy_env.yml` is load-bearing.** The pre-trained `svc.pkl` was pickled with sklearn 0.19.x; loading under modern sklearn silently miscomputes or crashes. (`svm\run_pretrained_svm_inference.py` works around this by reading `.npy` sidecars.)
4. **WSL path hardcodes** in live code: `StaPep\run_test_stapep_md.py:21`, `StaPep\_paper_vs_dramp_compare.py`, the top-level `run_*_stapep.py` driver docstrings.
5. **`stapled_amps_features_training_XZ_md50ns.csv` is THE canonical training input** per the HDC handoff brief — it must never be overwritten without a new MD run.

## 9. Duplication patterns to address in Phase 2

| Pattern | Files where it appears | Suggested helper |
|---|---|---|
| `STAPEP_COLS` 17-element list | `svm\run_stapep_svm.py`, `mlp\run_stapep_mlp.py`, `svm\run_combined_svm.py`, `comparison\compare_ngc_scores.py`, `comparison\run_feature_fusion_experiments.py`, `svm\predict_mic_svm.py`, `regression\run_mic_rf.py`, `svm\run_stapep_svm_no_loop.py` | central `features\stapep_columns.py` |
| `QSAR_COLS` 12-element list | same files as above | same module |
| E. coli MIC regex block | `regression\predict_pmic_regression.py`, `predict_pmic_stapled_variants.py`, `predict_pmic_all_organisms.py`, `figures\lyticity_vs_mic.py`, `regression\run_mic_classifier.py`, `comparison\compare_buf_pmic.py` | `data\mic_parser.py` (only after confirming all callers agree on edge cases — see §High-risk in REFACTOR_PLAN) |
| `Path(__file__).resolve().parent.parent` to find project root | 15+ files | `utils\paths.py` with `BASE`, `STAPEP_DIR`, etc. |
| Hardcoded `BUF_WT_FEATURES` / `SP_FEATURES` numeric blocks for native Buforin | `predict_pmic_regression.py`, `predict_pmic_stapled_variants.py`, `predict_mic_svm.py`, `compare_ngc_scores.py`, `compare_buf_pmic.py`, `predict_hemolysis_regression.py` | `data\reference_peptides.py` |
| Identical GCN-on-StaPep-candidates body | `gnn\predict_gcn_single.py` ≈ `gnn\predict_stapep_candidates.py` (differ only in ensemble depth) | merge with `--ensemble-size` flag |
| Duplicated tree | `seq2svm\descriptors\descripGen_12_py3.py` ↔ `pretrained_svm\…` (byte-for-byte). Same for `predictSVC.py`, `make_seqs_windows.py`, `run_sequence_svm.py` | leave the snapshot frozen; only edit live copies |

## 10. Refactor progress

### Slice 1+2: QSAR/SVM — COMPLETE (2026-05-27)

**Shared modules created:**
- `seq2svm\utils\paths.py` — `PROJECT_ROOT`, `STAPEP_DIR`, `PRETRAINED_PARAMS_DIR`, `PRETRAINED_SVC_PKL`, `PRETRAINED_ZSCORE_CSV`, etc. Replaces 15+ instances of `Path(__file__).resolve().parent.parent`.
- `seq2svm\features\stapep_columns.py` — `STAPEP_COLS` (17), `STAPEP_COLS_WITH_HSASA` (18), `QSAR_COLS` (12), `TEST_NAMES`, `QSAR_TEST_NAME_MAP`, `SVM_PARAM_GRID`. Replaces 8+ verbatim copies.
- `seq2svm\utils\pretrained_svm.py` — `PretrainedSVM` dataclass + `load_pretrained_svm()` factory. Encapsulates the verified `svc.pkl_NN.npy` sidecar layout (`pkl_03 → dual_coef`, `pkl_07 → support_vectors`, `pkl_10 → intercept = -0.01187876`, `pkl_04 → probA = -3.29162142`, `pkl_11 → probB = +0.03014156`), Z-score CSV loading, linear-kernel `w = SV.T @ dual_coef[0]` reconstruction, and Platt-scaled probability. Replaces the verbatim copy of this loader that lived in three SVM scripts and the sklearn-0.19.x boundary handling that they each tried independently.

**Scripts migrated to shared modules (all 7 svm/ scripts done):**
- `seq2svm\svm\run_stapep_svm.py` — `STAPEP_DIR`, `STAPEP_COLS`, `QSAR_COLS`, `TEST_NAMES`, `QSAR_TEST_NAME_MAP`, `SVM_PARAM_GRID`.
- `seq2svm\svm\run_combined_svm.py` — same set + `load_pretrained_svm`. Inline `pretrained_qsar_inference` body collapsed from ~46 lines to 12; unused `joblib` and `os` imports removed implicitly.
- `seq2svm\svm\run_mic_svm.py` — `STAPEP_DIR`, `STAPEP_COLS`, `TEST_NAMES`, `SVM_PARAM_GRID`.
- `seq2svm\svm\run_stapep_svm_no_loop.py` — **also fixed a pre-existing path bug** where `os.path.dirname(os.path.abspath(__file__))` resolved BASE to `svm/` instead of project root (so `data/training_dataset/StaPep/...` was being looked up at `svm/data/...`). Now uses `STAPEP_DIR`.
- `seq2svm\svm\predict_mic_svm.py` — `STAPEP_DIR` + `STAPEP_COLS_WITH_HSASA`. Module-level `sys.stdout = io.TextIOWrapper(...)` hack moved inside `main()` so the module is now import-safe (regression-tested via `test_import_svm_predict_mic_svm`). MIC regex kept inline per PROJECT_MAP §7e high-risk rule.
- `seq2svm\svm\run_pretrained_svm_inference.py` — same path-bug fix as `no_loop`; body wrapped in `main()` + `if __name__ == "__main__":` so import-safe; pretrained loader replaced with `load_pretrained_svm()`; sklearn-0.19.x pickle now read only for diagnostic printing of `kernel` and `C` and falls back safely if it raises.
- `seq2svm\svm\run_pretrained_svm_low_loop.py` — same treatment; `run_svm()` helper now takes a `PretrainedSVM` argument instead of bare module-level globals.

**Smoke-test harness:**
- `seq2svm\pytest.ini`, `seq2svm\tests\conftest.py` (skip markers for `torch`/`torch_geometric`/`biopython`/`esm`/`sklearn`/Windows).
- `seq2svm\tests\test_imports.py` — 11 tests: shared modules + all 7 SVM scripts + a regression assertion that `predict_mic_svm` does NOT clobber `sys.stdout` on import + a behavioral check that `load_pretrained_svm()` reconstructs the verified intercept/probA/probB constants + `features.geometric_features` import.
- `seq2svm\tests\test_entrypoints.py` — 3 `--help` subprocess tests for the argparse-driven SVM scripts.

**Verification status (2026-05-27):**
- `conda run -n esm_env python -m pytest tests/ -v` → **17 passed in 3.24 s, 0 failed, 0 skipped**.
- Includes `test_pretrained_svm_loader_reproduces_known_constants` which asserts the .npy sidecar reconstruction matches the values cross-checked in the original `run_pretrained_svm_inference.py` docstring.

**Cross-folder cleanup still pending (out of this slice):**
- `STAPEP_COLS` / `QSAR_COLS` still copy-pasted in `seq2svm\mlp\run_stapep_mlp.py`, `seq2svm\comparison\compare_ngc_scores.py`, `seq2svm\comparison\run_feature_fusion_experiments.py`, `seq2svm\regression\run_mic_rf.py`. Migrate as those slices come up.
- MIC parsing regex still duplicated across `regression\` and `comparison\` files and the inline copy in `svm\predict_mic_svm.py`. Centralize only after the regression slice when callers can be cross-validated.

**pytest:** `pytest 9.0.3` installed into `esm_env` on 2026-05-27. Other envs (`base`, `stap`, `skl_legacy`, `VAE_env`) still lack it; install per-env on demand as later slices reach into them.

### Slice 3: MLP — COMPLETE (2026-05-27)

**Scripts migrated (all 5 mlp/ scripts done):**
- `seq2svm\mlp\run_stapep_mlp.py` — uses `STAPEP_DIR`, `STAPEP_COLS`, `QSAR_COLS`, `TEST_NAMES`, `QSAR_TEST_NAME_MAP`. (Direct counterpart of `svm/run_stapep_svm.py`; same migration.)
- `seq2svm\mlp\run_mic_mlp.py` — uses `STAPEP_DIR`, `STAPEP_COLS as FEATURES`, `TEST_NAMES`. Script-local `BINS` / `TIER_LABELS` / `SHORT` / `MLP_GRID` kept inline; MIC regex kept inline (high-risk per PROJECT_MAP §7e).
- `seq2svm\mlp\predict_mic_mlp.py` — `STAPEP_DIR` + `STAPEP_COLS_WITH_HSASA`. Module-level `sys.stdout = io.TextIOWrapper(...)` hack moved inside `main()` so the module is import-safe (regression-tested via `test_import_mlp_predict_mic_mlp`). MIC regex + `BUF_WT_FEATURES` + `LITERATURE_MIC` kept inline.
- `seq2svm\mlp\run_nn_training.py` — uses `PROJECT_ROOT` + `DATA_DIR` to replace inline `Path(__file__).resolve().parent.parent / "data" / "training_dataset"` blocks. PyTorch only; no shared column constants to extract.
- `seq2svm\mlp\run_nn_training_pnas_style.py` — uses `DATA_DIR` for the `--data` argparse default and `RESULTS_DIR` for the default output folder. PyTorch only.

**Smoke tests added:**
- 5 import tests (`mlp.run_stapep_mlp`, `mlp.run_mic_mlp`, `mlp.predict_mic_mlp` + sys.stdout regression assertion, `mlp.run_nn_training` (`@requires_torch`), `mlp.run_nn_training_pnas_style` (`@requires_torch`)).
- 3 `--help` subprocess tests (`run_stapep_mlp.py`, `run_mic_mlp.py`, `run_nn_training_pnas_style.py`).
- Default `_help()` subprocess timeout raised from 15 s → 60 s to accommodate torch cold-start in fresh subprocesses (~10-20 s on Windows when OS file cache is cold).

**Verification status (2026-05-27):**
- `conda run -n esm_env python -m pytest tests/ -v` → **25 passed in 7.25 s, 0 failed, 0 skipped**.

**Cross-folder cleanup still pending (out of this slice):**
- `STAPEP_COLS` / `QSAR_COLS` still copy-pasted in `seq2svm\comparison\compare_ngc_scores.py`, `seq2svm\comparison\run_feature_fusion_experiments.py`. Migrate as the comparison slice comes up.
- MIC parsing regex still duplicated across `regression\`, `comparison\`, and inline in `svm\predict_mic_svm.py` and `mlp\predict_mic_mlp.py` (5 distinct variants). Centralize only after the comparison slice when all callers can be cross-validated.

### Slice 4: regression — COMPLETE (2026-05-27)

**New shared modules:**
- `seq2svm\utils\mic_units.py` — pure log10 conversions: `pmic_to_mic_uM`, `mic_to_pmic_uM`, `pmic_to_mic_ugml`, `mic_to_pmic_ugml`. Replaces inline copies in 5 scripts (3 regression + svm + mlp). Math-only — does NOT parse free-text MIC fields.
- `seq2svm\features\reference_peptides.py` — `LITERATURE_MIC_ECOLI` dict (11 Buf variants, μg/mL + MW). Verified byte-identical across `regression\predict_pmic_stapled_variants.py`, `svm\predict_mic_svm.py`, `mlp\predict_mic_mlp.py`.
- New constant `STAPEP_COLS_PAPER_14` in `features\stapep_columns.py` — 14-feature subset matching StaPep paper Fig. 6 (= `STAPEP_COLS` minus `lyticity_index`, `sheet_percent`, `sasa`). Used by 3 regression scripts.

**Scripts migrated (all 6 regression/ scripts done):**
- `seq2svm\regression\predict_pmic_regression.py` — uses `STAPEP_DIR`, `STAPEP_COLS_PAPER_14`, `pmic_to_mic_uM`. Inline 14-col `FEATURE_COLS` + `pmic_to_mic_uM` helper removed.
- `seq2svm\regression\predict_pmic_all_organisms.py` — same set + `mic_to_pmic_uM`. Inline `pmic_to_mic` / `mic_to_pmic` helpers removed.
- `seq2svm\regression\predict_hemolysis_regression.py` — uses `STAPEP_DIR`, `STAPEP_COLS_PAPER_14`. Inline 14-col `FEATURE_COLS` removed.
- `seq2svm\regression\predict_pmic_stapled_variants.py` — uses `STAPEP_DIR`, `STAPEP_COLS_WITH_HSASA`, `mic_to_pmic_ugml`, `pmic_to_mic_ugml`, `LITERATURE_MIC_ECOLI`. `sys.stdout` hack moved into `main()` (regression-tested via `test_import_regression_predict_pmic_stapled_variants`).
- `seq2svm\regression\run_mic_classifier.py` — uses `STAPEP_DIR`, `STAPEP_COLS`. (MIC regex kept inline — variant D, different from the other 4 variants.)
- `seq2svm\regression\run_mic_rf.py` — uses `STAPEP_DIR`, `STAPEP_COLS as SP_COLS`, `QSAR_COLS`, `TEST_NAMES as TEST_ORDER`, `QSAR_TEST_NAME_MAP`. (MIC regex kept inline.)

**Cross-slice consolidation:**
- `seq2svm\svm\predict_mic_svm.py` (slice 2) — now also imports `mic_to_pmic_ugml` and `pmic_to_mic_ugml` from `utils.mic_units` + `LITERATURE_MIC_ECOLI` from `features.reference_peptides`. Inline copies removed.
- `seq2svm\mlp\predict_mic_mlp.py` (slice 3) — same treatment. Inline `LITERATURE_MIC` dict and `mic_to_pmic` helper removed.

**MIC parser status (5 distinct variants identified, all kept inline — see PROJECT_MAP §7e):**
| Variant | Files using it | Behavior |
|---|---|---|
| A | `regression\predict_pmic_regression.py` | Returns μM; handles `>` `<` censored values as None |
| B | `regression\predict_pmic_all_organisms.py` | Parameterized per organism; returns μM |
| C | `regression\predict_pmic_stapled_variants.py`, `svm\predict_mic_svm.py`, `mlp\predict_mic_mlp.py` | Returns μg/mL with MW conversion; has fallback patterns; handles `≥` |
| D | `regression\run_mic_classifier.py`, `regression\run_mic_rf.py`, `mlp\run_mic_mlp.py` | Requires `=` after MIC; returns raw value + unit (no fallback, no censored handling) |

The five variants have different censored-value handling, different unit prioritization (μM vs μg/mL primary returns), different match strictness (require `=` vs not), and different fallback strategies. Centralization deferred per the high-risk rule — needs a separate slice with side-by-side numerical comparison on the full canonical dataset.

**Native peptide feature dict status:**
- `BUF_WT_FEATURES` (Buforin native, 18-col) — kept inline in `regression\predict_pmic_stapled_variants.py`, `svm\predict_mic_svm.py`, `mlp\predict_mic_mlp.py`.
- 14-col `NATIVE` dict (Buforin II + Magainin II) — kept inline in `regression\predict_pmic_regression.py`, `predict_pmic_all_organisms.py`, `predict_hemolysis_regression.py`.
- These exist at TWO precision tiers (rounded to 5 dp vs full numpy float precision). Unifying would silently change ML inputs. Defer to a separate slice with experimental validation.

**Smoke tests added:**
- 6 import tests (`regression.predict_pmic_regression`, `predict_pmic_all_organisms`, `predict_hemolysis_regression`, `predict_pmic_stapled_variants` + sys.stdout regression assertion, `run_mic_classifier`, `run_mic_rf`).
- 3 `--help` subprocess tests (`predict_pmic_regression.py`, `predict_pmic_all_organisms.py`, `predict_hemolysis_regression.py`).
- 3 behavioral tests for the new shared modules: `test_mic_units_roundtrip`, `test_reference_peptides_literature_mic`, `test_stapep_columns_paper_14_is_strict_subset`.

**Verification status (2026-05-27):**
- `conda run -n esm_env python -m pytest tests/ -v` → **37 passed in 10.00 s, 0 failed, 0 skipped**.

### Slice 5: comparison — COMPLETE (2026-05-27)

**Scripts migrated (all 5 comparison/ scripts done):**
- `seq2svm\comparison\compare_anomalous_features.py` — uses `STAPEP_DIR`. **Module-level execution wrapped in `main()`** so the module is now import-safe (was previously reading 3 CSVs at module level with a hardcoded `platform.system()` path switch, which Phase-0 flagged as a key blocker for any test that touched it). Removed the platform-branched hardcoded path entirely.
- `seq2svm\comparison\compare_buf_pmic.py` — uses `STAPEP_DIR`, `STAPEP_COLS_PAPER_14`, `mic_to_pmic_ugml`, `pmic_to_mic_uM`. ADVISOR_FEAT path now resolves via `PROJECT_ROOT.parent.parent` instead of inline relative joins. Inline MIC regex kept inline — it's variant E (yet another variant: `(?:MIC(?:99\.9)?(?:[\d.]*)?\s*[=≥>]?\s*([\d.]+)\s*([μu]g/mL|[μu]M))`), distinct from variants A–D in the regression slice.
- `seq2svm\comparison\compare_ngc_scores.py` — uses `STAPEP_DIR`, `STAPEP_COLS`, `QSAR_COLS`, `load_pretrained_svm`. Inline `_npy` helper + `pretrained_qsar_scores` body collapsed from ~22 lines to 8 by delegating to `utils.pretrained_svm.PretrainedSVM`. Hardcoded `SP_FEATURES` and `QSAR_FEATURES` dicts (full-precision per-variant feature dicts for Buf WT / Buf 12 / Buf 13 / Buf Q9K) kept inline — same precision-tier concern as `BUF_WT_FEATURES`.
- `seq2svm\comparison\run_dataset_comparison.py` — uses `PROJECT_ROOT`, `STAPEP_DIR as DATA_DIR`. CSV-swap semantics preserved unchanged (still `shutil.copy2(LEGACY_CSV, BACKUP_CSV)` → swap → restore in a try/finally). **Permanently skipped in `test_entrypoints.py`** — the swap leaves the canonical dataset corrupted if interrupted (PROJECT_MAP.md §7f landmine).
- `seq2svm\comparison\run_feature_fusion_experiments.py` — uses `DATA_DIR`, `RESULTS_DIR`, `QSAR_COLS`. The inline `compute_qsar12_descriptors` function and the script-specific `TwoTowerFusionMLP` architecture stay inline (one-of-a-kind utilities).

**Smoke tests added:**
- 5 import tests (`compare_anomalous_features` now import-safe — the biggest win this slice, since Phase-0 flagged it as unimportable; `compare_buf_pmic`, `compare_ngc_scores`, `run_dataset_comparison` import-only, `run_feature_fusion_experiments` `@requires_torch`).
- 1 `--help` subprocess test for `run_feature_fusion_experiments.py`.
- 1 PERMANENT skip marker for `run_dataset_comparison.py` with a clear reason citing PROJECT_MAP.md §7f and SMOKE_TESTS.md Part D.

**Cross-slice consolidation tracked across now-all-5 model folders:**
- `utils.paths` (`PROJECT_ROOT`, `STAPEP_DIR`, `DATA_DIR`, `RESULTS_DIR`, `PRETRAINED_PARAMS_DIR`, etc.) — used by all migrated svm/, mlp/, regression/, comparison/ scripts.
- `features.stapep_columns` (`STAPEP_COLS`, `STAPEP_COLS_WITH_HSASA`, `STAPEP_COLS_PAPER_14`, `QSAR_COLS`, `TEST_NAMES`, `QSAR_TEST_NAME_MAP`, `SVM_PARAM_GRID`) — used by all model-training scripts.
- `utils.pretrained_svm.load_pretrained_svm()` — used by 4 scripts (`svm/run_combined_svm.py`, `svm/run_pretrained_svm_inference.py`, `svm/run_pretrained_svm_low_loop.py`, `comparison/compare_ngc_scores.py`).
- `utils.mic_units` — used by 6 scripts (3 regression + svm/predict_mic_svm + mlp/predict_mic_mlp + comparison/compare_buf_pmic).
- `features.reference_peptides.LITERATURE_MIC_ECOLI` — used by 3 scripts (regression/predict_pmic_stapled_variants + svm/predict_mic_svm + mlp/predict_mic_mlp).

**Verification status (2026-05-27):**
- `conda run -n esm_env python -m pytest tests/ -v` → **43 passed, 1 skipped in 11.90 s, 0 failed**.
- The single skip is the permanent guard on `run_dataset_comparison.py` (intentional, documented).

### Slice 6: feature_extraction — COMPLETE (2026-05-27)

**Scripts migrated (all 5 feature_extraction/ scripts done):**
- `seq2svm\feature_extraction\build_geometric_features.py` — argparse CLI, takes paths as args; already had correct sys.path bootstrap. No constants to extract — operates on user-provided dirs. Verified import-safe and `--help` works.
- `seq2svm\feature_extraction\build_stapep_features.py` — uses `PROJECT_ROOT`/`STAPEP_DIR`/`AAINDEX_DIR`. Removed the duplicate `sys.path.insert(0, str(BASE))` (now handled by the top-level bootstrap).
- `seq2svm\feature_extraction\build_stapep_geo.py` — uses `STAPEP_DIR`. Inline 8-entry `TEST_NAMES = {1: "Buf12", ...}` dict replaced with `{i+1: name for i, name in enumerate(features.stapep_columns.TEST_NAMES)}` so the int-keyed mapping now derives from the canonical list — fixes drift risk.
- `seq2svm\feature_extraction\extract_stapep_qsar.py` — uses `PROJECT_ROOT`/`STAPEP_DIR`/`DESCRIPTORS_DIR`/`AAINDEX_DIR`. The QSAR-style `TEST_NAMES` dict (`"Buf(i+4)_12"` etc.) kept inline — it's the COUNTERPART vocabulary mapped by `QSAR_TEST_NAME_MAP` (not the same as the model-script `TEST_NAMES`). Added a docstring note clarifying the two-vocabulary bridge.
- `seq2svm\feature_extraction\generate_stapep_structures.py` — uses `STAPEP_DIR` in the `CONFIG` dict. Torch + transformers imports kept at module top (gated by `@requires_torch` in the test marker).

**Smoke tests added:**
- New `requires_propy` marker in `conftest.py` (propy is system-installed in `esm_env`; the marker is a defensive guard for other envs).
- 5 import tests:
  - `feature_extraction.build_geometric_features` (`@requires_bio`)
  - `feature_extraction.build_stapep_features` (`@requires_bio @requires_propy`)
  - `feature_extraction.build_stapep_geo` (`@requires_bio`)
  - `feature_extraction.extract_stapep_qsar` (`@requires_propy`)
  - `feature_extraction.generate_stapep_structures` (`@requires_torch`) — import-safe but `main()` is GPU+ESMFold-download expensive, never invoked in tests.
- 2 `--help` subprocess tests:
  - `feature_extraction/build_geometric_features.py` (`@requires_bio`)
  - `feature_extraction/generate_stapep_structures.py` (`@requires_torch`; `--help` exits before any model load, so safe even though `main()` would download ~15 GB).

**Output safety reminder:** All four data-generation scripts in this folder (`build_stapep_features`, `build_stapep_geo`, `extract_stapep_qsar`, `generate_stapep_structures`) write **canonical CSVs / PDB files** to `data/training_dataset/StaPep/`. Running them regenerates artifacts that previously took GPU-hours. Per PROJECT_MAP §3, the existing canonical files must not be casually overwritten.

**Verification status (2026-05-27):**
- `conda run -n esm_env python -m pytest tests/ -v` → **50 passed, 1 skipped in 13.27 s, 0 failed**.

**Folders not yet touched (in order of remaining work):**
- `gnn/` (8 files) — largest remaining cluster; mostly GPU+WSL experiments.
- `models/` (13 files) — ESMFold scripts + utility tools.
- `data/training_dataset/StaPep/*.py` (17 diagnostic scripts) — most are pandas-only sanity checkers; would mostly need path migration.
- `figures/` (2 files), `debug_checks/` (7 files), top-level `run_*_stapep.py` (4 WSL+stap files), `cli/` (3 TODO stubs), `stapep_package/` (vendored library — should not be modified).
- Deferred items: MIC parser centralization (5 inline variants); native peptide feature dict consolidation (2 precision tiers).

### Slice 7: models — COMPLETE (2026-07-06)

**New shared code:**
- `utils/sequence_io.py` — `parse_sequence_file()` returning `(idx, seq)` pairs. Replaces 3 inline copies: `models/batch_esmfold.py` and `models/esm_sequence_processor.py` (byte-identical) use it directly; `models/run_esmfold_peptides.py` now wraps it to build its `(unique_id, idx, seq, label)` 4-tuples. Byte-for-byte behavior verified by `test_parse_sequence_file_pairs` + `_run_esmfold_wrapper`.
- New constants in `utils/paths.py`: `MODELS_DIR`, `ESMFOLD_LOCAL_DIR` (= `MODELS_DIR/esmfold_v1_local`), `TORCH_HUB_CHECKPOINTS`, `HF_CACHE`.

**Scripts migrated (models/):**
- `batch_esmfold.py`, `run_esmfold_peptides.py`, `esm_sequence_processor.py` — 3 ESMFold drivers: `parse_sequence_file` → shared helper; `Path(__file__).parent/"esmfold_v1_local"` → `ESMFOLD_LOCAL_DIR` (same resolved path; dir doesn't exist so still HF-fallback — now location-stable). `run_esmfold_peptides.py` data-file argparse defaults → `DATA_DIR` (was `Path(__file__).parent.parent/"data"/...` — the folder move left it correct but fragile).
- `check_cache.py`, `diagnose_model.py` — hardcoded torch-hub cache path → `TORCH_HUB_CHECKPOINTS`.
- **Left as-is** (can't-run one-offs; no GPU to verify): `download_esmfold.py`, `download_esmfold_simple.py`, `download_from_huggingface.py`, `fix_hf_cache.py`, `test_gpu_esmfold.py`. The inline `~/.cache/huggingface` paths in the download scripts are a trivial pending follow-up.
- TODO stubs `mic_predictor.py`, `train_mic.py` left as inert scaffolding (implementing them = the ESMFold-integration project, not this refactor).

**Cross-folder fix surfaced by this slice:**
- `nn_pipeline/train.py` did `sys.path.insert(0, <its own dir>)` + bare `from models import ...` / `from feature_dataset import ...`, which registered `nn_pipeline/models.py` as the **global** `models` module — shadowing the `models/` ESMFold package for the rest of any process that imported it and breaking every `import models.X`. Fixed: bootstrap `PROJECT_ROOT` + package-qualified `from nn_pipeline.models import ...` / `from nn_pipeline.feature_dataset import ...`. It was the only bare-`models` importer in the tree.

**Duplication deferred (conservative scope — GPU inference paths not runnable to verify):**
- Shared ESMFold loader (`esmfold_v1_local` + FP16 + `local_files_only` + HF-fallback) still copy-pasted across the 3 drivers → future `utils/pretrained_esm.py` once a GPU run can validate.
- Checkpoint/resume plumbing (`load_checkpoint`/`save_checkpoint`/`predict_single_structure`/`estimate_time`) still duplicated between `batch_esmfold.py` and `run_esmfold_peptides.py`.
- ESM-2 embedding extraction shared between `esm_sequence_processor.py` and `test_gpu_esmfold.py`.

**Smoke tests added:**
- 13 import tests (all models/ scripts; torch importers `@requires_torch`; `download_esmfold` guarded on `requests`). No main() ever invoked.
- 4 `--help` subprocess tests: `extract_structure_features.py` (CPU-only) + the 3 ESMFold drivers (`@requires_torch`; `--help` exits before model load and proves the standalone bootstrap resolves).
- 2 parser behavioral tests; 4 new path constants added to `test_import_utils_paths`.

**Verification status (2026-07-06):**
- `conda run -n esm_env python -m pytest tests/ -q` → **69 passed, 1 skipped, 0 failed**.

### Slice 8: gnn — COMPLETE (2026-07-06)

**Environment note:** gnn scripts run under the WSL `venv` (torch + torch_geometric).
`torch_geometric` is NOT installed in the Windows `esm_env` used for the pytest
run, so all 10 new gnn tests **skip** there (not fail). They were validated for
real under the **WSL `esm_env`** conda env (which *does* have torch + torch_geometric
+ Bio) via a standalone script — every gnn module imports, all migrations resolve,
`run_gnn_training.py --help` exits 0.

**Core library (no changes needed beyond one dedup — already clean):**
- `gnn/data_utils.py` — extracted the 24-element geometric-feature column list to a
  module-level constant `DEFAULT_GEO_FEATURE_COLS` (single source of truth).
  `PeptideGraphDataset` now defaults to `list(DEFAULT_GEO_FEATURE_COLS)`.
  **Deliberately NOT reused** `features.geometric_features.get_feature_names()` — that
  list includes the categorical `ss_method` marker (25 cols), which would change
  `geo_feature_dim` from 24 to 25 and break the models.
- `gnn/models.py`, `gnn/train.py`, `gnn/__init__.py` — no path/constant duplication; left as-is.

**Scripts migrated (all 5 gnn/ driver scripts):**
- `gnn/run_gnn_training.py` — argparse CLI. Removed unused `import joblib`. `--csv_path`
  / `--pdb_dir` / `--output_dir` argparse defaults migrated from cwd-relative strings to
  `str(DATA_DIR / ...)` / `str(RESULTS_DIR / 'gnn')` (now project-root-absolute).
- `gnn/run_gnn_comparison.py` — `CONFIG` paths → `DATA_DIR`; output dirs → `RESULTS_DIR`;
  inline 12-element QSAR list → `features.stapep_columns.QSAR_COLS` (verified byte-identical);
  inline 24-element geo list → `gnn.data_utils.DEFAULT_GEO_FEATURE_COLS`.
- `gnn/predict_gcn_single.py` — `CONFIG` paths → `STAPEP_DIR`; ESMFold local path →
  `ESMFOLD_LOCAL_DIR`; inline `parse_sequence_file` → `utils.sequence_io.parse_sequence_file`.
- `gnn/predict_stapep_candidates.py` — same treatment; output dir → `RESULTS_DIR / 'stapep_predictions'`.
- `gnn/run_stapep_gnn_comparison.py` — same treatment; `output_dir` CONFIG → `STAPEP_DIR`;
  output dirs → `RESULTS_DIR / 'stapep_gnn'`. Kept its unique `clean_sequence` staple-notation cleaner inline.

**Consolidation this slice:**
- `utils.sequence_io.parse_sequence_file` — now used by 3 more scripts (the gnn drivers), 4 inline
  copies removed total this slice. All were behaviorally identical (`line.split(None,1)` makes the
  `parts[0].strip()` vs unstripped-idx difference a no-op).
- `features.stapep_columns.QSAR_COLS` — the last remaining verbatim copy (flagged in §9) is now gone.
- New `gnn.data_utils.DEFAULT_GEO_FEATURE_COLS` — dedups the two identical 24-col copies (data_utils + run_gnn_comparison).

**Models-shadowing gotcha (watched per the refactor-status memo): NOT present in gnn.**
All gnn scripts already used `sys.path.insert(<project root>)` + package-qualified
`from gnn.X import ...` (never a bare `from models import ...`), so the `nn_pipeline/train.py`
hazard does not recur here. The `sys.path.insert` bootstrap is retained (needed for standalone
`python gnn/run_gnn_training.py` execution; it also enables the new `utils.*`/`features.*` imports).

**Deferred (conservative scope — GPU/venv training paths not runnable to verify, per the
same rule the models/ slice applied to the ESMFold drivers):**
- Merging `predict_gcn_single.py` and `predict_stapep_candidates.py` (§9 "merge with --ensemble-size").
  They are NOT "differ only in ensemble depth": single-GCN-full-train vs GCN+GAT+EGNN 5-fold ensemble
  with σ_GNN calibration + JSON output. A behavior-changing merge of unverifiable ML scripts is out of scope.
- `predict_gcn_single.py` uses `dtype=` and `predict_stapep_candidates.py` uses `torch_dtype=` on
  `EsmForProteinFolding.from_pretrained` — left unchanged (path-only migration; kwarg divergence is pre-existing).

**Smoke tests added:**
- `conftest.requires_tg` marker wired into both test modules' imports (was already defined, now used).
- 9 import tests (`gnn.data_utils` + `DEFAULT_GEO_FEATURE_COLS` assertions, `gnn.models`, `gnn.train`,
  `gnn` package, + all 5 drivers). Gated `@requires_torch @requires_tg @requires_bio` (importing any
  gnn submodule runs `gnn/__init__.py`, which pulls in `data_utils` → `Bio.PDB`). Regression asserts:
  `run_gnn_comparison.QSAR_COLS == QSAR_COLS`, shared geo list identity, and `parse_sequence_file is`
  the shared helper in all 3 rewritten scripts.
- 1 `--help` subprocess test for `run_gnn_training.py` (the only argparse-driven gnn script).

**Verification status (2026-07-06):**
- Windows `esm_env`: `conda run -n esm_env python -m pytest tests/ -q` → **69 passed, 11 skipped, 0 failed**
  (11 skips = 1 permanent `run_dataset_comparison` guard + 10 gnn tests skipped for missing torch_geometric).
- WSL `esm_env` (has torch_geometric): standalone validation script — all gnn imports + migration
  assertions pass; `run_gnn_training.py --help` exits 0; argparse defaults resolve project-root-absolute.

### Phase 4 (partial): archived 6 top-level Buforin scripts (2026-07-06)

During the folder reorganization, six top-level scripts were deleted from the working tree
**without a subfolder home** (not part of any slice's migrated list). Restored from git
`HEAD` (`b12974a`) into `seq2svm\archive\` and frozen rather than deleted (REFACTOR_PLAN.md
Phase 4). They are NOT on the live path and NOT smoke-tested. See `seq2svm\archive\README.md`
and PROJECT_MAP.md §10 for per-file provenance.

- `predict_buf_hemolysis.py` (v1, superseded by `_v2`), `predict_buf_hemolysis_v2.py` (18-feat, un-migrated),
  `predict_buf_hemolysis_lyticity.py` (lyticity-only), `predict_buf_pmic_v2.py` (18-feat pMIC, un-migrated),
  `plot_hydrophobic_vs_mic.py`, `plot_lyticity_vs_hemolysis.py` (figure scripts).
- Open decision for a later slice: whether the two 18-feature `_v2` models should be promoted into
  `regression\` as an alternative feature configuration alongside the 14-feature paper-subset scripts.

### Slice 9: nn_pipeline / figures / debug_checks / StaPep diagnostics / cli / top-level — COMPLETE (2026-07-06)

Final Phase-3 pass over the six remaining folders. Verified: Windows `esm_env`
`pytest tests/ -q` → **96 passed, 12 skipped** (was 69/11); the 12 skips = 1 permanent
`run_dataset_comparison` guard + 11 torch_geometric-gated gnn tests. The gnn-touching
change (below) + all gnn tests were re-validated under **WSL `esm_env`** (has torch_geometric)
via a standalone script — all pass, `gnn/run_gnn_training.py --help` exits 0.

**New shared module:**
- `features/geometric_columns.py` — dependency-free single source of truth for the **Geo-24**
  numeric column list (`GEO_FEATURE_COLS`) + its 6 semantic sub-groups (`PLDDT_COLS`,
  `COMPACTNESS_COLS`, `SECONDARY_STRUCTURE_COLS`, `SASA_COLS`, `SEQUENCE_COLS`, `CURVATURE_COLS`).
  Dedups the 24-col list that was copied verbatim in **three** places. Kept torch-free on purpose
  so the pandas-only debug scripts can import it (gnn.data_utils needs torch_geometric,
  nn_pipeline.feature_dataset needs torch). Still deliberately ≠ `features.geometric_features.get_feature_names()`
  (that carries the 25th `ss_method` marker).
- `gnn/data_utils.py` `DEFAULT_GEO_FEATURE_COLS` is now `list(GEO_FEATURE_COLS)` (was a literal) —
  the gnn-slice "single source of truth" now re-exports the shared one. `run_gnn_comparison`'s
  `is`-identity with it still holds (verified WSL).
- `nn_pipeline/feature_dataset.py` imports the sub-groups + `GEO_FEATURE_COLS` from the shared module;
  `GEOMETRIC_FEATURE_COLS = list(GEO_FEATURE_COLS)` (byte-identical to the old concat).

**nn_pipeline/ (5 files):**
- `train.py` — **fixed a latent runtime bug**: two function-local imports (`from feature_dataset import AMPDataset`
  in `train_final_model()`, `from prepare_clusters import create_simple_clusters` in `main()`) were left bare
  when the top-of-file bootstrap was changed (slice 7) to insert PROJECT_ROOT instead of the `nn_pipeline/` dir.
  They now ModuleNotFound when the module is imported rather than run as a script. Both fixed to
  package-qualified `from nn_pipeline.X import ...`. Regression-locked by `test_nn_pipeline_train_uses_package_qualified_imports`.
- `feature_dataset.py` — added a PROJECT_ROOT bootstrap (for standalone runs), wired to the shared geo cols
  and `utils.paths.DATA_DIR` (main() demo path). Behavior byte-identical (verified `GEOMETRIC_FEATURE_COLS == GEO_FEATURE_COLS`).
- `models.py`, `prepare_clusters.py`, `__init__.py` — self-contained; no changes needed.

**figures/ (2 `.py` files; 3 `.ipynb` notebooks left untouched — research artifacts):**
- `plot_mic_distribution.py` — `BASE`/`STAPEP` → `utils.paths.STAPEP_DIR`; inline 17-col `FEATURES` → `STAPEP_COLS`
  and 14-col `PAPER_FEATURES` → `STAPEP_COLS_PAPER_14` (both verified byte-identical). Added PROJECT_ROOT bootstrap.
  Inline `FEATURE_LABELS` + `parse_ecoli_mic` MIC regex (variant F) kept inline.
- `lyticity_vs_mic.py` — **module-level `sys.stdout = io.TextIOWrapper(...)` moved into main()** so the module
  is now import-safe (regression-tested); `BASE`/`DATA` → `STAPEP_DIR`, output PNG path → `PROJECT_ROOT`. Inline
  MIC regex + `BUF_WT` + 8-entry `LITERATURE_MIC` kept inline (distinct from `LITERATURE_MIC_ECOLI`).

**debug_checks/ (7 files) — all now import-safe:**
- `check_feature_overlap.py`, `check_feature_overlap_detailed.py`, `check_features.py` — wrapped module-level bodies
  in `main()` + guard; relative/hardcoded CSV paths → `utils.paths.DATA_DIR`. `check_feature_overlap.py` inline
  `qsar_feats`/`geo_feats` → `QSAR_COLS` / `GEO_FEATURE_COLS` (verified end-to-end: still finds the single
  netCharge↔net_charge overlap, r=0.9516). `_detailed` keeps its per-feature meaning dicts (documentation).
- `check_features_fixed.py` — already guarded; path → `DATA_DIR`.
- `debug_secondary_structure.py` — already guarded; paths → `utils.paths.STRUCTURES_DIR`.
- `check_pyg.py` — trivial diagnostic, wrapped in `main()` for import-testability.
- `test_esmfold_quick.py` — model load moved into `main()` (was loading ESMFold at module level → unimportable);
  now import-safe. (Not a pytest test — no `test_*` functions; pytest only collects from `tests/`.)

**StaPep diagnostics (`data/training_dataset/StaPep/`) — 7 pandas-only sanity checkers made import-safe + smoke-tested:**
- Wrapped in `main()` + removed hardcoded absolute `c:/Users/bioin/...` paths (→ script-relative `HERE = Path(__file__).parent`):
  `_check_all_nan.py`, `_check_mag_i7.py`, `_check_feature_coverage.py`, `_verify_training_md4ns.py`.
- Already guarded (no change): `_compare_replicate_md.py`, `_inspect_md50ns_in_progress.py`, `sanity_check_md_features.py`.
- They live under a non-package dir (no `__init__.py` chain), so the smoke test loads each by file path via
  `importlib.util.spec_from_file_location` (parametrized `test_stapep_diagnostic_import_safe`). Verified `_check_all_nan`
  runs end-to-end (188 rows, flags DRAMP21556 NaN-energy).
- **Intentionally NOT touched / NOT smoke-tested** (per REFACTOR_PLAN "preserve behavior; never regenerate canonical data"):
  data-generation scripts (`extract_amp_features`, `_drop_dramp21556` [module-level canonical WRITE — never import],
  `combine_stapled_paper_dataset`, `filter_drop_k_stapled`, `build_paper_supplement_mag115`, `convert_to_txt` [module-level write],
  `_build_replicate_subset` [module-level write]); MD scripts (`run_amp_md_features`, `run_test_stapep_md` [WSL path hardcode §7d]);
  WSL DSSP probe (`_probe_dssp_staples`); personal-path compare (`_paper_vs_dramp_compare` [reads a `~/Downloads` file, §7d]).

**cli/ (3 TODO stubs):** `extract_features.py`, `predict_mic.py`, `train_model.py` left as inert scaffolding
(implementing them = the ESMFold-integration project, not this refactor — same call as models/ `mic_predictor.py`/`train_mic.py`).
Added 3 import smoke tests confirming they import as no-ops.

**top-level WSL StaPep drivers (repo root, OUTSIDE the package):** documented + classified, **not rewired** (they are
standalone WSL+stap production drivers; their `stapep_package`-locating logic is correct and coupling them to the seq2svm
package would add path fragility for near-zero benefit — same conservative call as the archived `_v2` scripts).
- `run_buforin_stapep.py` — argparse; stapep imports are function-local, so `--help` parses+exits without the stap env.
  Added subprocess `--help` smoke test (`test_help_run_buforin_stapep`).
- `run_buf_variants_stapep.py`, `test_buf_variant_single.py` — no argparse (run MD directly); import-safe but not smoke-tested.
- `test_wsl_stapep.py` — **NOT import-safe by design** (module-level `import pytraj` + `sys.exit(1)` on failure); pure WSL diagnostic.

**Tests added this slice:** +28 (1 skips on Windows): shared geo module; 5 nn_pipeline (incl. the train.py regression guard);
2 figures (incl. lyticity sys.stdout guard); 7 debug_checks; 3 cli; 7 StaPep (parametrized); 2 `--help` subprocess
(`plot_mic_distribution`, top-level `run_buforin_stapep`); 1 WSL-gated gnn-shared-cols equality.

**Still pending (deferred, unchanged):**
- MIC parsing regex — 5 distinct variants still inline across `regression/`, `comparison/compare_buf_pmic.py`, `svm/predict_mic_svm.py`, `mlp/predict_mic_mlp.py`. Centralization deferred per PROJECT_MAP §7e.
- Native peptide feature dicts (`BUF_WT_FEATURES`, `NATIVE`, `SP_FEATURES`) — still in 3-4 scripts at two precision tiers. Deferred per the same rule.
- StaPep data-generation / MD / WSL-only scripts (listed above) — logic intentionally frozen.
- `stapep_package/` (vendored library — do not modify). Root-level `predict_mic_single.py` (seq2svm CLI, out of this task's scope).

**All Phase-3 code folders are now done:** svm, mlp, regression, comparison, feature_extraction, models, gnn, nn_pipeline, figures, debug_checks, StaPep diagnostics, cli.
