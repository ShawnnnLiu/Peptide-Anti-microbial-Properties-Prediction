# SMOKE_TESTS.md

Plan for the Phase-1 smoke-test harness. **No test code has been written yet** — this document is the design contract the next agent should implement against. Read `RUNNABLE_FILES.md` for the per-file inventory and `PROJECT_MAP.md` for the data/env context.

## 1. Goals and non-goals

**Goals**
- A `python -m pytest` invocation that finishes in under 60 seconds on a vanilla machine with biopython + numpy + pandas + sklearn installed.
- Confirm that "import-clean" modules really do import without side effects.
- Confirm that argparse CLIs print help without crashing.
- Surface broken / Python-2 / TODO-stub files explicitly so they don't get treated as healthy.
- Cleanly skip (not fail) anything that needs CUDA, WSL+OpenMM, fair-esm, torch-geometric, or the legacy `skl_legacy` sklearn 0.19.x.

**Non-goals**
- Do not start MD simulations, ESMFold batch runs, full GNN/MLP training, or the dataset-swap comparison runner.
- Do not download model weights.
- Do not exercise the `pretrained_svm\` snapshot tree.
- Do not validate scientific correctness — that is for the per-slice refactor verification.

## 2. Current state

### Existing test files
- `tests\test_geometric_features.py` — ad-hoc runner for `features\geometric_features.py`. Three pytest-collectible `test_*` functions plus a `demo` + `main()`. Imports only `Bio.PDB` at module top, so import is safe given biopython. Two of the three tests skip vacuously when no PDB files are present under `data\training_dataset\structures\`; `test_edge_cases()` runs unconditionally.

### Test-config files
- None. No `pytest.ini`, `conftest.py`, `pyproject.toml`, `setup.cfg`, or `tox.ini` anywhere.

## 3. Proposed test layout

```
seq2svm\
├── pytest.ini                        # NEW (see §6)
└── tests\
    ├── conftest.py                   # NEW — shared skip helpers
    ├── test_imports.py               # NEW — see §4
    ├── test_entrypoints.py           # NEW — see §5
    └── test_geometric_features.py    # KEEP — already pytest-collectible
```

## 4. `tests\test_imports.py` — import-only checks

Goal: every module that *claims* to be importable can be imported in a clean process without side effects (no CSV reads at module top, no GPU touches, no env writes). Use `importlib.import_module` so failures are clean assertion messages, not `ImportError` stack-trace noise.

### Group 1 — Always-safe (stdlib / numpy / pandas / biopython only)

| Module | Guard |
|---|---|
| `features.geometric_features` | `pytest.importorskip("Bio")` |
| `features.feature_extractor`, `features.feature_utils` | none (TODO stub) |
| `utils.caching`, `utils.data_prep`, `utils.evaluation` | none (TODO stub) |
| `structure.esmfold_predictor`, `structure.pdb_parser`, `structure.structure_features` | none (TODO stub) |
| `models.mic_predictor`, `models.train_mic` | none (TODO stub) |
| `cli.extract_features`, `cli.predict_mic`, `cli.train_model` | none (TODO stub) |

### Group 2 — Requires torch (CPU-only is fine)
| Module | Guard |
|---|---|
| `nn_pipeline.models` | `pytest.importorskip("torch")` |
| `nn_pipeline.feature_dataset` | `pytest.importorskip("torch")` + `pytest.importorskip("sklearn")` |
| `nn_pipeline.train` | `pytest.importorskip("torch")` |

### Group 3 — Requires torch + torch_geometric
| Module | Guard |
|---|---|
| `gnn.models`, `gnn.data_utils`, `gnn.train` | `requires_torch` + `requires_tg` |

### Group 4 — Requires fair-esm (`esm_env` active)
Use `requires_esm` (see §7). These modules import `esm` or `transformers.EsmForProteinFolding` at module top:
- `models.download_esmfold_simple` (top-level `from transformers import EsmForProteinFolding`)
- `models.download_from_huggingface` (top-level `from huggingface_hub import ...`)

`models.esm_sequence_processor`, `models.batch_esmfold`, `models.run_esmfold_peptides`, `models.download_esmfold` defer their heavy imports to function bodies; they import cleanly without `esm`. They go in Group 2 (torch-only).

### Group 5 — Requires WSL + `stap` env (must skip on Windows)
These trigger an `openmm` / `pytraj` / `parmed` import chain through `stapep`:
- `data.training_dataset.StaPep.run_amp_md_features`
- `data.training_dataset.StaPep.run_test_stapep_md`
- `data.training_dataset.StaPep.extract_amp_features`
- `data.training_dataset.StaPep.convert_to_txt`
- `data.training_dataset.StaPep._probe_dssp_staples`
- `feature_extraction.build_stapep_features` (transitive)
- `feature_extraction.generate_stapep_structures` (also needs `esm`)

Guard at module level:
```python
pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="StaPep stack requires WSL + stap env")
```

### Group 6 — Modules to NOT import at all (side-effects at module level)
These will execute code or read CSVs as soon as you `import` them. Cover them with `--help` subprocess tests in `test_entrypoints.py` instead, or wait until the refactor moves their `__main__` body behind `if __name__ == "__main__":`.

| Module | Reason |
|---|---|
| `svm.predict_mic_svm` | Sets `sys.stdout = io.TextIOWrapper(...)` at module top |
| `comparison.compare_anomalous_features` | Reads `stapled_amps_features.csv` at module top |
| `comparison.compare_buf_pmic` | Suspected same pattern (verify before testing) |
| `comparison.compare_ngc_scores` | Suspected same pattern (verify before testing) |
| Most `regression\*.py` and `mlp\predict_mic_mlp.py` | Many lack `if __name__` guards — verify each before importing |

Mark these as `@pytest.mark.skip(reason="executes at module load; refactor to guard with __main__ first")` so they appear in the report.

### Group 7 — Legacy sklearn 0.19.x pickle (`svc.pkl`)
Any test that actually unpickles `predictionsParameters\svc.pkl` must be guarded by:
```python
import sklearn
@pytest.mark.skipif(
    tuple(int(x) for x in sklearn.__version__.split(".")[:2]) > (0, 19),
    reason="svc.pkl requires scikit-learn 0.19.x (skl_legacy env)"
)
```
Note: `svm\run_pretrained_svm_inference.py` rebuilds the SVM from `.npy` sidecars and may not need this guard. Verify on first run.

## 5. `tests\test_entrypoints.py` — `--help` subprocess checks

Pattern:
```python
def _help(script_rel: str, timeout: int = 10):
    result = subprocess.run(
        [sys.executable, str(ROOT / script_rel), "--help"],
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{script_rel} --help exited {result.returncode}\n"
        f"stderr: {result.stderr[:500]}"
    )
```

### Sub-group A — Always safe
- `scripts\run_sequence_svm.py`
- `scripts\make_seqs_windows.py`
- `feature_extraction\build_geometric_features.py` (guard `requires_bio`)
- `models\extract_structure_features.py`
- `regression\predict_pmic_regression.py`
- `regression\predict_pmic_all_organisms.py`
- `predict_mic_single.py`
- `figures\plot_mic_distribution.py`
- `comparison\run_feature_fusion_experiments.py`
- `nn_pipeline\prepare_clusters.py`
- `mlp\run_stapep_mlp.py`
- `svm\run_stapep_svm.py`
- `svm\run_combined_svm.py`

### Sub-group B — Requires torch (`requires_torch`)
- `models\batch_esmfold.py`
- `models\esm_sequence_processor.py`
- `models\run_esmfold_peptides.py`
- `mlp\run_nn_training_pnas_style.py`
- `nn_pipeline\train.py` (verify argparse first)

### Sub-group C — Requires torch_geometric (`requires_torch` + `requires_tg`)
- `gnn\run_gnn_training.py`

### Sub-group D — Verify argparse exists before adding a `--help` test
Scripts in `regression\` and `mlp\` that may lack argparse: `predict_hemolysis_regression.py`, `predict_pmic_stapled_variants.py`, `run_mic_classifier.py`, `run_mic_rf.py`, `run_mic_mlp.py`, `predict_mic_mlp.py`, several `svm\*` scripts (`run_mic_svm.py`, `run_stapep_svm_no_loop.py`, `run_pretrained_svm_*.py`), `figures\lyticity_vs_mic.py`. Read the bottom of each script during Phase-3 slice work; if argparse exists, add to Sub-group A or B.

## 6. `pytest.ini` (proposed)

Place at `seq2svm\pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts =
    --tb=short
    -ra
    --ignore=../pretrained_svm
    --ignore=descriptors
    --ignore=debug_checks
    --ignore=data
    --ignore=figures
    --ignore=comparison
    --ignore=10ns vs 50ns comparison
markers =
    slow:        marks tests that run training loops or MD simulations (deselect with -m "not slow")
    gpu:         marks tests that require CUDA
    wsl:         marks tests that require WSL + stap env
    legacy_skl:  marks tests that require scikit-learn 0.19.x
filterwarnings =
    ignore::DeprecationWarning
    ignore::UserWarning:Bio
```

Notes:
- `--ignore=../pretrained_svm` keeps pytest from collecting tests under the frozen snapshot.
- The other `--ignore` entries keep collection cheap by excluding subtrees that have side-effect-on-import modules; tests for individual scripts under those folders should live under `tests\` and call them as subprocesses.
- Add `--rootdir=.` if invoked from elsewhere; otherwise pytest derives root from `pytest.ini` location.

## 7. `tests\conftest.py` (proposed)

```python
import importlib.util
import sys
import pytest

def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None

HAS_TORCH  = _has("torch")
HAS_TG     = _has("torch_geometric")
HAS_BIO    = _has("Bio")
HAS_ESM    = _has("esm")
ON_WIN     = sys.platform == "win32"

requires_torch   = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_tg      = pytest.mark.skipif(not HAS_TG,    reason="torch_geometric not installed")
requires_bio     = pytest.mark.skipif(not HAS_BIO,   reason="biopython not installed")
requires_esm     = pytest.mark.skipif(not HAS_ESM,   reason="fair-esm (esm_env) not active")
requires_wsl     = pytest.mark.skipif(ON_WIN,        reason="requires WSL + stap env")
```

## 8. Permanently-skipped scripts (never run in CI, even when env is available)

Mark each of these with `@pytest.mark.skip(reason=...)` so the report shows them as deliberately skipped, not silently ignored.

| Script | Reason |
|---|---|
| `data\training_dataset\StaPep\run_amp_md_features.py` | 50 ns MD × 130–188 peptides: hours–days |
| `data\training_dataset\StaPep\run_test_stapep_md.py` | 50 ns MD × 8 peptides: hours |
| `data\training_dataset\StaPep\_drop_dramp21556.py` | Overwrites canonical CSV |
| `comparison\run_dataset_comparison.py` | Swaps `stapled_amps_features.csv` in place; if interrupted, corrupts the dataset (see PROJECT_MAP §7f) |
| `feature_extraction\generate_stapep_structures.py` | ESMFold × 541 stapled sequences + StaPep MD: hours + GPU + WSL |
| `models\run_esmfold_peptides.py` `main()` | Full ESMFold inference: hours of GPU time (the `--help` smoke test is still fine) |
| `models\batch_esmfold.py` `main()` | Same |
| `models\test_gpu_esmfold.py` | Uses `input()` — will hang non-interactive runners |
| `models\download_esmfold.py`, `download_esmfold_simple.py`, `download_from_huggingface.py`, `fix_hf_cache.py` | Network downloads (~15 GB) |
| `gnn\run_gnn_comparison.py`, `gnn\run_stapep_gnn_comparison.py`, `gnn\predict_gcn_single.py`, `gnn\predict_stapep_candidates.py` | Full GNN training: 1–2 h on GPU |
| `mlp\run_nn_training.py`, `mlp\run_nn_training_pnas_style.py` `main()` | Full PyTorch MLP training (the `--help` smoke is still fine) |
| `run_buforin_stapep.py`, `run_buf_variants_stapep.py`, `test_buf_variant_single.py` (root) | WSL+stap; hours of MD |
| Both `seqWindowConstructor.py` files | Python 2 syntax — cannot import on Py 3 |
| `stapep_pkg\example\predictor.py` | Uses deprecated `alphafold=` kwarg; raises TypeError |
| `data\training_dataset\StaPep\_paper_vs_dramp_compare.py` | Hardcoded personal `Downloads/` path |

## 9. Invocation

From `seq2svm\`:

```
python -m pytest                               # full smoke run; <60 s on a clean machine
python -m pytest -v                            # verbose
python -m pytest -m "not slow and not gpu"     # explicit filter
python -m pytest tests\test_imports.py         # imports only
python -m pytest tests\test_entrypoints.py     # --help only
```

If pytest is not installed: do **not** install it. Document the gap and report back. The user has multiple Conda envs; the planner should not be installing packages globally.

## 10. Open questions to resolve during Phase 1 implementation

1. **`svm\run_pretrained_svm_inference.py` import safety** — does the workaround `.npy` path mean it can be imported under modern sklearn? Read the top 80 lines on first attempt.
2. **`regression\` and `mlp\` scripts without `__name__` guards** — audit each to find which read CSVs at module top. Those go in Group 6 (skip-import); the rest go in Group 1.
3. **`nn_pipeline\train.py` argparse** — only the first 15 lines were sampled. Confirm argparse exists before adding a `--help` test.
4. **`gnn\predict_stapep_candidates.py`, `gnn\run_stapep_gnn_comparison.py`** — verify they top-import `torch_geometric` (likely yes) and whether they have argparse.
5. **`figures\lyticity_vs_mic.py`** — argparse? module-level CSV reads?
6. **CI baseline assumption** — this plan assumes Python 3.9+ with numpy/pandas/biopython/sklearn available. If even leaner, also gate Group 1's `features.geometric_features` test under `requires_bio` (already in the plan).
7. **`pretrained_svm\…\svc.pkl` existence** — confirm tracked in git before any test references it.
8. **`debug_checks\check_pyg.py`** — is it useful as a fast `requires_tg` check, or duplicative once `tests\test_imports.py` covers `gnn.models`? Likely duplicative; delete or import-test only.
