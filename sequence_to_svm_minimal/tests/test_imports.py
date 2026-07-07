"""Smoke test: import-only checks for modules that must remain side-effect-free.

Covers the Phase-2 shared modules and the SVM-slice scripts that are safe to
import (they all use the ``sys.path.insert(...) + if __name__ == "__main__"``
pattern, so their bodies do not execute on import).
"""
from __future__ import annotations

import importlib

import pytest

from conftest import requires_skl, requires_torch, requires_bio, requires_propy


# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 shared modules
# ──────────────────────────────────────────────────────────────────────────────

def test_import_utils_paths():
    mod = importlib.import_module("utils.paths")
    # Core constants must exist
    for name in (
        "PROJECT_ROOT", "DATA_DIR", "STAPEP_DIR", "STRUCTURES_DIR",
        "DESCRIPTORS_DIR", "AAINDEX_DIR", "RESULTS_DIR", "FIGURES_DIR",
        "PRETRAINED_PARAMS_DIR", "PRETRAINED_SVC_PKL",
        "PRETRAINED_ZSCORE_CSV", "LIVE_PARAMS_DIR",
        # models/ slice additions
        "MODELS_DIR", "ESMFOLD_LOCAL_DIR",
        "TORCH_HUB_CHECKPOINTS", "HF_CACHE",
    ):
        assert hasattr(mod, name), f"utils.paths missing {name}"


def test_utils_paths_resolves_to_project_root():
    from utils.paths import PROJECT_ROOT, STAPEP_DIR
    # PROJECT_ROOT should be sequence_to_svm_minimal/
    assert (PROJECT_ROOT / "REFACTOR_PLAN.md").exists(), (
        "PROJECT_ROOT did not resolve to sequence_to_svm_minimal/ — "
        f"got {PROJECT_ROOT}"
    )
    # STAPEP_DIR should hold the canonical CSV named in HDC_HANDOFF_BRIEF.md
    canonical = STAPEP_DIR / "stapled_amps_features_training_XZ_md50ns.csv"
    assert canonical.exists(), (
        f"Canonical StaPep CSV missing at {canonical}"
    )


def test_import_features_stapep_columns():
    mod = importlib.import_module("features.stapep_columns")
    assert mod.STAPEP_COLS[0] == "length"
    assert len(mod.STAPEP_COLS) == 17
    assert mod.STAPEP_COLS_WITH_HSASA[-1] == "hydrophobic_sasa"
    assert len(mod.STAPEP_COLS_WITH_HSASA) == 18
    assert mod.QSAR_COLS[0] == "netCharge"
    assert len(mod.QSAR_COLS) == 12
    assert len(mod.TEST_NAMES) == 8
    assert mod.TEST_NAMES[0] == "Buf12"
    assert mod.QSAR_TEST_NAME_MAP["Buf(i+4)_12"] == "Buf12"
    assert "svc__C" in mod.SVM_PARAM_GRID
    assert "svc__gamma" in mod.SVM_PARAM_GRID


# ──────────────────────────────────────────────────────────────────────────────
# SVM-slice scripts: import side effects must be limited to the constants block
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_import_svm_run_stapep_svm():
    importlib.import_module("svm.run_stapep_svm")


@requires_skl
def test_import_svm_run_combined_svm():
    importlib.import_module("svm.run_combined_svm")


@requires_skl
def test_import_svm_run_mic_svm():
    importlib.import_module("svm.run_mic_svm")


@requires_skl
def test_import_svm_predict_mic_svm():
    """Should import without overriding sys.stdout (the hack now lives in main())."""
    import sys as _sys
    saved = _sys.stdout
    importlib.import_module("svm.predict_mic_svm")
    assert _sys.stdout is saved, (
        "svm.predict_mic_svm replaced sys.stdout on import — the Unicode hack "
        "must stay inside main()"
    )


def test_import_svm_run_pretrained_svm_inference():
    importlib.import_module("svm.run_pretrained_svm_inference")


def test_import_svm_run_pretrained_svm_low_loop():
    importlib.import_module("svm.run_pretrained_svm_low_loop")


# ──────────────────────────────────────────────────────────────────────────────
# Pretrained-SVM loader: reproduces the verified intercept / probA / probB
# constants documented in svm/run_pretrained_svm_inference.py.
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# mlp/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_import_mlp_run_stapep_mlp():
    importlib.import_module("mlp.run_stapep_mlp")


@requires_skl
def test_import_mlp_run_mic_mlp():
    importlib.import_module("mlp.run_mic_mlp")


@requires_skl
def test_import_mlp_predict_mic_mlp():
    """Should import without overriding sys.stdout (the hack now lives in main())."""
    import sys as _sys
    saved = _sys.stdout
    importlib.import_module("mlp.predict_mic_mlp")
    assert _sys.stdout is saved, (
        "mlp.predict_mic_mlp replaced sys.stdout on import — the Unicode hack "
        "must stay inside main()"
    )


@requires_torch
def test_import_mlp_run_nn_training():
    importlib.import_module("mlp.run_nn_training")


@requires_torch
def test_import_mlp_run_nn_training_pnas_style():
    importlib.import_module("mlp.run_nn_training_pnas_style")


# ──────────────────────────────────────────────────────────────────────────────
# regression/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_import_regression_predict_pmic_regression():
    importlib.import_module("regression.predict_pmic_regression")


@requires_skl
def test_import_regression_predict_pmic_all_organisms():
    importlib.import_module("regression.predict_pmic_all_organisms")


@requires_skl
def test_import_regression_predict_hemolysis_regression():
    importlib.import_module("regression.predict_hemolysis_regression")


@requires_skl
def test_import_regression_predict_pmic_stapled_variants():
    """Should import without overriding sys.stdout (the hack now lives in main())."""
    import sys as _sys
    saved = _sys.stdout
    importlib.import_module("regression.predict_pmic_stapled_variants")
    assert _sys.stdout is saved, (
        "regression.predict_pmic_stapled_variants replaced sys.stdout on import"
    )


@requires_skl
def test_import_regression_run_mic_classifier():
    importlib.import_module("regression.run_mic_classifier")


@requires_skl
def test_import_regression_run_mic_rf():
    importlib.import_module("regression.run_mic_rf")


# ──────────────────────────────────────────────────────────────────────────────
# comparison/ slice
# ──────────────────────────────────────────────────────────────────────────────

def test_import_comparison_compare_anomalous_features():
    importlib.import_module("comparison.compare_anomalous_features")


@requires_skl
def test_import_comparison_compare_buf_pmic():
    importlib.import_module("comparison.compare_buf_pmic")


@requires_skl
def test_import_comparison_compare_ngc_scores():
    importlib.import_module("comparison.compare_ngc_scores")


def test_import_comparison_run_dataset_comparison():
    """Orchestrator script — must NEVER be invoked in tests (PERMANENT skip in
    test_entrypoints.py: it swaps the canonical CSV and leaves it corrupted
    if interrupted). Import-only check confirms no syntax errors and proper
    main guard."""
    importlib.import_module("comparison.run_dataset_comparison")


@requires_torch
def test_import_comparison_run_feature_fusion_experiments():
    importlib.import_module("comparison.run_feature_fusion_experiments")


# ──────────────────────────────────────────────────────────────────────────────
# feature_extraction/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_bio
def test_import_feature_extraction_build_geometric_features():
    importlib.import_module("feature_extraction.build_geometric_features")


@requires_bio
@requires_propy
def test_import_feature_extraction_build_stapep_features():
    importlib.import_module("feature_extraction.build_stapep_features")


@requires_bio
def test_import_feature_extraction_build_stapep_geo():
    importlib.import_module("feature_extraction.build_stapep_geo")


@requires_propy
def test_import_feature_extraction_extract_stapep_qsar():
    importlib.import_module("feature_extraction.extract_stapep_qsar")


@requires_torch
def test_import_feature_extraction_generate_stapep_structures():
    """Imports cleanly; main() is guarded — never invoked in tests (would
    require GPU + ~15 GB ESMFold download)."""
    importlib.import_module("feature_extraction.generate_stapep_structures")


# ──────────────────────────────────────────────────────────────────────────────
# Phase-2 shared helpers (new in regression slice)
# ──────────────────────────────────────────────────────────────────────────────

def test_mic_units_roundtrip():
    """Conversions are inverses of each other and produce known values."""
    from utils.mic_units import (
        pmic_to_mic_uM, mic_to_pmic_uM,
        pmic_to_mic_ugml, mic_to_pmic_ugml,
    )
    # pMIC 6.0 ↔ 1 μM (definition)
    assert abs(pmic_to_mic_uM(6.0) - 1.0) < 1e-9
    assert abs(mic_to_pmic_uM(1.0) - 6.0) < 1e-9
    # Round-trip with MW
    mw = 2473.829  # Buforin II
    for pmic in (4.0, 5.5, 6.7):
        assert abs(mic_to_pmic_ugml(pmic_to_mic_ugml(pmic, mw), mw) - pmic) < 1e-9


def test_reference_peptides_literature_mic():
    from features.reference_peptides import LITERATURE_MIC_ECOLI
    # Spot-check a few entries that exist in the source data
    assert LITERATURE_MIC_ECOLI["Buf12"]["mic_ugml"] == 6.25
    assert LITERATURE_MIC_ECOLI["Buf13"]["mic_ugml"] == 100.0
    assert LITERATURE_MIC_ECOLI["Buf12_V15K_L19K"]["mic_ugml"] is None
    assert len(LITERATURE_MIC_ECOLI) == 11


def test_stapep_columns_paper_14_is_strict_subset():
    """The 14-feature paper subset must be a strict subset of STAPEP_COLS."""
    from features.stapep_columns import STAPEP_COLS, STAPEP_COLS_PAPER_14
    assert set(STAPEP_COLS_PAPER_14).issubset(set(STAPEP_COLS))
    assert len(STAPEP_COLS_PAPER_14) == 14
    # The 3 features dropped from the 17-col set:
    dropped = set(STAPEP_COLS) - set(STAPEP_COLS_PAPER_14)
    assert dropped == {"lyticity_index", "sheet_percent", "sasa"}


# ──────────────────────────────────────────────────────────────────────────────
# Pretrained-SVM loader sanity check
# ──────────────────────────────────────────────────────────────────────────────

def test_pretrained_svm_loader_reproduces_known_constants():
    """Sanity-check the .npy sidecar reconstruction against the values
    cross-checked in the original inference docstring."""
    from utils.pretrained_svm import load_pretrained_svm
    model = load_pretrained_svm()
    # Verified values from the docstring of run_pretrained_svm_inference.py:
    assert abs(model.intercept - (-0.01187876)) < 1e-6, model.intercept
    assert abs(model.probA     - (-3.29162142)) < 1e-6, model.probA
    assert abs(model.probB     - (+0.03014156)) < 1e-6, model.probB
    # Shape of the linear-kernel weight vector matches the 12-descriptor model.
    assert model.w.shape == (12,)
    assert len(model.desc_names) == 12
    assert model.z_means.shape == (12,)
    assert model.z_stds.shape == (12,)


# ──────────────────────────────────────────────────────────────────────────────
# Misc: confirm the existing feature module imports cleanly when biopython
# is present (this re-verifies the legacy tests/test_geometric_features.py
# environment from a clean angle).
# ──────────────────────────────────────────────────────────────────────────────

def test_import_features_geometric_features():
    Bio = pytest.importorskip("Bio", reason="biopython not installed")
    importlib.import_module("features.geometric_features")


# ──────────────────────────────────────────────────────────────────────────────
# models/ slice — ESMFold drivers, ESM-2 processor, cache/download tools.
# All are import-safe (heavy work + input()/CUDA/downloads live inside main()).
# The torch-importing ones are gated; their main() is NEVER invoked in tests.
# ──────────────────────────────────────────────────────────────────────────────

def test_import_models_check_cache():
    importlib.import_module("models.check_cache")


@requires_torch
def test_import_models_diagnose_model():
    importlib.import_module("models.diagnose_model")


def test_import_models_download_esmfold():
    pytest.importorskip("requests", reason="requests not installed")
    importlib.import_module("models.download_esmfold")


def test_import_models_download_esmfold_simple():
    importlib.import_module("models.download_esmfold_simple")


def test_import_models_download_from_huggingface():
    importlib.import_module("models.download_from_huggingface")


def test_import_models_fix_hf_cache():
    importlib.import_module("models.fix_hf_cache")


def test_import_models_extract_structure_features():
    """CPU-only (numpy/pandas); the safest models/ CLI to smoke-test."""
    importlib.import_module("models.extract_structure_features")


@requires_torch
def test_import_models_batch_esmfold():
    """Imports cleanly; main() is GPU/ESMFold — never invoked in tests."""
    importlib.import_module("models.batch_esmfold")


@requires_torch
def test_import_models_run_esmfold_peptides():
    importlib.import_module("models.run_esmfold_peptides")


@requires_torch
def test_import_models_esm_sequence_processor():
    importlib.import_module("models.esm_sequence_processor")


@requires_torch
def test_import_models_test_gpu_esmfold():
    """Import-safe; main() needs CUDA + calls input() — never invoked in tests."""
    importlib.import_module("models.test_gpu_esmfold")


def test_import_models_mic_predictor():
    """TODO scaffolding — must at least import as a no-op."""
    importlib.import_module("models.mic_predictor")


def test_import_models_train_mic():
    """TODO scaffolding — must at least import as a no-op."""
    importlib.import_module("models.train_mic")


# ──────────────────────────────────────────────────────────────────────────────
# Shared sequence-file parser (new in models/ slice) — must reproduce the
# behavior of the three inline copies it replaced, byte-for-byte.
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_sequence_file_pairs(tmp_path):
    from utils.sequence_io import parse_sequence_file
    p = tmp_path / "seqs.txt"
    p.write_text(
        "# a comment line\n"
        "\n"
        "1 MKTAYIAK\n"
        "2   GVVDSDD\n"       # multiple spaces between index and sequence
        "ACDEFG\n"            # index-less line -> auto 1-based position
        "   \n"               # whitespace-only line, skipped
        "7 LASTONE\n"
    )
    assert parse_sequence_file(str(p)) == [
        ("1", "MKTAYIAK"),
        ("2", "GVVDSDD"),
        ("3", "ACDEFG"),      # 3rd parsed entry -> index "3", not line number
        ("7", "LASTONE"),
    ]


@requires_torch
def test_parse_sequence_file_run_esmfold_wrapper(tmp_path):
    """run_esmfold_peptides wraps the shared parser into 4-tuples with a
    prefix/label — confirm it matches the previous inline implementation."""
    mod = importlib.import_module("models.run_esmfold_peptides")
    p = tmp_path / "amp.txt"
    p.write_text("1 MKTAYIAK\nGVVDSDD\n")
    assert mod.parse_sequence_file(str(p), label=1, prefix="AMP") == [
        ("AMP_1", "1", "MKTAYIAK", 1),
        ("AMP_2", "2", "GVVDSDD", 1),
    ]
