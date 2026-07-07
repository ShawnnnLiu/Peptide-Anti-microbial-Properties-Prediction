"""Smoke test: import-only checks for modules that must remain side-effect-free.

Covers the Phase-2 shared modules and the SVM-slice scripts that are safe to
import (they all use the ``sys.path.insert(...) + if __name__ == "__main__"``
pattern, so their bodies do not execute on import).
"""
from __future__ import annotations

import importlib

import pytest

from conftest import (
    PROJECT_ROOT,
    requires_skl, requires_torch, requires_tg, requires_bio, requires_propy,
)


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


# ──────────────────────────────────────────────────────────────────────────────
# gnn/ slice — GNN library (data_utils, models, train) + 5 driver scripts.
# All require torch + torch_geometric; importing any gnn submodule also runs
# gnn/__init__.py, which pulls in data_utils -> Bio.PDB, hence @requires_bio too.
# torch_geometric is only present in the WSL `venv`, so these SKIP under esm_env.
# main()/GPU training is NEVER invoked — imports are side-effect-free (guarded).
# ──────────────────────────────────────────────────────────────────────────────

@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_data_utils():
    mod = importlib.import_module("gnn.data_utils")
    # The centralised Geo-24 column list — single source of truth shared with
    # run_gnn_comparison.py. Must be exactly 24 numeric columns and must NOT
    # carry the categorical ``ss_method`` marker (which would make it 25 and
    # break geo_feature_dim=24).
    cols = mod.DEFAULT_GEO_FEATURE_COLS
    assert len(cols) == 24
    assert cols[0] == "plddt_mean"
    assert cols[-1] == "torsion_std"
    assert "ss_method" not in cols


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_models():
    mod = importlib.import_module("gnn.models")
    assert set(mod.PeptideGNN.ARCHITECTURES) == {"gcn", "gat", "egnn"}


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_train():
    importlib.import_module("gnn.train")


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_package():
    """gnn/__init__.py re-exports the public API from the three submodules."""
    mod = importlib.import_module("gnn")
    for name in ("PeptideGNN", "PeptideGraphDataset", "run_training", "evaluate"):
        assert hasattr(mod, name), f"gnn package missing {name}"


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_run_gnn_training():
    """argparse CLI; main() guarded (GPU training) — never invoked in tests."""
    importlib.import_module("gnn.run_gnn_training")


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_run_gnn_comparison():
    """Reuses the shared QSAR_COLS and DEFAULT_GEO_FEATURE_COLS constants."""
    mod = importlib.import_module("gnn.run_gnn_comparison")
    from features.stapep_columns import QSAR_COLS
    from gnn.data_utils import DEFAULT_GEO_FEATURE_COLS
    # load_data_with_features returns the shared QSAR list verbatim.
    assert mod.QSAR_COLS == QSAR_COLS
    assert mod.DEFAULT_GEO_FEATURE_COLS is DEFAULT_GEO_FEATURE_COLS


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_predict_gcn_single():
    """The inline parse_sequence_file copy was replaced with the shared helper."""
    mod = importlib.import_module("gnn.predict_gcn_single")
    from utils.sequence_io import parse_sequence_file
    assert mod.parse_sequence_file is parse_sequence_file


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_predict_stapep_candidates():
    mod = importlib.import_module("gnn.predict_stapep_candidates")
    from utils.sequence_io import parse_sequence_file
    assert mod.parse_sequence_file is parse_sequence_file


@requires_torch
@requires_tg
@requires_bio
def test_import_gnn_run_stapep_gnn_comparison():
    mod = importlib.import_module("gnn.run_stapep_gnn_comparison")
    from utils.sequence_io import parse_sequence_file
    assert mod.parse_sequence_file is parse_sequence_file


# ──────────────────────────────────────────────────────────────────────────────
# Shared Geo-24 column module (new in the nn_pipeline/figures/debug slice).
# Torch-free single source of truth for the geometric-feature column list.
# ──────────────────────────────────────────────────────────────────────────────

def test_import_features_geometric_columns():
    mod = importlib.import_module("features.geometric_columns")
    cols = mod.GEO_FEATURE_COLS
    assert len(cols) == 24
    assert cols[0] == "plddt_mean"
    assert cols[-1] == "torsion_std"
    assert "ss_method" not in cols
    # sub-groups concatenate to the flat list, in order
    assert (
        mod.PLDDT_COLS + mod.COMPACTNESS_COLS + mod.SECONDARY_STRUCTURE_COLS
        + mod.SASA_COLS + mod.SEQUENCE_COLS + mod.CURVATURE_COLS
    ) == cols


@requires_tg
@requires_torch
@requires_bio
def test_gnn_default_geo_cols_come_from_shared_module():
    """gnn.data_utils.DEFAULT_GEO_FEATURE_COLS is now a copy of the shared list."""
    from gnn.data_utils import DEFAULT_GEO_FEATURE_COLS
    from features.geometric_columns import GEO_FEATURE_COLS
    assert DEFAULT_GEO_FEATURE_COLS == list(GEO_FEATURE_COLS)


# ──────────────────────────────────────────────────────────────────────────────
# nn_pipeline/ slice — PyTorch MLP pipeline.
# feature_dataset + train + models require torch (present in esm_env);
# prepare_clusters is pandas-only.
# ──────────────────────────────────────────────────────────────────────────────

def test_import_nn_pipeline_prepare_clusters():
    """argparse CLI, pandas-only — no torch needed to import."""
    importlib.import_module("nn_pipeline.prepare_clusters")


@requires_torch
def test_import_nn_pipeline_models():
    importlib.import_module("nn_pipeline.models")


@requires_torch
def test_import_nn_pipeline_feature_dataset():
    """GEOMETRIC_FEATURE_COLS now derives from the shared Geo-24 list."""
    mod = importlib.import_module("nn_pipeline.feature_dataset")
    from features.geometric_columns import GEO_FEATURE_COLS
    assert mod.GEOMETRIC_FEATURE_COLS == list(GEO_FEATURE_COLS)


@requires_torch
def test_import_nn_pipeline_train():
    importlib.import_module("nn_pipeline.train")


def test_nn_pipeline_train_uses_package_qualified_imports():
    """Regression guard: train.py must NOT use bare ``from feature_dataset import``
    or ``from prepare_clusters import`` (those broke once the bootstrap was
    changed to insert PROJECT_ROOT instead of the nn_pipeline/ dir)."""
    from pathlib import Path
    src = (PROJECT_ROOT / "nn_pipeline" / "train.py").read_text(encoding="utf-8")
    assert "from feature_dataset import" not in src
    assert "from prepare_clusters import" not in src
    assert "from nn_pipeline.feature_dataset import" in src
    assert "from nn_pipeline.prepare_clusters import" in src


# ──────────────────────────────────────────────────────────────────────────────
# figures/ slice — plotting scripts (import-safe; heavy work in main()).
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_import_figures_plot_mic_distribution():
    """FEATURES / PAPER_FEATURES now alias the shared STAPEP column lists."""
    mod = importlib.import_module("figures.plot_mic_distribution")
    from features.stapep_columns import STAPEP_COLS, STAPEP_COLS_PAPER_14
    assert mod.FEATURES == STAPEP_COLS
    assert mod.PAPER_FEATURES == STAPEP_COLS_PAPER_14


def test_import_figures_lyticity_vs_mic():
    """Should import without overriding sys.stdout (the UTF-8 hack now lives in
    main(), matching the predict_mic_* scripts)."""
    import sys as _sys
    saved = _sys.stdout
    importlib.import_module("figures.lyticity_vs_mic")
    assert _sys.stdout is saved, (
        "figures.lyticity_vs_mic replaced sys.stdout on import — the UTF-8 hack "
        "must stay inside main()"
    )


# ──────────────────────────────────────────────────────────────────────────────
# debug_checks/ slice — diagnostics. All now import-safe (CSV reads / model
# loads live inside main()); the torch import in test_esmfold_quick is also
# inside main(), so importing it needs no torch.
# ──────────────────────────────────────────────────────────────────────────────

def test_import_debug_check_pyg():
    importlib.import_module("debug_checks.check_pyg")


def test_import_debug_check_feature_overlap():
    """Uses the shared QSAR_COLS + GEO_FEATURE_COLS constants."""
    mod = importlib.import_module("debug_checks.check_feature_overlap")
    assert hasattr(mod, "main")


def test_import_debug_check_feature_overlap_detailed():
    importlib.import_module("debug_checks.check_feature_overlap_detailed")


def test_import_debug_check_features():
    importlib.import_module("debug_checks.check_features")


def test_import_debug_check_features_fixed():
    importlib.import_module("debug_checks.check_features_fixed")


@requires_bio
def test_import_debug_secondary_structure():
    importlib.import_module("debug_checks.debug_secondary_structure")


def test_import_debug_test_esmfold_quick():
    """Import-safe: the torch/ESMFold load lives inside main()."""
    importlib.import_module("debug_checks.test_esmfold_quick")


# ──────────────────────────────────────────────────────────────────────────────
# cli/ stubs — empty TODO scaffolding. Must at least import as no-ops.
# ──────────────────────────────────────────────────────────────────────────────

def test_import_cli_extract_features():
    importlib.import_module("cli.extract_features")


def test_import_cli_predict_mic():
    importlib.import_module("cli.predict_mic")


def test_import_cli_train_model():
    importlib.import_module("cli.train_model")


# ──────────────────────────────────────────────────────────────────────────────
# StaPep diagnostic scripts — live under data/training_dataset/StaPep/, which is
# NOT an importable package (no __init__.py chain), so they are loaded by file
# path. Importing them must be side-effect-free: every CSV read now lives inside
# main() (some were previously executing at module level with hardcoded absolute
# paths). The data-generation / MD scripts in that folder are deliberately NOT
# listed here — several write canonical CSVs at module level and must never be
# imported.
# ──────────────────────────────────────────────────────────────────────────────

_STAPEP_DIAGNOSTICS = [
    "_check_all_nan.py",
    "_check_mag_i7.py",
    "_check_feature_coverage.py",
    "_verify_training_md4ns.py",
    "_compare_replicate_md.py",
    "_inspect_md50ns_in_progress.py",
    "sanity_check_md_features.py",
]


@pytest.mark.parametrize("script_name", _STAPEP_DIAGNOSTICS)
def test_stapep_diagnostic_import_safe(script_name):
    """Each pandas-only StaPep diagnostic loads without reading any CSV (its
    body is guarded by main())."""
    import importlib.util
    path = PROJECT_ROOT / "data" / "training_dataset" / "StaPep" / script_name
    assert path.exists(), f"missing {path}"
    spec = importlib.util.spec_from_file_location(script_name[:-3], str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs module body; main() is not called
    assert hasattr(mod, "main"), f"{script_name} has no main() guard"
