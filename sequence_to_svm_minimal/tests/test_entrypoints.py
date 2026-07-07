"""Smoke test: ``--help`` runs cleanly for argparse-driven CLI scripts.

Only covers scripts in the current refactor slice (QSAR/SVM). Each test
invokes the script as a subprocess so we don't accidentally execute its
body via import.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from conftest import requires_skl, requires_torch, requires_tg, requires_bio

ROOT = Path(__file__).resolve().parent.parent  # sequence_to_svm_minimal/


def _help(script_rel: str, timeout: int = 60) -> None:
    path = ROOT / script_rel
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True, text=True, timeout=timeout, cwd=str(ROOT),
    )
    assert result.returncode == 0, (
        f"{script_rel} --help exited {result.returncode}\n"
        f"stderr: {result.stderr[:600]}"
    )
    # argparse prints something
    assert "usage" in (result.stdout + result.stderr).lower(), (
        f"{script_rel} --help produced no usage text"
    )


@requires_skl
def test_help_run_stapep_svm():
    _help("svm/run_stapep_svm.py")


@requires_skl
def test_help_run_combined_svm():
    _help("svm/run_combined_svm.py")


@requires_skl
def test_help_run_mic_svm():
    _help("svm/run_mic_svm.py")


# ──────────────────────────────────────────────────────────────────────────────
# mlp/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_help_run_stapep_mlp():
    _help("mlp/run_stapep_mlp.py")


@requires_skl
def test_help_run_mic_mlp():
    _help("mlp/run_mic_mlp.py")


@requires_torch
def test_help_run_nn_training_pnas_style():
    _help("mlp/run_nn_training_pnas_style.py")


# ──────────────────────────────────────────────────────────────────────────────
# regression/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_help_predict_pmic_regression():
    _help("regression/predict_pmic_regression.py")


@requires_skl
def test_help_predict_pmic_all_organisms():
    _help("regression/predict_pmic_all_organisms.py")


@requires_skl
def test_help_predict_hemolysis_regression():
    _help("regression/predict_hemolysis_regression.py")


# ──────────────────────────────────────────────────────────────────────────────
# comparison/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_torch
def test_help_run_feature_fusion_experiments():
    _help("comparison/run_feature_fusion_experiments.py")


# PERMANENT skip: rewrites the canonical CSV in place. Leaves dataset corrupted
# if interrupted mid-run. See PROJECT_MAP.md §7f and SMOKE_TESTS.md Part D.
@pytest.mark.skip(reason="PERMANENT: rewrites canonical stapled_amps_features.csv "
                         "via in-place swap; corrupts dataset if interrupted")
def test_NEVER_run_dataset_comparison():
    _help("comparison/run_dataset_comparison.py")


# ──────────────────────────────────────────────────────────────────────────────
# feature_extraction/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_bio
def test_help_build_geometric_features():
    _help("feature_extraction/build_geometric_features.py")


@requires_torch
def test_help_generate_stapep_structures():
    """--help exits before any model load (no GPU/download needed)."""
    _help("feature_extraction/generate_stapep_structures.py")


# ──────────────────────────────────────────────────────────────────────────────
# models/ slice
# ──────────────────────────────────────────────────────────────────────────────

def test_help_extract_structure_features():
    """CPU-only (numpy/pandas) — no torch, no model load."""
    _help("models/extract_structure_features.py")


@requires_torch
def test_help_batch_esmfold():
    """--help exits during parse_args(), before any ESMFold load. Also proves
    the standalone sys.path bootstrap + utils imports resolve outside pytest."""
    _help("models/batch_esmfold.py")


@requires_torch
def test_help_run_esmfold_peptides():
    _help("models/run_esmfold_peptides.py")


@requires_torch
def test_help_esm_sequence_processor():
    _help("models/esm_sequence_processor.py")


# ──────────────────────────────────────────────────────────────────────────────
# gnn/ slice
# ──────────────────────────────────────────────────────────────────────────────
# Only run_gnn_training.py is argparse-driven; the other four gnn scripts run
# main() directly from a hardcoded CONFIG (no --help). --help exits during
# parse_args(), before any dataset load or GPU work. Requires torch_geometric
# (import of gnn.data_utils), so SKIPS under esm_env — runs under the WSL venv.


@requires_torch
@requires_tg
@requires_bio
def test_help_run_gnn_training():
    _help("gnn/run_gnn_training.py")


# ──────────────────────────────────────────────────────────────────────────────
# figures/ slice
# ──────────────────────────────────────────────────────────────────────────────

@requires_skl
def test_help_plot_mic_distribution():
    """argparse (--save); --help exits before any data load. Imports sklearn +
    matplotlib(Agg) + scipy at module top, so it also proves the standalone
    utils/features bootstrap resolves outside pytest."""
    _help("figures/plot_mic_distribution.py")


# ──────────────────────────────────────────────────────────────────────────────
# top-level StaPep MD drivers (live OUTSIDE the package, at the repo root).
# Only run_buforin_stapep.py is argparse-driven; its stapep/OpenMM imports are
# function-local, so --help parses and exits without needing the WSL stap env.
# The other three (run_buf_variants_stapep, test_buf_variant_single,
# test_wsl_stapep) have no argparse and would start real MD / import stapep at
# module level, so they are intentionally not smoke-tested here.
# ──────────────────────────────────────────────────────────────────────────────

def test_help_run_buforin_stapep():
    # Top-level drivers live at the workspace root: seq2svm/ -> ...Prediction/ ->
    # SVM_ESM_Peptides/. That's two parents up from ROOT (= sequence_to_svm_minimal).
    repo_root = ROOT.parent.parent
    script = repo_root / "run_buforin_stapep.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=60, cwd=str(repo_root),
    )
    assert result.returncode == 0, (
        f"run_buforin_stapep.py --help exited {result.returncode}\n"
        f"stderr: {result.stderr[:600]}"
    )
    assert "usage" in (result.stdout + result.stderr).lower()
