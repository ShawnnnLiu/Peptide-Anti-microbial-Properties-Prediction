"""Pytest configuration shared by all test modules.

- Adds the project root to ``sys.path`` so tests can ``import utils.paths``,
  ``import features.stapep_columns``, etc.
- Exposes ``requires_*`` skip markers for optional heavy dependencies so the
  smoke suite cleanly skips (not fails) on machines that don't have them.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


HAS_TORCH = _has("torch")
HAS_TG    = _has("torch_geometric")
HAS_BIO   = _has("Bio")
HAS_ESM   = _has("esm")
HAS_SKL   = _has("sklearn")
HAS_PROPY = _has("propy")
ON_WIN    = sys.platform == "win32"

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_tg    = pytest.mark.skipif(not HAS_TG,    reason="torch_geometric not installed")
requires_bio   = pytest.mark.skipif(not HAS_BIO,   reason="biopython not installed")
requires_esm   = pytest.mark.skipif(not HAS_ESM,   reason="fair-esm (esm_env) not active")
requires_skl   = pytest.mark.skipif(not HAS_SKL,   reason="scikit-learn not installed")
requires_propy = pytest.mark.skipif(not HAS_PROPY, reason="propy (QSAR descriptor library) not installed")
requires_wsl   = pytest.mark.skipif(ON_WIN,        reason="requires WSL + stap env")
