"""Pure pMIC ↔ MIC unit conversions.

These are math-only helpers (log10 transforms); they do NOT parse free-text
MIC fields. MIC-string parsing is intentionally kept inline in each caller
per PROJECT_MAP §7e — five script families currently use slightly different
regex semantics, and centralizing them risks silent label changes.
"""
from __future__ import annotations

import math


def pmic_to_mic_uM(pmic: float) -> float:
    """pMIC = -log10(MIC in mol/L)  →  MIC in μM."""
    return 10 ** (6 - pmic)


def mic_to_pmic_uM(mic_uM: float) -> float:
    """MIC in μM  →  pMIC = -log10(MIC in mol/L) = 6 - log10(MIC_μM)."""
    return 6.0 - math.log10(mic_uM)


def pmic_to_mic_ugml(pmic: float, mw_da: float) -> float:
    """pMIC + molecular weight (Da)  →  MIC in μg/mL."""
    mic_uM = 10 ** (6.0 - pmic)
    return mic_uM * mw_da / 1000.0


def mic_to_pmic_ugml(mic_ugml: float, mw_da: float) -> float:
    """MIC in μg/mL + molecular weight (Da)  →  pMIC."""
    mic_uM = mic_ugml / mw_da * 1000.0
    return 6.0 - math.log10(mic_uM)
