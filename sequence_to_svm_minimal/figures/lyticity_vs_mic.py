#!/usr/bin/env python3
"""
lyticity_vs_mic.py
==================
Lyticity Index (StaPep amphipathic helical patterning score) vs MIC (E. coli).

Three-panel figure:
  Panel A – Full training set with exponential-decay fit and behaviour zones
  Panel B – Transition zone (lyticity 400-700) with exponential-decay fit
  Panel C – Buforin variants: lyticity index vs literature MIC

Usage:
    conda run -n esm_env python lyticity_vs_mic.py
"""

import io, re, sys, math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# Bootstrap project root so a standalone ``python figures/lyticity_vs_mic.py``
# can reach the shared ``utils`` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import PROJECT_ROOT, STAPEP_DIR

warnings.filterwarnings("ignore")

DATA = STAPEP_DIR

# ── MIC parsing (same as predict_mic_svm.py) ────────────────────────────────
_COLI = r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+\w+)*\s*"
_MIC_UM   = r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M"
_MIC_UGML = r"\([^)]*?MIC[\w.]*\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL"
_RE_UM    = re.compile(_COLI + _MIC_UM,   re.IGNORECASE)
_RE_UGML  = re.compile(_COLI + _MIC_UGML, re.IGNORECASE)
_GENERIC_UM   = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]M", re.I)
_GENERIC_UGML = re.compile(r"E\.?\s*coli[^)]*?MIC[\w.]*\s*=?\s*([><=\u2265]?)\s*([\d.]+)\s*[\u00b5\u03bcuU]g/mL", re.I)


def _parse(m):
    if m is None:
        return None
    mod = m.group(1).strip()
    if mod in (">", "<", "\u2265"):
        return None
    try:
        return float(m.group(2))
    except Exception:
        return None


def get_mic_ugml(text, mw):
    if not isinstance(text, str):
        return None
    v = _parse(_RE_UGML.search(text))
    if v is not None:
        return v
    v = _parse(_RE_UM.search(text))
    if v is not None and mw > 0:
        return v * mw / 1000.0
    v = _parse(_GENERIC_UGML.search(text))
    if v is not None:
        return v
    v = _parse(_GENERIC_UM.search(text))
    if v is not None and mw > 0:
        return v * mw / 1000.0
    return None


def pearson_safe(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    return stats.pearsonr(x, y)


def spearman_safe(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    return stats.spearmanr(x, y)


# ── Exponential decay: MIC = a * exp(-b * lyticity) + c ─────────────────────
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c


def fit_exp_decay(x, y, p0=None):
    """Fit exponential decay, return (popt, pcov) or None on failure."""
    if p0 is None:
        p0 = [np.max(y), 0.005, np.min(y)]
    try:
        popt, pcov = curve_fit(exp_decay, x, y, p0=p0,
                               maxfev=20000, method="trf",
                               bounds=([0, 0, 0], [np.inf, 1.0, np.inf]))
        return popt, pcov
    except Exception:
        pass
    try:
        popt, pcov = curve_fit(exp_decay, x, y, p0=p0, maxfev=20000)
        return popt, pcov
    except Exception:
        return None


# ── Buf WT features ─────────────────────────────────────────────────────────
BUF_WT = {
    "lyticity_index": 300.106,
    "weight": 2473.829,
}

LITERATURE_MIC = {
    "Buf_i4_16_F10W":  {"mic_ugml": 5.2,   "mw": 2429.9},
    "Buf_i4_14_F10W":  {"mic_ugml": 29.2,  "mw": 2453.8},
    "Buf_i4_4_F10W":   {"mic_ugml": 100.0, "mw": 2523.0},
    "Buf_i4_3_F10W":   {"mic_ugml": 6.3,   "mw": 2579.1},
    "Buf_i7_9_F10W":   {"mic_ugml": 3.1,   "mw": 2500.0},
    "Buf_i7_6_F10W":   {"mic_ugml": 22.9,  "mw": 2637.2},
    "Buf12":           {"mic_ugml": 6.25,  "mw": 2491.93},
    "Buf13":           {"mic_ugml": 100.0, "mw": 2514.96},
}


def main():
    # UTF-8 stdout for the μ/ρ glyphs below. Kept inside main() (not at module
    # top) so importing this module never mutates the global sys.stdout.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 72)
    print("  Lyticity Index vs MIC (E. coli)")
    print("  StaPep lyticity_index = amphipathic helical patterning score")
    print("=" * 72)

    # ── 1. Load training AMPs with MIC ───────────────────────────────────────
    meta = pd.read_csv(DATA / "stapled_amps.csv")
    feat = pd.read_csv(DATA / "stapled_amps_features.csv")

    df = pd.merge(
        meta[["DRAMP_ID", "Target_Organism"]],
        feat[["DRAMP_ID", "lyticity_index", "weight"]],
        on="DRAMP_ID", how="inner",
    )
    df["mic_ugml"] = df.apply(
        lambda r: get_mic_ugml(r["Target_Organism"], r["weight"]), axis=1
    )
    df = df[df["mic_ugml"].notna() & (df["mic_ugml"] > 0)].copy()
    df = df.dropna(subset=["lyticity_index"]).reset_index(drop=True)

    lyt_train = df["lyticity_index"].values
    mic_train = df["mic_ugml"].values
    n_train = len(df)

    print(f"\n  Training AMPs with E. coli MIC & lyticity_index: {n_train}")
    print(f"  Lyticity range: {lyt_train.min():.0f} – {lyt_train.max():.0f}")
    print(f"  MIC range: {mic_train.min():.1f} – {mic_train.max():.1f} ug/mL")

    r_full, p_full = pearson_safe(lyt_train, mic_train)
    rho_full, prho_full = spearman_safe(lyt_train, mic_train)
    print(f"\n  Full set:  Pearson R = {r_full:.2f},  Spearman rho = {rho_full:.2f}")

    # ── 2. Load Buforin test variants ────────────────────────────────────────
    test_f10w = pd.read_csv(DATA / "test_buf_specific_stapep_features.csv")
    test_orig = pd.read_csv(DATA / "test_stapled_features.csv")
    test_orig = test_orig[test_orig["peptide_id"].str.startswith("Buf")].copy()

    buf_variants = []

    for _, row in test_f10w.iterrows():
        pid = row["peptide_id"]
        lit = LITERATURE_MIC.get(pid, {})
        mic = lit.get("mic_ugml") if lit else None
        if row.get("lit_mic_ugml") and not pd.isna(row.get("lit_mic_ugml")):
            mic = row["lit_mic_ugml"]
        if mic is None:
            continue
        group = "i+7" if "i7" in pid else "i+4"
        short = pid.replace("Buf_", "").replace("_F10W", "")
        buf_variants.append({
            "name": pid, "short": short + " (F10W)",
            "lyticity": row["lyticity_index"],
            "mic_ugml": mic,
            "group": group,
        })

    for _, row in test_orig.iterrows():
        pid = row["peptide_id"]
        lit = LITERATURE_MIC.get(pid, {})
        mic = lit.get("mic_ugml") if lit else None
        if mic is None:
            continue
        buf_variants.append({
            "name": pid, "short": pid,
            "lyticity": row["lyticity_index"],
            "mic_ugml": mic,
            "group": "Original",
        })

    n_buf = len(buf_variants)
    print(f"\n  Buforin variants with literature MIC: {n_buf}")
    for v in buf_variants:
        print(f"    {v['name']:<22}  lyticity={v['lyticity']:.0f}  "
              f"MIC={v['mic_ugml']:.1f} ug/mL  ({v['group']})")

    buf_lyt = np.array([v["lyticity"] for v in buf_variants])
    buf_mic = np.array([v["mic_ugml"] for v in buf_variants])

    r_buf, p_buf = pearson_safe(buf_lyt, buf_mic)
    rho_buf, _ = spearman_safe(buf_lyt, buf_mic)
    print(f"\n  Buforin:  Pearson R = {r_buf:.2f},  Spearman rho = {rho_buf:.2f}")

    # ── 3. Exponential decay fits ────────────────────────────────────────────
    fit_full = fit_exp_decay(lyt_train, mic_train)
    if fit_full:
        popt_f, _ = fit_full
        print(f"\n  Full-set exp decay: a={popt_f[0]:.1f}, b={popt_f[1]:.5f}, c={popt_f[2]:.1f}")

    mask_tz = (lyt_train >= 400) & (lyt_train <= 700)
    lyt_tz = lyt_train[mask_tz]
    mic_tz = mic_train[mask_tz]
    n_tz = mask_tz.sum()
    fit_tz = fit_exp_decay(lyt_tz, mic_tz) if n_tz >= 5 else None

    r_tz, p_tz = pearson_safe(lyt_tz, mic_tz)
    rho_tz, _ = spearman_safe(lyt_tz, mic_tz)
    print(f"  Transition zone (400–700): n={n_tz}, R={r_tz:.2f}, rho={rho_tz:.2f}")

    fit_buf = fit_exp_decay(buf_lyt, buf_mic) if n_buf >= 3 else None
    if fit_buf:
        popt_b, _ = fit_buf
        print(f"  Buforin exp decay: a={popt_b[0]:.1f}, b={popt_b[1]:.5f}, c={popt_b[2]:.1f}")

    # Also fit linear on log(MIC) for R² comparison
    slope_f, int_f, r_log_f, _, _ = stats.linregress(lyt_train, np.log10(mic_train))
    r2_lin_f = r_log_f ** 2
    print(f"\n  Full set log-linear: R = {r_log_f:.3f}, R² = {r2_lin_f:.3f}")

    if n_tz >= 3:
        slope_tz, int_tz, r_log_tz, _, _ = stats.linregress(lyt_tz, np.log10(mic_tz))
        r2_lin_tz = r_log_tz ** 2
    else:
        r_log_tz = r2_lin_tz = np.nan

    # R² for exponential fits
    if fit_full:
        y_pred_f = exp_decay(lyt_train, *popt_f)
        ss_res_f = np.sum((mic_train - y_pred_f) ** 2)
        ss_tot_f = np.sum((mic_train - np.mean(mic_train)) ** 2)
        r2_exp_f = 1 - ss_res_f / ss_tot_f
        print(f"  Full set exp-decay R² = {r2_exp_f:.3f}")
    else:
        r2_exp_f = np.nan

    if fit_tz:
        popt_tz, _ = fit_tz
        y_pred_tz = exp_decay(lyt_tz, *popt_tz)
        ss_res_tz = np.sum((mic_tz - y_pred_tz) ** 2)
        ss_tot_tz = np.sum((mic_tz - np.mean(mic_tz)) ** 2)
        r2_exp_tz = 1 - ss_res_tz / ss_tot_tz
    else:
        r2_exp_tz = np.nan

    # ═════════════════════════════════════════════════════════════════════════
    #  FIGURE
    # ═════════════════════════════════════════════════════════════════════════
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle(
        "Lyticity Index vs MIC (E. coli)\n"
        "StaPep lyticity_index = amphipathic helical patterning score",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # ── Panel A: Full Training Set ───────────────────────────────────────────
    ax = ax_a

    ax.axvspan(0, 400, alpha=0.08, color="green")
    ax.axvspan(400, 700, alpha=0.08, color="orange")
    ax.axvspan(700, 900, alpha=0.08, color="red")

    ax.scatter(lyt_train, mic_train, alpha=0.5, s=40, color="#5499C7",
               edgecolors="white", linewidths=0.3, zorder=3)

    if fit_full:
        x_fit = np.linspace(max(50, lyt_train.min() - 30),
                            lyt_train.max() + 30, 300)
        y_fit = exp_decay(x_fit, *popt_f)
        ax.plot(x_fit, y_fit, "-", color="#d62728", lw=2.5, alpha=0.8,
                label=f"Exp decay (R²={r2_exp_f:.3f})")

    x_lin = np.linspace(lyt_train.min() - 30, lyt_train.max() + 30, 300)
    y_lin = 10 ** (slope_f * x_lin + int_f)
    ax.plot(x_lin, y_lin, "--", color="#2ca02c", lw=1.5, alpha=0.6,
            label=f"Linear fit (R²={r2_lin_f:.3f})")

    stat_txt = (f"Pearson R  = {r_full:.2f}\n"
                f"Spearman ρ = {rho_full:.2f}\n"
                f"n = {n_train}")
    ax.text(0.03, 0.97, stat_txt, transform=ax.transAxes,
            fontsize=9, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#999", alpha=0.9))

    ax.set_xlabel("Lyticity Index", fontsize=11)
    ax.set_ylabel("MIC (μg/mL, E. coli)", fontsize=11)
    ax.set_title(f"Panel A — Full Training Set\nExponential decay fit",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)

    ymax_a = min(mic_train.max() * 1.05, 400)
    ax.set_ylim(bottom=-5, top=ymax_a)

    ax.text(200, ymax_a * 0.95, "< 400\nLow lyticity",
            ha="center", va="top", fontsize=7.5, color="green",
            fontweight="bold", alpha=0.8)
    ax.text(550, ymax_a * 0.95, "400-700\nTransition",
            ha="center", va="top", fontsize=7.5, color="darkorange",
            fontweight="bold", alpha=0.8)
    ax.text(780, ymax_a * 0.95, "> 700\nHighly lytic",
            ha="center", va="top", fontsize=7.5, color="red",
            fontweight="bold", alpha=0.8)

    # ── Panel B: Transition Zone (400-700) ───────────────────────────────────
    ax = ax_b

    ax.scatter(lyt_tz, mic_tz, alpha=0.55, s=50, color="#E67E22",
               edgecolors="white", linewidths=0.3, zorder=3,
               label=f"Zone II peptides (n={n_tz})")

    if fit_tz:
        x_fit2 = np.linspace(390, 710, 200)
        y_fit2 = exp_decay(x_fit2, *popt_tz)
        ax.plot(x_fit2, y_fit2, "-", color="#d62728", lw=2.5, alpha=0.8,
                label=f"Exp decay (R²={r2_exp_tz:.3f})")

    if n_tz >= 3:
        x_lin2 = np.linspace(390, 710, 200)
        y_lin2 = 10 ** (slope_tz * x_lin2 + int_tz)
        ax.plot(x_lin2, y_lin2, "--", color="#2ca02c", lw=1.5, alpha=0.6,
                label=f"Linear fit (R²={r2_lin_tz:.3f})")

    stat_txt2 = (f"Pearson R  = {r_tz:.2f}\n"
                 f"Spearman ρ = {rho_tz:.2f}\n"
                 f"n = {n_tz}")
    ax.text(0.03, 0.97, stat_txt2, transform=ax.transAxes,
            fontsize=9, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#999", alpha=0.9))

    ax.set_xlabel("Lyticity Index", fontsize=11)
    ax.set_ylabel("MIC (μg/mL, E. coli)", fontsize=11)
    ax.set_title(f"Panel B — Transition Zone (400–700)\nExponential decay fit on lyticity 400–700",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2)
    ax.set_xlim(390, 710)

    # ── Panel C: Buforin Variants ────────────────────────────────────────────
    ax = ax_c

    colors_map = {"i+4": "#4363d8", "i+7": "#e6194b", "Original": "#3cb44b"}
    markers_map = {"i+4": "o", "i+7": "^", "Original": "s"}

    ax.scatter(lyt_train, mic_train, alpha=0.12, s=15,
               color="#bbbbbb", edgecolors="none", zorder=1,
               label=f"Training AMPs (n={n_train})")

    for v in buf_variants:
        ax.scatter(v["lyticity"], v["mic_ugml"], s=120,
                   marker=markers_map.get(v["group"], "o"),
                   color=colors_map.get(v["group"], "#333"),
                   edgecolors="k", linewidths=0.8, zorder=5)
        ax.annotate(v["short"], (v["lyticity"], v["mic_ugml"]),
                    textcoords="offset points", xytext=(7, 5),
                    fontsize=7, fontweight="bold", alpha=0.9)

    if fit_buf:
        x_fit3 = np.linspace(max(50, buf_lyt.min() - 60),
                             buf_lyt.max() + 60, 300)
        y_fit3 = exp_decay(x_fit3, *popt_b)
        y_fit3 = np.clip(y_fit3, 0, None)
        ax.plot(x_fit3, y_fit3, "-", color="#d62728", lw=2.5, alpha=0.8,
                label="Exp decay fit")

    if n_buf >= 3:
        sl3, int3, r_log_b, _, _ = stats.linregress(buf_lyt, np.log10(buf_mic))
        x_lin3 = np.linspace(buf_lyt.min() - 60, buf_lyt.max() + 60, 200)
        y_lin3 = 10 ** (sl3 * x_lin3 + int3)
        ax.plot(x_lin3, y_lin3, "--", color="#2ca02c", lw=1.5, alpha=0.5,
                label=f"Linear fit (R²={r_log_b**2:.2f})")

    stat_txt3 = (f"Pearson R  = {r_buf:.2f}\n"
                 f"Spearman ρ = {rho_buf:.2f}\n"
                 f"n = {n_buf}")
    ax.text(0.03, 0.97, stat_txt3, transform=ax.transAxes,
            fontsize=9, va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#999", alpha=0.9))

    buf_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4363d8",
               markeredgecolor="k", markersize=9, label="i+4 stapled (F10W)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#e6194b",
               markeredgecolor="k", markersize=9, label="i+7 stapled (F10W)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3cb44b",
               markeredgecolor="k", markersize=9, label="Original Buf variants"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbb",
               markeredgecolor="none", markersize=7, label="Training AMPs"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    all_handles = handles + buf_legend
    ax.legend(handles=all_handles, fontsize=7.5, loc="upper right")

    ax.set_xlabel("Lyticity Index", fontsize=11)
    ax.set_ylabel("MIC (μg/mL, E. coli)", fontsize=11)
    ax.set_title(f"Panel C — Buforin Variants\nLyticity Index vs Literature MIC",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = PROJECT_ROOT / "lyticity_vs_mic.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
