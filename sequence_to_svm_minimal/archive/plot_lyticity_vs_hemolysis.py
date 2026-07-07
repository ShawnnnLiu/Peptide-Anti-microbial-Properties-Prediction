#!/usr/bin/env python3
"""
plot_lyticity_vs_hemolysis.py
=============================
Two-panel scatter plot:
  Panel A — Training set: lyticity_index vs % hemolysis (normalised)
  Panel B — Buforin variants: lyticity_index vs literature % hemolysis
            with labels for each variant
"""

import re, sys, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress
from pathlib import Path

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent
DATA_DIR = BASE / "data" / "training_dataset" / "StaPep"

AMPS_META = DATA_DIR / "stapled_amps.csv"
AMPS_FEAT = DATA_DIR / "stapled_amps_features.csv"
F10W_FEAT = DATA_DIR / "test_buf_specific_stapep_features.csv"
TEST_FEAT = DATA_DIR / "test_stapled_features.csv"

# ── Hemolysis parsing (reuse from v2) ────────────────────────────────────────
def _strip_op(s):
    return re.sub(r'^[<>≤≥\s]+', '', s)

def _to_uM(val, unit, mw_da):
    unit = unit.lower().strip()
    if unit in ("μm", "um"):
        return val
    if unit in ("μg/ml", "ug/ml") and mw_da > 0:
        return val / mw_da * 1000.0
    return None

_HEM_SINGLE_RE = re.compile(
    r"([<>≤]?\s*[\d.]+)\s*%\s*hemol"
    r"[^.;#\n]*?"
    r"\bat\s+(?:peptide\s+)?(?:concentration\s+of\s+)?"
    r"([<>≤]?\s*[\d.]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

_LIST_PCT_RE  = re.compile(r"([<>≤]?\s*[\d.]+)\s*%")
_LIST_CONC_RE = re.compile(r"([\d.]+)")

_LC50_RE = re.compile(
    r"(?:LC50|HC50|LC\s*50|HC\s*50)\s*[=:>< ]+\s*([><=]?\s*[\d.]+)\s*([μu]M|[μu]g/mL)",
    re.IGNORECASE,
)

_LIST_AGAINST_RE = re.compile(
    r"([\d.<>≤%, ]+)\s+against\s+.*?\bat\s+(?:peptide\s+)?(?:concentrations?\s+of\s+)?"
    r"([\d.,\s]+)\s*([μu]g/mL|[μu]M)",
    re.IGNORECASE,
)

def extract_hemolysis(text, mw_da, ref_conc_uM):
    if not isinstance(text, str):
        return None

    for m in _HEM_SINGLE_RE.finditer(text):
        pct_s, conc_s, unit = m.group(1), m.group(2), m.group(3)
        try:
            pct  = float(_strip_op(pct_s))
            conc = float(_strip_op(conc_s))
        except ValueError:
            continue
        conc_uM = _to_uM(conc, unit, mw_da)
        if conc_uM and conc_uM > 0:
            return pct, conc_uM, "single"

    m_hemol = re.search(r"hemol\w*\s+against", text, re.I)
    if m_hemol:
        before = text[:m_hemol.start()]
        after  = text[m_hemol.end():]
        pct_vals = [float(_strip_op(m.group(1))) for m in _LIST_PCT_RE.finditer(before)
                    if _strip_op(m.group(1))]
        m_at = re.search(
            r"\bat\s+(?:peptide\s+)?(?:concentrations?\s+of\s+)?"
            r"([\d.,\s]+(?:\s+and\s+[\d.]+)?)\s*([μu]g/mL|[μu]M)",
            after, re.I
        )
        if m_at and pct_vals:
            conc_block = m_at.group(1).replace(" and ", ",")
            unit = m_at.group(2)
            conc_vals = [float(mc.group(1)) for mc in _LIST_CONC_RE.finditer(conc_block)]
            if conc_vals:
                n_pairs = min(len(pct_vals), len(conc_vals))
                pairs = []
                for i in range(n_pairs):
                    c_uM = _to_uM(conc_vals[i], unit, mw_da)
                    if c_uM and c_uM > 0:
                        pairs.append((pct_vals[i], c_uM))
                if pairs:
                    best = min(pairs, key=lambda x: abs(x[1] - ref_conc_uM))
                    return best[0], best[1], "list"

    m_ag = _LIST_AGAINST_RE.search(text)
    if m_ag:
        pct_block = m_ag.group(1)
        conc_block = m_ag.group(2).replace(" and ", ",")
        unit = m_ag.group(3)
        pct_vals = [float(_strip_op(m.group(1))) for m in _LIST_PCT_RE.finditer(pct_block)
                    if _strip_op(m.group(1))]
        conc_vals = [float(m.group(1)) for m in _LIST_CONC_RE.finditer(conc_block)]
        if pct_vals and conc_vals:
            n_pairs = min(len(pct_vals), len(conc_vals))
            pairs = []
            for i in range(n_pairs):
                c_uM = _to_uM(conc_vals[i], unit, mw_da)
                if c_uM and c_uM > 0:
                    pairs.append((pct_vals[i], c_uM))
            if pairs:
                best = min(pairs, key=lambda x: abs(x[1] - ref_conc_uM))
                return best[0], best[1], "list_nohem"

    m_lc = _LC50_RE.search(text)
    if m_lc:
        lc_s, unit = m_lc.group(1), m_lc.group(2)
        try:
            lc_val = float(_strip_op(lc_s))
        except ValueError:
            return None
        lc_uM = _to_uM(lc_val, unit, mw_da)
        if lc_uM and lc_uM > 0:
            pct_at_ref = 100.0 * ref_conc_uM / (ref_conc_uM + lc_uM)
            return pct_at_ref, ref_conc_uM, "lc50_hill"

    return None


# ── Literature hemolysis for Buforin variants (raw % at 50 ug/mL) ────────────
BUFORIN_DATA = [
    # (label,                   lyticity_index, hemo_raw%, MW)
    ("Buf WT\n(native)",         300.1,  None,   2473.8),
    ("Buf(i+4)16\n(F10W)",       508.4,  12.6,   2429.9),
    ("Buf(i+4)14\n(F10W)",       489.5,   2.9,   2453.8),
    ("Buf(i+4)4\n(F10W)",        461.3,   2.4,   2523.0),
    ("Buf(i+4)3\n(F10W)",        474.9,   3.1,   2579.1),
    ("Buf(i+7)9\n(F10W)",        652.1,  57.0,   2500.0),
    ("Buf(i+7)6\n(F10W)",        396.5,   3.0,   2637.2),
    ("Buf(i+7)1\n(F10W)",        322.0,   2.3,   2551.0),
    ("Buf(i+4)12",               502.9,   9.23,  2491.9),
    ("Buf(i+4)13",               403.1,   3.84,  2515.0),
    ("Buf(i+4)13\nQ9K",          403.1,  None,   2515.0),
    ("Buf(i+4)12\nV15K,L19K",    302.3,  None,   2536.0),
]


def main():
    print("=" * 70)
    print("  Lyticity Index vs % Hemolysis — Correlation Plot")
    print("=" * 70)

    # ── 1. Load training data ────────────────────────────────────────────────
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    feat["hydrophobic_sasa"] = feat["sasa"] - feat["psa"]

    df = pd.merge(
        meta[["DRAMP_ID", "Hemolytic_Activity"]],
        feat[["DRAMP_ID", "lyticity_index", "weight"]],
        on="DRAMP_ID", how="inner",
    ).dropna(subset=["lyticity_index"])

    # First pass for ref_conc
    first_pass = df.apply(
        lambda r: extract_hemolysis(str(r["Hemolytic_Activity"]), r["weight"], 10.0),
        axis=1,
    )
    temp_concs = [x[1] for x in first_pass if x is not None and x[2] == "single"]
    ref_conc = float(np.median(temp_concs)) if temp_concs else 10.0

    # Second pass
    extracted = df.apply(
        lambda r: extract_hemolysis(str(r["Hemolytic_Activity"]), r["weight"], ref_conc),
        axis=1,
    )
    df["pct_raw"]  = extracted.apply(lambda x: x[0] if x else None)
    df["conc_uM"]  = extracted.apply(lambda x: x[1] if x else None)
    df["method"]   = extracted.apply(lambda x: x[2] if x else None)
    df_parsed = df.dropna(subset=["pct_raw", "conc_uM"]).copy()

    df_parsed["pct_norm"] = df_parsed.apply(
        lambda r: r["pct_raw"] if r["method"] == "lc50_hill"
                  else r["pct_raw"] * (ref_conc / r["conc_uM"]),
        axis=1,
    ).clip(0, 100)

    n_train = len(df_parsed)
    lyti_train = df_parsed["lyticity_index"].values
    hemo_train = df_parsed["pct_norm"].values

    r_train, p_train = pearsonr(lyti_train, hemo_train)
    rho_train, p_rho = spearmanr(lyti_train, hemo_train)
    print(f"\n  Training set (n={n_train}):")
    print(f"    Pearson R  = {r_train:.3f}  (p={p_train:.2e})")
    print(f"    Spearman r = {rho_train:.3f}  (p={p_rho:.2e})")

    # ── 2. Buforin variant data ──────────────────────────────────────────────
    buf_with_lit = [(l, ly, h) for l, ly, h, _ in BUFORIN_DATA if h is not None]
    buf_no_lit   = [(l, ly, h) for l, ly, h, _ in BUFORIN_DATA if h is None]

    buf_lyti = np.array([b[1] for b in buf_with_lit])
    buf_hemo = np.array([b[2] for b in buf_with_lit])

    r_buf, p_buf = pearsonr(buf_lyti, buf_hemo)
    rho_buf, p_rho_buf = spearmanr(buf_lyti, buf_hemo)
    print(f"\n  Buforin variants (n={len(buf_with_lit)}):")
    print(f"    Pearson R  = {r_buf:.3f}  (p={p_buf:.4f})")
    print(f"    Spearman r = {rho_buf:.3f}  (p={p_rho_buf:.4f})")

    # ── 3. Figure (3 panels) ────────────────────────────────────────────────
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(22, 7),
                                         gridspec_kw={"width_ratios": [1, 1, 1]})

    # ── Panel A: Full training set with 3-zone annotation ────────────────────
    ax0.scatter(lyti_train, hemo_train, alpha=0.5, s=40, color="#2c7bb6",
                edgecolors="white", linewidths=0.3, zorder=3,
                label=f"All training (n={n_train})")

    # Shade the 3 zones
    ax0.axvspan(ax0.get_xlim()[0] if ax0.get_xlim()[0] < 100 else 80, 400,
                alpha=0.08, color="#2ca02c", zorder=1)
    ax0.axvspan(400, 700, alpha=0.08, color="#ff7f0e", zorder=1)
    ax0.axvspan(700, 850, alpha=0.08, color="#d62728", zorder=1)

    ax0.axvline(400, color="#333", lw=1.0, ls="--", alpha=0.5)
    ax0.axvline(700, color="#333", lw=1.0, ls="--", alpha=0.5)

    ax0.text(250, 98, "Zone I\n< 400\nNo hemolysis", ha="center",
             fontsize=8.5, fontweight="bold", color="#2ca02c", alpha=0.8)
    ax0.text(550, 98, "Zone II\n400\u2013700\nTransition", ha="center",
             fontsize=8.5, fontweight="bold", color="#ff7f0e", alpha=0.8)
    ax0.text(750, 98, "Zone III\n> 700\nHighly lytic", ha="center",
             fontsize=8.5, fontweight="bold", color="#d62728", alpha=0.8)

    # Linear fit for reference
    sl, ic, *_ = linregress(lyti_train, hemo_train)
    x_full = np.linspace(100, 830, 300)
    ax0.plot(x_full, np.clip(sl * x_full + ic, 0, 100), "--", color="#999",
             lw=1.0, alpha=0.6, label=f"Linear ref (R={r_train:.2f})")

    ax0.text(0.05, 0.60,
             f"Pearson R  = {r_train:.2f}\n"
             f"Spearman \u03c1 = {rho_train:.2f}\n"
             f"n = {n_train}",
             transform=ax0.transAxes, fontsize=10, va="top",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="lightyellow", alpha=0.85))

    ax0.set_xlabel("Lyticity Index", fontsize=11)
    ax0.set_ylabel(f"% Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=11)
    ax0.set_title("Panel A \u2014 Full Training Set\n"
                  "Three behaviour zones",
                  fontsize=11, fontweight="bold")
    ax0.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.48))
    ax0.grid(alpha=0.2, lw=0.5)
    ax0.set_ylim(-5, 105)
    ax0.set_xlim(80, 830)

    # ── Panel B: Transition zone (400-700) with quadratic fit ────────────────
    zone_mask = (lyti_train >= 400) & (lyti_train <= 700)
    lyti_zone = lyti_train[zone_mask]
    hemo_zone = hemo_train[zone_mask]
    n_zone = len(lyti_zone)

    ax1.scatter(lyti_zone, hemo_zone, alpha=0.6, s=50, color="#2c7bb6",
                edgecolors="white", linewidths=0.3, zorder=3,
                label=f"Zone II peptides (n={n_zone})")

    # Quadratic (parabolic) fit
    coeffs = np.polyfit(lyti_zone, hemo_zone, 2)
    x_zone = np.linspace(390, 710, 300)
    y_quad = np.polyval(coeffs, x_zone)
    ax1.plot(x_zone, np.clip(y_quad, 0, 100), "-", color="#d62728", lw=2.5,
             zorder=4, label=f"Quadratic fit (y = {coeffs[0]:.4f}x\u00b2 ...)")

    # Linear fit for comparison
    sl_z, ic_z, *_ = linregress(lyti_zone, hemo_zone)
    ax1.plot(x_zone, sl_z * x_zone + ic_z, "--", color="#999", lw=1.2,
             alpha=0.7, label=f"Linear fit (slope={sl_z:.3f})")

    # Stats for zone
    r_zone, p_zone = pearsonr(lyti_zone, hemo_zone)
    rho_zone, _ = spearmanr(lyti_zone, hemo_zone)

    # R^2 for quadratic vs linear
    y_pred_quad = np.polyval(coeffs, lyti_zone)
    ss_res_quad = np.sum((hemo_zone - y_pred_quad) ** 2)
    ss_tot = np.sum((hemo_zone - np.mean(hemo_zone)) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else 0

    y_pred_lin = sl_z * lyti_zone + ic_z
    ss_res_lin = np.sum((hemo_zone - y_pred_lin) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0

    ax1.text(0.05, 0.95,
             f"Pearson R  = {r_zone:.2f}\n"
             f"Spearman \u03c1 = {rho_zone:.2f}\n"
             f"n = {n_zone}\n\n"
             f"R\u00b2 linear    = {r2_lin:.3f}\n"
             f"R\u00b2 quadratic = {r2_quad:.3f}",
             transform=ax1.transAxes, fontsize=9.5, va="top",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="lightyellow", alpha=0.85))

    ax1.set_xlabel("Lyticity Index", fontsize=11)
    ax1.set_ylabel(f"% Hemolysis (norm. to {ref_conc:.0f} uM)", fontsize=11)
    ax1.set_title("Panel B \u2014 Transition Zone (400\u2013700)\n"
                  "Quadratic fit on lyticity 400\u2013700",
                  fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.55))
    ax1.grid(alpha=0.2, lw=0.5)
    ax1.set_xlim(390, 720)
    ax1.set_ylim(-3, max(hemo_zone) * 1.2 + 5)

    # ── Panel B: Buforin variants ────────────────────────────────────────────
    # Variants with literature hemolysis
    i4_mask = ["i+4" in b[0] for b in buf_with_lit]
    i7_mask = ["i+7" in b[0] for b in buf_with_lit]
    other_mask = [not (a or b) for a, b in zip(i4_mask, i7_mask)]

    for mask, color, marker, label in [
        (i4_mask, "#4363d8", "o", "i+4 staples"),
        (i7_mask, "#e6194b", "^", "i+7 staples"),
        (other_mask, "#3cb44b", "s", "Other"),
    ]:
        xl = [buf_lyti[i] for i, m in enumerate(mask) if m]
        yl = [buf_hemo[i] for i, m in enumerate(mask) if m]
        if xl:
            ax2.scatter(xl, yl, s=120, marker=marker, color=color,
                        edgecolors="k", linewidths=0.7, zorder=5, label=label)

    # Annotate each point
    for name, ly, he in buf_with_lit:
        x_off, y_off = 8, 5
        # Avoid overlap for clustered low-hemo points
        if he < 5 and ly > 450:
            y_off = -12
        if he > 40:
            x_off = -80
            y_off = -10
        if "Buf(i+4)12" == name:
            y_off = 8
        ax2.annotate(name.replace("\n", " "), (ly, he),
                     textcoords="offset points", xytext=(x_off, y_off),
                     fontsize=7, fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color="#777", lw=0.5))

    # Variants WITHOUT literature (show as open markers at y=0 area)
    for name, ly, _, _ in BUFORIN_DATA:
        if name in [b[0] for b in buf_no_lit]:
            ax2.scatter(ly, -3, s=100, marker="D", facecolors="none",
                        edgecolors="#999", linewidths=1.5, zorder=5)
            ax2.annotate(name.replace("\n", " "), (ly, -3),
                         textcoords="offset points", xytext=(8, -8),
                         fontsize=6.5, color="#999", style="italic")

    # Quadratic fit on Buforin variants
    buf_coeffs = np.polyfit(buf_lyti, buf_hemo, 2)
    x_buf = np.linspace(280, 680, 200)
    y_buf_quad = np.polyval(buf_coeffs, x_buf)
    ax2.plot(x_buf, np.clip(y_buf_quad, -5, 70), "-", color="#d62728",
             lw=2.0, zorder=4, label="Quadratic fit")

    # Linear fit (faded reference)
    sl2, ic2, *_ = linregress(buf_lyti, buf_hemo)
    ax2.plot(x_buf, np.clip(sl2 * x_buf + ic2, -5, 70), "--", color="#999",
             lw=1.0, alpha=0.6, label=f"Linear ref (R={r_buf:.2f})")

    ax2.text(0.05, 0.95,
             f"Pearson R  = {r_buf:.2f}\n"
             f"Spearman \u03c1 = {rho_buf:.2f}\n"
             f"n = {len(buf_with_lit)}",
             transform=ax2.transAxes, fontsize=10, va="top",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="lightyellow", alpha=0.85))

    ax2.set_xlabel("Lyticity Index", fontsize=11)
    ax2.set_ylabel("% Hemolysis (raw, at 50 ug/mL)", fontsize=11)
    ax2.set_title("Panel C \u2014 Buforin Variants\n"
                  "Lyticity Index vs Literature % Hemolysis",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.76))
    ax2.grid(alpha=0.2, lw=0.5)
    ax2.set_ylim(-8, max(buf_hemo) * 1.15)
    ax2.axhline(0, color="#333", lw=0.5, ls="-")

    # Hemolysis threshold lines
    for ax in [ax1, ax2]:
        for yv, lbl, lc in [(5, "5%", "#2ca02c"), (15, "15%", "#ff7f0e"),
                             (40, "40%", "#d62728")]:
            if yv < ax.get_ylim()[1]:
                ax.axhline(yv, color=lc, ls=":", lw=0.8, alpha=0.6)
                ax.text(ax.get_xlim()[1], yv + 1, f" {lbl}", fontsize=7,
                        color=lc, va="bottom", ha="right")

    fig.suptitle(
        "Lyticity Index vs % Hemolysis\n"
        "StaPep lyticity_index = amphipathic helical patterning score",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = BASE / "lyticity_vs_hemolysis.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved -> {out}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
