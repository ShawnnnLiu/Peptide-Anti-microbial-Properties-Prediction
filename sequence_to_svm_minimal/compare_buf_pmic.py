#!/usr/bin/env python3
"""
compare_buf_pmic.py

Side-by-side comparison of:
  LEFT  — Literature pMIC values for the 7 stapled Buforin variants
          from the advisor's table (MIC in μg/mL, E. coli, corrected from mg/mL typo).
  RIGHT — RF-predicted pMIC for our 4 Buforin variants
          (Buf12, Buf13, Buf13_Q9K, Buf12_V15K_L19K) using StaPep MD features.

Note: The two sets are DIFFERENT staple-position variants of the same parent
      Buforin II sequence, so this is a family-level comparison, not 1-to-1.
"""

import math, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = "data/training_dataset/StaPep"
AMPS_META = f"{DATA_DIR}/stapled_amps.csv"
AMPS_FEAT = f"{DATA_DIR}/stapled_amps_features.csv"
TEST_FEAT = f"{DATA_DIR}/test_stapled_features.csv"

FEATURE_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent", "mean_bfactor", "mean_gyrate",
    "num_hbonds", "psa",
]

# ─── Literature table (advisor's table, MIC corrected: mg/mL → μg/mL) ────────
# Source: stapled Buforin variants tested against E. coli
# pMIC = 6 - log10(MIC_uM)  where  MIC_uM = MIC_ugml * 1000 / MW_Da
LITERATURE = [
    # name,             MW_Da,   MIC_ugml (corrected),  hemo_pct (at 50 μg/mL)
    ("Buf(i+4)16\n(F10W)", 2429.9, 5.2,   12.6),
    ("Buf(i+4)14\n(F10W)", 2453.8, 29.2,   2.9),
    ("Buf(i+4)4\n(F10W)",  2523.0, 100.0,  2.4),
    ("Buf(i+4)3\n(F10W)",  2579.1, 6.3,    3.1),
    ("Buf(i+7)9\n(F10W)",  2500.0, 3.1,   57.0),
    ("Buf(i+7)6\n(F10W)",  2637.2, 22.9,   3.0),
    ("Buf(i+7)1\n(F10W)",  2551.0, None,   2.3),  # >100, skip for pMIC
]

def mic_ugml_to_pmic(mic_ugml, mw_da):
    """Convert MIC in μg/mL to pMIC (= 6 - log10(MIC in μM))."""
    if mic_ugml is None:
        return None
    mic_uM = mic_ugml * 1000.0 / mw_da
    return 6.0 - math.log10(mic_uM)

def pmic_to_mic_uM(pmic):
    return 10 ** (6 - pmic)

def tier(pmic):
    mic = pmic_to_mic_uM(pmic)
    if mic <  2: return "Very Strong"
    if mic <  5: return "Strong"
    if mic < 10: return "Moderate"
    return "Weak"

TIER_COLORS = {
    "Very Strong": "#1a9641",
    "Strong":      "#a6d96a",
    "Moderate":    "#fdae61",
    "Weak":        "#d7191c",
}

def bar_color(pmic):
    return TIER_COLORS[tier(pmic)]

# ─── Native features ──────────────────────────────────────────────────────────
NATIVE_FEAT = {
    "length": 21, "weight": 2473.829, "hydrophobic_index": -0.8142857142857142,
    "charge": 6.094, "aromaticity": 0.047619, "isoelectric_point": 11.9999,
    "fraction_arginine": 0.23810, "fraction_lysine": 0.04762,
    "helix_percent": 0.17819, "loop_percent": 0.82105,
    "mean_bfactor": 573.434, "mean_gyrate": 12.001,
    "num_hbonds": 0, "psa": 1064.217,
}

# ─── Train RF on full stapled AMP dataset ─────────────────────────────────────
def load_and_train():
    import re

    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    df   = pd.merge(meta[["DRAMP_ID","Target_Organism","Sequence"]],
                    feat[["DRAMP_ID"] + FEATURE_COLS],
                    on="DRAMP_ID", how="inner").dropna(subset=FEATURE_COLS)

    _ECOLI_MIC_RE = re.compile(
        r"(?:E\.?\s?coli|Escherichia\s+coli)(?:\s+ATCC\s+\d+)?\s*"
        r"\(MIC(?:99\.9)?(?:[\d.]*)?\s*[=≥>]?\s*([\d.]+)\s*([μu]g/mL|[μu]M)\s*[,);]",
        re.IGNORECASE,
    )

    def get_mic_uM(row):
        text = str(row["Target_Organism"])
        m    = _ECOLI_MIC_RE.search(text)
        if not m: return None
        v, u = float(m.group(1)), m.group(2).lower()
        if u in ("μm","um"):      return v
        mw = row.get("weight", 0)
        if u in ("μg/ml","ug/ml") and mw > 0: return v / mw * 1000.0
        return None

    df["mic_uM"]  = df.apply(get_mic_uM, axis=1)
    df = df.dropna(subset=["mic_uM"])
    df["pMIC"]    = 6 - np.log10(df["mic_uM"].clip(lower=1e-6))

    X = df[FEATURE_COLS].values.astype(float)
    y = df["pMIC"].values

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # CV for R
    rf_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    cv    = KFold(n_splits=5, shuffle=True, random_state=42)
    y_cv  = cross_val_predict(rf_cv, X, y, cv=cv)
    r_cv, _ = stats.pearsonr(y, y_cv)

    print(f"  Training set: n={len(df)}  |  5-fold CV Pearson R = {r_cv:.3f}")
    return rf, r_cv

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Buforin pMIC: Literature vs. RF Model Predictions")
    print("=" * 65)

    print("\n  Training RF model …")
    rf, r_cv = load_and_train()

    # ── Literature pMIC ──────────────────────────────────────────────────────
    lit_pmics = []
    for name, mw, mic_ugml, hemo in LITERATURE:
        pm = mic_ugml_to_pmic(mic_ugml, mw)
        lit_pmics.append(pm)

    # ── Our model predictions ────────────────────────────────────────────────
    test = pd.read_csv(TEST_FEAT).dropna(subset=FEATURE_COLS)
    buf_test = test[test["peptide_id"].str.lower().str.startswith("buf")]

    our_labels = ["Buforin II\n(native)"] + buf_test["peptide_id"].tolist()
    our_weights = [NATIVE_FEAT["weight"]] + buf_test["weight"].tolist()

    our_Xs = [np.array([[NATIVE_FEAT[f] for f in FEATURE_COLS]])]
    for _, row in buf_test.iterrows():
        our_Xs.append(np.array([[row[f] for f in FEATURE_COLS]]))

    our_pmics = [float(rf.predict(x)[0]) for x in our_Xs]

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n  ── Literature (from advisor's table, corrected to μg/mL) ──")
    print(f"  {'Peptide':<22}  {'MIC μg/mL':>10}  {'MIC μM':>8}  {'pMIC':>6}  Tier")
    print("  " + "─" * 60)
    for (name, mw, mic_ugml, hemo), pm in zip(LITERATURE, lit_pmics):
        tag   = ">100 μg/mL" if mic_ugml is None else f"{mic_ugml:.1f}"
        mic_v = f"{mic_ugml*1000/mw:.2f}" if mic_ugml else "—"
        pm_s  = f"{pm:.3f}" if pm else "—"
        t     = tier(pm) if pm else "—"
        print(f"  {name.replace(chr(10),' '):<22}  {tag:>10}  {mic_v:>8}  {pm_s:>6}  {t}")

    print("\n  ── RF Model Predictions (our 4 Buf test variants) ──")
    print(f"  {'Peptide':<24}  {'pMIC':>6}  {'MIC μM':>8}  {'MIC μg/mL':>10}  Tier")
    print("  " + "─" * 62)
    for lab, pm, mw in zip(our_labels, our_pmics, our_weights):
        mic_uM   = pmic_to_mic_uM(pm)
        mic_ugml = mic_uM * mw / 1000.0
        t = tier(pm)
        print(f"  {lab.replace(chr(10),' '):<24}  {pm:.3f}  {mic_uM:>8.2f}  {mic_ugml:>10.2f}  {t}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax_lit, ax_our) = plt.subplots(1, 2, figsize=(15, 6),
                                          gridspec_kw={"wspace": 0.42})

    # ── Panel A: Literature ───────────────────────────────────────────────────
    lit_names  = [r[0] for r in LITERATURE]
    lit_hemo   = [r[3] for r in LITERATURE]
    valid_lit  = [(i, pm) for i, pm in enumerate(lit_pmics) if pm is not None]
    invalid_li = [i for i, pm in enumerate(lit_pmics) if pm is None]

    y_pos = np.arange(len(LITERATURE))
    colors_lit = [bar_color(pm) if pm else "#bbbbbb" for pm in lit_pmics]
    bars_lit = ax_lit.barh(y_pos, [pm if pm else 4.3 for pm in lit_pmics],
                            color=colors_lit, edgecolor="black", linewidth=0.7,
                            height=0.55)

    # Mark >100 bar with arrow
    for i in invalid_li:
        bars_lit[i].set_hatch("///")
        ax_lit.text(4.35, i, "  >100 μg/mL\n  (>pMIC 4.4)", va="center",
                    fontsize=7.5, color="#555")

    for i, pm in valid_lit:
        mw       = LITERATURE[i][1]
        mic_ugml = LITERATURE[i][2]
        mic_uM   = mic_ugml * 1000 / mw
        hemo     = lit_hemo[i]
        ax_lit.text(pm + 0.02, i,
                    f"  {pm:.2f}  ({mic_ugml:.1f} μg/mL / {mic_uM:.2f} μM)"
                    f"  |  {hemo:.0f}% hem.",
                    va="center", fontsize=7.5)

    ax_lit.set_yticks(y_pos)
    ax_lit.set_yticklabels(lit_names, fontsize=8.5)
    ax_lit.invert_yaxis()
    ax_lit.set_xlim(4.0, 6.5)
    ax_lit.set_xlabel("pMIC  (= 6 − log₁₀[MIC in μM])", fontsize=10)
    ax_lit.set_title("Panel A — Literature Values\n(Advisor's table, μg/mL corrected)",
                     fontsize=10, fontweight="bold")

    # Tier reference lines
    for mic_thresh, label in [(2, "2 μM"), (5, "5 μM"), (10, "10 μM")]:
        pm_line = 6 - math.log10(mic_thresh)
        ax_lit.axvline(pm_line, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax_lit.text(pm_line, -0.6, label, ha="center", fontsize=7, color="gray")

    # Annotate staple type groups
    ax_lit.axhline(3.5, color="#555", lw=0.8, ls=":")
    ax_lit.text(4.02, 1.5, "i+4 staples", fontsize=7.5, color="#333",
                style="italic", va="center")
    ax_lit.text(4.02, 5.0, "i+7 staples", fontsize=7.5, color="#333",
                style="italic", va="center")

    # ── Panel B: Our RF predictions ───────────────────────────────────────────
    y_pos2   = np.arange(len(our_labels))
    our_cols = [bar_color(pm) for pm in our_pmics]
    hatches  = ["///" if "native" in l.lower() else "" for l in our_labels]

    bars2 = ax_our.barh(y_pos2, our_pmics, color=our_cols,
                         edgecolor="black", linewidth=0.7, height=0.55)
    for bar, h in zip(bars2, hatches):
        if h: bar.set_hatch(h)

    for i, (pm, mw, lab) in enumerate(zip(our_pmics, our_weights, our_labels)):
        mic_uM   = pmic_to_mic_uM(pm)
        mic_ugml = mic_uM * mw / 1000.0
        ax_our.text(pm + 0.02, i,
                    f"  {pm:.2f}  ({mic_uM:.2f} μM / {mic_ugml:.1f} μg/mL)",
                    va="center", fontsize=7.5)

    ax_our.set_yticks(y_pos2)
    ax_our.set_yticklabels(our_labels, fontsize=8.5)
    ax_our.invert_yaxis()
    ax_our.set_xlim(4.0, 6.5)
    ax_our.set_xlabel("Predicted pMIC  (= 6 − log₁₀[MIC in μM])", fontsize=10)
    ax_our.set_title(f"Panel B — RF Model Predictions\n"
                     f"(5-fold CV Pearson R = {r_cv:.2f}, our Buf variants)",
                     fontsize=10, fontweight="bold")

    for mic_thresh, label in [(2, "2 μM"), (5, "5 μM"), (10, "10 μM")]:
        pm_line = 6 - math.log10(mic_thresh)
        ax_our.axvline(pm_line, color="gray", lw=0.8, ls="--", alpha=0.6)
        ax_our.text(pm_line, -0.6, label, ha="center", fontsize=7, color="gray")

    # ── Shared legend ─────────────────────────────────────────────────────────
    tier_patches = [mpatches.Patch(color=c, label=f"{t} (MIC: {m})")
                    for t, (c, m) in {
                        "Very Strong": ("#1a9641", "<2 μM"),
                        "Strong":      ("#a6d96a", "2–5 μM"),
                        "Moderate":    ("#fdae61", "5–10 μM"),
                        "Weak":        ("#d7191c", ">10 μM"),
                    }.items()]
    tier_patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                        hatch="///", label="Non-stapled / native"))
    fig.legend(handles=tier_patches, loc="lower center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9,
               title="MIC Tier Color Code", title_fontsize=8.5)

    fig.suptitle(
        "Buforin Stapled Variants — Literature MIC vs. RF-Predicted pMIC\n"
        "Note: Panel A and B are DIFFERENT staple-position variants of the same parent "
        "Buforin II sequence\n(F10W mutation present in literature variants)",
        fontsize=9.5, fontweight="bold",
    )

    out = "buforin_pmic_comparison.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved → {out}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
