#!/usr/bin/env python3
"""
predict_pmic_stapled_variants.py

Runs the trained pMIC RF regression model on all 8 stapled Buforin/Magainin
variants (already MD-simulated, features in test_stapled_features.csv) and
compares them with the non-stapled native peptides.

Usage:
    python predict_pmic_stapled_variants.py
    python predict_pmic_stapled_variants.py --save stapled_pmic.png
"""

import re, math, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data/training_dataset/StaPep"
AMPS_META  = f"{DATA_DIR}/stapled_amps.csv"
AMPS_FEAT  = f"{DATA_DIR}/stapled_amps_features.csv"
TEST_FEAT  = f"{DATA_DIR}/test_stapled_features.csv"

FEATURE_COLS = [
    "length", "weight", "hydrophobic_index", "charge", "aromaticity",
    "isoelectric_point", "fraction_arginine", "fraction_lysine",
    "helix_percent", "loop_percent", "mean_bfactor", "mean_gyrate",
    "num_hbonds", "psa",
]

# ─── Non-stapled (native) features from 5 ns MD runs ─────────────────────────
NATIVE = {
    "Buforin II\n(native)": {
        "length": 21, "weight": 2473.829, "hydrophobic_index": -0.8142857142857142,
        "charge": 6.094, "aromaticity": 0.047619, "isoelectric_point": 11.9999,
        "fraction_arginine": 0.23810, "fraction_lysine": 0.04762,
        "helix_percent": 0.17819, "loop_percent": 0.82105,
        "mean_bfactor": 573.434, "mean_gyrate": 12.001,
        "num_hbonds": 0, "psa": 1064.217,
    },
    "Magainin II\n(native)": {
        "length": 23, "weight": 2466.832, "hydrophobic_index": 0.08261,
        "charge": 3.095, "aromaticity": 0.13043, "isoelectric_point": 10.0014,
        "fraction_arginine": 0.0, "fraction_lysine": 0.17391,
        "helix_percent": 0.16787, "loop_percent": 0.80809,
        "mean_bfactor": 895.020, "mean_gyrate": 12.213,
        "num_hbonds": 1, "psa": 918.001,
    },
}

# ─── MIC / pMIC helpers ───────────────────────────────────────────────────────
_COLI_ORG  = r"(?:Escherichia\s+coli|E\.?\s*coli)(?:\s+\w+)*\s*"
_MIC_UM    = r"\([^)]*?MIC[\w.]*\s*([><=≥]?)\s*([\d.]+)\s*[μu]M"
_MIC_UGML  = r"\([^)]*?MIC[\w.]*\s*([><=≥]?)\s*([\d.]+)\s*[μu]g/mL"
_RE_UM     = re.compile(_COLI_ORG + _MIC_UM,   re.IGNORECASE)
_RE_UGML   = re.compile(_COLI_ORG + _MIC_UGML, re.IGNORECASE)

def _parse(m):
    if m is None: return None
    if m.group(1).strip() in (">", "<"): return None
    try:    return float(m.group(2))
    except: return None

def get_mic_uM(text, mw):
    if not isinstance(text, str): return None
    v = _parse(_RE_UM.search(text))
    if v is not None: return v
    v = _parse(_RE_UGML.search(text))
    if v is not None and mw > 0: return v / mw * 1000
    return None

def pmic_to_mic(pmic): return 10 ** (6 - pmic)
def mic_to_pmic(mic_uM): return 6.0 - math.log10(mic_uM)

def tier(mic_uM):
    if mic_uM < 2:  return "Very Strong"
    if mic_uM < 5:  return "Strong"
    if mic_uM < 10: return "Moderate"
    return "Weak"

TIER_COLOR = {
    "Very Strong": "#1f77b4",
    "Strong":      "#2ca02c",
    "Moderate":    "#ff7f0e",
    "Weak":        "#d62728",
}

# ─── Train RF on all 147 stapled AMPs ────────────────────────────────────────
def train_model():
    meta = pd.read_csv(AMPS_META)
    feat = pd.read_csv(AMPS_FEAT)
    df   = pd.merge(meta[["DRAMP_ID","Target_Organism"]],
                    feat[["DRAMP_ID"] + FEATURE_COLS],
                    on="DRAMP_ID", how="inner").dropna(subset=FEATURE_COLS)
    df["mic_uM"] = df.apply(lambda r: get_mic_uM(r["Target_Organism"], r["weight"]), axis=1)
    df = df[df["mic_uM"].notna() & (df["mic_uM"] > 0)].copy()
    df["pMIC"] = df["mic_uM"].apply(lambda x: 6.0 - math.log10(x))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(df[FEATURE_COLS].values.astype(float), df["pMIC"].values)
    print(f"  RF trained on {len(df)} stapled AMPs with E. coli MIC")
    return rf, df["pMIC"].mean(), df["pMIC"].std()

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="stapled_pmic.png")
    args = parser.parse_args()

    print("=" * 65)
    print("  pMIC Inference — Stapled Buforin & Magainin Variants")
    print("=" * 65)

    print("\n  Training RF model on full dataset …")
    rf, train_mean_pmic, train_std_pmic = train_model()

    # ── Load test peptide features ───────────────────────────────────────────
    test = pd.read_csv(TEST_FEAT)
    test = test.dropna(subset=FEATURE_COLS)
    X_test = test[FEATURE_COLS].values.astype(float)
    preds  = rf.predict(X_test)

    # ── Predict native (non-stapled) ─────────────────────────────────────────
    native_preds = {}
    for name, feats in NATIVE.items():
        x = np.array([[feats[f] for f in FEATURE_COLS]])
        native_preds[name] = float(rf.predict(x)[0])

    # ── Print table ─────────────────────────────────────────────────────────
    print("\n" + "─" * 75)
    print(f"  {'Peptide':<26}  {'pMIC':>6}  {'MIC (μM)':>9}  {'MIC (μg/mL)':>12}  Tier")
    print("─" * 75)

    all_entries = []

    # Group: Buforin
    buf_ids = [r for r in test["peptide_id"] if r.lower().startswith("buf")]
    mag_ids = [r for r in test["peptide_id"] if r.lower().startswith("mag")]

    for group_ids, group_label in [(buf_ids, "— Buforin —"), (mag_ids, "— Magainin —")]:
        print(f"\n  {group_label}")
        nat_key = [k for k in NATIVE if group_label.lower().split()[1].lower() in k.lower()][0]
        nat_pmic  = native_preds[nat_key]
        nat_mic   = pmic_to_mic(nat_pmic)
        nat_mw    = NATIVE[nat_key]["weight"]
        nat_ugml  = nat_mic * nat_mw / 1000.0
        label_clean = nat_key.replace("\n", " ")
        print(f"  {'  '+label_clean:<24}  {nat_pmic:>6.3f}  {nat_mic:>9.2f}  {nat_ugml:>12.2f}  {tier(nat_mic)}  [non-stapled]")
        all_entries.append({
            "id": label_clean, "group": group_label.strip("— "),
            "stapled": False, "pMIC": nat_pmic, "mic_uM": nat_mic,
            "mic_ugml": nat_ugml,
        })

        for pid in group_ids:
            row   = test[test["peptide_id"] == pid].iloc[0]
            idx   = list(test["peptide_id"]).index(pid)
            pm    = float(preds[idx])
            mic   = pmic_to_mic(pm)
            mw    = float(row["weight"])
            ugml  = mic * mw / 1000.0
            print(f"  {'  '+pid:<24}  {pm:>6.3f}  {mic:>9.2f}  {ugml:>12.2f}  {tier(mic)}")
            all_entries.append({
                "id": pid, "group": group_label.strip("— "),
                "stapled": True, "pMIC": pm, "mic_uM": mic,
                "mic_ugml": ugml,
            })

    print("─" * 75)
    print(f"\n  Training set: pMIC mean={train_mean_pmic:.2f} "
          f"(MIC≈{pmic_to_mic(train_mean_pmic):.1f} μM), "
          f"std={train_std_pmic:.2f}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             sharey=False,
                             gridspec_kw={"wspace": 0.35})

    for ax_i, (group, ax) in enumerate(zip(["Buforin", "Magainin"], axes)):
        entries = [e for e in all_entries if e["group"] == group]
        labels  = [e["id"]      for e in entries]
        pmics   = [e["pMIC"]    for e in entries]
        mics    = [e["mic_uM"]  for e in entries]
        ugmls   = [e["mic_ugml"] for e in entries]
        stapled = [e["stapled"] for e in entries]

        y_pos   = np.arange(len(labels))

        # Horizontal bars coloured by tier
        bar_colors = [TIER_COLOR[tier(m)] for m in mics]
        bars = ax.barh(y_pos, pmics, color=bar_colors,
                       edgecolor="black", linewidth=0.7, height=0.55)

        # Hatch non-stapled bars differently
        for bar, is_st in zip(bars, stapled):
            if not is_st:
                bar.set_hatch("///")
                bar.set_edgecolor("#333")

        # Annotate bars with pMIC, μM, and μg/mL
        for y, pm, mic, ugml, is_st in zip(y_pos, pmics, mics, ugmls, stapled):
            ax.text(pm + 0.03, y,
                    f" {pm:.2f}  ({mic:.1f} μM / {ugml:.1f} μg/mL)",
                    va="center", fontsize=8.0,
                    fontweight="bold" if not is_st else "normal")

        # Tier boundary lines
        for xv, lbl, lc in [(5.0, "Mod|Weak\n10μM",  "#d62728"),
                              (5.3, "Str\n5μM",        "#ff7f0e"),
                              (5.7, "VS\n2μM",         "#1f77b4")]:
            ax.axvline(xv, color=lc, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text(xv, len(labels) - 0.1, lbl,
                    ha="center", va="bottom", fontsize=6, color=lc)

        # Training mean shading
        ax.axvspan(train_mean_pmic - train_std_pmic,
                   train_mean_pmic + train_std_pmic,
                   alpha=0.07, color="gray", label=f"Train μ±σ ({train_mean_pmic:.2f})")
        ax.axvline(train_mean_pmic, color="gray", linestyle=":",
                   linewidth=1.3, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Predicted pMIC", fontsize=11)
        ax.set_xlim(4.0, max(pmics) + 1.8)
        ax.set_title(f"{group} — Native vs. Stapled Variants\n"
                     f"Predicted pMIC (E. coli, RF regression)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")

    # Shared tier legend
    patches = [mpatches.Patch(color=c, label=t)
               for t, c in TIER_COLOR.items()]
    patches.append(mpatches.Patch(facecolor="white", edgecolor="#333",
                                   hatch="///", label="Non-stapled (native)"))
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9, title="MIC Tier / Style", title_fontsize=8)

    plt.suptitle("Stapled vs. Non-Stapled Buforin II & Magainin II\n"
                 "pMIC Predicted by RF Regression (trained on 147 stapled AMPs)",
                 fontsize=11, fontweight="bold", y=1.01)

    plt.savefig(args.save, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved  →  {args.save}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
