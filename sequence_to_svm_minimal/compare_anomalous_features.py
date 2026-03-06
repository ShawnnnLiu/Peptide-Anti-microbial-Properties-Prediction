"""
compare_anomalous_features.py
------------------------------
Compares all 8 anomalous StaPep features for Buf13 (and Buf12) against the
AMP and Decoy training distributions, with ASCII histogram bars.
"""
import pandas as pd
import numpy as np

import os, platform
if platform.system() == "Windows":
    BASE = r"C:\Users\bioin\Documents\SVM_ESM_Peptides\Peptide-Anti-microbial-Properties-Prediction\sequence_to_svm_minimal\data\training_dataset\StaPep"
else:
    BASE = "/mnt/c/Users/bioin/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep"

amp  = pd.read_csv(f"{BASE}/stapled_amps_features.csv")
dec  = pd.read_csv(f"{BASE}/stapled_decoys.csv")
test = pd.read_csv(f"{BASE}/test_stapled_features.csv").set_index("peptide_id")

ANOMALOUS = {
    # ── Previously identified ──────────────────────────────────────────────
    "fraction_lysine"   : ("LOW",  "AMPs score higher — Buf13 has almost no Lys"),
    "loop_percent"      : ("LOW",  "AMPs have more loop/coil — Buf13 is too ordered"),
    "helix_percent"     : ("HIGH", "Decoys are more helical → high helix = decoy signal"),
    "fraction_arginine" : ("HIGH", "Decoys have more Arg → high Arg = decoy signal"),
    "psa"               : ("HIGH", "Decoys have higher PSA → high PSA = decoy signal"),
    # ── Newly added (found by full 17-feature scan) ────────────────────────
    "isoelectric_point" : ("HIGH", "Buf13 pI very high (12.0) — AMPs also high, decoys low; AMP-like but extreme"),
    "sheet_percent"     : ("HIGH", "Buf13 has elevated β-sheet (1.7%) vs AMP mean (0.6%) — minor decoy signal"),
    "mean_bfactor"      : ("HIGH", "High B-factor (562 vs AMP 307) — Buf13 is more flexible/disordered in MD"),
}

def bar(val, lo, hi, width=30, marker="█"):
    """ASCII progress bar scaled between lo and hi."""
    frac  = np.clip((val - lo) / (hi - lo + 1e-9), 0, 1)
    filled = int(round(frac * width))
    return marker * filled + "·" * (width - filled)

def pct_rank(series, val):
    return (series < val).mean() * 100

print()
print("╔" + "═"*76 + "╗")
print("║  Anomalous Feature Deep-Dive: Buf12 / Buf13 / Mag31 vs Distributions    ║")
print("║  (Mag31 = next lowest loop_percent after Buf13 among all 8 test peps)   ║")
print("╚" + "═"*76 + "╝")

for feat, (direction, reason) in ANOMALOUS.items():
    li_amp = amp[feat].dropna()
    li_dec = dec[feat].dropna()

    amp_mean, amp_std = li_amp.mean(), li_amp.std()
    dec_mean, dec_std = li_dec.mean(), li_dec.std()

    buf12  = float(test.loc["Buf12",  feat])
    buf13  = float(test.loc["Buf13",  feat])
    mag31  = float(test.loc["Mag31",  feat])

    pct12_amp = pct_rank(li_amp, buf12)
    pct13_amp = pct_rank(li_amp, buf13)
    pctm31_amp = pct_rank(li_amp, mag31)
    pct12_dec = pct_rank(li_dec, buf12)
    pct13_dec = pct_rank(li_dec, buf13)
    pctm31_dec = pct_rank(li_dec, mag31)

    lo = min(li_amp.min(), li_dec.min(), buf12, buf13, mag31)
    hi = max(li_amp.max(), li_dec.max(), buf12, buf13, mag31)

    flag = "⚠  LOW" if direction == "LOW" else "⚠  HIGH"

    print()
    print(f"┌{'─'*76}┐")
    print(f"│  {feat.upper():<40}  [{flag} for AMP]{'':>10}│")
    print(f"│  {reason:<74}│")
    print(f"├{'─'*76}┤")
    print(f"│  {'Group':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Median':>8} {'Max':>8}  │")
    print(f"│  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}  │")
    print(f"│  {'AMP  (n='+str(len(li_amp))+')':<14} "
          f"{amp_mean:>8.4f} {amp_std:>8.4f} {li_amp.min():>8.4f} "
          f"{li_amp.median():>8.4f} {li_amp.max():>8.4f}  │")
    print(f"│  {'Decoy(n='+str(len(li_dec))+')':<14} "
          f"{dec_mean:>8.4f} {dec_std:>8.4f} {li_dec.min():>8.4f} "
          f"{li_dec.median():>8.4f} {li_dec.max():>8.4f}  │")
    print(f"├{'─'*76}┤")

    # Distribution bar chart
    print(f"│  Distribution bar  [{lo:.4f} ──────────────────────── {hi:.4f}]  │")
    print(f"│  AMP  mean  {amp_mean:>8.4f}  |{bar(amp_mean, lo, hi)}|  │")
    print(f"│  Decoy mean {dec_mean:>8.4f}  |{bar(dec_mean, lo, hi)}|  │")
    print(f"│  Buf12      {buf12:>8.4f}  |{bar(buf12,   lo, hi, marker='①')}|  │")
    print(f"│  Buf13      {buf13:>8.4f}  |{bar(buf13,   lo, hi, marker='②')}|  │")
    print(f"│  Mag31      {mag31:>8.4f}  |{bar(mag31,   lo, hi, marker='③')}|  │")

    print(f"├{'─'*76}┤")
    print(f"│  Buf12 ①  {pct12_amp:5.1f}th %-ile of AMPs   ({pct12_dec:5.1f}th of Decoys)                   │")
    print(f"│  Buf13 ②  {pct13_amp:5.1f}th %-ile of AMPs   ({pct13_dec:5.1f}th of Decoys)                   │")
    print(f"│  Mag31 ③  {pctm31_amp:5.1f}th %-ile of AMPs   ({pctm31_dec:5.1f}th of Decoys)                   │")

    # Verdict
    sym  = "<<" if direction == "LOW" else ">>"
    cmp  = f"Buf12={buf12:.3f}  Buf13={buf13:.3f}  Mag31={mag31:.3f}  {sym} AMP mean={amp_mean:.3f}"
    print(f"│  {cmp:<74}│")
    print(f"└{'─'*76}┘")

# ── Summary table ─────────────────────────────────────────────────────────────
print()
W = 104
print("╔" + "═"*W + "╗")
print("║  Summary: All 8 Outlier Features — Buf12 / Buf13 / Mag31 vs AMP & Decoy Distributions" + " "*17 + "║")
print("╠" + "═"*W + "╣")
print(f"║  {'Feature':<22} {'Buf12':>8} {'①AMP%':>7} {'Buf13':>8} {'②AMP%':>7} {'Mag31':>8} {'③AMP%':>7}  "
      f"{'Direction':>9}  {'AMP>Decoy?':>9}  ║")
print(f"║  {'─'*22} {'─'*8} {'─'*7} {'─'*8} {'─'*7} {'─'*8} {'─'*7}  {'─'*9}  {'─'*9}  ║")

for feat, (direction, _) in ANOMALOUS.items():
    li_amp = amp[feat].dropna()
    li_dec = dec[feat].dropna()
    v12    = float(test.loc["Buf12", feat])
    v13    = float(test.loc["Buf13", feat])
    vm31   = float(test.loc["Mag31", feat])
    p12    = pct_rank(li_amp, v12)
    p13    = pct_rank(li_amp, v13)
    pm31   = pct_rank(li_amp, vm31)
    amp_hi = "AMP>Decoy" if li_amp.mean() > li_dec.mean() else "Decoy>AMP"
    arrow  = "↓ LOW" if direction == "LOW" else "↑ HIGH"
    # mark any peptide that differs >10 pct-pts from Buf12
    d13  = "◄" if abs(p12 - p13)  > 10 else " "
    dm31 = "◄" if abs(p12 - pm31) > 10 else " "
    print(f"║  {feat:<22} {v12:>8.3f} {p12:>6.1f}% {v13:>8.3f} {p13:>6.1f}%{d13} {vm31:>8.3f} {pm31:>6.1f}%{dm31}"
          f"  {arrow:>9}  {amp_hi:>9}  ║")

print("╠" + "═"*W + "╣")
print("║  ◄ = differs from Buf12 by >10 percentile points" + " "*55 + "║")
print("║  Mag31 loop_percent = 0.895 — less extreme than Buf13 (0.849) but still below AMP mean." + " "*15 + "║")
print("║  Mag31 has NO helix content at all, unlike Buf13 — its low loop is from β-sheet instead." + " "*13 + "║")
print("╚" + "═"*W + "╝")
print()
