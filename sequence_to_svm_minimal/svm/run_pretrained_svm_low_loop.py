"""
run_pretrained_svm_low_loop.py
-------------------------------
Runs the pretrained 2016 PNAS SVM on:
  • The 8 test peptides  (from qsar_stapled_test.csv)
  • The 10 training AMPs with the lowest loop_percent
    (dynamically picked from stapled_amps_features.csv)

Outputs both tables for comparison. SVM loading logic is provided by
``utils.pretrained_svm.load_pretrained_svm`` — bypasses the legacy
0.19.2 ``svc.pkl`` entirely using the .npy sidecar arrays.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make utils/ importable regardless of cwd. Also fixes a pre-refactor bug
# where BASE = svm/ (not project root), causing data paths to resolve to
# svm/data/... which doesn't exist.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import STAPEP_DIR
from utils.pretrained_svm import PretrainedSVM, load_pretrained_svm

QSAR_TEST       = STAPEP_DIR / "qsar_stapled_test.csv"
QSAR_AMP        = STAPEP_DIR / "qsar_stapled_amps.csv"
STAPEP_FEATURES = STAPEP_DIR / "stapled_amps_features.csv"


def run_svm(
    df_in: pd.DataFrame,
    id_col: str,
    seq_col: str,
    model: PretrainedSVM,
) -> pd.DataFrame:
    """Z-score, predict, and return a sorted DataFrame of results."""
    names = df_in[id_col].tolist()
    seqs  = df_in[seq_col].tolist()
    pred  = model.predict_from_descriptors(df_in)
    label = np.where(pred.decision > 0, "+1 AMP", "-1 non")
    return pd.DataFrame({
        "name":     names,
        "sequence": seqs,
        "pred":     label,
        "P(AMP)":   pred.prob_amp,
        "f(x)":     pred.decision,
    }).sort_values("P(AMP)", ascending=False).reset_index(drop=True)


def main() -> None:
    model = load_pretrained_svm()

    # ── Table 1: 8 test peptides ────────────────────────────────────────────
    test_df  = pd.read_csv(QSAR_TEST)
    res_test = run_svm(test_df, "peptide_id", "sequence", model)

    # ── Table 2: 10 lowest-loop AMPs (dynamically ranked) ───────────────────
    N_LOW = 10
    amp_feat_full = pd.read_csv(STAPEP_FEATURES)[
        ["DRAMP_ID", "loop_percent", "helix_percent"]
    ]
    low_ids = (amp_feat_full.sort_values("loop_percent")
                            .head(N_LOW)["DRAMP_ID"].tolist())

    amp_qsar = pd.read_csv(QSAR_AMP)
    low_df   = amp_qsar[amp_qsar["peptide_id"].isin(low_ids)].copy()

    amp_feat = amp_feat_full.rename(columns={"DRAMP_ID": "peptide_id"})
    low_df   = low_df.merge(amp_feat, on="peptide_id", how="left")

    res_low  = run_svm(low_df, "peptide_id", "sequence", model)
    loop_map = (low_df.set_index("peptide_id")
                      [["loop_percent", "helix_percent"]].to_dict("index"))

    # ── Print Table 1 ───────────────────────────────────────────────────────
    W = 74
    print()
    print("=" * W)
    print("  TABLE 1 — Pretrained 2016 PNAS SVM: 8 Test Peptides (QSAR)")
    print("  (linear kernel · Z-score normalised · Platt-scaled probabilities)")
    print("=" * W)
    print(f"  {'Peptide':<30} {'Pred':>8} {'P(AMP)':>8} {'f(x)':>10}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*10}")
    for _, r in res_test.iterrows():
        print(f"  {r['name']:<30} {r['pred']:>8} "
              f"{r['P(AMP)']:>8.4f} {r['f(x)']:>10.4f}")
    print("=" * W)

    # ── Print Table 2 ───────────────────────────────────────────────────────
    print()
    print("=" * W)
    print(f"  TABLE 2 — Same SVM: {N_LOW} Training AMPs With Lowest loop_percent")
    print("  (confirmed AMPs — bottom 10 by loop_percent in training set)")
    print("  Buf13 shown at bottom for reference (loop=84.9%)")
    print("=" * W)
    print(f"  {'DRAMP_ID':<14} {'loop%':>6} {'helix%':>7} "
          f"{'Pred':>8} {'P(AMP)':>8} {'f(x)':>10}  Sequence")
    print(f"  {'─'*14} {'─'*6} {'─'*7} {'─'*8} {'─'*8} {'─'*10}  {'─'*30}")

    for _, r in res_low.iterrows():
        info = loop_map.get(r["name"], {})
        lp   = info.get("loop_percent",  float("nan"))
        hp   = info.get("helix_percent", float("nan"))
        seq  = r["sequence"][:35]
        print(f"  {r['name']:<14} {lp:>5.1%} {hp:>6.1%} "
              f"{r['pred']:>8} {r['P(AMP)']:>8.4f} {r['f(x)']:>10.4f}  {seq}")

    # Buf13 reference row
    buf13 = test_df[test_df["peptide_id"] == "Buf13"]
    if not buf13.empty:
        b = run_svm(buf13, "peptide_id", "sequence", model).iloc[0]
        print(f"  {'─'*14} {'─'*6} {'─'*7} {'─'*8} {'─'*8} {'─'*10}  {'─'*30}")
        print(f"  {'Buf13 (ref)':<14} {'84.9%':>6} {'13.4%':>7} {b['pred']:>8} "
              f"{b['P(AMP)']:>8.4f} {b['f(x)']:>10.4f}  {b['sequence'][:35]}")

    print("=" * W)
    print()
    print("  KEY QUESTION: Do confirmed low-loop AMPs score high on the pretrained SVM?")
    print("  If yes → SVM CAN recognise low-loop AMPs → Buf13's low score is about")
    print("  OTHER features (helix%, fraction_arginine, psa), not loop_percent alone.")
    print("  If no  → the pretrained SVM is simply blind to low-loop AMPs entirely.")
    print()


if __name__ == "__main__":
    main()
