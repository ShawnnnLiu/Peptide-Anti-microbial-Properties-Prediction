"""
run_pretrained_svm_low_loop.py
-------------------------------
Runs the pretrained 2016 PNAS SVM on:
  • The 8 test peptides  (from qsar_stapled_test.csv)
  • The 5 training AMPs with the lowest loop_percent  (from qsar_stapled_amps.csv)
    DRAMP29033, DRAMP21489, DRAMP21619, DRAMP21617, DRAMP28998

Outputs both tables side-by-side for comparison.
All SVM loading logic is identical to run_pretrained_svm_inference.py.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import joblib, platform

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
QSAR_TEST = os.path.join(BASE, "data", "training_dataset", "StaPep", "qsar_stapled_test.csv")
QSAR_AMP  = os.path.join(BASE, "data", "training_dataset", "StaPep", "qsar_stapled_amps.csv")
STAPEP    = os.path.join(BASE, "data", "training_dataset", "StaPep", "stapled_amps_features.csv")
PP_DIR    = os.path.join(BASE, "..", "pretrained_svm",
                         "sequence_to_svm_minimal", "predictionsParameters")
Z_FILE    = os.path.join(PP_DIR, "Z_score_mean_std__intersect_noflip.csv")
SVC_PKL   = os.path.join(PP_DIR, "svc.pkl")

# ── Load Z-score params ───────────────────────────────────────────────────────
with open(Z_FILE) as f:
    desc_names = f.readline().strip().split(",")
    z_means    = np.array([float(x) for x in f.readline().strip().split(",")])
    z_stds     = np.array([float(x) for x in f.readline().strip().split(",")])

# ── Load pretrained SVM (identical to run_pretrained_svm_inference.py) ────────
sys.modules.setdefault('sklearn.externals.joblib', joblib)
try:
    import sklearn.svm._classes as _c
    sys.modules.setdefault('sklearn.svm.classes', _c)
except Exception:
    pass

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    svc = joblib.load(SVC_PKL)

def _npy(n):
    return np.load(os.path.join(PP_DIR, f"svc.pkl_{n:02d}.npy"), allow_pickle=False)

support_vectors_ = _npy(7)
dual_coef_       = _npy(3)
intercept_       = _npy(10)
probA_           = _npy(4)
probB_           = _npy(11)

w = support_vectors_.T @ dual_coef_[0]   # linear kernel weight vector

def run_svm(df_in, id_col, seq_col):
    """Z-score, predict, return sorted DataFrame with results."""
    names = df_in[id_col].tolist()
    seqs  = df_in[seq_col].tolist()
    X     = df_in[desc_names].values.astype(float)
    X     = (X - z_means) / z_stds
    dec   = X @ w + intercept_[0]
    fval  = probA_[0] * dec + probB_[0]
    prob  = 1.0 / (1.0 + np.exp(fval))
    pred  = np.where(dec > 0, "+1 AMP", "-1 non")
    return pd.DataFrame({
        "name":     names,
        "sequence": seqs,
        "pred":     pred,
        "P(AMP)":   prob,
        "f(x)":     dec,
    }).sort_values("P(AMP)", ascending=False).reset_index(drop=True)

# ── Table 1: 8 test peptides ──────────────────────────────────────────────────
test_df = pd.read_csv(QSAR_TEST)
res_test = run_svm(test_df, "peptide_id", "sequence")

# ── Table 2: 10 lowest-loop AMPs (dynamically ranked) ────────────────────────
N_LOW = 10
amp_feat_full = pd.read_csv(STAPEP)[["DRAMP_ID", "loop_percent", "helix_percent"]]
low_ids = amp_feat_full.sort_values("loop_percent").head(N_LOW)["DRAMP_ID"].tolist()

amp_qsar = pd.read_csv(QSAR_AMP)
low_df   = amp_qsar[amp_qsar["peptide_id"].isin(low_ids)].copy()

# attach loop_percent for context
amp_feat = amp_feat_full.rename(columns={"DRAMP_ID": "peptide_id"})
low_df   = low_df.merge(amp_feat, on="peptide_id", how="left")

res_low = run_svm(low_df, "peptide_id", "sequence")

# re-attach loop info for display
loop_map = low_df.set_index("peptide_id")[["loop_percent","helix_percent"]].to_dict("index")

# ── Print Table 1 ─────────────────────────────────────────────────────────────
W = 74
print()
print("=" * W)
print("  TABLE 1 — Pretrained 2016 PNAS SVM: 8 Test Peptides (QSAR)")
print("  (linear kernel · Z-score normalised · Platt-scaled probabilities)")
print("=" * W)
print(f"  {'Peptide':<30} {'Pred':>8} {'P(AMP)':>8} {'f(x)':>10}")
print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*10}")
for _, r in res_test.iterrows():
    print(f"  {r['name']:<30} {r['pred']:>8} {r['P(AMP)']:>8.4f} {r['f(x)']:>10.4f}")
print("=" * W)

# ── Print Table 2 ─────────────────────────────────────────────────────────────
print()
print("=" * W)
print(f"  TABLE 2 — Same SVM: {N_LOW} Training AMPs With Lowest loop_percent")
print("  (confirmed AMPs — bottom 10 by loop_percent in training set)")
print("  Buf13 shown at bottom for reference (loop=84.9%)")
print("=" * W)
print(f"  {'DRAMP_ID':<14} {'loop%':>6} {'helix%':>7} {'Pred':>8} {'P(AMP)':>8} {'f(x)':>10}  Sequence")
print(f"  {'─'*14} {'─'*6} {'─'*7} {'─'*8} {'─'*8} {'─'*10}  {'─'*30}")

for _, r in res_low.iterrows():
    info = loop_map.get(r["name"], {})
    lp   = info.get("loop_percent",  float("nan"))
    hp   = info.get("helix_percent", float("nan"))
    seq  = r["sequence"][:35]
    print(f"  {r['name']:<14} {lp:>5.1%} {hp:>6.1%} {r['pred']:>8} {r['P(AMP)']:>8.4f} {r['f(x)']:>10.4f}  {seq}")

# Buf13 reference row
buf13 = test_df[test_df["peptide_id"] == "Buf13"]
if not buf13.empty:
    b = run_svm(buf13, "peptide_id", "sequence").iloc[0]
    print(f"  {'─'*14} {'─'*6} {'─'*7} {'─'*8} {'─'*8} {'─'*10}  {'─'*30}")
    print(f"  {'Buf13 (ref)':<14} {'84.9%':>6} {'13.4%':>7} {b['pred']:>8} {b['P(AMP)']:>8.4f} {b['f(x)']:>10.4f}  {b['sequence'][:35]}")

print("=" * W)
print()
print("  KEY QUESTION: Do confirmed low-loop AMPs score high on the pretrained SVM?")
print("  If yes → SVM CAN recognise low-loop AMPs → Buf13's low score is about")
print("  OTHER features (helix%, fraction_arginine, psa), not loop_percent alone.")
print("  If no  → the pretrained SVM is simply blind to low-loop AMPs entirely.")
print()
