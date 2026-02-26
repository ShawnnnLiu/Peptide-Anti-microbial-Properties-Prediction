"""
extract_stapep_qsar.py
======================
Extract the 12 QSAR descriptors used in the original 2016 PNAS SVM paper
(Fjell et al. / Cherkasov lab) for all stapled peptides:

  - 188 training AMPs       → qsar_stapled_amps.csv
  - 354 training Decoys     → qsar_stapled_decoys.csv
  - 8   test peptides       → qsar_stapled_test.csv

Feature method
--------------
Identical to the non-stapled pipeline: backbone-only standard-AA sequences
are passed through descripGen_bespoke(), which computes exactly the 12
descriptors published in the 2016 PNAS paper:

  netCharge, FC, LW, DP, NK, AE, pcMK,
  _SolventAccessibilityD1025,
  tau2_GRAR740104, tau4_GRAR740104,
  QSO50_GRAR740104, QSO29_GRAR740104

Staple positions (X→A, Z/R8→K) are already substituted in the seq files,
so only standard AAs reach propy – exactly as the original paper intended.

Usage
-----
    conda run -n skl_legacy python extract_stapep_qsar.py
or
    python extract_stapep_qsar.py  (inside skl_legacy / any env with numpy/pandas/propy)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).parent.resolve()
STAPEP_DIR   = SCRIPT_DIR / "data" / "training_dataset" / "StaPep"
DESC_DIR     = SCRIPT_DIR / "descriptors"
AAINDEX_DIR  = DESC_DIR / "aaindex"

AMP_SEQS_TXT   = STAPEP_DIR / "seqs_AMP_stapep.txt"
DECOY_SEQS_TXT = STAPEP_DIR / "seqs_DECOY_stapep.txt"
TEST_SEQS_TXT  = STAPEP_DIR / "seqs_test_stapled.txt"

AMP_CSV        = STAPEP_DIR / "stapled_amps.csv"
DECOY_CSV      = STAPEP_DIR / "stapled_decoys.csv"

OUT_AMP_QSAR   = STAPEP_DIR / "qsar_stapled_amps.csv"
OUT_DECOY_QSAR = STAPEP_DIR / "qsar_stapled_decoys.csv"
OUT_TEST_QSAR  = STAPEP_DIR / "qsar_stapled_test.csv"

# ---------------------------------------------------------------------------
# Bootstrap propy import (vendored inside descriptors/)
# ---------------------------------------------------------------------------
_vendor = DESC_DIR / "propy3_vendor"
_src    = DESC_DIR / "propy3_src"
for p in [str(_vendor), str(_src), str(SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from propy import AAIndex                        # type: ignore
from propy import ProCheck                       # type: ignore
from propy.PyPro import GetProDes                # type: ignore

# Test-peptide name map (from comments in seqs_test_stapled.txt)
TEST_NAMES = {
    1: "Buf(i+4)_12",
    2: "Buf(i+4)_13",
    3: "Buf(i+4)_13_Q9K",
    4: "Buf(i+4)_12_V15K_L19K",
    5: "Mag_20",
    6: "Mag_25",
    7: "Mag_31",
    8: "Mag_36",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map non-standard / staple AA codes → nearest standard AA for propy compatibility
_NON_STANDARD = str.maketrans({
    "X": "A",   # S5 / R5 staple → alanine (alpha-methyl backbone, similar size)
    "Z": "K",   # R8 staple      → lysine  (long side-chain cross-link)
    "J": "L",   # norleucine     → leucine  (iso-structural)
    "B": "N",   # Asx ambiguity  → asparagine
    "O": "K",   # pyrrolysine    → lysine
    "U": "C",   # selenocysteine → cysteine
})

STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def clean_for_qsar(seq: str) -> str:
    """Upper-case and map any non-standard residues to the closest standard AA."""
    seq = seq.upper().translate(_NON_STANDARD)
    # Remove any remaining characters that propy doesn't know
    return "".join(aa if aa in STANDARD_AAS else "A" for aa in seq)


def load_seqs_txt(path: Path, skip_comments: bool = False) -> dict[int, str]:
    """Return {int_index: cleaned_sequence} from a whitespace-delimited two-column file."""
    result = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if skip_comments and line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            result[idx] = clean_for_qsar(parts[1])
    return result


def _compute_12_descriptors(seq: str, aap_dict: dict) -> tuple[list[str], np.ndarray]:
    """
    Inline replica of descripGen_bespoke() — identical 12 features from the
    2016 PNAS SVM paper, but raises ValueError instead of sys.exit on bad input.
    """
    v_nlag   = 30
    v_weight = 0.05

    if ProCheck.ProteinCheck(seq) == 0:
        raise ValueError(f"Non-standard residues remain in '{seq}'")

    seqLen = len(seq)
    Des    = GetProDes(seq)

    names:  list[str]   = []
    values: np.ndarray  = np.empty([0,])

    # 1. netCharge
    chargeDict = {"A":0,"C":0,"D":-1,"E":-1,"F":0,"G":0,"H":1,"I":0,"K":1,
                  "L":0,"M":0,"N":0,"P":0,"Q":0,"R":1,"S":0,"T":0,"V":0,"W":0,"Y":0}
    names.append("netCharge")
    values = np.append(values, sum(chargeDict.get(x, 0) for x in seq))

    # 2-6. Dipeptide compositions: FC, LW, DP, NK, AE
    dpc = Des.GetDPComp()
    for handle in ["FC", "LW", "DP", "NK", "AE"]:
        names.append(handle)
        values = np.append(values, float("%.2f" % dpc[handle]))

    # 7. pcMK
    nM = sum(1 for x in seq if x == "M")
    nK = sum(1 for x in seq if x == "K")
    names.append("pcMK")
    values = np.append(values, 0.0 if nM == 0 else float(nM) / float(nM + nK))

    # 8. _SolventAccessibilityD1025
    ctd = Des.GetCTD()
    names.append("_SolventAccessibilityD1025")
    values = np.append(values, ctd["_SolventAccessibilityD1025"])

    # 9-10. tau2 / tau4 GRAR740104
    socn_p = Des.GetSOCNp(maxlag=v_nlag, distancematrix=aap_dict)
    for handle in ["tau2", "tau4"]:
        delta = float(handle[3:])
        val   = 0.0 if delta > (seqLen - 1) else socn_p[handle] / float(seqLen - delta)
        names.append(f"{handle}_GRAR740104")
        values = np.append(values, val)

    # 11-12. QSO50 / QSO29 GRAR740104
    qso_p = Des.GetQSOp(maxlag=v_nlag, weight=v_weight, distancematrix=aap_dict)
    for handle in ["QSO50", "QSO29"]:
        names.append(f"{handle}_GRAR740104")
        values = np.append(values, qso_p[handle])

    return names, values


def run_qsar(
    seqs: dict[int, str],
    id_map: dict[int, str],
    aap_dict: dict,
    desc: str,
) -> pd.DataFrame:
    """
    Compute the 12 QSAR descriptors for all sequences.

    Parameters
    ----------
    seqs     : {int_index: cleaned_sequence}
    id_map   : {int_index: peptide_id_string}  e.g. {1: "DRAMP21468"}
    aap_dict : pre-loaded GRAR740104 AAIndex dict
    desc     : tqdm bar label
    """
    rows: list[dict] = []
    feature_names: list[str] | None = None

    for idx, seq in tqdm(seqs.items(), desc=desc, unit="pep"):
        pid = id_map.get(idx, f"UNK_{idx}")
        try:
            names, values = _compute_12_descriptors(seq, aap_dict)
            if feature_names is None:
                feature_names = names
            rows.append({"peptide_id": pid, "sequence": seq,
                         **dict(zip(names, values.tolist()))})
        except Exception as exc:
            print(f"\n  ⚠  [{pid}] {seq}: {exc}")
            if feature_names is not None:
                rows.append({"peptide_id": pid, "sequence": seq,
                             **{n: np.nan for n in feature_names}})

    df = pd.DataFrame(rows)
    if feature_names:
        col_order = ["peptide_id", "sequence"] + feature_names
        df = df[[c for c in col_order if c in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  QSAR Feature Extraction  (2016 PNAS SVM paper method)")
    print("=" * 60)

    # 1. Load AAIndex (expensive – do once)
    print("\nLoading GRAR740104 AAIndex …")
    aap_dict = AAIndex.GetAAIndex23("GRAR740104", path=str(AAINDEX_DIR))
    print("  ✓ AAIndex loaded.")

    # 2. AMP sequences + DRAMP_ID map
    print(f"\nReading AMP sequences from {AMP_SEQS_TXT.name} …")
    amp_seqs = load_seqs_txt(AMP_SEQS_TXT)

    # Build index→DRAMP_ID from stapled_amps.csv (row order matches txt indices)
    amp_meta = pd.read_csv(AMP_CSV, usecols=["DRAMP_ID"])
    amp_id_map = {i + 1: str(row["DRAMP_ID"]) for i, row in amp_meta.iterrows()}

    print(f"  {len(amp_seqs)} AMP sequences loaded.")

    df_amp = run_qsar(amp_seqs, amp_id_map, aap_dict, desc="AMPs ")
    df_amp["label"] = 1
    df_amp.to_csv(OUT_AMP_QSAR, index=False)
    print(f"\n  ✓ Saved → {OUT_AMP_QSAR}")
    print(f"    Shape : {df_amp.shape}  |  Columns: {list(df_amp.columns)}")

    # 3. Decoy sequences + COMPOUND_ID map
    print(f"\nReading Decoy sequences from {DECOY_SEQS_TXT.name} …")
    decoy_seqs = load_seqs_txt(DECOY_SEQS_TXT)

    decoy_meta = pd.read_csv(DECOY_CSV, usecols=["COMPOUND_ID"])
    decoy_id_map = {i + 1: str(row["COMPOUND_ID"]) for i, row in decoy_meta.iterrows()}

    print(f"  {len(decoy_seqs)} Decoy sequences loaded.")

    df_decoy = run_qsar(decoy_seqs, decoy_id_map, aap_dict, desc="Decoys")
    df_decoy["label"] = 0
    df_decoy.to_csv(OUT_DECOY_QSAR, index=False)
    print(f"\n  ✓ Saved → {OUT_DECOY_QSAR}")
    print(f"    Shape : {df_decoy.shape}  |  Columns: {list(df_decoy.columns)}")

    # 4. Test sequences (skip # comment lines)
    print(f"\nReading Test sequences from {TEST_SEQS_TXT.name} …")
    test_seqs = load_seqs_txt(TEST_SEQS_TXT, skip_comments=True)
    test_id_map = {idx: TEST_NAMES.get(idx, f"TEST_{idx}") for idx in test_seqs}

    print(f"  {len(test_seqs)} Test sequences loaded.")

    df_test = run_qsar(test_seqs, test_id_map, aap_dict, desc="Test  ")
    # No label for test peptides (unknown)
    df_test.to_csv(OUT_TEST_QSAR, index=False)
    print(f"\n  ✓ Saved → {OUT_TEST_QSAR}")
    print(f"    Shape : {df_test.shape}  |  Columns: {list(df_test.columns)}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("  Done!  Three QSAR CSV files written to StaPep directory:")
    print(f"    • {OUT_AMP_QSAR.name}     ({len(df_amp)} rows × {len(df_amp.columns)} cols)")
    print(f"    • {OUT_DECOY_QSAR.name}   ({len(df_decoy)} rows × {len(df_decoy.columns)} cols)")
    print(f"    • {OUT_TEST_QSAR.name}    ({len(df_test)} rows × {len(df_test.columns)} cols)")
    print(f"\n  Features (12 QSAR descriptors from 2016 PNAS paper):")
    for i, name in enumerate(list(df_amp.columns[2:-1]), 1):   # skip peptide_id, sequence, label
        print(f"    {i:>2}. {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
