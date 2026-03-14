#!/usr/bin/env python3
"""
Extract geometric features from ESMFold PDB files for StaPep training data
and the 8 test peptides.

Outputs (all to data/training_dataset/StaPep/):
  stapep_amp_geometric.csv      — 187 training AMPs
  stapep_decoy_geometric.csv    — 354 training Decoys
  test_stapled_geometric.csv    —   8 test peptides

Run with:  conda run -n esm_env python build_stapep_geo.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

BASE   = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

sys.path.insert(0, str(BASE))
from features.geometric_features import extract_all_features, get_feature_names

# ── sequence lookup files ──────────────────────────────────────────────────────
AMP_SEQ_FILE   = STAPEP / "seqs_AMP_stapep.txt"
DECOY_SEQ_FILE = STAPEP / "seqs_DECOY_stapep.txt"
TEST_SEQ_FILE  = STAPEP / "seqs_test_stapled.txt"

# ── AMP → DRAMP_ID mapping (same row order as seq file) ──────────────────────
AMP_CSV = STAPEP / "stapled_amps.csv"


def load_seq_file(path: Path) -> dict:
    """Return {int_index: sequence} ignoring comment lines."""
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1].strip()
    return mapping


def load_amp_ids(amp_csv: Path) -> dict:
    """Return {1-based_row_index: DRAMP_ID} from the AMP CSV."""
    df = pd.read_csv(amp_csv)
    return {i + 1: row["DRAMP_ID"] for i, row in df.iterrows()}


def parse_pdb_index(name: str) -> int:
    """structure_42.pdb → 42"""
    return int(Path(name).stem.split('_')[-1])


def geo_from_dir(pdb_dir: Path, seq_map: dict, label: int,
                 id_map: dict = None, split_name: str = "") -> pd.DataFrame:
    """
    Extract geometric features from all structure_N.pdb files in pdb_dir.

    id_map: optional {seq_index: external_id} (e.g. DRAMP_ID)
    """
    pdb_files = sorted(pdb_dir.glob("structure_*.pdb"),
                       key=lambda p: parse_pdb_index(p.name))
    print(f"\n[{split_name or pdb_dir.name}]  {len(pdb_files)} PDB files  (label={label})")

    rows, failed = [], 0
    for pdb in tqdm(pdb_files, unit="pdb"):
        idx = parse_pdb_index(pdb.name)
        seq = seq_map.get(idx)
        try:
            feats = extract_all_features(
                pdb_path=str(pdb),
                peptide_id=pdb.stem,
                sequence=seq,
            )
            feats['label']     = label
            feats['seq_index'] = idx
            feats['pdb_file']  = pdb.name
            if seq:
                feats['sequence'] = seq
            if id_map and idx in id_map:
                feats['external_id'] = id_map[idx]
            rows.append(feats)
        except Exception as e:
            warnings.warn(f"  ❌ {pdb.name}: {e}")
            failed += 1

    df = pd.DataFrame(rows)
    feat_cols = [c for c in get_feature_names(include_optional=True) if c in df.columns]
    id_cols   = ['peptide_id', 'external_id', 'seq_index', 'sequence', 'pdb_file', 'label']
    id_cols   = [c for c in id_cols if c in df.columns]
    other     = [c for c in df.columns if c not in id_cols + feat_cols]
    df = df[id_cols + feat_cols + other]

    print(f"  ✅ {len(df)} rows  ({failed} failed)")
    return df


# ── TEST peptide names ────────────────────────────────────────────────────────
TEST_NAMES = {
    1: "Buf12",
    2: "Buf13",
    3: "Buf13_Q9K",
    4: "Buf12_V15K_L19K",
    5: "Mag20",
    6: "Mag25",
    7: "Mag31",
    8: "Mag36",
}

def geo_test(pdb_dir: Path, seq_map: dict) -> pd.DataFrame:
    """Extract geometric features for test peptides (no label)."""
    pdb_files = sorted(pdb_dir.glob("structure_*.pdb"),
                       key=lambda p: parse_pdb_index(p.name))
    print(f"\n[TEST]  {len(pdb_files)} PDB files")

    rows, failed = [], 0
    for pdb in tqdm(pdb_files, unit="pdb"):
        idx  = parse_pdb_index(pdb.name)
        seq  = seq_map.get(idx)
        name = TEST_NAMES.get(idx, f"test_{idx}")
        try:
            feats = extract_all_features(
                pdb_path=str(pdb),
                peptide_id=name,
                sequence=seq,
            )
            feats['seq_index']  = idx
            feats['peptide_id'] = name
            feats['pdb_file']   = pdb.name
            if seq:
                feats['sequence'] = seq
            rows.append(feats)
        except Exception as e:
            warnings.warn(f"  ❌ {pdb.name}: {e}")
            failed += 1

    df = pd.DataFrame(rows)
    feat_cols = [c for c in get_feature_names(include_optional=True) if c in df.columns]
    id_cols   = ['peptide_id', 'seq_index', 'sequence', 'pdb_file']
    id_cols   = [c for c in id_cols if c in df.columns]
    other     = [c for c in df.columns if c not in id_cols + feat_cols]
    df = df[id_cols + feat_cols + other]

    print(f"  ✅ {len(df)} rows  ({failed} failed)")
    return df


if __name__ == "__main__":
    print("=" * 65)
    print("  StaPep Geometric Feature Builder")
    print("=" * 65)

    amp_seqs   = load_seq_file(AMP_SEQ_FILE)
    decoy_seqs = load_seq_file(DECOY_SEQ_FILE)
    test_seqs  = load_seq_file(TEST_SEQ_FILE)
    amp_ids    = load_amp_ids(AMP_CSV)

    print(f"  AMP seqs   : {len(amp_seqs)}   |  DRAMP IDs: {len(amp_ids)}")
    print(f"  Decoy seqs : {len(decoy_seqs)}")
    print(f"  Test seqs  : {len(test_seqs)}")

    # Training AMP structures
    df_amp = geo_from_dir(
        STAPEP / "structures" / "AMP",
        amp_seqs, label=1,
        id_map=amp_ids, split_name="AMP (train)")
    df_amp.to_csv(STAPEP / "stapep_amp_geometric.csv", index=False)

    # Training Decoy structures
    df_dec = geo_from_dir(
        STAPEP / "structures" / "DECOY",
        decoy_seqs, label=0,
        split_name="DECOY (train)")
    df_dec.to_csv(STAPEP / "stapep_decoy_geometric.csv", index=False)

    # Test peptide structures
    df_test = geo_test(STAPEP / "structures" / "TEST", test_seqs)
    df_test.to_csv(STAPEP / "test_stapled_geometric.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    for df, name, out in [
        (df_amp,  "AMP geometric (train)",   "stapep_amp_geometric.csv"),
        (df_dec,  "Decoy geometric (train)", "stapep_decoy_geometric.csv"),
        (df_test, "Test geometric",          "test_stapled_geometric.csv"),
    ]:
        print(f"  {name:30s}: {df.shape[0]:4d} × {df.shape[1]:3d}  →  StaPep/{out}")
    print("\n✅ Done!\n")
