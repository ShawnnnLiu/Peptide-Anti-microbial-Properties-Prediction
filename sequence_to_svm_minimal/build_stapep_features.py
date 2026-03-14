#!/usr/bin/env python3
"""
Build Geometric + QSAR features for StaPep AMP and Decoy peptides.

Reads:
  - data/training_dataset/StaPep/seqs_AMP_stapep.txt   (index sequence)
  - data/training_dataset/StaPep/seqs_DECOY_stapep.txt (index sequence)
  - data/training_dataset/StaPep/structures/AMP/       (structure_N.pdb)
  - data/training_dataset/StaPep/structures/DECOY/     (structure_N.pdb)

Writes to data/training_dataset/StaPep/:
  - stapep_amp_geometric.csv
  - stapep_decoy_geometric.csv
  - stapep_amp_qsar.csv
  - stapep_decoy_qsar.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
STAPEP = BASE / "data" / "training_dataset" / "StaPep"

SEQ_AMP   = STAPEP / "seqs_AMP_stapep.txt"
SEQ_DECOY = STAPEP / "seqs_DECOY_stapep.txt"
PDB_AMP   = STAPEP / "structures" / "AMP"
PDB_DECOY = STAPEP / "structures" / "DECOY"

OUT_GEO_AMP   = STAPEP / "stapep_amp_geometric.csv"
OUT_GEO_DECOY = STAPEP / "stapep_decoy_geometric.csv"
OUT_QSAR_AMP  = STAPEP / "stapep_amp_qsar.csv"
OUT_QSAR_DECOY= STAPEP / "stapep_decoy_qsar.csv"

AAINDEX_DIR = BASE / "descriptors" / "aaindex"

# ── helpers ────────────────────────────────────────────────────────────────────

def load_seq_file(path: Path) -> dict:
    """Return {int_index: sequence} from whitespace-delimited txt file."""
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


def parse_structure_index(pdb_name: str) -> int:
    """structure_42.pdb → 42"""
    stem = Path(pdb_name).stem          # "structure_42"
    return int(stem.split('_')[-1])


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — GEOMETRIC FEATURES
# ══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(BASE))
from features.geometric_features import extract_all_features, get_feature_names


def build_geometric(pdb_dir: Path, seq_map: dict, label: int, out_path: Path):
    pdb_files = sorted(pdb_dir.glob("structure_*.pdb"),
                       key=lambda p: parse_structure_index(p.name))
    print(f"\n[Geometric] {pdb_dir.name}: {len(pdb_files)} PDB files → {out_path.name}")

    rows = []
    failed = 0
    for pdb_path in tqdm(pdb_files, unit="pdb"):
        idx = parse_structure_index(pdb_path.name)
        seq = seq_map.get(idx, None)

        try:
            feats = extract_all_features(
                pdb_path=str(pdb_path),
                peptide_id=pdb_path.stem,
                sequence=seq,
            )
            feats['label']      = label
            feats['seq_index']  = idx
            feats['pdb_file']   = pdb_path.name
            if seq:
                feats['sequence'] = seq
            rows.append(feats)
        except Exception as e:
            warnings.warn(f"  Geometric failed for {pdb_path.name}: {e}")
            failed += 1

    df = pd.DataFrame(rows)

    # reorder: id cols first
    id_cols = ['peptide_id', 'seq_index', 'sequence', 'pdb_file', 'label']
    feat_cols = [c for c in get_feature_names(include_optional=True) if c in df.columns]
    other = [c for c in df.columns if c not in id_cols + feat_cols]
    df = df[[c for c in id_cols if c in df.columns] + feat_cols + other]

    df.to_csv(out_path, index=False)
    print(f"  ✅ {len(df)} rows saved  ({failed} failed)  →  {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — QSAR FEATURES
# ══════════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(BASE / "descriptors"))
from descriptors.descripGen_12_py3 import descripGen_bespoke   # noqa: E402
from propy import AAIndex                                       # noqa: E402

# Standard 20 amino acids only — stapled residues kept as hidden sequence
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def sanitize(seq: str) -> str:
    """Keep only standard AA letters (upper-case)."""
    return "".join(c for c in seq.upper() if c in STANDARD_AA)


def build_qsar(seq_map: dict, label: int, out_path: Path):
    print(f"\n[QSAR] {len(seq_map)} sequences → {out_path.name}")

    aap_dict = AAIndex.GetAAIndex23('GRAR740104', path=str(AAINDEX_DIR))

    rows = []
    failed = 0
    for idx, raw_seq in tqdm(sorted(seq_map.items()), unit="seq"):
        seq = sanitize(raw_seq)
        if len(seq) < 8:
            warnings.warn(f"  Skipping seq_index={idx}: too short after sanitise ({seq!r})")
            failed += 1
            continue
        try:
            desc_names, desc_values = descripGen_bespoke(seq, aap_dict)
            row = {'seq_index': idx, 'sequence': raw_seq, 'label': label}
            row.update(dict(zip(desc_names, desc_values.tolist())))
            rows.append(row)
        except Exception as e:
            warnings.warn(f"  QSAR failed for seq_index={idx} ({seq!r}): {e}")
            failed += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  ✅ {len(df)} rows saved  ({failed} failed)  →  {out_path}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  StaPep Feature Builder  —  Geometric + QSAR")
    print("=" * 65)

    amp_seqs   = load_seq_file(SEQ_AMP)
    decoy_seqs = load_seq_file(SEQ_DECOY)
    print(f"  AMP sequences  : {len(amp_seqs)}")
    print(f"  Decoy sequences: {len(decoy_seqs)}")

    # ── Geometric ────────────────────────────────────────────────────────────
    geo_amp   = build_geometric(PDB_AMP,   amp_seqs,   label=1, out_path=OUT_GEO_AMP)
    geo_decoy = build_geometric(PDB_DECOY, decoy_seqs, label=0, out_path=OUT_GEO_DECOY)

    # ── QSAR ─────────────────────────────────────────────────────────────────
    qsar_amp   = build_qsar(amp_seqs,   label=1, out_path=OUT_QSAR_AMP)
    qsar_decoy = build_qsar(decoy_seqs, label=0, out_path=OUT_QSAR_DECOY)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Summary")
    print("=" * 65)
    for df, name in [(geo_amp,   "AMP Geometric"),
                     (geo_decoy, "Decoy Geometric"),
                     (qsar_amp,  "AMP QSAR"),
                     (qsar_decoy,"Decoy QSAR")]:
        print(f"  {name:20s}: {df.shape[0]:4d} rows × {df.shape[1]:3d} cols")
    print("\n✅ All done!\n")
