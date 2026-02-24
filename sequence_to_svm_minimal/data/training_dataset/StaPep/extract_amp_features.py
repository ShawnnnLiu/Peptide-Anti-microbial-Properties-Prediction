"""
StaPep feature extraction for stapled_amps.csv
------------------------------------------------
Computes the same sequence-level features as those pre-computed in
stapled_decoys.csv:
    length, weight, hydrophobic_index, charge, aromaticity,
    isoelectric_point, fraction_arginine, fraction_lysine, lyticity_index

Sequence preprocessing rules (mapping AMP Hiden_Sequence → StaPep format):
  - Z  → R8   (R8 olefin/hydrocarbon staple, single char → StaPep multi-char token)
  - J  → B    (Norleucine, alternative single-letter → StaPep's 'B')
  - lowercase → uppercase  (D-amino acids treated as backbone L-AA for sequence features)
  - N_terminal_Modification == 'Acetylation' → prepend 'Ac'
  - C_terminal_Modification == 'Amidation'   → append  'NH2'
  - K-staple rows (Fmoc-Lys, no X/Z/J/B)    → K used as regular Lys (backbone only)

MD-based structure features (helix_percent, sheet_percent, loop_percent,
mean_bfactor, mean_gyrate, num_hbonds, psa, sasa) require AmberTools/pytraj
(Linux) and are left as NaN columns to preserve schema parity with decoys.

Output: stapled_amps_features.csv  (same directory as this script)
"""

import sys
import io
import warnings
import traceback

import pandas as pd
import numpy as np

# Make stdout UTF-8 safe on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
import os
HERE = os.path.dirname(os.path.abspath(__file__))
AMP_CSV  = os.path.join(HERE, 'stapled_amps.csv')
OUT_CSV  = os.path.join(HERE, 'stapled_amps_features.csv')

# ── StaPep ProtParamsSeq ───────────────────────────────────────────────────────
from stapep.utils import ProtParamsSeq


# ── Sequence translation helpers ───────────────────────────────────────────────
def build_stapep_seq(hiden_seq: str,
                     n_mod: str,
                     c_mod: str) -> str:
    """
    Convert an AMP Hiden_Sequence string into a StaPep-parseable sequence.

    Token mapping:
      X  → S5  (auto-handled by StaPep, passed through)
      Z  → R8  (R8 olefin staple, StaPep multi-char token)
      J  → B   (Norleucine)
      lowercase → uppercase  (treat D-AA as backbone L-AA for seq features)

    N/C terminal modifications are prepended/appended as Ac/NH2.
    """
    seq = str(hiden_seq).strip()

    # ① replace non-StaPep single-char codes
    seq = seq.replace('Z', 'R8')   # R8 olefin staple
    seq = seq.replace('J', 'B')    # Norleucine

    # ② uppercase (treats D-amino acids as backbone L-amino acids)
    seq = seq.upper()

    # ③ terminal modifications
    prefix = 'Ac' if str(n_mod).strip().lower() == 'acetylation' else ''
    suffix = 'NH2' if str(c_mod).strip().lower() == 'amidation' else ''

    if prefix:
        seq = prefix + seq
    if suffix:
        seq = seq + suffix

    return seq


# ── Feature extraction for one row ────────────────────────────────────────────
def extract_features(stapep_seq: str, c_mod: str):
    """
    Run ProtParamsSeq and return a feature dict.
    Returns None on error (logged to stderr).
    """
    amide = str(c_mod).strip().lower() == 'amidation'
    try:
        pps = ProtParamsSeq(stapep_seq)
        return {
            'length':            pps.seq_length,
            'weight':            pps.weight,
            'hydrophobic_index': pps.hydrophobicity_index,
            'charge':            pps.calc_charge(pH=7.0, amide=amide),
            'aromaticity':       pps.aromaticity,
            'isoelectric_point': pps.isoelectric_point,
            'fraction_arginine': pps.fraction_arginine,
            'fraction_lysine':   pps.fraction_lysine,
            'lyticity_index':    pps.lyticity_index,
        }
    except Exception as e:
        return {'_error': str(e)}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f'Reading {AMP_CSV}')
    amp = pd.read_csv(AMP_CSV, encoding='utf-8-sig')
    print(f'  {len(amp)} rows loaded')

    records = []
    n_ok = n_err = 0

    for idx, row in amp.iterrows():
        dramp_id    = row['DRAMP_ID']
        hiden_seq   = row['Hiden_Sequence']
        n_mod       = row.get('N_terminal_Modification', '')
        c_mod       = row.get('C_terminal_Modification', '')
        seq_label   = row.get('Sequence', hiden_seq)   # original display seq

        # Build StaPep sequence
        stapep_seq = build_stapep_seq(hiden_seq, n_mod, c_mod)

        # Extract features
        feats = extract_features(stapep_seq, c_mod)
        err = feats.pop('_error', None) if feats else 'None returned'

        record = {
            'DRAMP_ID':       dramp_id,
            'Sequence':       seq_label,
            'Hiden_Sequence': hiden_seq,
            'stapep_seq':     stapep_seq,
            'N_terminal_Modification': n_mod,
            'C_terminal_Modification': c_mod,
            'label':          1,    # AMP = 1
            # MD-based structure features (Linux/AmberTools only → NaN)
            'helix_percent':  np.nan,
            'sheet_percent':  np.nan,
            'loop_percent':   np.nan,
            'mean_bfactor':   np.nan,
            'mean_gyrate':    np.nan,
            'num_hbonds':     np.nan,
            'psa':            np.nan,
            'sasa':           np.nan,
            'extraction_error': err,
        }

        if feats:
            record.update(feats)
            n_ok += 1
        else:
            n_err += 1
            print(f'  [WARN] {dramp_id}: {err}  (seq: {stapep_seq})')

        records.append(record)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(records)

    # reorder columns to mirror decoy schema
    seq_feat_cols = ['length', 'weight', 'hydrophobic_index', 'charge',
                     'aromaticity', 'isoelectric_point', 'fraction_arginine',
                     'fraction_lysine', 'lyticity_index']
    md_feat_cols  = ['helix_percent', 'sheet_percent', 'loop_percent',
                     'mean_bfactor', 'mean_gyrate', 'num_hbonds', 'psa', 'sasa']
    id_cols       = ['DRAMP_ID', 'Sequence', 'Hiden_Sequence', 'stapep_seq',
                     'N_terminal_Modification', 'C_terminal_Modification', 'label']
    all_cols      = id_cols + seq_feat_cols + md_feat_cols + ['extraction_error']
    out_df = out_df[[c for c in all_cols if c in out_df.columns]]

    out_df.to_csv(OUT_CSV, index=False)
    print(f'\nDone!  {n_ok} ok / {n_err} errors')
    print(f'Saved → {OUT_CSV}')

    # ── Quick stats ───────────────────────────────────────────────────────────
    print('\n── Feature summary (sequence-level) ──')
    print(out_df[seq_feat_cols].describe().round(3).to_string())

    # Compare with decoys
    dec_path = os.path.join(HERE, 'stapled_decoys.csv')
    if os.path.exists(dec_path):
        dec = pd.read_csv(dec_path)
        dec_feat_cols = [c for c in seq_feat_cols if c in dec.columns]
        print('\n── Decoy feature ranges (for comparison) ──')
        print(dec[dec_feat_cols].describe().round(3).to_string())

        print('\n── Feature overlap check (AMP mean vs Decoy mean) ──')
        print(f'{"Feature":<25} {"AMP mean":>12} {"Decoy mean":>12} {"overlap?":>10}')
        print('-' * 65)
        for feat in seq_feat_cols:
            if feat not in dec.columns:
                continue
            amp_mean = out_df[feat].mean()
            dec_mean = dec[feat].mean()
            amp_std  = out_df[feat].std()
            dec_std  = dec[feat].std()
            # simple overlap heuristic: |means| < 2*(std_amp + std_dec)
            overlap = abs(amp_mean - dec_mean) < 2 * (amp_std + dec_std)
            print(f'{feat:<25} {amp_mean:>12.3f} {dec_mean:>12.3f} {"YES" if overlap else "NO":>10}')

    print('\n── Error summary ──')
    errs = out_df[out_df['extraction_error'].notna() & (out_df['extraction_error'] != 'None')]
    print(f'{len(errs)} rows had errors.')
    if len(errs):
        for _, r in errs.head(10).iterrows():
            print(f'  {r["DRAMP_ID"]}: {r["Hiden_Sequence"]} → {r["extraction_error"]}')


if __name__ == '__main__':
    main()
