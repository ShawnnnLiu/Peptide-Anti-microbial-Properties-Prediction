#!/usr/bin/env python3
"""
READ-ONLY probe: how does StaPep's DSSP classify the staple residues?

Loads the trajectory + topology from /tmp/stapep_md/<DRAMP_ID>/ (already on
disk from the 4 ns CUDA run) and prints, per residue:

    residue_name   helix_frac   sheet_frac   loop_frac   is_staple?

Repeats for several representative peptides. No StaPep code is modified.
Run inside WSL `stap` env from this directory:

    conda activate stap
    python _probe_dssp_staples.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

try:
    import pytraj as pt
except ImportError:
    print("[ERROR] pytraj not available — run inside WSL `stap` env.", file=sys.stderr)
    sys.exit(1)

WORK = "/tmp/stapep_md"
PROBES = [
    "DRAMP21540",  # Pleu(i+4)1,15(A9K)   — i,i+4 hydrocarbon staple x2
    "DRAMP21541",  # CAP(i+4)1,23(L17K)   — long, 32-mer
    "DRAMP21542",  # Mag(i+4)1,15(A9K)    — Mag(i+4) reference
    "DRAMP21482",  # S-6K-F17             — short 17-mer with 2 X residues
    "PAPER_31427820_MAG1_15_A9K_B21A",  # supplement: shows PAPER_* coverage
]

# Residue codes that StaPep / tleap use for staple amino acids
STAPLE_RESNAMES = {"PS3", "PS5", "PS8", "PR3", "PR5", "PR8",
                   "S3", "S5", "S8", "R3", "R5", "R8",
                   # also nonstandard non-staple codes worth flagging:
                   "NLE", "AIB", "B", "Aib"}


def probe_one(dramp_id: str) -> None:
    work = os.path.join(WORK, dramp_id)
    top  = os.path.join(work, "pep_vac.prmtop")
    trj  = os.path.join(work, "traj.dcd")
    if not (os.path.exists(top) and os.path.exists(trj)):
        print(f"\n[SKIP] {dramp_id}: missing topology or trajectory at {work}")
        return

    traj = pt.load(trj, top=top)
    res_names_pt, ss, ss_int = pt.dssp(traj, simplified=True)
    # pytraj returns res_names like ['ALA:1', 'PS5:2', ...]
    res_names = [r.split(":")[0] for r in res_names_pt]
    ss_arr = np.array(ss)  # shape (n_residues, n_frames)

    n_res, n_frames = ss_arr.shape
    print(f"\n=== {dramp_id}  ({n_res} residues, {n_frames} frames) ===")
    print(f"  unique SS codes returned: {sorted(set(ss_arr.flatten().tolist()))}")
    print(f"  {'idx':>3} {'resname':<8} {'helix':>7} {'sheet':>7} {'loop':>7}  {'staple?':<8}")
    helix_per_res = []
    for i, rn in enumerate(res_names):
        h = float((ss_arr[i] == 'H').sum()) / n_frames
        e = float((ss_arr[i] == 'E').sum()) / n_frames
        c = float((ss_arr[i] == 'C').sum()) / n_frames
        helix_per_res.append(h)
        flag = "STAPLE" if rn in STAPLE_RESNAMES else ""
        print(f"  {i+1:>3} {rn:<8} {h:>7.3f} {e:>7.3f} {c:>7.3f}  {flag:<8}")
    overall = np.mean(helix_per_res)
    print(f"  -- mean helix_percent (StaPep formula) = {overall:.3f}")

    # Subset comparison: helix on standard residues only vs staples only
    std_h = [h for h, rn in zip(helix_per_res, res_names) if rn not in STAPLE_RESNAMES]
    stp_h = [h for h, rn in zip(helix_per_res, res_names) if rn in STAPLE_RESNAMES]
    if std_h:
        print(f"     standard residues (n={len(std_h)})  mean helix = {np.mean(std_h):.3f}")
    if stp_h:
        print(f"     staple/nonstd      (n={len(stp_h)})  mean helix = {np.mean(stp_h):.3f}")


def main() -> None:
    for did in PROBES:
        probe_one(did)


if __name__ == "__main__":
    main()
