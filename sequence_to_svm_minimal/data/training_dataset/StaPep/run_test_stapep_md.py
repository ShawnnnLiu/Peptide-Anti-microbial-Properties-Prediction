#!/usr/bin/env python3
"""
Run StaPep MD feature extraction for the 8 stapled-peptide test candidates.

Run inside WSL with the 'stap' conda environment:
  conda activate stap
  cd /mnt/c/Users/bioin/Documents/SVM_ESM_Peptides/stapep_package
  python ../Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep/run_test_stapep_md.py

Output: test_stapled_features.csv  (same directory as this script)
"""

import os, sys, time, warnings, traceback, platform, tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── add stapep package to path ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
STAPEP_PKG = Path("/mnt/c/Users/bioin/Documents/SVM_ESM_Peptides/stapep_package")
sys.path.insert(0, str(STAPEP_PKG))

from stapep.utils import ProtParamsSeq, PhysicochemicalPredictor

# ── OpenMM platform auto-detect ────────────────────────────────────────────────
def get_best_platform() -> str:
    try:
        import openmm as omm
        platforms = [omm.Platform.getPlatform(i).getName()
                     for i in range(omm.Platform.getNumPlatforms())]
        for pref in ("CUDA", "OpenCL"):
            if pref in platforms:
                return pref
        return "CPU"
    except Exception:
        return "CPU"

DEFAULT_PLATFORM = get_best_platform()

# ── Test peptide definitions ───────────────────────────────────────────────────
# X = S5 (S-configured, 5-carbon olefin staple attachment)
# 8 = R8 (R-configured, 8-carbon olefin staple attachment)
# C-terminus = amide (-NH2) is typical for these peptides; adjust if not.

TEST_PEPTIDES = [
    {
        "id":     "Buf12",
        "stapep": "TRSSRAGLQWPS5GRVS5RLLRK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Buf13",
        "stapep": "TRSSRAGLQWPVS5RVHS5LLRK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Buf13_Q9K",
        "stapep": "TRSSRAGLKWPVS5RVHS5LLRK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Buf12_V15K_L19K",
        "stapep": "TRSSRAGLQWPS5GRKS5RLKRK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Mag20",
        "stapep": "GIGKFLHSKKR8FGKAFVS5EIAKK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Mag25",
        "stapep": "S5KGKS5LHSKKKFGKAS5VGES5AKK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Mag31",
        "stapep": "GS5GKFS5HSKKKKGKAS5VGES5AKK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
    {
        "id":     "Mag36",
        "stapep": "GS5GKFS5HSKKKKR8KAFKGES5AKK",
        "c_mod":  "Amidation",
        "n_mod":  "Free",
    },
]

OUT_CSV  = SCRIPT_DIR / "test_stapled_features.csv"
MD_DIR   = SCRIPT_DIR / "structures" / "TEST_MD"
MD_NSTEPS    = 250_000
MD_INTERVAL  = 2_500
MD_TIMESTEP  = 2          # fs
MD_TEMP      = 300        # K
MD_FRICTION  = 1.0        # ps⁻¹
MD_TYPE      = "implicit"
PLATFORM     = DEFAULT_PLATFORM


# ── sequence-level features ────────────────────────────────────────────────────
def seq_features(stapep_seq: str, c_mod: str) -> dict:
    """Use the same property/method names as run_amp_md_features.py."""
    amide = (c_mod == "Amidation")
    pps = ProtParamsSeq(stapep_seq)
    return {
        "length":            pps.seq_length,
        "weight":            pps.weight,
        "hydrophobic_index": pps.hydrophobicity_index,   # sequence-level; overwritten by MD if successful
        "charge":            pps.calc_charge(pH=7.0, amide=amide),
        "aromaticity":       pps.aromaticity,
        "isoelectric_point": pps.isoelectric_point,
        "fraction_arginine": pps.fraction_arginine,
        "fraction_lysine":   pps.fraction_lysine,
        "lyticity_index":    pps.lyticity_index,
    }


# ── full MD + structure features ───────────────────────────────────────────────
def md_features(pid: str, stapep_seq: str, c_mod: str, out_dir: Path) -> dict:
    """
    Mirrors the working pipeline in run_amp_md_features.py exactly.
    Returns a dict of MD-based features (may be partial if individual calls fail).
    """
    import os as _os
    from stapep.molecular_dynamics import PrepareProt, Simulation

    work = str(out_dir)
    _os.makedirs(work, exist_ok=True)

    # 1. AmberTools topology (tleap) ───────────────────────────────────────────
    pp = PrepareProt(stapep_seq, work, method=None)
    pp._gen_prmtop_and_inpcrd_file()

    topology_file = _os.path.join(work, "pep_vac.prmtop")
    if not _os.path.exists(topology_file):
        raise RuntimeError("tleap did not produce pep_vac.prmtop")

    # 2. OpenMM MD simulation ──────────────────────────────────────────────────
    sim = Simulation(work)
    sim.setup(
        type=MD_TYPE,
        solvent="water",
        temperature=MD_TEMP,
        friction=MD_FRICTION,
        timestep=MD_TIMESTEP,
        interval=MD_INTERVAL,
        nsteps=MD_NSTEPS,
        platform=PLATFORM,
    )
    sim.minimize()
    sim.run()

    traj_file = _os.path.join(work, "traj.dcd")
    if not _os.path.exists(traj_file):
        raise RuntimeError("MD did not produce traj.dcd")

    # 3. Analysis via PhysicochemicalPredictor ─────────────────────────────────
    pcp = PhysicochemicalPredictor(
        sequence=stapep_seq,
        topology_file=topology_file,
        trajectory_file=traj_file,
        start_frame=0,
    )
    mean_pdb = _os.path.join(work, "mean_structure.pdb")
    pcp._save_mean_structure(mean_pdb)

    def _safe(fn, *args, name="?"):
        try:
            return fn(*args)
        except Exception as exc:
            warnings.warn(f"  [{pid}] {name}: {exc}", RuntimeWarning)
            return None

    feats = {
        "hydrophobic_index": _safe(pcp.calc_hydrophobic_index, mean_pdb,        name="hydrophobic_index"),
        "helix_percent":     _safe(pcp.calc_helix_percent,                       name="helix_percent"),
        "sheet_percent":     _safe(pcp.calc_extend_percent,                      name="sheet_percent"),
        "loop_percent":      _safe(pcp.calc_loop_percent,                        name="loop_percent"),
        "mean_bfactor":      _safe(pcp.calc_mean_bfactor,                        name="mean_bfactor"),
        "mean_gyrate":       _safe(pcp.calc_mean_gyrate,                         name="mean_gyrate"),
        "num_hbonds":        _safe(pcp.calc_n_hbonds,                            name="num_hbonds"),
        "psa":               _safe(pcp.calc_psa,          mean_pdb,              name="psa"),
        "sasa":              _safe(pcp.calc_mean_molsurf,                        name="sasa"),
    }
    # Filter out None values — seq-level fallback already in row
    return {k: v for k, v in feats.items() if v is not None}


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  StaPep MD Feature Extraction — Test Peptides")
    print("=" * 60)
    print(f"  Platform : {PLATFORM}")
    print(f"  Steps    : {MD_NSTEPS:,}  interval={MD_INTERVAL}")
    print(f"  Output   : {OUT_CSV}")

    # Resume logic
    done_ok: set = set()
    all_rows: list = []
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done_ok = set(prev[prev.get('extraction_error', pd.Series(dtype='str')).isna()]['peptide_id'].astype(str))
        all_rows = prev.to_dict('records')
        print(f"  Resuming — {len(done_ok)} already done OK")

    pending = [p for p in TEST_PEPTIDES if p['id'] not in done_ok]
    print(f"  Pending  : {len(pending)} / {len(TEST_PEPTIDES)}\n")

    for pep in tqdm(pending, desc="Test MD", unit="peptide"):
        pid      = pep['id']
        stapep   = pep['stapep']
        c_mod    = pep['c_mod']
        n_mod    = pep['n_mod']
        out_dir  = MD_DIR / pid
        t0       = time.time()

        row = {
            "peptide_id":       pid,
            "stapep_seq":       stapep,
            "n_mod":            n_mod,
            "c_mod":            c_mod,
            "label":            1,       # these are all AMPs
            "extraction_error": None,
        }

        try:
            s_feats = seq_features(stapep, c_mod)
            m_feats = md_features(pid, stapep, c_mod, out_dir)
            # MD hydrophobic_index overwrites seq-level if successful
            row.update(s_feats)
            row.update(m_feats)
        except Exception as e:
            tb = traceback.format_exc()
            warnings.warn(f"[{pid}] ERROR: {e}\n{tb}")
            row['extraction_error'] = str(e)
            try:
                s_feats = seq_features(stapep, c_mod)
                row.update(s_feats)
            except Exception:
                pass

        row['elapsed_s'] = round(time.time() - t0, 1)

        # Overwrite/append row in all_rows
        all_rows = [r for r in all_rows if str(r.get('peptide_id')) != pid]
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        tqdm.write(f"  ✓ {pid:25s}  {row['elapsed_s']:.0f}s  "
                   f"{'ERR: ' + str(row['extraction_error'])[:40] if row['extraction_error'] else 'OK'}")

    print(f"\n✅  Saved → {OUT_CSV}")
    df = pd.read_csv(OUT_CSV)
    ok  = df['extraction_error'].isna().sum()
    err = df['extraction_error'].notna().sum()
    print(f"   Rows: {ok} OK  /  {err} errored")


if __name__ == "__main__":
    main()
