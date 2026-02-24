#!/usr/bin/env python3
"""
Full StaPep MD feature extraction for stapled_amps.csv  (run in WSL)
----------------------------------------------------------------------
Extracts ALL 17 StaPep features for each of the 188 stapled AMPs:

  Sequence features (ProtParamsSeq — fast, no MD):
    length, weight, hydrophobic_index, charge, aromaticity,
    isoelectric_point, fraction_arginine, fraction_lysine, lyticity_index

  MD/structure features (AmberTools tleap + OpenMM + pytraj):
    helix_percent, sheet_percent, loop_percent,
    mean_bfactor, mean_gyrate, num_hbonds, psa, sasa

MD settings (implicit GB/OBC2 solvent, CPU platform):
  timestep  = 2 fs
  nsteps    = 250 000  → 500 ps  (~40–60 s per peptide on CPU)
  interval  = 2 500   → frame every 5 ps → 100 frames total

Usage (WSL, stap conda env):
    conda activate stap
    python /mnt/c/Users/bioin/Documents/SVM_ESM_Peptides/\
Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/\
data/training_dataset/StaPep/run_amp_md_features.py

    Optional flags:
      --steps N        Override MD steps (default 250000)
      --platform CUDA  Use GPU (default CPU)
      --seq-only       Skip MD; extract only the 9 sequence-level features

Resumable: rows already in the output CSV are skipped automatically.
"""

from __future__ import annotations
import os
import sys
import argparse
import traceback
import warnings
import time
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── tqdm (required) ───────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    print("[ERROR] tqdm not installed.  Run:  pip install tqdm", file=sys.stderr)
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
AMP_CSV  = os.path.join(HERE, "stapled_amps.csv")
OUT_CSV  = os.path.join(HERE, "stapled_amps_features.csv")
WORK_DIR = "/tmp/stapep_md"          # Linux tmp dir for MD artefacts
os.makedirs(WORK_DIR, exist_ok=True)

# ── MD defaults ───────────────────────────────────────────────────────────────
DEFAULT_MD_NSTEPS   = 250_000        # 500 ps at 2 fs timestep
DEFAULT_MD_INTERVAL = 2_500          # frame every 5 ps → 100 frames
MD_TIMESTEP         = 2              # fs
MD_TEMP             = 300            # K
MD_FRICTION         = 1.0            # ps⁻¹
MD_TYPE             = "implicit"

# ── GPU auto-detection ────────────────────────────────────────────────────────
def detect_best_platform() -> str:
    """
    Return the fastest OpenMM platform available on this machine.
    Priority: CUDA > OpenCL > CPU
    CUDA is ~10-50x faster than CPU for MD simulations.
    """
    try:
        import openmm as mm
        available = [mm.Platform.getPlatform(i).getName()
                     for i in range(mm.Platform.getNumPlatforms())]
        for pname in ("CUDA", "OpenCL", "CPU"):
            if pname in available:
                return pname
    except Exception:
        pass
    return "CPU"


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    best = detect_best_platform()
    p = argparse.ArgumentParser(description="StaPep MD feature extraction for all 188 AMPs")
    p.add_argument("--steps",    type=int, default=DEFAULT_MD_NSTEPS,
                   help=f"MD steps (default {DEFAULT_MD_NSTEPS})")
    p.add_argument("--platform", type=str, default=best,
                   choices=["CPU", "CUDA", "OpenCL"],
                   help=f"OpenMM platform (auto-detected: {best})")
    p.add_argument("--seq-only", action="store_true",
                   help="Extract only the 9 sequence-level features, skip MD")
    p.add_argument("--out", type=str, default=OUT_CSV,
                   help="Output CSV path")
    return p.parse_args()


# ── Sequence preprocessing ────────────────────────────────────────────────────
def build_stapep_seq(hiden_seq: str, n_mod: str, c_mod: str) -> str:
    """
    Convert AMP Hiden_Sequence → StaPep token format.

      X  → S5   (S5 olefin staple; X is the AMP dataset's single-char code)
      Z  → R8   (R8 olefin staple)
      J  → B    (Norleucine)
      lowercase → uppercase  (D-amino acids treated as backbone L-AA for
                               sequence features; StaPep uses uppercase)
      Acetylation N-term  → prepend 'Ac'
      Amidation   C-term  → append  'NH2'
    """
    seq = str(hiden_seq).strip()
    seq = seq.replace("Z", "R8")    # R8 olefin staple
    seq = seq.replace("J", "B")     # Norleucine
    seq = seq.upper()               # D-AA treated as L-AA backbone
    prefix = "Ac"  if str(n_mod).strip().lower() == "acetylation" else ""
    suffix = "NH2" if str(c_mod).strip().lower() == "amidation"   else ""
    return f"{prefix}{seq}{suffix}"


# ── Sequence-only features (9 features, fast) ─────────────────────────────────
def seq_features(stapep_seq: str, c_mod: str) -> dict:
    """Return the 9 sequence-level StaPep features. Raises on failure."""
    from stapep.utils import ProtParamsSeq
    amide = str(c_mod).strip().lower() == "amidation"
    pps   = ProtParamsSeq(stapep_seq)
    return {
        "length":            pps.seq_length,
        "weight":            pps.weight,
        "hydrophobic_index": pps.hydrophobicity_index,
        "charge":            pps.calc_charge(pH=7.0, amide=amide),
        "aromaticity":       pps.aromaticity,
        "isoelectric_point": pps.isoelectric_point,
        "fraction_arginine": pps.fraction_arginine,
        "fraction_lysine":   pps.fraction_lysine,
        "lyticity_index":    pps.lyticity_index,
    }


# ── Full MD pipeline (17 features) ────────────────────────────────────────────
def full_pipeline(dramp_id: str, stapep_seq: str, c_mod: str,
                  nsteps: int, interval: int, platform: str,
                  bar: tqdm) -> dict:
    """
    Run tleap → OpenMM MD → pytraj analysis for one peptide.
    Returns a flat feature dict or a dict with key 'error'.
    """
    from stapep.molecular_dynamics import PrepareProt, Simulation
    from stapep.utils import ProtParamsSeq, PhysicochemicalPredictor

    work  = os.path.join(WORK_DIR, dramp_id)
    os.makedirs(work, exist_ok=True)
    amide = str(c_mod).strip().lower() == "amidation"

    try:
        # 1. Sequence features ─────────────────────────────────────────────────
        bar.set_postfix_str(f"{dramp_id} | seq features")
        pps = ProtParamsSeq(stapep_seq)
        s_feats = {
            "length":            pps.seq_length,
            "weight":            pps.weight,
            # hydrophobic_index here is the sequence-based Kyte-Doolittle value.
            # It acts as a fallback if the DSSP-based MD version fails.
            "hydrophobic_index": pps.hydrophobicity_index,
            "charge":            pps.calc_charge(pH=7.0, amide=amide),
            "aromaticity":       pps.aromaticity,
            "isoelectric_point": pps.isoelectric_point,
            "fraction_arginine": pps.fraction_arginine,
            "fraction_lysine":   pps.fraction_lysine,
            "lyticity_index":    pps.lyticity_index,
        }

        # 2. AmberTools topology ───────────────────────────────────────────────
        bar.set_postfix_str(f"{dramp_id} | tleap")
        pp = PrepareProt(stapep_seq, work, method=None)
        pp._gen_prmtop_and_inpcrd_file()

        topology_file = os.path.join(work, "pep_vac.prmtop")
        if not os.path.exists(topology_file):
            return {"error": "tleap did not produce pep_vac.prmtop"}

        # 3. MD simulation ─────────────────────────────────────────────────────
        bar.set_postfix_str(f"{dramp_id} | minimize")
        sim = Simulation(work)
        sim.setup(
            type=MD_TYPE,
            solvent="water",
            temperature=MD_TEMP,
            friction=MD_FRICTION,
            timestep=MD_TIMESTEP,
            interval=interval,
            nsteps=nsteps,
            platform=platform,
        )
        sim.minimize()
        bar.set_postfix_str(f"{dramp_id} | MD {nsteps:,} steps ({platform})…")
        sim.run()

        traj_file = os.path.join(work, "traj.dcd")
        if not os.path.exists(traj_file):
            return {"error": "MD did not produce traj.dcd"}

        # 4. MD-based structure features ───────────────────────────────────────
        bar.set_postfix_str(f"{dramp_id} | MD analysis")
        pcp = PhysicochemicalPredictor(
            sequence=stapep_seq,
            topology_file=topology_file,
            trajectory_file=traj_file,
            start_frame=0,
        )
        mean_pdb = os.path.join(work, "mean_structure.pdb")
        pcp._save_mean_structure(mean_pdb)

        def _safe(fn, *args, name="?"):
            """Call fn(*args), return None and print warning on failure."""
            try:
                return fn(*args)
            except Exception as exc:
                tqdm.write(f"  [WARN] {dramp_id} {name}: {exc}")
                return None

        m_feats = {
            "hydrophobic_index": _safe(pcp.calc_hydrophobic_index, mean_pdb, name="hydrophobic_index"),
            "helix_percent":     _safe(pcp.calc_helix_percent,                name="helix_percent"),
            "sheet_percent":     _safe(pcp.calc_extend_percent,               name="sheet_percent"),
            "loop_percent":      _safe(pcp.calc_loop_percent,                 name="loop_percent"),
            "mean_bfactor":      _safe(pcp.calc_mean_bfactor,                 name="mean_bfactor"),
            "mean_gyrate":       _safe(pcp.calc_mean_gyrate,                  name="mean_gyrate"),
            "num_hbonds":        _safe(pcp.calc_n_hbonds,                     name="num_hbonds"),
            "psa":               _safe(pcp.calc_psa, mean_pdb,                name="psa"),
            "sasa":              _safe(pcp.calc_mean_molsurf,                 name="sasa"),
        }

        # Drop None values so s_feats fallbacks (e.g. seq-based hydrophobic_index)
        # are not overwritten when a feature fails.
        m_feats_ok = {k: v for k, v in m_feats.items() if v is not None}

        # If ALL MD features are None, treat as an error
        if not m_feats_ok:
            return {"error": "All MD feature extractions failed (check WARN lines above)"}

        return {**s_feats, **m_feats_ok}

    except Exception:
        return {"error": traceback.format_exc()}


# ── Persistence helpers ────────────────────────────────────────────────────────
ID_COLS   = ["DRAMP_ID", "Sequence", "Hiden_Sequence", "stapep_seq",
             "N_terminal_Modification", "C_terminal_Modification", "label"]
SEQ_FEATS = ["length", "weight", "hydrophobic_index", "charge", "aromaticity",
             "isoelectric_point", "fraction_arginine", "fraction_lysine",
             "lyticity_index"]
MD_FEATS  = ["helix_percent", "sheet_percent", "loop_percent",
             "mean_bfactor", "mean_gyrate", "num_hbonds", "psa", "sasa"]
ALL_FEAT_COLS = SEQ_FEATS + MD_FEATS


def _ordered_cols(df: pd.DataFrame, seq_only: bool) -> list[str]:
    feat_cols = SEQ_FEATS if seq_only else ALL_FEAT_COLS
    ordered   = ID_COLS + feat_cols + ["elapsed_s", "extraction_error"]
    return [c for c in ordered if c in df.columns]


def _save(records: list[dict], existing: pd.DataFrame | None,
          path: str, seq_only: bool) -> None:
    new_df = pd.DataFrame(records)
    if existing is not None and len(existing):
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="DRAMP_ID", keep="last")
    else:
        combined = new_df
    combined[_ordered_cols(combined, seq_only)].to_csv(path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    md_nsteps   = args.steps
    md_interval = max(1, md_nsteps // 100)   # always keep ~100 frames
    platform    = args.platform
    seq_only    = args.seq_only
    out_csv     = args.out

    print("=" * 65)
    print("  StaPep feature extraction — 188 stapled AMPs")
    print("=" * 65)
    print(f"  Input  : {AMP_CSV}")
    print(f"  Output : {out_csv}")
    if seq_only:
        print("  Mode   : sequence-only (9 features, no MD)")
    else:
        gpu_tag = " ⚡ GPU" if platform in ("CUDA", "OpenCL") else ""
        print(f"  Mode   : full MD  ({md_nsteps:,} steps, platform={platform}{gpu_tag})")
    print()

    # Load AMPs
    amp = pd.read_csv(AMP_CSV, encoding="utf-8-sig")
    total = len(amp)
    print(f"  Loaded {total} rows from CSV")

    # Resuming — only skip rows that completed WITHOUT errors
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        ok_mask  = existing["extraction_error"].isna() | (existing["extraction_error"] == "")
        done_ids = set(existing.loc[ok_mask, "DRAMP_ID"].astype(str))
        n_prev_err = len(existing) - len(done_ids)
        print(f"  Resuming — {len(done_ids)} rows done OK, "
              f"{n_prev_err} errored rows will be retried, "
              f"{total - len(done_ids)} remaining")
    else:
        existing  = None
        done_ids  = set()

    print()

    records  = []
    n_ok     = 0
    n_err    = 0
    n_skip   = len(done_ids)
    errors   = []          # (dramp_id, short_error)

    bar = tqdm(
        total=total,
        initial=n_skip,
        unit="peptide",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        desc="StaPep",
    )

    for _, row in amp.iterrows():
        dramp_id  = str(row["DRAMP_ID"])
        hiden_seq = row["Hiden_Sequence"]
        n_mod     = row.get("N_terminal_Modification", "")
        c_mod     = row.get("C_terminal_Modification", "")
        seq_label = row.get("Sequence", hiden_seq)

        if dramp_id in done_ids:
            continue   # already counted in bar.initial

        stapep_seq = build_stapep_seq(hiden_seq, n_mod, c_mod)

        t0 = time.time()
        if seq_only:
            bar.set_postfix_str(f"{dramp_id} | seq")
            try:
                result = seq_features(stapep_seq, c_mod)
                err    = None
            except Exception as e:
                result = {}
                err    = str(e)
        else:
            result = full_pipeline(dramp_id, stapep_seq, c_mod,
                                   md_nsteps, md_interval, platform, bar)
            err = result.pop("error", None)

        elapsed = round(time.time() - t0, 1)

        record = {
            "DRAMP_ID":                dramp_id,
            "Sequence":                seq_label,
            "Hiden_Sequence":          hiden_seq,
            "stapep_seq":              stapep_seq,
            "N_terminal_Modification": n_mod,
            "C_terminal_Modification": c_mod,
            "label":                   1,
            "elapsed_s":               elapsed,
            "extraction_error":        err,
        }
        record.update(result)

        if err:
            n_err += 1
            short_err = err.splitlines()[-1][:80]
            errors.append((dramp_id, short_err))
            bar.set_postfix_str(f"✗ {dramp_id}  ok={n_ok}  err={n_err}")
        else:
            n_ok += 1
            bar.set_postfix_str(f"✓ {dramp_id}  ok={n_ok}  err={n_err}")

        records.append(record)
        bar.update(1)

        # Incremental save every 10 rows
        if len(records) % 10 == 0:
            _save(records, existing, out_csv, seq_only)

    bar.set_postfix_str(f"done — ok={n_ok}  err={n_err}")
    bar.close()

    # Final save
    _save(records, existing, out_csv, seq_only)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"  DONE  —  {n_ok} succeeded / {n_err} failed / {n_skip} skipped")
    print(f"  Output saved → {out_csv}")
    print("=" * 65)

    if errors:
        print(f"\n  ── Failed rows ({len(errors)}) ──")
        for did, msg in errors:
            print(f"    {did}: {msg}")

    # Feature stats table
    out = pd.read_csv(out_csv)
    feat_cols = [c for c in ALL_FEAT_COLS if c in out.columns]
    if feat_cols:
        print("\n  ── Feature summary (all processed AMPs) ──")
        print(out[feat_cols].describe().round(3).to_string())

    # Overlap check with decoys
    dec_path = os.path.join(HERE, "stapled_decoys.csv")
    if os.path.exists(dec_path):
        dec = pd.read_csv(dec_path)
        shared = [c for c in feat_cols if c in dec.columns]
        if shared:
            print("\n  ── AMP vs Decoy mean feature comparison ──")
            header = f"{'Feature':<25} {'AMP mean':>12} {'Decoy mean':>12}  overlap?"
            print(header)
            print("-" * len(header))
            for feat in shared:
                amp_mu  = out[feat].mean()
                amp_sd  = out[feat].std()
                dec_mu  = dec[feat].mean()
                dec_sd  = dec[feat].std()
                overlap = abs(amp_mu - dec_mu) < 2 * (amp_sd + dec_sd)
                print(f"  {feat:<23} {amp_mu:>12.3f} {dec_mu:>12.3f}  {'YES' if overlap else 'NO'}")


if __name__ == "__main__":
    main()
