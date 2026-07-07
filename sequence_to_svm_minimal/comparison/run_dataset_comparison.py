#!/usr/bin/env python3
"""
run_dataset_comparison.py
=========================
Side-by-side benchmark of 6 prominent prediction scripts on TWO training feature
datasets:

  Legacy   -- stapled_amps_features.csv                        (147 rows w/ MIC)
  MD-50ns  -- stapled_amps_features_training_XZ_md50ns.csv     (130 rows w/ MIC)

Workflow
--------
1. Backup the legacy CSV.
2. Run each of the 6 scripts in turn -> capture stdout/stderr + any new figures
   into "10ns vs 50ns comparison/legacy_147rows/".
3. Replace the legacy CSV with the MD-50ns CSV (in-place file swap so every
   script picks it up without code changes).
4. Run each of the 6 scripts again -> capture into "md50ns_130rows/".
5. Restore the legacy CSV from backup.

All scripts are launched with cwd = sequence_to_svm_minimal/ so any relative
output paths (e.g. predict_pmic_regression.py default --save) land at the
project root, which is also what we snapshot for figure capture.

Usage
-----
    conda activate stap   # or whichever env has sklearn etc.
    python comparison/run_dataset_comparison.py
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import time
from pathlib import Path

# Make utils/ importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.paths import PROJECT_ROOT, STAPEP_DIR as DATA_DIR

LEGACY_CSV   = DATA_DIR / "stapled_amps_features.csv"
MD50_CSV     = DATA_DIR / "stapled_amps_features_training_XZ_md50ns.csv"
BACKUP_CSV   = DATA_DIR / "stapled_amps_features.csv.legacy_backup"

OUT_ROOT     = PROJECT_ROOT / "10ns vs 50ns comparison"

SCRIPTS = [
    # (category, filename without .py, default --save argument or None)
    ("svm",        "predict_mic_svm",                None),
    ("svm",        "run_stapep_svm",                 None),
    ("mlp",        "predict_mic_mlp",                None),
    ("mlp",        "run_stapep_mlp",                 None),
    ("regression", "predict_pmic_stapled_variants",  None),
    ("regression", "predict_pmic_regression",        "pmic_regression.png"),
]


def snapshot_pngs() -> dict[str, float]:
    """Map of png_name -> mtime for all PNGs at the project root."""
    return {p.name: p.stat().st_mtime for p in PROJECT_ROOT.glob("*.png")}


def collect_new_pngs(before: dict[str, float]) -> list[Path]:
    """Return PNGs at PROJECT_ROOT that are new OR were modified since 'before'."""
    out: list[Path] = []
    for p in PROJECT_ROOT.glob("*.png"):
        prev = before.get(p.name)
        if prev is None or p.stat().st_mtime > prev + 0.01:
            out.append(p)
    return out


def run_one(category: str, name: str, save_arg: str | None,
            label: str, out_dir: Path) -> None:
    script_path = PROJECT_ROOT / category / f"{name}.py"
    log_path    = out_dir / f"{category}_{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(script_path)]
    if save_arg:
        # Force the script to save its figure with a unique name so we capture it
        unique_save = f"{category}_{name}.png"
        cmd += ["--save", unique_save]

    print(f"\n[{label}] running {category}/{name}.py ...")
    t0 = time.time()
    pre_pngs = snapshot_pngs()

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    dt = time.time() - t0

    header = (
        f"=== {category}/{name}.py  |  dataset={label}  |  "
        f"exit={proc.returncode}  |  wall={dt:.1f}s ===\n"
        f"cmd: {' '.join(cmd)}\n\n"
    )
    body = proc.stdout + "\n----- STDERR -----\n" + proc.stderr
    log_path.write_text(header + body, encoding="utf-8")

    new_pngs = collect_new_pngs(pre_pngs)
    for png in new_pngs:
        target = out_dir / f"{category}_{name}_{png.name}"
        # Move (not copy) so root stays clean; legacy figures will be regenerated below
        shutil.move(str(png), str(target))
        print(f"   captured figure -> {target.name}")

    print(f"   exit={proc.returncode}  wall={dt:.1f}s  log={log_path.name}"
          + (f"  figs={len(new_pngs)}" if new_pngs else "  (terminal only)"))


def run_all(label: str) -> None:
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat, name, save_arg in SCRIPTS:
        run_one(cat, name, save_arg, label, out_dir)


def write_summary() -> None:
    summary = OUT_ROOT / "README.md"
    summary.write_text(
        "# Dataset comparison: legacy vs MD-50ns\n\n"
        "Scripts were each run twice -- once with the legacy 147-row\n"
        "`stapled_amps_features.csv` and once with the new 130-row\n"
        "`stapled_amps_features_training_XZ_md50ns.csv` (50 ns implicit-GB MD).\n\n"
        "Folder layout:\n\n"
        "- `legacy_147rows/`  ... stdout logs + figures from the legacy run\n"
        "- `md50ns_130rows/`  ... stdout logs + figures from the 50 ns run\n\n"
        "Filenames inside each folder follow the pattern:\n\n"
        "    <category>_<script_name>.log\n"
        "    <category>_<script_name>_<original_figure_name>.png\n\n"
        "so the same metric/figure can be diffed across folders by name.\n",
        encoding="utf-8",
    )


def main() -> int:
    if not LEGACY_CSV.exists():
        print(f"ERROR: missing {LEGACY_CSV}", file=sys.stderr)
        return 2
    if not MD50_CSV.exists():
        print(f"ERROR: missing {MD50_CSV}", file=sys.stderr)
        return 2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUT_ROOT}")

    print("Backing up legacy CSV ...")
    shutil.copy2(LEGACY_CSV, BACKUP_CSV)

    try:
        # Phase 1: legacy
        print("\n========== Phase 1/2: LEGACY (147 rows) ==========")
        run_all("legacy_147rows")

        # Phase 2: swap in the MD-50ns CSV under the legacy name
        print("\nSwapping legacy CSV -> MD-50ns CSV (in-place) ...")
        shutil.copy2(MD50_CSV, LEGACY_CSV)

        print("\n========== Phase 2/2: MD-50ns (130 rows) ==========")
        run_all("md50ns_130rows")
    finally:
        print("\nRestoring legacy CSV from backup ...")
        shutil.copy2(BACKUP_CSV, LEGACY_CSV)
        BACKUP_CSV.unlink()

    write_summary()
    print(f"\nDone.  See {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
