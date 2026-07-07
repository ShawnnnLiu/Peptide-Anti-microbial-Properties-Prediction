# archive/

Scripts moved here during the Phase-3 folder reorganization instead of being deleted.
They lived at the top level of `sequence_to_svm_minimal/` and were removed from the
working tree by a previous refactor pass **without a subfolder home**. Restored from
git `HEAD` (commit `b12974a`) on **2026-07-06** and quarantined here to preserve
provenance per REFACTOR_PLAN.md Phase 4 ("archive after approval, never delete").

These are **frozen references**, not part of the live code path. They are **not**
covered by the smoke-test suite. Do not edit them in place — if one is revived, copy
it out and fix its paths first (see "Reviving" below).

## What each file is

| File | Lines | What it is | Relationship to live code |
|---|---:|---|---|
| `predict_buf_hemolysis.py` | 506 | **v1** RF regression for % hemolysis (StaPep MD features) → predicts hemolysis for 12 Buforin variants; 3-panel figure. | Superseded by `predict_buf_hemolysis_v2.py`. The migrated `regression/predict_hemolysis_regression.py` is a *different* model (14-feature paper subset, not these features). |
| `predict_buf_hemolysis_v2.py` | 664 | **v2**: 18-feature RF (adds `lyticity_index`, `sasa`, `sheet_percent`, `hydrophobic_sasa`) + improved hemolysis parsing (single-value / list / LC50-HC50 via Hill n=1). | **Not migrated.** Distinct from `regression/predict_hemolysis_regression.py` (14-feat). Latest of the hemolysis-RF line. |
| `predict_buf_hemolysis_lyticity.py` | 616 | Hemolysis regression using **only** `lyticity_index` (quadratic + exponential parametric fits). | Distinct single-feature experiment; no live counterpart. |
| `predict_buf_pmic_v2.py` | 618 | **v2** pMIC RF (E. coli), 18 features, mirrors the hemolysis-v2 structure. | Original is `regression/predict_pmic_regression.py` (migrated, 14-feat) / `regression/predict_pmic_stapled_variants.py`. This 18-feat version was not migrated. |
| `plot_hydrophobic_vs_mic.py` | 258 | Figure: hydrophobic SASA (Total−PSA) and Kyte-Doolittle index vs MIC for 7 F10W Buforin variants. Reads `test_buf_specific_stapep_features.csv`. | Generates the research PNG `hydrophobic_vs_mic.png` (listed in PROJECT_MAP §5). No counterpart in `figures/`. |
| `plot_lyticity_vs_hemolysis.py` | 426 | Figure: `lyticity_index` vs % hemolysis (training set + Buforin variants). | Distinct from `figures/lyticity_vs_mic.py`, which plots lyticity vs **MIC**, not hemolysis. |

## Why archived rather than kept live

Only `predict_buf_hemolysis.py` (v1) is cleanly superseded. The other five are either a
distinct feature set (the two 18-feature `_v2` models), a distinct single-feature model,
or figure scripts with no exact replacement — they would have been **lost analyses**, not
redundant duplicates. They were parked here pending a decision on whether the 18-feature
variants should be promoted into `regression/` as an alternative feature configuration.

## Reviving a script

Each script was written to run from the top-level `sequence_to_svm_minimal/` directory:
`BASE = Path(__file__).parent` and data resolved as `BASE / "data" / "training_dataset" / "StaPep"`.
From `archive/` that path is wrong (`archive/data/...` does not exist). To revive one:

1. Copy it back to `sequence_to_svm_minimal/` (or into the appropriate subfolder), **don't** edit it here.
2. If placed in a subfolder, migrate it to the shared modules like its siblings:
   `utils.paths` (`STAPEP_DIR`, `PROJECT_ROOT`), `features.stapep_columns`, `utils.mic_units`,
   `features.reference_peptides` — see RUNNABLE_FILES.md §10.
3. Add import + `--help` smoke tests to match the rest of the tree.
