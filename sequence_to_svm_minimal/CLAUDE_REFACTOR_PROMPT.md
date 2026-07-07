# Claude Code Refactor Prompt

Paste this into Claude Code from the `sequence_to_svm_minimal` directory.

```text
You are working in:
c:\Users\bioin\Documents\SVM_ESM_Peptides\Peptide-Anti-microbial-Properties-Prediction\sequence_to_svm_minimal

Read `REFACTOR_PLAN.md` first and follow it closely.

Goal:
Refactor this research codebase so every important Python file is either runnable, documented as expensive/manual, or intentionally marked legacy/broken. Start by creating the runnable inventory and smoke-test harness before changing scientific behavior.

Rules:
- Do not delete data, logs, figures, model files, or result artifacts.
- Do not run expensive jobs unless explicitly approved: 50 ns MD, ESMFold batches, full GNN training, full MLP training, or large dataset regeneration.
- Do not overwrite canonical CSVs unless explicitly approved.
- Preserve current scientific behavior during the inventory and smoke-test phases.
- Prefer small commits/changes grouped by module area.
- If a file is broken, document why instead of silently patching around unknown behavior.
- Be careful with Windows vs WSL paths.

First tasks:
1. Create `RUNNABLE_FILES.md`.
   - Inventory every `.py` file.
   - Classify each file as `core library`, `CLI script`, `experiment`, `data generation`, `expensive/manual`, `legacy/reference`, or `broken/unknown`.
   - For each runnable file, document command, required env, inputs, outputs, runtime estimate, and whether it is safe for smoke tests.

2. Create `PROJECT_MAP.md`.
   - Summarize major directories.
   - Summarize key datasets and result files.
   - Identify canonical files versus legacy/reference files.
   - Note high-risk workflows: StaPep MD, ESMFold, GNN training, MIC parsing, hemolysis parsing, dataset swapping.

3. Create an initial smoke-test harness.
   - Add `tests/test_imports.py`.
   - Add `tests/test_entrypoints.py` if practical.
   - Prefer import checks and `--help` checks.
   - Skip or xfail scripts that require CUDA, WSL-only tooling, OpenMM, AmberTools, pytraj, full MD, ESMFold, or long training.
   - Document skipped files in `SMOKE_TESTS.md`.

4. Run only safe checks.
   - Run `python -m pytest` if pytest is available.
   - If pytest is missing, document that and do not install packages unless asked.
   - Do not start expensive scientific jobs.

5. Stop and report.
   - Summarize inventory counts.
   - List broken or unknown files.
   - List proposed first refactor slice.
   - Recommend whether to start with QSAR/SVM, regression, StaPep utilities, MLP, or GNN.

Only after the inventory and smoke tests are in place should you begin code refactoring. If you do refactor, start with the QSAR/SVM path because it is fast and lower risk than StaPep MD, ESMFold, or GNN workflows.
```

