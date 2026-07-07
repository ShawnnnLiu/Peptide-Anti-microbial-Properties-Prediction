# Refactor Plan

## Goal

Refactor this research codebase so every important script is either runnable, documented as expensive/manual, or intentionally archived. The main outcome is a cleaner project where future AMP, stapled-peptide, MIC, hemolysis, SVM, MLP, GNN, ESMFold, and StaPep experiments can be run without rediscovering paths, environments, inputs, and outputs.

## Guiding Rules

- Preserve scientific behavior before improving style.
- Do not rerun expensive jobs such as 50 ns MD, ESMFold batches, or full GNN/MLP training unless explicitly requested.
- Treat data files, result logs, and generated figures as research artifacts. Do not delete or overwrite them during refactor.
- Separate "runnable now", "runnable with data/env", "expensive/manual", "legacy", and "broken/unknown".
- Prefer small vertical refactors over broad rewrites.
- After each refactor slice, run smoke tests and update the runnable inventory.

## Phase 0: Baseline Inventory

Deliverables:

- `RUNNABLE_FILES.md`
- `PROJECT_MAP.md`
- `SMOKE_TESTS.md`

Tasks:

1. Enumerate every Python file in `sequence_to_svm_minimal`.
2. Classify each file:
   - `core library`
   - `CLI script`
   - `experiment`
   - `data generation`
   - `expensive/manual`
   - `legacy/reference`
   - `broken/unknown`
3. For each runnable script, record:
   - command
   - required environment
   - required input files
   - expected output files
   - approximate runtime
   - whether it is safe for automated smoke tests
4. Identify duplicated logic:
   - path setup
   - MIC parsing
   - StaPep feature columns
   - QSAR feature columns
   - model training helpers
   - plotting/output helpers

Acceptance criteria:

- A new agent can identify what to run and what not to run.
- No source behavior has been changed yet.

## Phase 1: Smoke Test Harness

Deliverables:

- `tests/test_imports.py`
- `tests/test_entrypoints.py`
- optional `pytest.ini`

Tasks:

1. Add import tests for modules that should import cleanly.
2. Add CLI `--help` tests where scripts expose argparse.
3. Add non-executing checks for expensive scripts.
4. Mark or skip scripts requiring WSL, CUDA, OpenMM, AmberTools, pytraj, or full data.
5. Document how to run:

```bash
python -m pytest
```

Acceptance criteria:

- Smoke tests distinguish runnable failures from intentionally skipped expensive jobs.
- Tests do not start MD, ESMFold batches, or long training runs.

## Phase 2: Shared Configuration And Constants

Deliverables:

- shared path/config helper module
- shared feature-column definitions
- shared MIC/hemolysis parsing helpers where safe

Tasks:

1. Centralize project-root discovery.
2. Centralize `StaPep`, `QSAR`, `Geo24`, and regression feature lists.
3. Centralize repeated MIC parsing only after confirming all callers expect the same behavior.
4. Keep old script interfaces intact.

Acceptance criteria:

- Scripts no longer need repeated fragile `Path(__file__).resolve().parent.parent` blocks.
- Existing outputs remain behaviorally consistent on representative smoke/sample runs.

## Phase 3: Refactor By Vertical Slice

Suggested order:

1. Descriptors and QSAR extraction.
2. SVM classifiers.
3. StaPep feature extraction and dataset utilities.
4. Regression scripts for pMIC/MIC and hemolysis.
5. MLP scripts.
6. GNN scripts.
7. Comparison and plotting scripts.

For each slice:

1. Read the relevant files.
2. Update `RUNNABLE_FILES.md`.
3. Extract only the duplication needed for that slice.
4. Run smoke tests.
5. Run one safe representative command if available.
6. Record unresolved blockers.

Acceptance criteria:

- Each slice ends in a runnable state.
- No unrelated rewrites or formatting churn.

## Phase 4: Archive Or Quarantine Legacy Code

Tasks:

1. Identify scripts superseded by newer versions.
2. Move only after user approval.
3. Preserve provenance in `PROJECT_MAP.md`.
4. Leave old result artifacts untouched.

Acceptance criteria:

- The active code path is clear.
- Historical experiments remain findable.

## High-Risk Areas

- Free-text MIC and hemolysis parsing can silently change labels.
- StaPep MD features are expensive and noisy; they should not be regenerated casually.
- Some scripts use legacy/current CSV filename swapping, especially in `10ns vs 50ns comparison`.
- Windows vs WSL paths can break otherwise valid scripts.
- GPU/CUDA/ESMFold dependencies are environment-sensitive.
- Family leakage in peptide variants can make validation metrics optimistic.

## Recommended First Refactor Target

Start with the runnable inventory and smoke tests, then refactor the SVM/QSAR path first. It is comparatively fast, has clear inputs and outputs, and provides a stable baseline before touching expensive StaPep, ESMFold, or GNN workflows.

