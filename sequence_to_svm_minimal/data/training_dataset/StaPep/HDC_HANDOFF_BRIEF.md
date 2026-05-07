# HDC Project Handoff Brief — Stapled-AMP Dataset

**Audience.** A new agent tasked with designing a hyperdimensional-computing (HDC) encoding + model for this stapled-peptide dataset.

**Your role.** Devise the encoding scheme and HDC architecture (item memory, binding, bundling, classifier/regressor head, training loop). The dataset, its quality, and its caveats are documented below — you should not need to re-investigate them.

**My role (the previous agent).** I built and validated the dataset; I did not pick the model. I am explicitly NOT recommending an encoding strategy — that is your call.

---

## 1. What the dataset is and why it exists

The training corpus is a curated set of **stapled antimicrobial peptides** (AMPs) drawn from the DRAMP database plus one paper supplement (PMID 31427820). All peptides are **hydrocarbon-stapled** (S5 / R8 olefin staples, denoted `X` and `Z`); lysine-tethered (`Ⓚ`) staples were removed at advisor's request because their cyclic chemistry differs fundamentally.

Every peptide has been processed through **StaPep** (a published peptide-descriptor pipeline that runs AmberTools `tleap` → OpenMM implicit-GB MD → pytraj/CPPTRAJ analysis) to extract a 17-feature descriptor vector.

The downstream task is **hemolysis (hemo) and minimum-inhibitory-concentration (MIC) regression**, plus optional AMP-vs-decoy classification.

---

## 2. File inventory (all under `…/StaPep/`)

### Canonical training inputs

| File | Rows | Role |
|---|---:|---|
| `stapled_amps_features_training_XZ_md50ns.csv` | 172 (in progress) | **THE training feature matrix.** 50 ns implicit-GB MD per peptide, K-stapled removed, NaN-failure peptide DRAMP21556 removed. |
| `stapled_amps_combined_paper_dataset_XZ_only.csv` | 172 | Identity / metadata for the same 172 peptides (sequences, names, source paper, hemolytic-activity text, etc.). Joins on `DRAMP_ID`. |
| `stapled_decoys.csv` | 355 | Decoy stapled peptides (label = 0). Same StaPep 17-feature schema. These are "stapled peptides that are NOT antimicrobial" — MDM2 inhibitors, AKAP disruptors, etc. — making them a strong negative class because they have identical chemistry to AMPs (staples included) but no antimicrobial function. |

### Reference / backup files (older protocols, do not use as primary)

| File | Rows | What it is |
|---|---:|---|
| `stapled_amps_features_training_XZ_md4ns.csv` | 172 | Same peptides, 4 ns MD. Lower-quality features. Useful for protocol-comparison studies. |
| `stapled_amps_features_training_XZ_md4ns_REP2.csv` | 3 | 4 ns replicate run on (DRAMP21558, DRAMP21542, DRAMP21541) with a fresh OpenMM RNG seed. Defines the per-peptide MD noise floor. |
| `stapled_amps_features.csv` | 188 | Original DRAMP-only feature run, mixed protocol history. Don't use. |
| `stapled_amps_features_combined_paper_dataset.csv` | 195 | Includes the 22 K-stapled peptides that were later dropped. Don't use. |

### Audit logs

| File | Purpose |
|---|---|
| `dropped_k_stapled.csv` | 22 K-stapled peptides removed from training set |
| `dropped_md_failures.csv` | 1 peptide (DRAMP21556 = Mag(i+7)13) dropped — MD blew up with "Energy is NaN" twice in a row |
| `paper_vs_dramp_side_by_side.csv` | Mapping of paper supplement peptides to DRAMP equivalents |

### Diagnostic scripts (already written, runnable in WSL `stap` env)

| Script | What it does |
|---|---|
| `_verify_training_md4ns.py` | Completeness check: counts NaN MD cells, identifies extraction errors |
| `sanity_check_md_features.py` | Internal-consistency + distributional sanity report; produces `sanity_report_*.txt` |
| `_inspect_md50ns_in_progress.py` | Read-only inspector that compares the in-progress 50 ns CSV against the 4 ns reference |
| `_compare_replicate_md.py` | Computes per-peptide noise floor between two replicate runs |
| `_probe_dssp_staples.py` | Per-residue DSSP probe — used to confirm staple residues are correctly classified |

---

## 3. Feature schema (17 features, all numeric, all populated for 172 / 172 rows)

### Sequence-only (9 features) — **deterministic, zero MD noise**

| Column | Definition | Range observed | Notes |
|---|---|---|---|
| `length` | residue count incl. caps | 7 – 32 (mean 17.8) | wide span, important for HDC sequence-length handling |
| `weight` | molecular weight (Da) | ~860 – 3850 | scales with length; mostly redundant with `length` |
| `hydrophobic_index` | Kyte-Doolittle mean | −1.2 to +2.2 | |
| `charge` | net charge at pH 7 | −2.9 to +14 (mean +5) | most peptides are cationic, expected for AMPs |
| `aromaticity` | F/W/Y fraction | 0 – 0.43 | |
| `isoelectric_point` | pI | 4.4 – 12.0 (mean 10.3) | |
| `fraction_arginine` | R / total | 0 – 0.43 | |
| `fraction_lysine` | K / total | 0 – 0.57 (mean 0.26) | |
| `lyticity_index` | StaPep's empirical lytic-propensity score | 136 – 949 (mean 495) | high values usually = highly lytic / hemolytic |

### MD-derived (8 features) — **noisy at the per-peptide level, see §6**

| Column | Definition | Notes for HDC |
|---|---|---|
| `helix_percent` | Mean over residues × frames of DSSP "H" classification | Most discriminative MD feature for AMP function. At 50 ns it's a usable per-peptide label; at 4 ns it was only a cohort-level signal. |
| `sheet_percent` | DSSP "E" fraction | ~0 for almost every peptide — uninformative; consider dropping |
| `loop_percent` | DSSP "C" fraction | = 1 − helix − sheet (closure verified to ±0.01 for every row); contains the same information as `helix_percent` for this cohort. Consider dropping one. |
| `mean_bfactor` | Mean atomic fluctuation across trajectory | Noisy (~45% replicate noise at 4 ns); proxy for flexibility |
| `mean_gyrate` | Time-averaged radius of gyration (Å) | Scales as ~2.2 · √length for folded helix; clean signal |
| `num_hbonds` | Mean intramolecular backbone H-bonds | Discrete-valued (integer-ish); rises strongly with helicity |
| `psa` | Polar surface area (Å²) | Geometric, low noise |
| `sasa` | Solvent-accessible surface area (Å²) | Geometric, low noise; scales with length |

### Bookkeeping columns also in the CSV

| Column | Notes |
|---|---|
| `DRAMP_ID` | Primary key. Format: `DRAMP21482` … `DRAMP29241`, plus 7 `PAPER_31427820_*` rows |
| `Sequence` | Display sequence with circled-letter staple glyphs (`Ⓧ`, `Ⓩ`) |
| `Hiden_Sequence` | ASCII-only sequence: `X` = S5 staple, `Z` = R8 staple, `J` = norleucine alt, `B` = norleucine, lowercase = D-amino acid |
| `stapep_seq` | StaPep tokenised form, with `Ac` / `NH2` caps inlined |
| `N_terminal_Modification`, `C_terminal_Modification` | text: "Free", "Acetylation", "Amidation" |
| `label` | always `1` for AMPs; you'll need `0` for decoys |
| `elapsed_s`, `extraction_error` | run telemetry; should always be empty error after the final 172/172 promotion |

---

## 4. Sequence alphabet — important for HDC encoding decisions

Peptides use a **24-letter expanded alphabet** beyond the canonical 20:

| Token | Meaning | Count in dataset |
|---|---|---|
| 20 standard AAs | A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y | thousands |
| `X` | S5 staple residue (S-2-(4'-pentenyl)alanine, hydrocarbon staple) | 384 |
| `Z` | R8 staple residue (R-2-(7'-octenyl)alanine, hydrocarbon staple) | 29 |
| `J` | norleucine variant | 66 |
| `B` | norleucine | 7 |
| lowercase letter | D-stereoisomer of that AA (rare; few entries) | tiny |

**Staple semantics that may matter for encoding.** Staples come in **pairs** that are chemically cross-linked — they are not independent residues. For an i,i+4 staple at positions p and p+4, the two `X`/`Z` residues are physically tethered. The metadata column `Special_Amino_Acid_and_Stapling_Position` describes the topology in plain English (e.g. `①Ⓧ (10) and Ⓧ (14) are cross-linked by hydrocarbon stapling`). HDC encoding could treat the staple as either:
- two independent `X` tokens (simplest, loses the bond),
- a single tagged "stapled-pair" feature bound at the midpoint position (richer),
- or a topological feature (i,i+4 vs i,i+7) bound separately.

That decision is yours.

**N/C-terminal caps** are present (`Ac` and `NH2`) for most peptides; they are encoded in `stapep_seq` and `*_terminal_Modification`. They affect charge but not topology.

---

## 5. Labels — read this carefully, this is the most important section

### Available *features*: 17 numeric columns. ✅ ready for HDC.
### Available *labels*: **NOT YET ENGINEERED.** ⚠️ This is the open task.

The identity CSV has TWO unstructured-text columns containing the regression targets, plus an **MIC** value buried in the `Activity` column:

```
Hemolytic_Activity = "[Ref.29275987] MHC = 3.8 μM against human red blood cells. Note: ..."
Activity           = "[Ref.29275987] Gram-negative bacteria: E. coli (MIC= 1.0 μM)"
```

**Both targets need to be regex-extracted into numeric columns before any model training.** The format varies wildly across rows — some entries cite IC50, some MHC, some EC50, some report multiple organisms. Several rows lack experimental values entirely.

For the 7 `PAPER_*` peptides, the source paper (PMID 31427820) provides clean MIC tables for E. coli / B. cereus / P. aeruginosa / S. aureus and a single hemolysis percentage at 25 μg/mL. Those need a different parser than the DRAMP free-text.

**My recommendation to the new agent: settle the label-engineering pipeline before encoding.** The HDC architecture (classifier vs regressor, multi-output vs single-output, log-scale vs linear) hinges on what fraction of the 172 peptides actually have usable numeric targets, which I have not characterised. A quick `pandas` parsing pass is the right first move.

### Decoy labels are simple
`stapled_decoys.csv` rows are by definition non-AMP. For binary classification: AMP=1, decoy=0. The decoy file does not contain hemo/MIC values (they would be undefined for non-AMPs).

---

## 6. Data-quality caveats the new agent must respect

### a. MD features have a noise floor

Replicate-MD (4 ns rep1 vs 4 ns rep2 with different RNG seed) gave per-peptide |Δ| / mean(rep1):

| Feature | 4 ns noise | 50 ns improvement |
|---|---:|---|
| psa, sasa, mean_gyrate, loop_percent | < 5 % | already clean |
| helix_percent | 23 % at 4 ns | drops to ~10 % at 50 ns |
| mean_bfactor, num_hbonds | 45–67 % at 4 ns | improves but still noisy |
| sheet_percent | huge in relative terms but absolute values ≈ 0 | still ≈ 0 |

Implication: even at 50 ns the MD features carry irreducible noise from the underlying single-trajectory implicit-GB MD. HDC's bundling-as-averaging can absorb this gracefully; just don't expect the model to predict per-peptide hemo/MIC to beyond ~10 % precision purely from these inputs.

### b. Implicit-GB systematic bias

Implicit-solvent MD over-stabilises α-helix by ~10–20 % vs explicit water and vs experimental CD. So the absolute helix_percent values are biased upward from what CD would measure. This is uniform across the cohort, which means: **fine for relative ranking and for cross-peptide regression, NOT fine for publishing absolute structural percentages.**

### c. Long peptides under-converged

Peptides ≥30 residues (CAP-(i+4)1,23 = DRAMP21541 at 32-mer; DRAMP29235–29238 at 30-mer) likely have not reached full equilibrium even at 50 ns. Implicit-GB folding scales linearly with chain length, and 50 ns is borderline for 30-mers. Their MD features may be noisier than the magainin-class core. ~5 peptides total.

### d. DSSP-on-staples is NOT broken

I investigated whether StaPep's DSSP mis-classifies the staple residues. **It doesn't.** Per-residue helix fractions for `PS5` / `PR8` track within ~5–10 % of flanking standard residues. The low cohort helix at 4 ns was a sampling problem, not a labeling problem. (The probe script `_probe_dssp_staples.py` reproduces this if you want to see for yourself.)

### e. Class imbalance / dataset size

172 AMPs + 355 decoys = **527 total samples**. This is small for deep learning but plenty for HDC, especially given HDC's strong sample efficiency. Don't underestimate the value of cleanly-engineered features over more samples here.

### f. Family redundancy

Of the 172 AMPs, ~57 are magainin variants (single point mutations of Mag(i+4)1,15(A9K)). They are highly correlated. Naive train/test splits will leak; use **family-level or sequence-clustered splits** (e.g. 80%-identity clustering with MMseqs2) to get honest generalization estimates.

---

## 7. Can HDC do regression? — quick answer

**Yes.** Several established approaches:

1. **Class-prototype distance regression.** Quantize the target (e.g. log-MIC) into bins, build one prototype hypervector per bin, predict via weighted similarity to all prototypes (kernel-regression style). Used in HD-classifier-as-regressor papers (e.g. RegHD).
2. **Bundle-with-tagged-magnitude.** Encode (sample, target) pairs with the target itself encoded as a thermometer or scalar hypervector and bound in. Decode at test time by binding the test sample with the inverse and projecting onto the target axis.
3. **Random-projection regression.** Use HDC encoding as a random feature map and feed into a linear ridge regression head — this is essentially what Random Vector Functional Link Networks do; works very well for small numeric datasets like yours.

Approach 3 is probably the lowest-risk first cut on this dataset given (a) the features are mostly already numeric and well-scaled, (b) your sample count is small, (c) regression noise floor is already ~10 %.

For classification (AMP vs decoy or hemolytic vs non-hemolytic at a threshold) HDC is straightforward and very well-trodden — that's the easy direction.

---

## 8. What is NOT settled / what you should decide first

In rough priority order:

1. **Label engineering.** Parse `Activity` and `Hemolytic_Activity` text into numeric `MIC_uM` (probably per-organism), `MHC_uM`, and `pct_hemolysis_25ugml` columns. Document the success rate (how many of 172 had any usable label). This determines whether you can do regression at all on the small subset that has good labels.
2. **Sequence vs feature-only encoding.** Pure 17-feature numeric input is the easiest baseline; HDC sequence encoding (binding position-tagged residue HVs) is the natural way to add the sequence signal. Decide whether to use both (concat) or sequence-only.
3. **Train/test split strategy.** Family-aware splits (cluster by sequence identity), not random — see §6f.
4. **Hypervector dimensionality.** Standard HDC uses 10k. Given 527 samples a smaller D (1–5k) may be sufficient and faster to iterate on.
5. **Encoding for staples.** Independent token vs paired-staple representation (see §4).
6. **Multi-output vs separate regressors** for hemo and MIC. They share most features; multi-task may help.

---

## 9. Quickstart commands (in WSL `stap` env)

```bash
cd /mnt/c/Users/bioin/Documents/SVM_ESM_Peptides/Peptide-Anti-microbial-Properties-Prediction/sequence_to_svm_minimal/data/training_dataset/StaPep

# Sanity-check the canonical features file:
python sanity_check_md_features.py

# Compare 50 ns vs 4 ns to confirm protocol delta:
python _inspect_md50ns_in_progress.py

# Verify completeness:
python _verify_training_md4ns.py   # works on whichever file is named in the script
```

`stap` env has pandas, numpy, sklearn, openmm, ambertools, pytraj. For HDC you may want to add `torchhd` or `hdlib` (or roll your own, which is ~50 lines of NumPy).

---

## 10. Hand-off checklist

If you build on this dataset, please log your model artefacts in this folder so the next agent doesn't redo work:

- [ ] Numeric labels CSV (e.g. `targets_hemo_mic.csv`) with per-row `MIC_*`, `MHC_uM`, parse-success flags
- [ ] Train/test split CSV (`splits_clustered.csv`) with `DRAMP_ID`, fold assignment
- [ ] HDC encoder script (`hdc_encoder.py`) that takes feature CSV + sequence column → hypervector batch
- [ ] HDC classifier / regressor model file (joblib or numpy)
- [ ] Holdout metrics report (`hdc_results.txt`)
- [ ] One-paragraph README describing your encoding scheme — for the next agent to extend

That's everything I know about this data. Good luck.
