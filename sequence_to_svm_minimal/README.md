# GenomeClassifier: One-Click Sequence → Descriptors → SVM Predictions

This repository lets you take a list of protein sequences and get SVM predictions in a single command. You do not need to write any code.

Key script: `scripts/run_sequence_svm.py`

- Input: a simple text file of sequences (`seqs.txt`), one per line with an index.
- It generates descriptors and runs a pre-trained SVM model.
- Output: CSV files with predictions and probabilities.

---

## What you need (once)

- A working Conda installation (Anaconda or Miniconda).
- This folder (`GenomeClassifier/`) with:
  - `scripts/run_sequence_svm.py` (already included)
  - `descriptors/descripGen_12_py3.py` (already included)
  - `descriptors/aaindex/` directory with the AAIndex files (already included)
  - `predictionsParameters/svc.pkl` (the pre-trained model)
  - `predictionsParameters/Z_score_mean_std__intersect_noflip.csv` (feature normalization)

We provide a ready-to-use Conda environment file.

---

## Quick setup

Create the environment once:

```bash
# root project folder
conda env create -f skl_legacy_env.yml -n skl_legacy
conda activate skl_legacy
# Install the descriptor library used by the generator
pip install propy3
pip install numpy
pip install scipy
pip install scikit-learn==0.19.2
```

You only need to do this the first time.

---

## Prepare your input sequences

Create a text file (for example `predictionsParameters/seqs.txt`) with two columns separated by spaces:

```
1 MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANN
2 GVVDSDDLPLVVAASNAGKSTVVQLLAAAG...
3 MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVF...
```

- Column 1: sequence index (an integer).
- Column 2: the amino-acid sequence (letters A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y).
- One sequence per line.

Tip: You can copy your sequences into this format with any text editor.

---

## Run the pipeline (one command)

From inside `GenomeClassifier/` and with the `skl_legacy` environment active:

IMPORTANT PART HERE!!!!!
:: Activate the conda environment
conda activate skl_legacy

cd \path\to\sequence_to_svm_minimal

:: Define experiment folder (example: exp1)
set EXPERIMENT=experiments\exp1

:: Create folders (if not exist)
mkdir %EXPERIMENT%
mkdir %EXPERIMENT%\predictionsParameters

(make sure raw.txt is in experiment folder before window generation)
:: Step 1: Generate sequence windows
python scripts\make_seqs_windows.py --in "%EXPERIMENT%\raw.txt" --out "%EXPERIMENT%\predictionsParameters\seqs_1.txt" --min-len 10 --max-len 35 --stride 1

:: Step 2: Run SVM on generated sequences
python scripts\run_sequence_svm.py --seqs "%EXPERIMENT%\predictionsParameters\seqs_1.txt" --aaindex descriptors\aaindex --output-dir "%EXPERIMENT%\predictionsParameters" --model-pkl predictionsParameters\svc.pkl --scaler-csv predictionsParameters\Z_score_mean_std__intersect_noflip.csv



That’s it. The script will:
- Generate `descriptors.csv` under your chosen `--output-dir`.
- Run the SVM model on those descriptors.
- Produce prediction CSVs.

---

## What the outputs are

All paths below are relative to your `--output-dir`.

- `descriptors.csv`
  - Numeric features computed from your sequences (12 descriptors per sequence).
- `descriptors_PREDICTIONS_unsorted.csv`
  - One line per input sequence: index, predicted label, distance to decision boundary, and class probabilities.
  - Columns: `seqIndex,prediction,distToMargin,P(-1),P(+1)`
- `descriptors_PREDICTIONS.csv`
  - Same as above, sorted by strongest confidence.

Example first few lines of the predictions file:

```
seqIndex,prediction,distToMargin,P(-1),P(+1)
1,1,0.93,0.0431,0.9569
2,1,2.22,0.000001,0.999999
...
```

- `prediction`: class label (`1` or `-1`).
- `distToMargin`: positive means class `1`; negative means class `-1`.
- `P(+1)`: probability of class `+1` (between 0 and 1).

---

## Optional: process a subset (advanced)

If your `--seqs` file is very large, you can run a subset of rows using 1-based indices:

```bash
python scripts/run_sequence_svm.py ... --start 1001 --stop 2000
```

If you omit `--start` and `--stop`, the script processes the entire file.

---

## Troubleshooting

- "Conda command not found": Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
- "Module not found: propy": Ensure you ran `pip install propy3` inside the `skl_legacy` environment.
- Warnings about `scikit-learn` or `joblib` versions: These are expected for this legacy model and are safe to ignore.
- If you see errors related to AAIndex files, make sure your `--aaindex` path points to `descriptors/aaindex`.

If something fails, copy the full error message and send it along.

---

## Reference: what each parameter means

- `--seqs` (required): Path to your input sequences file (two columns: index and amino-acid sequence).
- `--aaindex` (required): Path to the AAIndex directory (`descriptors/aaindex`).
- `--output-dir` (required): Where descriptor and prediction CSVs will be written.
- `--model-pkl` (required): Pre-trained SVM model file (e.g., `predictionsParameters/svc.pkl`).
- `--scaler-csv` (required): CSV with descriptor names, means, and stds (e.g., `predictionsParameters/Z_score_mean_std__intersect_noflip.csv`).
- `--start` (optional): 1-based start row in `--seqs` to process (default: 1).
- `--stop` (optional): 1-based stop row in `--seqs` to process (default: end of file).

---

## Environment (technical details)

We ship an `environment.yml` that pins compatible package versions (Python 3.7, scikit-learn 0.19.2). The descriptor generator uses `propy3` for computing features. The SVM model is a legacy pickle and is handled correctly by the included prediction script.

If you are packaging this for someone else, send the entire `GenomeClassifier/` folder and this README. Ask them to install Conda, create the environment as shown above, and run the one command.
