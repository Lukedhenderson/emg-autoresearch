# autoresearch: EMG edition

This repo is now an autoresearch benchmark for EMG gesture classification instead of language-model pretraining. The shape stays close to Karpathy's original idea:

- one machine
- one process
- one primary mutation surface: `train.py`
- one fixed wall-clock budget per experiment: 5 minutes
- simple, self-contained Python

The search space is intentionally biased toward signal processing and feature engineering. The benchmark is designed to reward strong handcrafted EMG pipelines over unnecessary model complexity.

## Dataset

The repo expects two local assets that are already present here:

- `EMG-data.csv.zip`
- `dataset_description.txt`

From the dataset description and the analytics-ready CSV, the benchmark assumes:

- eight MYO EMG channels: `channel1` through `channel8`
- `class` labels:
  - `0`: unmarked transition data
  - `1`: hand at rest
  - `2`: fist
  - `3`: wrist flexion
  - `4`: wrist extension
  - `5`: radial deviation
  - `6`: ulnar deviation
  - `7`: extended palm, only present for a small subset of subjects
- `label` is the subject ID
- negative `time` jumps within a subject indicate the second recording session
- the effective sample rate is inferred from positive time deltas and is approximately `1000 Hz`; the fallback is `1000 Hz`

Class `0` is excluded by default because it is unmarked/non-gesture data. Class `7` is disabled by default because only two subjects appear to contain it, which makes grouped evaluation unstable.

## Main Files

- `train.py` — the main experiment engine and mutation surface
- `prepare.py` — lightweight dataset extraction and metadata generation
- `program.md` — autoresearch instructions for the agent
- `tests/test_emg_pipeline.py` — unit and integration smoke tests

## Quick Start

```bash
uv sync
uv run prepare.py
uv run train.py
python -m unittest
```

`prepare.py` extracts the CSV into `.cache_emg/` and writes explicit metadata assumptions. `train.py` can trigger preparation implicitly if needed.

## Experiment Design

Each run:

1. loads the EMG dataset
2. reconstructs subject/session structure
3. segments the signal into fixed windows without crossing subject, session, or gesture-bout boundaries
4. extracts handcrafted EMG features
5. trains a compact classifier
6. evaluates with leakage-aware grouped splits
7. appends a JSONL record to `results.jsonl`

Primary metric:

- `f1_macro`

Also logged:

- `accuracy`
- `balanced_accuracy`
- per-class precision/recall/F1/support
- runtime
- model family
- preprocessing choices
- feature families
- windowing settings
- feature count and window count

## Feature Families

The benchmark includes:

- basic time-domain EMG features: MAV, RMS, VAR, WL, ZC, SSC
- additional EMG features: WAMP, IEMG, AAC, DASDV, log detector, SSI, v-order, peak-to-peak
- distribution features: skewness, kurtosis
- Hjorth parameters: activity, mobility, complexity
- autoregressive coefficients
- optional sample entropy
- frequency-domain features: MDF, MNF, total power, peak frequency, coarse bandpowers, spectral entropy
- optional across-channel summary statistics

## Model Families

The default benchmark focuses on lightweight models:

- logistic regression
- linear SVM
- RBF SVM
- random forest
- extra trees
- histogram gradient boosting
- small MLP

The intent is not to turn this into a deep-learning-heavy raw-signal repo. If a neural baseline is added later, it should remain secondary and lightweight.

## Logging

Each run appends one JSON object to `results.jsonl`. This is the append-only record that autoresearch should read before planning the next experiment.

## Testing

The test suite covers:

- session reconstruction assumptions
- boundary-safe window generation
- deterministic feature extraction on small synthetic signals
- end-to-end smoke execution on a tiny synthetic EMG fixture
# emg-autoresearch
# emg-autoresearch
