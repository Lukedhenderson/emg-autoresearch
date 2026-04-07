# EMG Gesture Classification

An organized public snapshot of the final EMG gesture-classification project. This repository contains the training code, the final notebook, the trained model artifact, and the recorded evaluation results.

This project was run as an autoresearch experiment inspired directly by Andrej Karpathy's `autoresearch` repository. It follows the same core design philosophy: one machine, one main training surface, compact self-contained code, fixed-time experiments, and an append-only results log that makes the research loop inspectable.

In this adaptation, the language-model benchmark idea is carried over into EMG gesture classification. The search space is biased toward signal processing, feature engineering, and lightweight classifiers rather than large training infrastructure.

## Final Result

The final exported model metrics are stored in [`artifacts/final_metrics.json`](/Users/lukehenderson/Documents/Projects/emg/artifacts/final_metrics.json):

- `accuracy`: `0.9284`
- `f1_macro`: `0.9283`
- `balanced_accuracy`: `0.9282`

The trained model is available in [`artifacts/final_model.joblib`](/Users/lukehenderson/Documents/Projects/emg/artifacts/final_model.joblib), and the full notebook used for the final pass is in [`notebooks/final_pipeline.ipynb`](/Users/lukehenderson/Documents/Projects/emg/notebooks/final_pipeline.ipynb).

## Autoresearch Framing

Like Karpathy's original `autoresearch` setup, this repo keeps the system deliberately simple and legible:

- one primary mutation surface in [`train.py`](/Users/lukehenderson/Documents/Projects/emg/train.py)
- one local machine and one compact Python codebase
- fixed-budget experiments rather than open-ended training infrastructure
- explicit logged outcomes in [`artifacts/experiment_history.jsonl`](/Users/lukehenderson/Documents/Projects/emg/artifacts/experiment_history.jsonl)
- emphasis on interpretable changes over framework-heavy abstraction

The main difference is domain. Instead of language-model pretraining, this repository applies the same experimental shape to EMG gesture classification.

## Repository Layout

- [`train.py`](/Users/lukehenderson/Documents/Projects/emg/train.py) contains the main feature-engineering and training pipeline.
- [`prepare.py`](/Users/lukehenderson/Documents/Projects/emg/prepare.py) extracts the bundled dataset and writes lightweight metadata.
- [`tests/test_emg_pipeline.py`](/Users/lukehenderson/Documents/Projects/emg/tests/test_emg_pipeline.py) covers the core pipeline logic with smoke tests.
- [`data/`](/Users/lukehenderson/Documents/Projects/emg/data) contains the raw dataset zip and dataset description.
- [`artifacts/`](/Users/lukehenderson/Documents/Projects/emg/artifacts) contains public outputs: metrics, model, progress plot, and experiment history.
- [`notebooks/`](/Users/lukehenderson/Documents/Projects/emg/notebooks) contains the final analysis notebook.

## Dataset

The project uses the bundled MYO armband EMG dataset in [`data/EMG-data.csv.zip`](/Users/lukehenderson/Documents/Projects/emg/data/EMG-data.csv.zip) with the companion description in [`data/dataset_description.txt`](/Users/lukehenderson/Documents/Projects/emg/data/dataset_description.txt).

The pipeline assumes:

- eight EMG channels: `channel1` through `channel8`
- `label` is the subject identifier
- negative `time` jumps within a subject mark a new session
- class `0` is transition/unmarked data and is excluded by default
- class `7` is sparse and excluded by default from grouped evaluation

## What The Pipeline Does

The training pipeline is built around handcrafted EMG features rather than a large end-to-end neural model. A typical run:

1. prepares and loads the dataset
2. reconstructs subject/session boundaries
3. builds gesture-safe windows
4. extracts EMG time-domain and frequency-domain features
5. trains a compact classifier
6. evaluates with grouped splits to reduce leakage
7. appends a structured record to [`artifacts/experiment_history.jsonl`](/Users/lukehenderson/Documents/Projects/emg/artifacts/experiment_history.jsonl)

The best grouped cross-validation result is summarized in [`artifacts/best_grouped_result.json`](/Users/lukehenderson/Documents/Projects/emg/artifacts/best_grouped_result.json).

## Reproduce

```bash
uv sync
uv run prepare.py
uv run train.py
uv run python -m unittest discover -s tests -p 'test_*.py'
```

`prepare.py` extracts the CSV into `.cache_emg/`, which is intentionally ignored from version control.

## Artifacts

- [`artifacts/final_model.joblib`](/Users/lukehenderson/Documents/Projects/emg/artifacts/final_model.joblib): final exported classifier
- [`artifacts/final_metrics.json`](/Users/lukehenderson/Documents/Projects/emg/artifacts/final_metrics.json): final held-out metrics
- [`artifacts/best_grouped_result.json`](/Users/lukehenderson/Documents/Projects/emg/artifacts/best_grouped_result.json): best grouped CV run summary
- [`artifacts/experiment_history.jsonl`](/Users/lukehenderson/Documents/Projects/emg/artifacts/experiment_history.jsonl): experiment log
- [`artifacts/progress.png`](/Users/lukehenderson/Documents/Projects/emg/artifacts/progress.png): progress visualization
