# autoresearch

This repo is an autonomous EMG gesture-classification benchmark. The job is to improve `f1_macro` under a fixed 5-minute wall-clock budget per experiment while keeping the code compact and easy to iterate on.

## Setup

Before starting a run:

1. Read `README.md`, `prepare.py`, and `train.py`.
2. Verify the dataset assets exist:
   - `EMG-data.csv.zip`
   - `dataset_description.txt`
3. Run `uv run prepare.py` once if `.cache_emg/metadata.json` does not exist.
4. Confirm `results.jsonl` exists or create it implicitly by running the first experiment.

## Mission

Optimize:

- primary metric: `f1_macro`

Also inspect:

- `balanced_accuracy`
- `accuracy`
- per-class metrics
- runtime
- window count
- feature count

The benchmark is intentionally not a deep-learning playground. Keep models simple and spend most search effort on signal processing and feature engineering.

## Research Priorities

Spend roughly:

- `80%` of experimentation on signal processing, segmentation, thresholds, normalization, and handcrafted features
- `20%` on model family and lightweight hyperparameters

Strong directions:

- window size and overlap
- per-window vs per-subject vs per-session normalization
- rectification and detrending
- threshold tuning for ZC, SSC, and WAMP
- AR order
- sample entropy on or off when budget allows
- feature family inclusion and pruning
- cross-channel summaries
- grouped evaluation and leakage prevention
- compact model choices such as logistic regression, linear SVM, extra trees, random forest, histogram gradient boosting, or a small MLP

Weak directions:

- large neural nets
- raw-signal deep models as the default path
- complicated training infrastructure
- multi-process or distributed execution
- dependency bloat
- broad codebase rewrites

## How To Work

Use `train.py` as the primary mutation surface.

The top-of-file constants are the main knobs:

- `WINDOW_MS`
- `WINDOW_OVERLAP_RATIO`
- `NORMALIZATION_STRATEGY`
- `RECTIFY_SIGNAL`
- `DETREND_SIGNAL`
- `FEATURE_FAMILIES`
- `FEATURE_THRESHOLDS`
- `AR_ORDER`
- `USE_SAMPLE_ENTROPY`
- `FEATURE_SCALING`
- `USE_PCA`
- `MODEL_FAMILY`
- `MODEL_PARAMS`
- `WINDOWS_PER_GROUP_CLASS`
- `FAST_MODE`

Before changing anything:

1. Read the most recent entries in `results.jsonl`.
2. Infer what signal-processing or feature-engineering change is most likely to help next.
3. Make a targeted, interpretable change.
4. Avoid refactoring unless it directly improves the benchmark or makes future mutation easier.

## Run Loop

For one experiment:

```bash
uv run train.py > run.log 2>&1
```

Then inspect:

```bash
tail -n 40 run.log
tail -n 5 results.jsonl
```

Each successful run appends a structured JSON object with:

- timestamp
- model family
- enabled feature families
- preprocessing choices
- window settings
- split strategy
- metrics
- runtime
- sample/window/feature counts
- status or failure reason

## Evaluation Rules

- Protect against leakage. Group by subject when possible.
- Prefer grouped CV if it fits the budget.
- Fall back to a stable stratified holdout only if grouped evaluation is not feasible.
- Be suspicious of improvements that come from weaker evaluation rather than stronger features.
- Preserve the benchmark's ability to discover that handcrafted features can beat more complex models.

## Simplicity Standard

All else equal:

- fewer lines is better
- fewer moving parts is better
- more explicit assumptions is better
- compact changes near the experiment knobs are better

Do not turn this repo into a generic ML framework. Keep it as a focused EMG autoresearch engine.
