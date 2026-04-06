"""
EMG gesture-classification benchmark for autoresearch.

This replaces the default language-model example while preserving the original
autoresearch shape:
  - one main editable training surface
  - one machine, one process
  - fixed wall-clock budget per experiment
  - simple, inspectable code

The search space is intentionally biased toward signal processing and feature
engineering. Models are kept compact so the system can discover that strong
handcrafted features often beat unnecessary model complexity.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from prepare import DEFAULT_SAMPLE_RATE_HZ, ensure_prepared, load_metadata


# ---------------------------------------------------------------------------
# Experiment knobs: this is the primary mutation surface for autoresearch.
# ---------------------------------------------------------------------------

TIME_BUDGET_SECONDS = 300
RANDOM_SEED = 42
RESULTS_LOG_PATH = Path(__file__).resolve().parent / "results.jsonl"

WINDOW_MS = 225
WINDOW_OVERLAP_RATIO = 0.4
INCLUDE_CLASS_ZERO = False
INCLUDE_CLASS_SEVEN = False
NORMALIZATION_STRATEGY = 'per_session'
RECTIFY_SIGNAL = False
DETREND_SIGNAL = False
REMOVE_DC_OFFSET = True
FEATURE_SCALING = True
USE_PCA = False
PCA_COMPONENTS = 32
AR_ORDER = 4
USE_SAMPLE_ENTROPY = False
SAMPLE_ENTROPY_M = 2
SAMPLE_ENTROPY_R_RATIO = 0.2
WINDOWS_PER_GROUP_CLASS = 96
FAST_MODE = False
MAX_WINDOWS_TOTAL = 30000
ENABLE_CROSS_CHANNEL_SUMMARY = True
ENABLE_CHANNEL_PAIR_FEATURES = False
CHANNEL_PAIR_LIMIT = 4

MODEL_FAMILY = 'extra_trees'  # logreg, linear_svm, rbf_svm, random_forest, extra_trees, hist_gb, small_mlp
MODEL_PARAMS = {
    "logreg": {"C": 1.5, "max_iter": 1000},
    "linear_svm": {"C": 1.0, "max_iter": 5000},
    "rbf_svm": {"C": 3.0, "gamma": "scale"},
    "random_forest": {"n_estimators": 250, "max_depth": None, "min_samples_leaf": 2, "n_jobs": -1},
    "extra_trees": {"n_estimators": 350, "max_depth": None, "min_samples_leaf": 1, "n_jobs": -1},
    "hist_gb": {"learning_rate": 0.08, "max_leaf_nodes": 31, "max_depth": None, "min_samples_leaf": 20},
    "small_mlp": {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "max_iter": 250},
}

SPLIT_STRATEGY = "group_kfold"  # group_kfold or stratified_holdout
CV_FOLDS = 2
HOLDOUT_RATIO = 0.2

FEATURE_FAMILIES = {
    "autoregressive": True,
    "distribution": True,
    "frequency": True,
    "hjorth": True,
    "sample_entropy": False,
    "time_basic": True,
    "time_emg": True
}

FEATURE_THRESHOLDS = {
    "ssc": 0.04,
    "wamp": 0.04,
    "zc": 0.04
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    time_budget_seconds: int = TIME_BUDGET_SECONDS
    random_seed: int = RANDOM_SEED
    window_ms: int = WINDOW_MS
    overlap_ratio: float = WINDOW_OVERLAP_RATIO
    include_class_zero: bool = INCLUDE_CLASS_ZERO
    include_class_seven: bool = INCLUDE_CLASS_SEVEN
    normalization_strategy: str = NORMALIZATION_STRATEGY
    rectify_signal: bool = RECTIFY_SIGNAL
    detrend_signal: bool = DETREND_SIGNAL
    remove_dc_offset: bool = REMOVE_DC_OFFSET
    feature_scaling: bool = FEATURE_SCALING
    use_pca: bool = USE_PCA
    pca_components: int = PCA_COMPONENTS
    ar_order: int = AR_ORDER
    use_sample_entropy: bool = USE_SAMPLE_ENTROPY
    sample_entropy_m: int = SAMPLE_ENTROPY_M
    sample_entropy_r_ratio: float = SAMPLE_ENTROPY_R_RATIO
    windows_per_group_class: int = WINDOWS_PER_GROUP_CLASS
    fast_mode: bool = FAST_MODE
    max_windows_total: int = MAX_WINDOWS_TOTAL
    model_family: str = MODEL_FAMILY
    split_strategy: str = SPLIT_STRATEGY
    cv_folds: int = CV_FOLDS
    holdout_ratio: float = HOLDOUT_RATIO
    enable_cross_channel_summary: bool = ENABLE_CROSS_CHANNEL_SUMMARY
    enable_channel_pair_features: bool = ENABLE_CHANNEL_PAIR_FEATURES
    channel_pair_limit: int = CHANNEL_PAIR_LIMIT
    feature_families: dict[str, bool] = field(default_factory=lambda: dict(FEATURE_FAMILIES))
    thresholds: dict[str, float] = field(default_factory=lambda: dict(FEATURE_THRESHOLDS))
    model_params: dict[str, dict[str, object]] = field(default_factory=lambda: {k: dict(v) for k, v in MODEL_PARAMS.items()})


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    metadata: dict
    sample_rate_hz: float
    channels: list[str]


@dataclass(frozen=True)
class WindowRecord:
    group: str
    subject: int
    session: int
    gesture: int
    bout_index: int
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class EvaluationArtifacts:
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]]
    feature_count: int
    window_count: int
    subject_count: int
    session_count: int
    split_strategy: str
    fold_summaries: list[dict[str, float]]
    evaluation_fallback_reason: str | None = None


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    return np.random.default_rng(seed)


def fit_standardizer(frame: pd.DataFrame, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    values = frame[channels].to_numpy(dtype=np.float64, copy=False)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def fit_group_standardizers(frame: pd.DataFrame, channels: list[str], key: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for group_value, group_frame in frame.groupby(key, sort=False):
        stats[int(group_value)] = fit_standardizer(group_frame, channels)
    return stats


def fit_subject_session_standardizers(
    frame: pd.DataFrame,
    channels: list[str],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    stats: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for (subject, session), group_frame in frame.groupby(["subject_id", "session_id"], sort=False):
        stats[(int(subject), int(session))] = fit_standardizer(group_frame, channels)
    return stats


def detrend_signal_array(signal: np.ndarray) -> np.ndarray:
    # A lightweight least-squares detrend keeps the dependency surface small.
    n_samples = signal.shape[0]
    if n_samples < 2:
        return signal.copy()
    x = np.arange(n_samples, dtype=np.float64)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom <= 0:
        return signal.copy()
    centered_x = x - x_mean
    centered_signal = signal - signal.mean(axis=0, keepdims=True)
    slope = np.sum(centered_x[:, None] * centered_signal, axis=0) / denom
    intercept = signal.mean(axis=0) - slope * x_mean
    trend = slope[None, :] * x[:, None] + intercept[None, :]
    return signal - trend


def apply_preprocessing(
    window: np.ndarray,
    config: ExperimentConfig,
    normalization_stats: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    processed = window.astype(np.float64, copy=True)
    if config.remove_dc_offset:
        processed = processed - processed.mean(axis=0, keepdims=True)
    if config.detrend_signal:
        processed = detrend_signal_array(processed)
    if config.rectify_signal:
        processed = np.abs(processed)

    if config.normalization_strategy in {"per_subject", "per_session"} and normalization_stats is not None:
        mean, std = normalization_stats
        processed = (processed - mean) / std
    elif config.normalization_strategy == "per_window":
        mean = processed.mean(axis=0, keepdims=True)
        std = processed.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        processed = (processed - mean) / std
    return processed.astype(np.float32)


def resolve_dynamic_threshold(values: np.ndarray, configured_threshold: float) -> float:
    if configured_threshold >= 1.0:
        return configured_threshold
    scale = float(np.std(values))
    if scale < 1e-12:
        scale = 1.0
    return configured_threshold * scale


def zero_crossings(signal_1d: np.ndarray, threshold: float) -> float:
    if signal_1d.size < 2:
        return 0.0
    lhs = signal_1d[:-1]
    rhs = signal_1d[1:]
    signs_changed = (lhs * rhs) < 0
    large_enough = np.abs(lhs - rhs) >= threshold
    return float(np.sum(signs_changed & large_enough))


def slope_sign_changes(signal_1d: np.ndarray, threshold: float) -> float:
    if signal_1d.size < 3:
        return 0.0
    diff1 = signal_1d[1:-1] - signal_1d[:-2]
    diff2 = signal_1d[1:-1] - signal_1d[2:]
    changed = (diff1 * diff2) > 0
    large_enough = (np.abs(diff1) >= threshold) | (np.abs(diff2) >= threshold)
    return float(np.sum(changed & large_enough))


def sample_entropy(signal_1d: np.ndarray, m: int, r_ratio: float) -> float:
    n = signal_1d.size
    if n <= m + 1:
        return 0.0
    std = float(np.std(signal_1d))
    if std < 1e-12:
        return 0.0
    r = r_ratio * std

    def _phi(order: int) -> float:
        templates = np.lib.stride_tricks.sliding_window_view(signal_1d, order)
        count = 0
        total = 0
        for index in range(len(templates) - 1):
            distance = np.max(np.abs(templates[index + 1 :] - templates[index]), axis=1)
            count += int(np.sum(distance <= r))
            total += len(distance)
        if total == 0:
            return 0.0
        return count / total

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0 or phi_m1 <= 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def autoregressive_coefficients(signal_1d: np.ndarray, order: int) -> np.ndarray:
    if order <= 0:
        return np.empty(0, dtype=np.float32)
    if signal_1d.size <= order:
        return np.zeros(order, dtype=np.float32)
    signal_centered = signal_1d - np.mean(signal_1d)
    y = signal_centered[order:]
    x = np.column_stack([signal_centered[order - lag - 1 : -(lag + 1)] for lag in range(order)])
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coeffs.astype(np.float32)


def hjorth_parameters(signal_1d: np.ndarray) -> tuple[float, float, float]:
    first_derivative = np.diff(signal_1d, prepend=signal_1d[0])
    second_derivative = np.diff(first_derivative, prepend=first_derivative[0])
    activity = float(np.var(signal_1d))
    first_var = float(np.var(first_derivative))
    second_var = float(np.var(second_derivative))
    mobility = math.sqrt(first_var / activity) if activity > 1e-12 else 0.0
    complexity = math.sqrt(second_var / first_var) / mobility if first_var > 1e-12 and mobility > 1e-12 else 0.0
    return activity, mobility, complexity


def bandpower_features(power: np.ndarray, freqs: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    nyquist = sample_rate_hz / 2.0
    max_band = max(10.0, nyquist)
    candidate_bands = [
        (0.0, min(40.0, max_band)),
        (40.0, min(80.0, max_band)),
        (80.0, min(150.0, max_band)),
        (150.0, min(250.0, max_band)),
        (250.0, min(400.0, max_band)),
    ]
    features: dict[str, float] = {}
    for low, high in candidate_bands:
        if high <= low:
            continue
        mask = (freqs >= low) & (freqs < high)
        key = f"bandpower_{int(low)}_{int(high)}"
        features[key] = float(power[mask].sum()) if np.any(mask) else 0.0
    return features


def summarize_distribution(signal_1d: np.ndarray) -> tuple[float, float]:
    centered = signal_1d - np.mean(signal_1d)
    std = float(np.std(centered))
    if std < 1e-12:
        return 0.0, 0.0
    standardized = centered / std
    skewness = float(np.mean(standardized**3))
    kurtosis = float(np.mean(standardized**4))
    return skewness, kurtosis


def v_order(signal_1d: np.ndarray, order: float = 2.0) -> float:
    return float(np.power(np.mean(np.abs(signal_1d) ** order), 1.0 / order))


def extract_channel_features(
    signal_1d: np.ndarray,
    sample_rate_hz: float,
    config: ExperimentConfig,
) -> dict[str, float]:
    values = signal_1d.astype(np.float64, copy=False)
    diff = np.diff(values)
    abs_values = np.abs(values)
    abs_diff = np.abs(diff)
    variance = float(np.var(values))
    rms = float(np.sqrt(np.mean(values**2)))
    wl = float(np.sum(abs_diff))
    zc_threshold = resolve_dynamic_threshold(values, config.thresholds["zc"])
    ssc_threshold = resolve_dynamic_threshold(values, config.thresholds["ssc"])
    wamp_threshold = resolve_dynamic_threshold(values, config.thresholds["wamp"])
    skewness, kurtosis = summarize_distribution(values)

    features: dict[str, float] = {}
    if config.feature_families["time_basic"]:
        features.update(
            {
                "mav": float(np.mean(abs_values)),
                "rms": rms,
                "var": variance,
                "wl": wl,
                "zc": zero_crossings(values, zc_threshold),
                "ssc": slope_sign_changes(values, ssc_threshold),
            }
        )
    if config.feature_families["time_emg"]:
        log_detector = float(np.exp(np.mean(np.log(abs_values + 1e-8))))
        features.update(
            {
                "wamp": float(np.sum(abs_diff >= wamp_threshold)),
                "iemg": float(np.sum(abs_values)),
                "aac": float(np.mean(abs_diff)) if diff.size else 0.0,
                "dasdv": float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
                "log_detector": log_detector,
                "peak_to_peak": float(np.ptp(values)),
                "ssi": float(np.sum(values**2)),
                "v_order": v_order(values),
            }
        )
    if config.feature_families["distribution"]:
        features.update({"skewness": skewness, "kurtosis": kurtosis})
    if config.feature_families["hjorth"]:
        activity, mobility, complexity = hjorth_parameters(values)
        features.update({"hjorth_activity": activity, "hjorth_mobility": mobility, "hjorth_complexity": complexity})
    if config.feature_families["autoregressive"]:
        coeffs = autoregressive_coefficients(values, config.ar_order)
        for idx, coefficient in enumerate(coeffs, start=1):
            features[f"ar_{idx}"] = float(coefficient)
    if config.feature_families["sample_entropy"]:
        features["sample_entropy"] = sample_entropy(values, config.sample_entropy_m, config.sample_entropy_r_ratio)
    if config.feature_families["frequency"]:
        centered = values - np.mean(values)
        spectrum = np.fft.rfft(centered)
        power = np.abs(spectrum) ** 2
        freqs = np.fft.rfftfreq(values.size, d=1.0 / sample_rate_hz)
        total_power = float(power.sum())
        if total_power <= 1e-12:
            mnf = 0.0
            mdf = 0.0
            peak_frequency = 0.0
            spectral_entropy = 0.0
        else:
            mnf = float(np.sum(freqs * power) / total_power)
            cumulative = np.cumsum(power)
            median_index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
            mdf = float(freqs[min(median_index, freqs.size - 1)])
            peak_frequency = float(freqs[int(np.argmax(power))])
            normalized_power = power / total_power
            spectral_entropy = float(-(normalized_power * np.log2(normalized_power + 1e-12)).sum())
        features.update(
            {
                "mnf": mnf,
                "mdf": mdf,
                "total_power": total_power,
                "peak_frequency": peak_frequency,
                "spectral_entropy": spectral_entropy,
            }
        )
        features.update(bandpower_features(power, freqs, sample_rate_hz))
    return features


def flatten_feature_dict(features: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in features.items()}


def summarize_across_channels(feature_row: dict[str, float], feature_names: Iterable[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for feature_name in feature_names:
        channel_values = [value for key, value in feature_row.items() if key.endswith(f"_{feature_name}")]
        if not channel_values:
            continue
        values = np.asarray(channel_values, dtype=np.float64)
        summary[f"summary_mean_{feature_name}"] = float(values.mean())
        summary[f"summary_std_{feature_name}"] = float(values.std())
        summary[f"summary_max_{feature_name}"] = float(values.max())
        summary[f"summary_min_{feature_name}"] = float(values.min())
    return summary


def pairwise_channel_features(window: np.ndarray, channels: list[str], pair_limit: int) -> dict[str, float]:
    features: dict[str, float] = {}
    if pair_limit <= 0:
        return features
    pair_count = 0
    for left in range(len(channels)):
        for right in range(left + 1, len(channels)):
            if pair_count >= pair_limit:
                return features
            corr = np.corrcoef(window[:, left], window[:, right])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            features[f"pair_corr_{channels[left]}_{channels[right]}"] = float(corr)
            pair_count += 1
    return features


def extract_window_features(
    window: np.ndarray,
    channels: list[str],
    sample_rate_hz: float,
    config: ExperimentConfig,
) -> dict[str, float]:
    row: dict[str, float] = {}
    feature_names: set[str] = set()
    for channel_index, channel_name in enumerate(channels):
        channel_features = extract_channel_features(window[:, channel_index], sample_rate_hz, config)
        row.update(flatten_feature_dict(channel_features, channel_name))
        feature_names.update(channel_features.keys())
    if config.enable_cross_channel_summary:
        row.update(summarize_across_channels(row, sorted(feature_names)))
    if config.enable_channel_pair_features:
        row.update(pairwise_channel_features(window, channels, config.channel_pair_limit))
    return row


def infer_sample_rate_hz(frame: pd.DataFrame, fallback_hz: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
    time_diff = frame.groupby("label", sort=False)["time"].diff()
    positive = time_diff[(time_diff > 0) & (time_diff < 20)]
    if positive.empty:
        return fallback_hz
    median_ms = float(np.median(positive.to_numpy()))
    if median_ms <= 0:
        return fallback_hz
    return 1000.0 / median_ms


def load_dataset() -> DatasetBundle:
    prepared = ensure_prepared()
    metadata = load_metadata(prepared.metadata_path)
    channels = [f"channel{i}" for i in range(1, 9)]
    dtype_map = {channel: np.float32 for channel in channels}
    dtype_map.update({"time": np.int32, "class": np.int8, "label": np.int16})
    frame = pd.read_csv(prepared.extracted_csv_path, usecols=["time", *channels, "class", "label"], dtype=dtype_map)
    # Preserve the original row order from the analytics-ready CSV. Session
    # reconstruction depends on negative time resets, which would be destroyed by
    # sorting on the time column.
    frame = frame.reset_index(drop=True)
    frame["subject_id"] = frame["label"].astype(np.int16)

    # Session reconstruction assumption:
    # the original description says each subject performed two recording series.
    # The analytics-ready CSV does not expose session IDs directly, so we infer
    # them from negative time jumps within each subject.
    time_diff = frame.groupby("subject_id", sort=False)["time"].diff()
    frame["session_id"] = (time_diff < 0).groupby(frame["subject_id"]).cumsum().fillna(0).astype(np.int16)
    sample_rate_hz = infer_sample_rate_hz(frame)

    return DatasetBundle(frame=frame, metadata=metadata, sample_rate_hz=sample_rate_hz, channels=channels)


def filter_frame(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    filtered = frame.copy()
    if not config.include_class_zero:
        filtered = filtered[filtered["class"] != 0]
    if not config.include_class_seven:
        filtered = filtered[filtered["class"] != 7]
    filtered = filtered.reset_index(drop=True)
    return filtered


def build_bouts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("No rows remain after class filtering.")
    boundary = (
        frame["subject_id"].ne(frame["subject_id"].shift())
        | frame["session_id"].ne(frame["session_id"].shift())
        | frame["class"].ne(frame["class"].shift())
    )
    frame = frame.copy()
    frame["bout_index"] = boundary.cumsum().astype(np.int32)
    bout_summary = (
        frame.groupby("bout_index", sort=False)
        .agg(
            subject_id=("subject_id", "first"),
            session_id=("session_id", "first"),
            gesture=("class", "first"),
            start_row=("bout_index", "size"),
        )
        .reset_index()
    )
    return frame


def enumerate_windows(frame: pd.DataFrame, sample_rate_hz: float, config: ExperimentConfig, rng: np.random.Generator) -> list[WindowRecord]:
    window_size = max(8, int(round(sample_rate_hz * config.window_ms / 1000.0)))
    step_size = max(1, int(round(window_size * (1.0 - config.overlap_ratio))))
    windows: list[WindowRecord] = []

    for (subject, session, gesture), group_frame in frame.groupby(["subject_id", "session_id", "class"], sort=False):
        boundaries = np.flatnonzero(group_frame["bout_index"].to_numpy()[1:] != group_frame["bout_index"].to_numpy()[:-1]) + 1
        start_positions = [0, *boundaries.tolist(), len(group_frame)]
        group_windows: list[WindowRecord] = []
        for start, stop in zip(start_positions[:-1], start_positions[1:]):
            bout = group_frame.iloc[start:stop]
            bout_index = int(bout["bout_index"].iloc[0])
            if len(bout) < window_size:
                continue
            for offset in range(0, len(bout) - window_size + 1, step_size):
                global_start = int(bout.index[offset])
                global_end = int(bout.index[offset + window_size - 1]) + 1
                group_windows.append(
                    WindowRecord(
                        group=f"subject_{int(subject)}",
                        subject=int(subject),
                        session=int(session),
                        gesture=int(gesture),
                        bout_index=bout_index,
                        start_idx=global_start,
                        end_idx=global_end,
                    )
                )
        if config.windows_per_group_class > 0 and len(group_windows) > config.windows_per_group_class:
            selected = rng.choice(len(group_windows), size=config.windows_per_group_class, replace=False)
            selected = np.sort(selected)
            group_windows = [group_windows[int(index)] for index in selected]
        windows.extend(group_windows)

    if not windows:
        raise ValueError("No windows were generated; adjust the class filter or window size.")
    if config.max_windows_total > 0 and len(windows) > config.max_windows_total:
        selected = rng.choice(len(windows), size=config.max_windows_total, replace=False)
        selected = np.sort(selected)
        windows = [windows[int(index)] for index in selected]
    return windows


def normalization_lookup(
    frame: pd.DataFrame,
    channels: list[str],
    config: ExperimentConfig,
) -> dict[tuple[int, int] | int, tuple[np.ndarray, np.ndarray]] | None:
    if config.normalization_strategy == "per_subject":
        return fit_group_standardizers(frame, channels, "subject_id")
    if config.normalization_strategy == "per_session":
        return fit_subject_session_standardizers(frame, channels)
    return None


def build_feature_table(
    frame: pd.DataFrame,
    windows: list[WindowRecord],
    bundle: DatasetBundle,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    del rng  # reserved for future stochastic feature pruning.
    normalization_stats = normalization_lookup(frame, bundle.channels, config)
    signal_values = frame.loc[:, bundle.channels].to_numpy(dtype=np.float32, copy=False)
    rows: list[dict[str, float | int | str]] = []
    append_row = rows.append

    for record in windows:
        raw_window = signal_values[record.start_idx : record.end_idx]
        stats_key: tuple[int, int] | int | None = None
        if config.normalization_strategy == "per_subject":
            stats_key = record.subject
        elif config.normalization_strategy == "per_session":
            stats_key = (record.subject, record.session)
        processed_window = apply_preprocessing(
            raw_window,
            config,
            normalization_stats.get(stats_key) if normalization_stats is not None and stats_key in normalization_stats else None,
        )
        feature_row = extract_window_features(processed_window, bundle.channels, bundle.sample_rate_hz, config)
        feature_row.update(
            {
                "label": record.gesture,
                "group": record.group,
                "subject_id": record.subject,
                "session_id": record.session,
            }
        )
        append_row(feature_row)

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        raise ValueError("Feature extraction produced no rows.")
    labels = feature_frame.pop("label").to_numpy(dtype=np.int32)
    groups = feature_frame.pop("group").to_numpy(dtype=object)
    feature_frame = feature_frame.drop(columns=["subject_id", "session_id"], errors="ignore")
    return feature_frame, labels, groups


def build_model(config: ExperimentConfig):
    model_name = config.model_family
    params = config.model_params[model_name]
    if model_name == "logreg":
        return LogisticRegression(random_state=config.random_seed, solver="lbfgs", class_weight="balanced", **params)
    if model_name == "linear_svm":
        return LinearSVC(random_state=config.random_seed, class_weight="balanced", **params)
    if model_name == "rbf_svm":
        return SVC(random_state=config.random_seed, class_weight="balanced", **params)
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=config.random_seed, class_weight="balanced_subsample", **params)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(random_state=config.random_seed, class_weight="balanced", **params)
    if model_name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=config.random_seed, **params)
    if model_name == "small_mlp":
        return MLPClassifier(random_state=config.random_seed, early_stopping=True, **params)
    raise ValueError(f"Unsupported model family: {model_name}")


def build_pipeline(config: ExperimentConfig) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if config.feature_scaling and config.model_family in {"logreg", "linear_svm", "rbf_svm", "small_mlp"}:
        steps.append(("scaler", StandardScaler()))
    if config.use_pca:
        steps.append(("pca", PCA(n_components=config.pca_components, random_state=config.random_seed)))
    steps.append(("model", build_model(config)))
    return Pipeline(steps)


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    metrics: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(labels):
        metrics[str(int(label))] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": float(support[idx]),
        }
    return metrics


def stable_mean(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    keys = metric_rows[0].keys()
    return {key: float(np.mean([row[key] for row in metric_rows])) for key in keys}


def remaining_budget_seconds(start_time: float, time_budget_seconds: int) -> float:
    if start_time <= 0:
        return float("inf")
    return time_budget_seconds - (time.time() - start_time)


def should_fallback_from_group_kfold(start_time: float, config: ExperimentConfig) -> bool:
    remaining_budget = remaining_budget_seconds(start_time, config.time_budget_seconds)
    if remaining_budget <= 0:
        return False
    estimated_group_fold_budget = config.time_budget_seconds / max(config.cv_folds, 1)
    return remaining_budget <= estimated_group_fold_budget


def build_split_iterator(
    feature_values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ExperimentConfig,
    split_strategy_name: str,
) -> tuple[Iterable[tuple[np.ndarray, np.ndarray]], str]:
    if split_strategy_name == "group_kfold" and len(np.unique(groups)) >= config.cv_folds:
        splitter = GroupKFold(n_splits=config.cv_folds)
        return splitter.split(feature_values, labels, groups), f"group_kfold_{config.cv_folds}"
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=config.holdout_ratio, random_state=config.random_seed)
    return splitter.split(feature_values, labels), f"stratified_holdout_{config.holdout_ratio:.2f}"


def run_split_evaluation(
    feature_values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    pipeline: Pipeline,
    config: ExperimentConfig,
    start_time: float,
    split_strategy_name: str,
) -> tuple[list[dict[str, float]], list[dict[str, dict[str, float]]], list[dict[str, float]], str]:
    metric_rows: list[dict[str, float]] = []
    per_class_rows: list[dict[str, dict[str, float]]] = []
    fold_summaries: list[dict[str, float]] = []
    split_iterator, resolved_split_strategy = build_split_iterator(feature_values, labels, groups, config, split_strategy_name)

    for fold_index, (train_idx, test_idx) in enumerate(split_iterator, start=1):
        if remaining_budget_seconds(start_time, config.time_budget_seconds) <= 0:
            break
        model = clone(pipeline)
        model.fit(feature_values[train_idx], labels[train_idx])
        predictions = model.predict(feature_values[test_idx])
        fold_metrics = {
            "f1_macro": float(f1_score(labels[test_idx], predictions, average="macro")),
            "accuracy": float(accuracy_score(labels[test_idx], predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], predictions)),
        }
        metric_rows.append(fold_metrics)
        per_class_rows.append(per_class_metrics(labels[test_idx], predictions))
        fold_summaries.append({"fold": float(fold_index), **fold_metrics})

    return metric_rows, per_class_rows, fold_summaries, resolved_split_strategy


def evaluate_features(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ExperimentConfig,
    start_time: float,
) -> EvaluationArtifacts:
    pipeline = build_pipeline(config)
    feature_values = features.to_numpy(dtype=np.float32)
    fallback_reason: str | None = None
    requested_split_strategy = config.split_strategy
    effective_split_strategy = requested_split_strategy
    if requested_split_strategy == "group_kfold" and should_fallback_from_group_kfold(start_time, config):
        effective_split_strategy = "stratified_holdout"
        fallback_reason = "budget"

    metric_rows, per_class_rows, fold_summaries, split_strategy = run_split_evaluation(
        feature_values,
        labels,
        groups,
        pipeline,
        config,
        start_time,
        effective_split_strategy,
    )

    if not metric_rows and requested_split_strategy == "group_kfold" and effective_split_strategy == "group_kfold":
        metric_rows, per_class_rows, fold_summaries, split_strategy = run_split_evaluation(
            feature_values,
            labels,
            groups,
            pipeline,
            config,
            start_time,
            "stratified_holdout",
        )
        if metric_rows:
            fallback_reason = "budget"

    if not metric_rows:
        raise TimeoutError(
            "Evaluation exceeded the time budget before any grouped or holdout split could complete."
        )

    averaged_per_class: dict[str, dict[str, float]] = {}
    for class_name in sorted({class_name for row in per_class_rows for class_name in row.keys()}, key=int):
        averaged_per_class[class_name] = {
            key: float(np.mean([row.get(class_name, {}).get(key, 0.0) for row in per_class_rows]))
            for key in ["precision", "recall", "f1", "support"]
        }

    return EvaluationArtifacts(
        metrics=stable_mean(metric_rows),
        per_class=averaged_per_class,
        feature_count=int(features.shape[1]),
        window_count=int(features.shape[0]),
        subject_count=int(len(np.unique(groups))),
        session_count=int(len(np.unique(groups))),  # subject grouping is the leakage-aware default
        split_strategy=split_strategy,
        fold_summaries=fold_summaries,
        evaluation_fallback_reason=fallback_reason,
    )


def append_results_log(payload: dict[str, object]) -> None:
    RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def build_run_payload(
    config: ExperimentConfig,
    bundle: DatasetBundle,
    artifacts: EvaluationArtifacts | None,
    runtime_seconds: float,
    status: str,
    failure_reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "failure_reason": failure_reason,
        "runtime_seconds": round(runtime_seconds, 3),
        "primary_metric": "f1_macro",
        "sample_rate_hz": round(bundle.sample_rate_hz, 3),
        "subject_count": int(bundle.frame["subject_id"].nunique()),
        "session_count": int(bundle.frame[["subject_id", "session_id"]].drop_duplicates().shape[0]),
        "model_family": config.model_family,
        "model_params": config.model_params[config.model_family],
        "feature_families": config.feature_families,
        "preprocessing": {
            "window_ms": config.window_ms,
            "overlap_ratio": config.overlap_ratio,
            "normalization_strategy": config.normalization_strategy,
            "rectify_signal": config.rectify_signal,
            "detrend_signal": config.detrend_signal,
            "remove_dc_offset": config.remove_dc_offset,
            "feature_scaling": config.feature_scaling,
            "use_pca": config.use_pca,
            "pca_components": config.pca_components,
            "ar_order": config.ar_order,
            "thresholds": config.thresholds,
            "include_class_zero": config.include_class_zero,
            "include_class_seven": config.include_class_seven,
            "fast_mode": config.fast_mode,
        },
        "split_strategy_requested": config.split_strategy,
        "seed": config.random_seed,
    }
    if artifacts is not None:
        payload.update(
            {
                "split_strategy": artifacts.split_strategy,
                "evaluation_fallback_reason": artifacts.evaluation_fallback_reason,
                "metrics": artifacts.metrics,
                "per_class_metrics": artifacts.per_class,
                "fold_summaries": artifacts.fold_summaries,
                "window_count": artifacts.window_count,
                "feature_count": artifacts.feature_count,
                "evaluated_subject_groups": artifacts.subject_count,
            }
        )
    return payload


def maybe_adjust_for_budget(config: ExperimentConfig) -> ExperimentConfig:
    adjusted = asdict(config)
    if config.fast_mode:
        adjusted["cv_folds"] = min(config.cv_folds, 2)
        adjusted["windows_per_group_class"] = min(config.windows_per_group_class, 48)
        adjusted["max_windows_total"] = min(config.max_windows_total, 12000)
        feature_families = dict(config.feature_families)
        feature_families["sample_entropy"] = False
        adjusted["feature_families"] = feature_families
        if config.model_family == "rbf_svm":
            adjusted["model_family"] = "linear_svm"
    return ExperimentConfig(**adjusted)


def run_experiment(config: ExperimentConfig, experiment_start_time: float) -> tuple[EvaluationArtifacts, DatasetBundle]:
    rng = seed_everything(config.random_seed)
    bundle = load_dataset()
    filtered_frame = filter_frame(bundle.frame, config)
    filtered_frame = build_bouts(filtered_frame)
    windows = enumerate_windows(filtered_frame, bundle.sample_rate_hz, config, rng)
    features, labels, groups = build_feature_table(filtered_frame, windows, bundle, config, rng)
    artifacts = evaluate_features(features, labels, groups, config, experiment_start_time)
    return artifacts, bundle


def main() -> None:
    config = maybe_adjust_for_budget(ExperimentConfig())
    start_time = time.time()
    bundle: DatasetBundle | None = None
    artifacts: EvaluationArtifacts | None = None
    status = "success"
    failure_reason: str | None = None

    try:
        artifacts, bundle = run_experiment(config, start_time)
        runtime_seconds = time.time() - start_time
        payload = build_run_payload(config, bundle, artifacts, runtime_seconds, status)
        append_results_log(payload)
        print("---")
        print(f"primary_metric:    f1_macro")
        print(f"f1_macro:          {artifacts.metrics['f1_macro']:.6f}")
        print(f"accuracy:          {artifacts.metrics['accuracy']:.6f}")
        print(f"balanced_accuracy: {artifacts.metrics['balanced_accuracy']:.6f}")
        print(f"runtime_seconds:   {runtime_seconds:.2f}")
        print(f"window_count:      {artifacts.window_count}")
        print(f"feature_count:     {artifacts.feature_count}")
        print(f"model_family:      {config.model_family}")
        print(f"split_strategy:    {artifacts.split_strategy}")
        print(f"results_log:       {RESULTS_LOG_PATH}")
    except Exception as exc:  # noqa: BLE001 - explicit logging for autoresearch.
        runtime_seconds = time.time() - start_time
        status = "failure"
        failure_reason = f"{type(exc).__name__}: {exc}"
        if bundle is None:
            bundle = load_dataset()
        payload = build_run_payload(config, bundle, artifacts, runtime_seconds, status, failure_reason=failure_reason)
        append_results_log(payload)
        print("---")
        print(f"status:            failure")
        print(f"failure_reason:    {failure_reason}")
        print(f"runtime_seconds:   {runtime_seconds:.2f}")
        print(f"results_log:       {RESULTS_LOG_PATH}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
