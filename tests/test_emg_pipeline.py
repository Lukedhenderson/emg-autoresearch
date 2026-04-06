from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from train import (
    DatasetBundle,
    ExperimentConfig,
    EvaluationArtifacts,
    build_bouts,
    build_feature_table,
    enumerate_windows,
    evaluate_features,
    extract_channel_features,
    infer_sample_rate_hz,
)


CHANNELS = [f"channel{i}" for i in range(1, 9)]


def make_synthetic_frame() -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for subject in (1, 2):
        for session in (0, 1):
            time_offset = 0
            for gesture, amplitude in ((1, 0.2 + subject * 0.05), (2, 1.0 + subject * 0.1)):
                for sample_index in range(40):
                    rows.append(
                        {
                            "time": time_offset + sample_index,
                            "class": gesture,
                            "label": subject,
                            "subject_id": subject,
                            "session_id": session,
                            **{
                                channel: amplitude * np.sin((sample_index + 1) / (idx + 2))
                                for idx, channel in enumerate(CHANNELS)
                            },
                        }
                    )
                time_offset += 40
    return pd.DataFrame(rows)


class EmgPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = make_synthetic_frame()
        self.bundle = DatasetBundle(
            frame=self.frame.copy(),
            metadata={},
            sample_rate_hz=1000.0,
            channels=CHANNELS,
        )

    def test_infer_sample_rate_from_time_deltas(self) -> None:
        frame = pd.DataFrame(
            {
                "time": [0, 1, 2, 3, 0, 1, 2],
                "label": [1, 1, 1, 1, 1, 1, 1],
            }
        )
        self.assertAlmostEqual(infer_sample_rate_hz(frame), 1000.0)

    def test_window_generation_respects_bout_boundaries(self) -> None:
        config = ExperimentConfig(window_ms=10, overlap_ratio=0.5, windows_per_group_class=8, max_windows_total=100)
        bout_frame = build_bouts(self.frame.copy())
        rng = np.random.default_rng(42)
        windows = enumerate_windows(bout_frame, self.bundle.sample_rate_hz, config, rng)
        self.assertGreater(len(windows), 0)
        for record in windows:
            bout_values = bout_frame.loc[record.start_idx : record.end_idx - 1, "bout_index"].unique()
            class_values = bout_frame.loc[record.start_idx : record.end_idx - 1, "class"].unique()
            self.assertEqual(len(bout_values), 1)
            self.assertEqual(len(class_values), 1)

    def test_feature_extraction_contains_required_emg_features(self) -> None:
        signal = np.sin(np.linspace(0, 2 * np.pi, 64, dtype=np.float32))
        config = ExperimentConfig(
            use_sample_entropy=False,
            feature_families={
                "time_basic": True,
                "time_emg": True,
                "hjorth": True,
                "autoregressive": True,
                "distribution": True,
                "frequency": True,
                "sample_entropy": False,
            },
        )
        features = extract_channel_features(signal, 1000.0, config)
        required = {
            "mav",
            "rms",
            "var",
            "wl",
            "zc",
            "ssc",
            "wamp",
            "iemg",
            "aac",
            "dasdv",
            "log_detector",
            "skewness",
            "kurtosis",
            "peak_to_peak",
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
            "ar_1",
            "mdf",
            "mnf",
            "total_power",
            "peak_frequency",
            "spectral_entropy",
        }
        self.assertTrue(required.issubset(features.keys()))

    def test_end_to_end_smoke(self) -> None:
        config = ExperimentConfig(
            window_ms=10,
            overlap_ratio=0.5,
            windows_per_group_class=8,
            max_windows_total=100,
            split_strategy="group_kfold",
            cv_folds=2,
            model_family="logreg",
            use_sample_entropy=False,
        )
        bout_frame = build_bouts(self.frame.copy())
        rng = np.random.default_rng(42)
        windows = enumerate_windows(bout_frame, self.bundle.sample_rate_hz, config, rng)
        features, labels, groups = build_feature_table(bout_frame, windows, self.bundle, config, rng)
        artifacts = evaluate_features(features, labels, groups, config, 0.0)
        self.assertIsInstance(artifacts, EvaluationArtifacts)
        self.assertIn("f1_macro", artifacts.metrics)
        self.assertGreater(artifacts.window_count, 0)
        self.assertGreater(artifacts.feature_count, 0)

    def test_grouped_timeout_falls_back_to_holdout(self) -> None:
        config = ExperimentConfig(
            window_ms=10,
            overlap_ratio=0.5,
            windows_per_group_class=8,
            max_windows_total=100,
            split_strategy="group_kfold",
            cv_folds=2,
            model_family="logreg",
            use_sample_entropy=False,
            time_budget_seconds=1,
        )
        bout_frame = build_bouts(self.frame.copy())
        rng = np.random.default_rng(42)
        windows = enumerate_windows(bout_frame, self.bundle.sample_rate_hz, config, rng)
        features, labels, groups = build_feature_table(bout_frame, windows, self.bundle, config, rng)
        with patch("train.remaining_budget_seconds", return_value=0.5):
            artifacts = evaluate_features(features, labels, groups, config, 99.5)
        self.assertEqual(artifacts.split_strategy, "stratified_holdout_0.20")
        self.assertEqual(artifacts.evaluation_fallback_reason, "budget")
        self.assertIn("f1_macro", artifacts.metrics)

    def test_completed_group_fold_returns_without_fallback(self) -> None:
        config = ExperimentConfig(
            window_ms=10,
            overlap_ratio=0.5,
            windows_per_group_class=8,
            max_windows_total=100,
            split_strategy="group_kfold",
            cv_folds=2,
            model_family="logreg",
            use_sample_entropy=False,
            time_budget_seconds=1,
        )
        bout_frame = build_bouts(self.frame.copy())
        rng = np.random.default_rng(42)
        windows = enumerate_windows(bout_frame, self.bundle.sample_rate_hz, config, rng)
        features, labels, groups = build_feature_table(bout_frame, windows, self.bundle, config, rng)
        with patch("train.remaining_budget_seconds", side_effect=[2.0, 1.0, -1.0]):
            artifacts = evaluate_features(features, labels, groups, config, 100.0)
        self.assertEqual(artifacts.split_strategy, "group_kfold_2")
        self.assertIsNone(artifacts.evaluation_fallback_reason)
        self.assertEqual(len(artifacts.fold_summaries), 1)

    def test_raises_if_no_evaluation_mode_can_start_within_budget(self) -> None:
        config = ExperimentConfig(
            window_ms=10,
            overlap_ratio=0.5,
            windows_per_group_class=8,
            max_windows_total=100,
            split_strategy="group_kfold",
            cv_folds=2,
            model_family="logreg",
            use_sample_entropy=False,
            time_budget_seconds=1,
        )
        bout_frame = build_bouts(self.frame.copy())
        rng = np.random.default_rng(42)
        windows = enumerate_windows(bout_frame, self.bundle.sample_rate_hz, config, rng)
        features, labels, groups = build_feature_table(bout_frame, windows, self.bundle, config, rng)
        with patch("train.remaining_budget_seconds", return_value=-1.0):
            with self.assertRaisesRegex(
                TimeoutError,
                "grouped or holdout split could complete",
            ):
                evaluate_features(features, labels, groups, config, 98.0)


if __name__ == "__main__":
    unittest.main()
