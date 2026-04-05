"""
Prepare the EMG gesture dataset for autoresearch experiments.

The original autoresearch repo used prepare.py for one-time corpus setup. This
EMG benchmark keeps the same spirit, but preparation is intentionally light:
we locate the dataset assets already checked into the repo, extract the CSV to a
local cache, and record a few explicit assumptions that train.py can reuse.

Usage:
    uv run prepare.py
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DATASET_ZIP_NAME = "EMG-data.csv.zip"
DATASET_DESCRIPTION_NAME = "dataset_description.txt"
CACHE_DIR = REPO_ROOT / ".cache_emg"
EXTRACTED_CSV_PATH = CACHE_DIR / "EMG-data.csv"
METADATA_PATH = CACHE_DIR / "metadata.json"
DEFAULT_SAMPLE_RATE_HZ = 1000.0


@dataclass(frozen=True)
class PreparedDataset:
    zip_path: Path
    description_path: Path
    extracted_csv_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class DatasetMetadata:
    csv_columns: list[str]
    row_count: int
    subject_count: int
    class_counts: dict[str, int]
    session_count: int
    inferred_sample_rate_hz: float
    sample_rate_fallback_hz: float
    assumptions: list[str]


def locate_dataset_assets(repo_root: Path = REPO_ROOT) -> tuple[Path, Path]:
    zip_path = repo_root / DATASET_ZIP_NAME
    description_path = repo_root / DATASET_DESCRIPTION_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found at {zip_path}")
    if not description_path.exists():
        raise FileNotFoundError(f"Dataset description not found at {description_path}")
    return zip_path, description_path


def extract_dataset_csv(zip_path: Path, destination: Path = EXTRACTED_CSV_PATH) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"Expected exactly one CSV inside {zip_path.name}, found {names}")
        member_name = names[0]
        if destination.exists() and destination.stat().st_size > 0:
            return destination
        archive.extract(member_name, path=CACHE_DIR)
        extracted = CACHE_DIR / member_name
        if extracted != destination:
            extracted.replace(destination)
    return destination


def infer_session_count(frame: pd.DataFrame) -> int:
    time_diff = frame.groupby("label", sort=False)["time"].diff()
    reset_count = int((time_diff < 0).sum())
    return int(frame["label"].nunique() + reset_count)


def infer_sample_rate_hz(frame: pd.DataFrame, fallback_hz: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
    time_diff = frame.groupby("label", sort=False)["time"].diff()
    positive = time_diff[(time_diff > 0) & (time_diff < 20)]
    if positive.empty:
        return fallback_hz
    median_ms = float(np.median(positive.to_numpy()))
    if median_ms <= 0:
        return fallback_hz
    return 1000.0 / median_ms


def build_metadata(csv_path: Path) -> DatasetMetadata:
    frame = pd.read_csv(
        csv_path,
        usecols=["time", "class", "label", "channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8"],
    )
    assumptions = [
        "The zip contains a single analytics-ready CSV named EMG-data.csv.",
        "Column 'label' identifies the subject who wore the MYO bracelet.",
        "Negative time jumps within a subject mark a new recording session.",
        "Median positive time delta is used to infer sample rate; 1000 Hz is the fallback.",
        "Class 0 is unmarked transition data and is excluded by default in train.py.",
        "Class 7 is sparse and optional because only a small subset of subjects performed it.",
    ]
    return DatasetMetadata(
        csv_columns=list(frame.columns),
        row_count=int(len(frame)),
        subject_count=int(frame["label"].nunique()),
        class_counts={str(int(k)): int(v) for k, v in frame["class"].value_counts().sort_index().items()},
        session_count=infer_session_count(frame),
        inferred_sample_rate_hz=infer_sample_rate_hz(frame),
        sample_rate_fallback_hz=fallback_hz_or_default(),
        assumptions=assumptions,
    )


def fallback_hz_or_default() -> float:
    return DEFAULT_SAMPLE_RATE_HZ


def write_metadata(metadata: DatasetMetadata, metadata_path: Path = METADATA_PATH) -> Path:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return metadata_path


def ensure_prepared() -> PreparedDataset:
    zip_path, description_path = locate_dataset_assets()
    csv_path = extract_dataset_csv(zip_path)
    if not METADATA_PATH.exists():
        metadata = build_metadata(csv_path)
        write_metadata(metadata)
    return PreparedDataset(
        zip_path=zip_path,
        description_path=description_path,
        extracted_csv_path=csv_path,
        metadata_path=METADATA_PATH,
    )


def load_metadata(metadata_path: Path = METADATA_PATH) -> dict[str, Any]:
    if not metadata_path.exists():
        ensure_prepared()
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def main() -> None:
    prepared = ensure_prepared()
    metadata = load_metadata(prepared.metadata_path)
    print(f"Dataset zip:      {prepared.zip_path}")
    print(f"Description:      {prepared.description_path}")
    print(f"Extracted CSV:    {prepared.extracted_csv_path}")
    print(f"Metadata JSON:    {prepared.metadata_path}")
    print(f"Rows:             {metadata['row_count']:,}")
    print(f"Subjects:         {metadata['subject_count']}")
    print(f"Sessions:         {metadata['session_count']}")
    print(f"Sample rate (Hz): {metadata['inferred_sample_rate_hz']:.2f}")
    print("Assumptions:")
    for assumption in metadata["assumptions"]:
        print(f"  - {assumption}")


if __name__ == "__main__":
    main()
