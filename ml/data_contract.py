"""Stable contract shared by dataset preparation and Task 8."""
import json
from pathlib import Path

import pandas as pd
import random

FEATURE_COLUMNS = [
    "accel_magnitude_mean", "accel_magnitude_standard_deviation",
    "accel_magnitude_median_absolute_deviation", "activity_count",
    "accel_zero_crossing_rate", "heart_rate_mean", "heart_rate_standard_deviation",
    "accel_valid_sample_ratio", "heart_rate_valid_sample_ratio",
]
REQUIRED_COLUMNS = ["subject_id", "epoch_start_s", "label", *FEATURE_COLUMNS, "split"]


def validate_epoch_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.duplicated(["subject_id", "epoch_start_s"]).any():
        raise ValueError("duplicate subject_id/epoch_start_s rows")
    if frame["label"].isna().any() or ~frame["label"].isin([0, 1]).all():
        raise ValueError("label must be binary and non-null")
    if ~frame["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("split must be train, validation, or test")


def map_stage(stages: pd.Series):
    values = stages.map({"LIGHT": 1, "AWAKE": 0, "DEEP": 0, "REM": 0}).astype(object)
    values = values.where(values.notna(), None)
    return values, int(values.isna().sum())


def split_subjects(subjects, seed: int = 42):
    unique = sorted(set(map(str, subjects)))
    if len(unique) < 3:
        names = {"train": unique, "validation": [], "test": []}
        return names
    shuffled = unique[:]
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, len(shuffled) // 5)
    validation_count = max(1, len(shuffled) // 5)
    return {
        "train": sorted(shuffled[: len(shuffled) - test_count - validation_count]),
        "validation": sorted(shuffled[len(shuffled) - test_count - validation_count : len(shuffled) - test_count]),
        "test": sorted(shuffled[len(shuffled) - test_count :]),
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
