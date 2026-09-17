"""Common causal feature representation shared by MESA and BIDSleep."""

from __future__ import annotations

import numpy as np
import pandas as pd


COMMON_FEATURE_COLUMNS = [
    "activity_count",
    "activity_availability",
    "heart_rate_mean",
    "heart_rate_standard_deviation",
    "heart_rate_availability",
    "elapsed_hours",
    "clock_sin",
    "clock_cos",
]

_IDENTITY_COLUMNS = ["subject_id", "epoch_start_s", "label", "split"]


def _require(frame: pd.DataFrame, columns: set[str], dataset: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset} epochs missing common-feature inputs: {missing}")


def adapt_mesa_common(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only features whose semantics can be represented in both datasets."""
    required = {
        *_IDENTITY_COLUMNS,
        "activity_count",
        "activity_observed",
        "heart_rate_mean",
        "heart_rate_standard_deviation",
        "heart_rate_valid_ratio",
        "elapsed_hours",
        "clock_sin",
        "clock_cos",
    }
    _require(frame, required, "MESA")
    result = frame.loc[:, _IDENTITY_COLUMNS].copy()
    result["activity_count"] = np.log1p(frame["activity_count"].clip(lower=0))
    result["activity_availability"] = frame["activity_observed"]
    result["heart_rate_mean"] = frame["heart_rate_mean"]
    result["heart_rate_standard_deviation"] = frame["heart_rate_standard_deviation"]
    result["heart_rate_availability"] = frame["heart_rate_valid_ratio"]
    result["elapsed_hours"] = frame["elapsed_hours"]
    result["clock_sin"] = frame["clock_sin"]
    result["clock_cos"] = frame["clock_cos"]
    return result.loc[:, [*_IDENTITY_COLUMNS, *COMMON_FEATURE_COLUMNS]]


def adapt_bidsleep_common(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the shared representation from BIDSleep engineered epochs."""
    required = {
        *_IDENTITY_COLUMNS,
        "activity_count",
        "accel_valid_sample_ratio",
        "heart_rate_mean",
        "heart_rate_standard_deviation",
        "heart_rate_valid_sample_ratio",
    }
    _require(frame, required, "BIDSleep")
    result = frame.sort_values(["split", "subject_id", "epoch_start_s"], kind="stable").reset_index(drop=True)
    result = result.loc[:, [*_IDENTITY_COLUMNS, "activity_count", "accel_valid_sample_ratio", "heart_rate_mean", "heart_rate_standard_deviation", "heart_rate_valid_sample_ratio"]].copy()
    elapsed = np.zeros(len(result), dtype=np.float64)
    for (_, _), indices in result.groupby(["split", "subject_id"], sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        starts = result.loc[positions, "epoch_start_s"].to_numpy(dtype=np.int64)
        session_start = starts[0]
        for offset, start in enumerate(starts):
            if offset and start - starts[offset - 1] != 30:
                session_start = start
            elapsed[positions[offset]] = (start - session_start) / 3600.0
    local = pd.to_datetime(result["epoch_start_s"], unit="s", utc=True).dt.tz_convert("America/New_York")
    local_seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    angle = 2 * np.pi * local_seconds.to_numpy(dtype=np.float64) / 86400
    result["activity_count"] = np.log1p(result["activity_count"].clip(lower=0))
    result["activity_availability"] = result.pop("accel_valid_sample_ratio")
    result["heart_rate_availability"] = result.pop("heart_rate_valid_sample_ratio")
    result["elapsed_hours"] = elapsed
    result["clock_sin"] = np.sin(angle)
    result["clock_cos"] = np.cos(angle)
    return result.loc[:, [*_IDENTITY_COLUMNS, *COMMON_FEATURE_COLUMNS]]


def build_common_sequences(
    frame: pd.DataFrame, sequence_epochs: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build common-feature windows without crossing subject, split, or gaps."""
    _require(frame, {*_IDENTITY_COLUMNS, *COMMON_FEATURE_COLUMNS}, "common")
    if sequence_epochs <= 0:
        raise ValueError("sequence_epochs must be positive")
    sequences, labels, subjects, splits = [], [], [], []
    ordered = frame.sort_values(["subject_id", "split", "epoch_start_s"], kind="stable")
    for (split, subject), group in ordered.groupby(["split", "subject_id"], sort=False):
        starts = group["epoch_start_s"].to_numpy(dtype=np.int64)
        values = group[COMMON_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        group_labels = group["label"].to_numpy(dtype=np.int64)
        boundaries = np.flatnonzero(np.diff(starts) != 30) + 1
        for positions in np.split(np.arange(len(group)), boundaries):
            if len(positions) < sequence_epochs:
                continue
            segment = values[positions]
            if not np.isfinite(segment).all():
                continue
            windows = np.lib.stride_tricks.sliding_window_view(
                segment, window_shape=sequence_epochs, axis=0
            ).transpose(0, 2, 1).copy()
            sequences.append(windows)
            end_positions = positions[sequence_epochs - 1 :]
            labels.append(group_labels[end_positions])
            subjects.append(np.repeat(str(subject), len(windows)))
            splits.append(np.repeat(str(split), len(windows)))
    shape = (0, sequence_epochs, len(COMMON_FEATURE_COLUMNS))
    return (
        np.concatenate(sequences) if sequences else np.empty(shape, dtype=np.float32),
        np.concatenate(labels) if labels else np.empty(0, dtype=np.int64),
        np.concatenate(subjects) if subjects else np.empty(0, dtype=str),
        np.concatenate(splits) if splits else np.empty(0, dtype=str),
    )
