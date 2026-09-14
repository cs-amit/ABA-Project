"""Causal raw-signal resampling for BIDSleep epochs."""

import numpy as np
import pandas as pd

from .data_contract import map_stage
from .features import load_bidsleep_night


def resample_raw_epoch(
    accel: pd.DataFrame,
    heart_rate: pd.DataFrame,
    start_s: float,
    end_s: float,
    bins: int = 30,
) -> np.ndarray:
    """Return per-bin xyz/availability and HR/availability without future samples."""
    if bins <= 0 or end_s <= start_s:
        raise ValueError("bins must be positive and epoch end must follow start")
    result = np.zeros((bins, 6), dtype=np.float32)
    width = (end_s - start_s) / bins
    a = accel[(accel.timestamp_s >= start_s) & (accel.timestamp_s < end_s)].dropna(subset=["timestamp_s", "x", "y", "z"])
    h = heart_rate[(heart_rate.timestamp_s >= start_s) & (heart_rate.timestamp_s < end_s)].dropna(subset=["timestamp_s", "heart_rate_bpm"])
    if len(a):
        a_bins = np.minimum(((a.timestamp_s.to_numpy() - start_s) / width).astype(int), bins - 1)
        a_counts = np.bincount(a_bins, minlength=bins)
        for column, output_column in (("x", 0), ("y", 1), ("z", 2)):
            result[:, output_column] = np.divide(
                np.bincount(a_bins, weights=a[column].to_numpy(), minlength=bins),
                a_counts,
                out=np.zeros(bins),
                where=a_counts != 0,
            )
        result[a_counts != 0, 3] = 1.0
    if len(h):
        h_bins = np.minimum(((h.timestamp_s.to_numpy() - start_s) / width).astype(int), bins - 1)
        h_counts = np.bincount(h_bins, minlength=bins)
        result[:, 4] = np.divide(
            np.bincount(h_bins, weights=h.heart_rate_bpm.to_numpy(), minlength=bins),
            h_counts,
            out=np.zeros(bins),
            where=h_counts != 0,
        )
        result[h_counts != 0, 5] = 1.0
    return result


def prepare_raw_sequences(raw_root, allocation: dict[str, list[str]], bins: int = 30) -> dict[str, np.ndarray]:
    """Build fixed raw epoch sequences using the existing subject-held-out split."""
    split_by_subject = {subject: split for split, subjects in allocation.items() for subject in subjects}
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    epoch_starts: list[float] = []
    subject_ids: list[str] = []
    session_ids: list[str] = []
    for subject_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and path.name in split_by_subject):
        for night_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            accel, heart_rate, epoch_labels = load_bidsleep_night(night_dir, subject_dir.name)
            accel_times = accel["timestamp_s"].to_numpy()
            heart_rate_times = heart_rate["timestamp_s"].to_numpy()
            for label in epoch_labels.itertuples(index=False):
                mapped, _ = map_stage(pd.Series([label.stage]))
                if pd.isna(mapped.iloc[0]):
                    continue
                start, end = label.epoch_start_s, label.epoch_start_s + 30
                a_start, a_end = accel_times.searchsorted([start, end])
                h_start, h_end = heart_rate_times.searchsorted([start, end])
                sequences.append(resample_raw_epoch(accel.iloc[a_start:a_end], heart_rate.iloc[h_start:h_end], start, end, bins))
                labels.append(int(mapped.iloc[0]))
                splits.append(split_by_subject[subject_dir.name])
                epoch_starts.append(float(start))
                subject_ids.append(subject_dir.name)
                # A night directory is a recording/session boundary even when
                # timestamp values happen to be continuous across exports.
                session_ids.append(f"{subject_dir.name}/{night_dir.name}")
    if not sequences:
        raise ValueError("no usable raw BIDSleep epochs found")
    return {
        "sequences": np.stack(sequences),
        "labels": np.asarray(labels, dtype=np.int64),
        "splits": np.asarray(splits),
        "epoch_starts": np.asarray(epoch_starts, dtype=np.float64),
        "subject_ids": np.asarray(subject_ids),
        "session_ids": np.asarray(session_ids),
    }
