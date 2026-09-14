"""Construction of fixed-length, causal raw-signal windows."""

from __future__ import annotations

import numbers

import numpy as np


def build_causal_windows(
    sequences,
    labels,
    splits,
    epoch_starts,
    window_epochs: int,
    max_gap_s: float,
):
    """Build windows ending at each eligible labelled epoch.

    Every returned window contains exactly ``window_epochs`` consecutive input
    epochs and ends at its corresponding label/split.  A candidate is omitted
    when it crosses a split boundary or any epoch-start gap greater than
    ``max_gap_s``.  No future epoch is included in a window.
    """
    if isinstance(window_epochs, (bool, np.bool_)) or not isinstance(window_epochs, numbers.Integral):
        raise ValueError("window_epochs must be a positive integer")
    if window_epochs <= 0:
        raise ValueError("window_epochs must be a positive integer")
    if isinstance(max_gap_s, (bool, np.bool_)) or not isinstance(max_gap_s, numbers.Real):
        raise ValueError("max_gap_s must be a positive finite number")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("max_gap_s must be a positive finite number")

    sequence_array = np.asarray(sequences)
    label_array = np.asarray(labels)
    split_array = np.asarray(splits)
    start_array = np.asarray(epoch_starts)

    if sequence_array.ndim < 2:
        raise ValueError("sequences must have an epoch axis and at least one feature axis")
    if label_array.ndim != 1 or split_array.ndim != 1 or start_array.ndim != 1:
        raise ValueError("labels, splits, and epoch_starts must be one-dimensional")
    if not np.issubdtype(label_array.dtype, np.number):
        raise ValueError("labels must be numeric binary values")
    if not np.isfinite(label_array).all() or not np.isin(label_array, [0, 1]).all():
        raise ValueError("labels must be finite and binary (0 or 1)")
    count = sequence_array.shape[0]
    if not (len(label_array) == len(split_array) == len(start_array) == count):
        raise ValueError("sequences, labels, splits, and epoch_starts must have equal lengths")
    if not np.issubdtype(start_array.dtype, np.number) or not np.isfinite(start_array).all():
        raise ValueError("epoch_starts must contain finite numeric values")

    windows = []
    endpoint_labels = []
    endpoint_splits = []
    for end in range(window_epochs - 1, count):
        begin = end - window_epochs + 1
        starts = start_array[begin : end + 1]
        # Non-increasing starts also mark a session/recording boundary; do not
        # let a window mix epochs from either side of that boundary.
        gaps = np.diff(starts)
        if np.any(gaps <= 0) or np.any(gaps > max_gap_s):
            continue
        if not np.all(split_array[begin : end + 1] == split_array[end]):
            continue
        windows.append(sequence_array[begin : end + 1])
        endpoint_labels.append(label_array[end])
        endpoint_splits.append(split_array[end])

    if windows:
        window_array = np.stack(windows, axis=0)
    else:
        window_array = np.empty((0, window_epochs, *sequence_array.shape[1:]), dtype=sequence_array.dtype)
    return (
        window_array,
        np.asarray(endpoint_labels, dtype=label_array.dtype),
        np.asarray(endpoint_splits, dtype=split_array.dtype),
    )
