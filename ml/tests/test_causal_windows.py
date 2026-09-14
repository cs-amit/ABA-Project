import numpy as np
import pytest

from ml.causal_windows import build_causal_windows


def test_builds_fixed_windows_using_only_past_and_current_epochs():
    sequences = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    labels = np.array([0, 1, 0, 1, 0])
    splits = np.array(["train"] * 5)
    starts = np.array([0.0, 30.0, 60.0, 90.0, 120.0])

    windows, endpoint_labels, endpoint_splits = build_causal_windows(
        sequences, labels, splits, starts, window_epochs=3, max_gap_s=30.0
    )

    assert windows.shape == (3, 3, 2)
    np.testing.assert_array_equal(windows[0], sequences[:3])
    np.testing.assert_array_equal(windows[-1], sequences[2:5])
    np.testing.assert_array_equal(endpoint_labels, [0, 1, 0])
    np.testing.assert_array_equal(endpoint_splits, ["train"] * 3)


def test_skips_windows_that_cross_a_gap_or_split_boundary():
    sequences = np.arange(6, dtype=np.float32).reshape(6, 1)
    labels = np.arange(6) % 2
    splits = np.array(["train", "train", "train", "validation", "validation", "validation"])
    starts = np.array([0.0, 30.0, 60.0, 90.0, 150.0, 180.0])

    windows, endpoint_labels, endpoint_splits = build_causal_windows(
        sequences, labels, splits, starts, window_epochs=2, max_gap_s=30.0
    )

    np.testing.assert_array_equal(windows[:, :, 0], [[0, 1], [1, 2], [4, 5]])
    np.testing.assert_array_equal(endpoint_labels, [1, 0, 1])
    np.testing.assert_array_equal(endpoint_splits, ["train", "train", "validation"])


def test_non_increasing_timestamp_starts_a_new_session():
    sequences = np.arange(5, dtype=np.float32).reshape(5, 1)
    labels = np.array([0, 1, 0, 1, 0])
    splits = np.array(["train"] * 5)
    starts = np.array([0.0, 30.0, 0.0, 30.0, 60.0])

    windows, endpoint_labels, endpoint_splits = build_causal_windows(
        sequences, labels, splits, starts, window_epochs=2, max_gap_s=30.0
    )

    np.testing.assert_array_equal(windows[:, :, 0], [[0, 1], [2, 3], [3, 4]])
    np.testing.assert_array_equal(endpoint_labels, [1, 1, 0])
    np.testing.assert_array_equal(endpoint_splits, ["train"] * 3)


@pytest.mark.parametrize("labels", [np.array([0, 2]), np.array([0.0, np.nan])])
def test_rejects_non_binary_or_non_finite_labels(labels):
    with pytest.raises(ValueError):
        build_causal_windows(
            np.zeros((2, 1, 1)), labels, np.array(["train"] * 2), np.arange(2.0), 1, 30.0
        )


def test_empty_output_preserves_window_shape_and_dtypes():
    windows, endpoint_labels, endpoint_splits = build_causal_windows(
        np.zeros((2, 3, 2), dtype=np.float32),
        np.array([0, 1], dtype=np.int64),
        np.array(["train", "train"]),
        np.array([0.0, 30.0]),
        window_epochs=3,
        max_gap_s=30.0,
    )

    assert windows.shape == (0, 3, 3, 2)
    assert windows.dtype == np.float32
    assert endpoint_labels.shape == (0,)
    assert endpoint_labels.dtype == np.int64
    assert endpoint_splits.shape == (0,)
    assert endpoint_splits.dtype.kind in "US"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_epochs": 0, "max_gap_s": 30.0},
        {"window_epochs": 2.5, "max_gap_s": 30.0},
        {"window_epochs": 2, "max_gap_s": 0.0},
        {"window_epochs": 2, "max_gap_s": float("inf")},
    ],
)
def test_rejects_invalid_window_parameters(kwargs):
    with pytest.raises(ValueError):
        build_causal_windows(
            np.zeros((2, 1)), np.zeros(2), np.array(["train"] * 2), np.arange(2.0), **kwargs
        )


def test_rejects_inconsistent_or_malformed_inputs():
    with pytest.raises(ValueError):
        build_causal_windows(
            np.zeros((2, 1, 1)), np.zeros(1), np.array(["train"] * 2), np.arange(2.0), 1, 30.0
        )
    with pytest.raises(ValueError):
        build_causal_windows(
            np.zeros((2, 1, 1)), np.zeros(2), np.array(["train"]), np.arange(2.0), 1, 30.0
        )
    with pytest.raises(ValueError):
        build_causal_windows(
            np.zeros((2, 1, 1)), np.zeros(2), np.array(["train"] * 2), np.array([0.0, np.nan]), 1, 30.0
        )
