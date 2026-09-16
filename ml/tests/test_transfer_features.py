import numpy as np
import pandas as pd

from ml.transfer_features import (
    COMMON_FEATURE_COLUMNS,
    adapt_bidsleep_common,
    adapt_mesa_common,
    build_common_sequences,
)


def test_adapters_emit_identical_declared_feature_order_without_fabricating_ibi_or_axes():
    mesa = pd.DataFrame(
        {
            "subject_id": ["m1"], "epoch_start_s": [0], "label": [1], "split": ["train"],
            "activity_count": [12.0], "activity_observed": [0.75],
            "heart_rate_mean": [61.0], "heart_rate_standard_deviation": [2.0],
            "heart_rate_valid_ratio": [0.5], "elapsed_hours": [1.0],
            "clock_sin": [0.25], "clock_cos": [-0.5], "ibi_rmssd_ms": [99.0],
        }
    )
    bidsleep = pd.DataFrame(
        {
            "subject_id": ["b1"], "epoch_start_s": [3600], "label": [0], "split": ["test"],
            "activity_count": [7.0], "accel_valid_sample_ratio": [0.8],
            "heart_rate_mean": [58.0], "heart_rate_standard_deviation": [3.0],
            "heart_rate_valid_sample_ratio": [0.6], "ibi_rmssd": [44.0],
        }
    )

    mesa_result = adapt_mesa_common(mesa)
    bidsleep_result = adapt_bidsleep_common(bidsleep)

    expected = ["subject_id", "epoch_start_s", "label", "split", *COMMON_FEATURE_COLUMNS]
    assert mesa_result.columns.tolist() == bidsleep_result.columns.tolist() == expected
    assert not any("ibi" in name or name.endswith(("_x", "_y", "_z")) for name in COMMON_FEATURE_COLUMNS)
    assert bidsleep_result.loc[0, "elapsed_hours"] == 0.0
    assert np.isclose(bidsleep_result.loc[0, "clock_sin"], np.sin(2 * np.pi * 20 / 24))


def test_bidsleep_clock_uses_new_york_local_time_in_winter_and_summer():
    frame = pd.DataFrame({
        "subject_id": ["a", "b"], "split": ["train", "validation"], "label": [0, 1],
        # Both timestamps are 06:00 local, despite different UTC offsets.
        "epoch_start_s": [pd.Timestamp("2024-01-15T11:00:00Z").timestamp(),
                          pd.Timestamp("2024-07-15T10:00:00Z").timestamp()],
        "activity_count": [0.0, 0.0], "accel_valid_sample_ratio": [1.0, 1.0],
        "heart_rate_mean": [60.0, 60.0], "heart_rate_standard_deviation": [2.0, 2.0],
        "heart_rate_valid_sample_ratio": [1.0, 1.0],
    })
    adapted = adapt_bidsleep_common(frame)
    np.testing.assert_allclose(adapted.clock_sin, [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(adapted.clock_cos, [0.0, 0.0], atol=1e-12)


def test_both_adapters_log_nonnegative_activity_before_scaling():
    common = {
        "subject_id": ["a"] * 3, "split": ["train"] * 3, "label": [0, 1, 0],
        "epoch_start_s": [0, 30, 60], "activity_count": [-2.0, 0.0, 9.0],
        "heart_rate_mean": [60.0] * 3, "heart_rate_standard_deviation": [2.0] * 3,
    }
    mesa = pd.DataFrame({**common, "activity_observed": [1.0] * 3,
        "heart_rate_valid_ratio": [1.0] * 3, "elapsed_hours": [0.0] * 3,
        "clock_sin": [0.0] * 3, "clock_cos": [1.0] * 3})
    bidsleep = pd.DataFrame({**common, "accel_valid_sample_ratio": [1.0] * 3,
        "heart_rate_valid_sample_ratio": [1.0] * 3})
    for result in (adapt_mesa_common(mesa), adapt_bidsleep_common(bidsleep)):
        np.testing.assert_allclose(result.activity_count, [0.0, 0.0, 2.302585092994046])


def test_bidsleep_elapsed_time_resets_at_gaps_and_sequences_never_cross_boundaries():
    rows = []
    for subject, split, starts in (("a", "train", [0, 30, 60, 300, 330, 360]), ("b", "test", [0, 30, 60])):
        for index, start in enumerate(starts):
            rows.append(
                {
                    "subject_id": subject, "epoch_start_s": start, "label": index % 2, "split": split,
                    "activity_count": float(index), "accel_valid_sample_ratio": 1.0,
                    "heart_rate_mean": 60.0, "heart_rate_standard_deviation": 2.0,
                    "heart_rate_valid_sample_ratio": 1.0,
                }
            )
    adapted = adapt_bidsleep_common(pd.DataFrame(rows))

    assert adapted.loc[(adapted.subject_id == "a") & (adapted.epoch_start_s == 300), "elapsed_hours"].item() == 0.0
    sequences, labels, subjects, splits = build_common_sequences(adapted, sequence_epochs=3)

    assert sequences.shape == (3, 3, len(COMMON_FEATURE_COLUMNS))
    assert labels.tolist() == [0, 1, 0]
    assert subjects.tolist() == ["a", "a", "b"]
    assert splits.tolist() == ["train", "train", "test"]
