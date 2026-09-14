import math

import pandas as pd

from ml.features import build_epoch_features, prepare_bidsleep_epochs


def test_activity_count_and_rmssd_match_android_formulas():
    accel = pd.DataFrame({"subject_id": ["p1"] * 4, "timestamp_s": [0, 1, 2, 3], "x": [0, 0.2, 0.2, 0.0], "y": [0] * 4, "z": [0] * 4})
    hr = pd.DataFrame({"subject_id": ["p1"] * 3, "timestamp_s": [0, 1, 2], "heart_rate_bpm": [60, 60, 60], "ibi_ms": [800, 820, 780]})
    got = build_epoch_features(accel, hr, 0, 30)
    assert got["activity_count"] == 2
    assert math.isclose(got["ibi_rmssd"], math.sqrt((20**2 + (-40)**2) / 2), rel_tol=1e-6)


def test_zero_crossing_rate_uses_adjacent_centered_magnitudes():
    accel = pd.DataFrame(
        {
            "subject_id": ["p1"] * 4,
            "timestamp_s": [0, 1, 2, 3],
            "x": [0, 2, 0, 2],
            "y": [0] * 4,
            "z": [0] * 4,
        },
    )
    heart_rate = pd.DataFrame(
        {
            "subject_id": ["p1"],
            "timestamp_s": [0],
            "heart_rate_bpm": [60],
            "ibi_ms": [800],
        },
    )

    features = build_epoch_features(accel, heart_rate, 0, 30)

    assert features["accel_zero_crossing_rate"] == 1.0


def test_bidsleep_adapter_maps_30_second_labels():
    accel = pd.DataFrame({"subject_id": ["p1"] * 2, "timestamp_s": [0, 29], "x": [0, 0], "y": [0, 0], "z": [1, 1]})
    hr = pd.DataFrame({"subject_id": ["p1"] * 2, "timestamp_s": [0, 29], "heart_rate_bpm": [60, 60], "ibi_ms": [1000, 1000]})
    labels = pd.DataFrame({"subject_id": ["p1"], "epoch_start_s": [0], "stage": ["LIGHT"]})
    got, manifest = prepare_bidsleep_epochs(accel, hr, labels, seed=1)
    assert got.iloc[0]["epoch_start_s"] == 0
    assert got.iloc[0]["label"] == 1
    assert manifest["unknown_stage_count"] == 0


def test_quality_invalid_samples_do_not_count_as_valid_and_off_body_is_preserved():
    accel = pd.DataFrame({"subject_id": ["p1"] * 2, "timestamp_s": [0, 1], "x": [0, 0], "y": [0, 0], "z": [1, 1], "quality": ["VALID", "OFF_BODY"]})
    hr = pd.DataFrame({"subject_id": ["p1"] * 2, "timestamp_s": [0, 1], "heart_rate_bpm": [60, 60], "ibi_ms": [1000, 1000], "quality": ["VALID", "INVALID"]})
    got = build_epoch_features(accel, hr, 0, 30)
    assert got["accel_valid_sample_ratio"] == 0.5
    assert got["heart_rate_valid_sample_ratio"] == 0.5
    assert got["off_body_flag"] == 1.0
