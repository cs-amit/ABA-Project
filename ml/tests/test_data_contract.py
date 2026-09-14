import pandas as pd
import pytest

from ml.data_contract import FEATURE_COLUMNS, map_stage, split_subjects, validate_epoch_frame


def frame(subjects=("p1", "p2", "p3")):
    rows = []
    for i, subject in enumerate(subjects):
        row = {"subject_id": subject, "epoch_start_s": i * 30, "label": i % 2, "split": "train"}
        row.update({name: 0.0 for name in FEATURE_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def test_contract_requires_android_feature_columns_and_rejects_duplicates():
    validate_epoch_frame(frame())
    duplicate = pd.concat([frame(("p1",)), frame(("p1",))], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_epoch_frame(duplicate)


def test_stage_mapping_and_unknown_count():
    mapped, dropped = map_stage(pd.Series(["LIGHT", "AWAKE", "DEEP", "REM", "UNKNOWN"]))
    assert mapped.tolist() == [1, 0, 0, 0, None]
    assert dropped == 1


def test_model_contract_uses_only_motion_and_heart_rate_features_available_in_bidsleep():
    assert FEATURE_COLUMNS == [
        "accel_magnitude_mean",
        "accel_magnitude_standard_deviation",
        "accel_magnitude_median_absolute_deviation",
        "activity_count",
        "accel_zero_crossing_rate",
        "heart_rate_mean",
        "heart_rate_standard_deviation",
        "accel_valid_sample_ratio",
        "heart_rate_valid_sample_ratio",
    ]


def test_subject_split_is_deterministic_and_disjoint():
    first = split_subjects(["p3", "p1", "p2", "p4", "p5"], seed=7)
    assert first == split_subjects(["p5", "p2", "p4", "p1", "p3"], seed=7)
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
