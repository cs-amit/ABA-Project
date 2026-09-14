import numpy as np
import pandas as pd
from scipy.io import savemat

from ml.raw_sequences import prepare_raw_sequences, resample_raw_epoch


def test_resample_raw_epoch_uses_only_current_epoch_samples_and_marks_missing_hr():
    accel = pd.DataFrame({
        "timestamp_s": [0.1, 0.9, 1.1, 2.1],
        "x": [1.0, 3.0, 9.0, 99.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0],
    })
    heart_rate = pd.DataFrame({"timestamp_s": [0.5, 2.0], "heart_rate_bpm": [60.0, 90.0]})

    got = resample_raw_epoch(accel, heart_rate, start_s=0.0, end_s=2.0, bins=2)

    assert got.shape == (2, 6)
    np.testing.assert_allclose(got[0], [2.0, 0.0, 0.0, 1.0, 60.0, 1.0])
    np.testing.assert_allclose(got[1], [9.0, 0.0, 0.0, 1.0, 0.0, 0.0])


def test_resample_raw_epoch_drops_non_finite_sensor_rows():
    accel = pd.DataFrame({"timestamp_s": [0.1], "x": [float("nan")], "y": [0.0], "z": [0.0]})
    heart_rate = pd.DataFrame({"timestamp_s": [0.5], "heart_rate_bpm": [float("nan")]})

    got = resample_raw_epoch(accel, heart_rate, start_s=0.0, end_s=1.0, bins=1)

    np.testing.assert_allclose(got, np.zeros((1, 6)))


def test_prepare_raw_sequences_preserves_subject_split(tmp_path):
    raw_root = tmp_path / "raw"
    for subject in ("Bidslab00", "Bidslab01", "Bidslab02"):
        night = raw_root / subject / "1"
        night.mkdir(parents=True)
        pd.DataFrame({"Timestamp": [0.1, 1.1, 30.1], "x": [1.0, 2.0, 3.0], "y": [0.0] * 3, "z": [0.0] * 3}).to_csv(night / "motion.csv", index=False)
        pd.DataFrame({"Timestamp": [0.5, 30.5], "hr": [60.0, 61.0]}).to_csv(night / "hr.csv", index=False)
        savemat(night / "labels.mat", {"recStart": "1969-12-31 19:00:00", "dreem_label": np.array([1, 3], dtype=np.uint8)})

    result = prepare_raw_sequences(raw_root, {"train": ["Bidslab00"], "validation": ["Bidslab01"], "test": ["Bidslab02"]})

    assert result["sequences"].shape == (6, 30, 6)
    assert result["labels"].tolist() == [1, 0, 1, 0, 1, 0]
    assert result["splits"].tolist() == ["train", "train", "validation", "validation", "test", "test"]


def test_prepare_raw_sequences_preserves_epoch_and_explicit_session_metadata(tmp_path):
    raw_root = tmp_path / "raw"
    for night_name in ("1", "2"):
        night = raw_root / "Bidslab00" / night_name
        night.mkdir(parents=True)
        pd.DataFrame({"Timestamp": [0.1, 30.1], "x": [1.0, 2.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}).to_csv(night / "motion.csv", index=False)
        pd.DataFrame({"Timestamp": [0.5, 30.5], "hr": [60.0, 61.0]}).to_csv(night / "hr.csv", index=False)
        savemat(night / "labels.mat", {"recStart": "1969-12-31 19:00:00", "dreem_label": np.array([1, 3], dtype=np.uint8)})

    result = prepare_raw_sequences(raw_root, {"train": ["Bidslab00"]})

    assert result["epoch_starts"].tolist() == [0.0, 30.0, 0.0, 30.0]
    assert result["subject_ids"].tolist() == ["Bidslab00"] * 4
    assert result["session_ids"].tolist() == ["Bidslab00/1", "Bidslab00/1", "Bidslab00/2", "Bidslab00/2"]
