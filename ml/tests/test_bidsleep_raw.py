import numpy as np
import pandas as pd
from scipy.io import savemat

from ml.features import load_bidsleep_night, prepare_official_bidsleep


def test_raw_night_adapter_maps_aasm_light_and_drops_unknown(tmp_path):
    pd.DataFrame({"Timestamp": [0.0, 30.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [1.0, 1.0]}).to_csv(tmp_path / "motion.csv", index=False)
    pd.DataFrame({"Timestamp": [0.0, 30.0], "hr": [60.0, 61.0]}).to_csv(tmp_path / "hr.csv", index=False)
    savemat(tmp_path / "labels.mat", {"recStart": "1969-12-31 19:00:00", "dreem_label": np.array([1, 2, 3, 4, 5], dtype=np.uint8)})

    accel, heart_rate, labels = load_bidsleep_night(tmp_path, "participant-night")

    assert accel.columns.tolist() == ["subject_id", "timestamp_s", "x", "y", "z"]
    assert heart_rate.columns.tolist() == ["subject_id", "timestamp_s", "heart_rate_bpm", "ibi_ms"]
    assert labels[["epoch_start_s", "stage"]].to_dict("records") == [
        {"epoch_start_s": 0, "stage": "LIGHT"},
        {"epoch_start_s": 30, "stage": "LIGHT"},
        {"epoch_start_s": 60, "stage": "DEEP"},
        {"epoch_start_s": 90, "stage": "REM"},
    ]


def test_raw_night_adapter_interprets_recording_start_as_eastern_time(tmp_path):
    pd.DataFrame({"Timestamp": [1638504587.0], "x": [0.0], "y": [0.0], "z": [1.0]}).to_csv(tmp_path / "motion.csv", index=False)
    pd.DataFrame({"Timestamp": [1638504587.0], "hr": [60.0]}).to_csv(tmp_path / "hr.csv", index=False)
    savemat(tmp_path / "labels.mat", {"recStart": "2021-12-02 23:11:25", "dreem_label": np.array([0, 0], dtype=np.uint8)})

    _, _, labels = load_bidsleep_night(tmp_path, "participant-night")

    assert labels.iloc[0]["epoch_start_s"] == 1638504685


def test_official_preparation_keeps_subjects_held_out_across_nights(tmp_path):
    raw_root = tmp_path / "raw"
    for subject in ("Bidslab00", "Bidslab01", "Bidslab02"):
        night = raw_root / subject / "1"
        night.mkdir(parents=True)
        pd.DataFrame({"Timestamp": [0.0, 30.0, 60.0], "x": [0.0] * 3, "y": [0.0] * 3, "z": [1.0] * 3}).to_csv(night / "motion.csv", index=False)
        pd.DataFrame({"Timestamp": [0.0, 30.0, 60.0], "hr": [60.0] * 3}).to_csv(night / "hr.csv", index=False)
        savemat(night / "labels.mat", {"recStart": "1969-12-31 19:00:00", "dreem_label": np.array([1, 0, 2], dtype=np.uint8)})

    epochs, manifest = prepare_official_bidsleep(raw_root, tmp_path / "artifacts", seed=7)

    assert set(epochs.subject_id) == {"Bidslab00", "Bidslab01", "Bidslab02"}
    assert epochs.groupby("subject_id").split.nunique().eq(1).all()
    assert (tmp_path / "artifacts" / "epochs.parquet").is_file()
    assert manifest["dataset"] == "BIDSleep"
