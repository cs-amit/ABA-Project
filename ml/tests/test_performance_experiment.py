import importlib
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from ml.data_contract import FEATURE_COLUMNS


def experiment_api():
    assert importlib.util.find_spec("ml.performance_experiment") is not None, "validation-only experiment loader is missing"
    return importlib.import_module("ml.performance_experiment")


@pytest.fixture
def allocation():
    return {"train": ["Bidslab01"], "validation": ["Bidslab02"], "test": ["Bidslab03"]}


@pytest.fixture
def epochs():
    rows = []
    for subject, split in (("Bidslab01", "train"), ("Bidslab02", "validation")):
        for start in (0, 30):
            rows.append({"subject_id": subject, "epoch_start_s": start, "label": int(start == 0),
                         "split": split, **{column: 1.0 for column in FEATURE_COLUMNS}})
    return pd.DataFrame(rows)


@pytest.fixture
def raw_and_output(tmp_path, allocation):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("ml/artifacts/\n", encoding="utf-8")
    raw = tmp_path / "raw"
    for subject in allocation["train"] + allocation["validation"]:
        night = raw / subject / "1"
        night.mkdir(parents=True)
        pd.DataFrame({"Timestamp": [0, 1, 2, 3, 30, 31, 32, 33],
                      "x": [0, 2, 0, 2] * 2, "y": [0] * 8, "z": [0] * 8}).to_csv(night / "motion.csv", index=False)
        pd.DataFrame({"Timestamp": [0, 30], "hr": [60, 61]}).to_csv(night / "hr.csv", index=False)
        savemat(night / "labels.mat", {"recStart": "1969-12-31 19:00:00", "dreem_label": np.array([1, 0])})
    # A test directory with unreadable/missing source files must never be opened.
    (raw / "Bidslab03" / "1").mkdir(parents=True)
    return raw, tmp_path / "ml" / "artifacts" / "corrected"


def test_regeneration_recovers_zcr_without_reading_test_subjects(raw_and_output, allocation):
    raw, output = raw_and_output
    frame, manifest = experiment_api().regenerate_corrected_bidsleep(raw, output, allocation)
    assert set(frame.subject_id) == {"Bidslab01", "Bidslab02"}
    assert frame.accel_zero_crossing_rate.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert manifest["splits"] == allocation
    assert manifest["included_splits"] == ["train", "validation"]
    assert json.loads((output / "splits.json").read_text()) == allocation
    pd.testing.assert_frame_equal(pd.read_parquet(output / "epochs.parquet"), frame)


def test_loader_rejects_test_rows_before_label_or_feature_access(epochs, allocation):
    frame = pd.concat([epochs, pd.DataFrame([{"subject_id": "Bidslab03", "split": "test"}])], ignore_index=True)
    with pytest.raises(ValueError, match="test rows"):
        experiment_api().load_validation_epochs(frame, allocation)


@pytest.mark.parametrize("subject,split", [("Bidslab03", "train"), ("unknown", "train"), ("Bidslab01", "validation")])
def test_loader_enforces_frozen_allocation(epochs, allocation, subject, split):
    epochs.loc[0, ["subject_id", "split"]] = [subject, split]
    with pytest.raises(ValueError, match="frozen"):
        experiment_api().load_validation_epochs(epochs, allocation)


@pytest.mark.parametrize("column,value", [(FEATURE_COLUMNS[0], np.inf), (FEATURE_COLUMNS[1], np.nan), ("epoch_start_s", np.nan)])
def test_loader_rejects_nonfinite_values(epochs, allocation, column, value):
    epochs[column] = epochs[column].astype(float)
    epochs.loc[0, column] = value
    with pytest.raises(ValueError, match="finite"):
        experiment_api().load_validation_epochs(epochs, allocation)


def test_loader_requires_all_frozen_train_validation_subjects(epochs, allocation):
    with pytest.raises(ValueError, match="missing.*frozen"):
        experiment_api().load_validation_epochs(epochs[epochs.split == "train"], allocation)


def test_loader_rejects_overlapping_allocation(epochs, allocation):
    allocation["test"].append("Bidslab01")
    with pytest.raises(ValueError, match="multiple splits"):
        experiment_api().load_validation_epochs(epochs, allocation)


def test_loader_loads_separate_artifact_and_preserves_gaps(epochs, allocation, tmp_path):
    epochs.loc[1, "epoch_start_s"] = 90
    source = tmp_path / "epochs.parquet"
    epochs.to_parquet(source, index=False)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps(allocation), encoding="utf-8")
    result = experiment_api().load_validation_epochs(source, splits)
    pd.testing.assert_frame_equal(result, epochs)


def test_regeneration_refuses_to_overwrite_previous_artifact(raw_and_output, allocation):
    raw, output = raw_and_output
    output.mkdir(parents=True)
    old = output / "epochs.parquet"
    old.write_bytes(b"preserve this previous artifact")
    with pytest.raises(FileExistsError):
        experiment_api().regenerate_corrected_bidsleep(raw, output, allocation)
    assert old.read_bytes() == b"preserve this previous artifact"


def test_regeneration_refuses_unignored_output(raw_and_output, allocation):
    raw, _ = raw_and_output
    with pytest.raises(ValueError, match="ignored"):
        experiment_api().regenerate_corrected_bidsleep(raw, raw.parent / "tracked-output", allocation)


def test_regeneration_cli(raw_and_output, allocation):
    raw, output = raw_and_output
    splits = raw.parent / "frozen.json"
    splits.write_text(json.dumps(allocation), encoding="utf-8")
    experiment_api().main(["regenerate", "--raw-root", str(raw), "--output-dir", str(output), "--frozen-splits", str(splits)])
    result = pd.read_parquet(output / "epochs.parquet")
    assert len(result) == 4
    assert set(result.split) == {"train", "validation"}
