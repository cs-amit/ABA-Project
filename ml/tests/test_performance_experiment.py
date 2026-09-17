import importlib
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
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


def test_output_guard_rejects_ignore_rules_that_cover_only_regeneration_sentinels(tmp_path):
    api = experiment_api()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(
        "run/epochs.parquet\nrun/splits.json\nrun/dataset_manifest.json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Git-ignored"):
        api._require_ignored_output(tmp_path / "run", api._VALIDATION_LADDER_OUTPUT_NAMES)


def test_regeneration_accepts_ignore_rules_for_only_its_outputs(raw_and_output, allocation):
    raw, output = raw_and_output
    output.parent.parent.parent.joinpath(".gitignore").write_text(
        "ml/artifacts/corrected/epochs.parquet\n"
        "ml/artifacts/corrected/splits.json\n"
        "ml/artifacts/corrected/dataset_manifest.json\n",
        encoding="utf-8",
    )

    frame, _ = experiment_api().regenerate_corrected_bidsleep(raw, output, allocation)

    assert len(frame) == 4


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


def candidate(name, macro, pooled, balanced):
    return {"name": name, "participant_macro_f1": macro, "pooled": {"f1": pooled},
            "participant_macro_balanced_accuracy": balanced}


def test_ranking_uses_macro_and_advancement_enforces_all_three_limits():
    api = experiment_api()
    baseline = candidate("native_logistic", .60, .65, .62)
    rows = [baseline, candidate("pooled_only", .609, .9, .8),
            candidate("pooled_regression", .70, .649, .7),
            candidate("ba_regression", .69, .66, .614),
            candidate("eligible", .61, .65, .615)]
    ranking, frozen = api.rank_validation_candidates(rows, "native_logistic")
    assert ranking == ["pooled_regression", "ba_regression", "eligible", "pooled_only", "native_logistic"]
    assert frozen == "eligible"
    assert api.rank_validation_candidates(rows[:2], "native_logistic")[1] is None


def test_freeze_persists_complete_allocation_checkpoint_and_scaler_contract(tmp_path):
    api = experiment_api()
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"model bytes")
    scaler = tmp_path / "chosen_scaler.joblib"
    scaler.write_bytes(b"scaler bytes")
    allocation = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    selected = {**candidate("chosen", .7, .7, .7), "checkpoint_path": str(checkpoint),
                "validation_threshold": .42, "feature_columns": ["f"],
                "model_config": {"seed": 17},
                "scaler": {"type": "RobustScaler", "fit_split": "train",
                           "center": [1.0], "scale": [2.0]},
                "participants": {"train": ["a"], "validation": ["b"]}}
    path = api.persist_frozen_candidate(selected, tmp_path, allocation)
    result = json.loads(path.read_text())
    assert result["candidate"] == selected
    assert result["checkpoint_sha256"] == "9cb7487000bc86ac36ce83c4acfabe8878552be99572a6770f65ab1d048a5c48"
    assert result["subject_allocation"] == allocation
    assert result["subject_allocation_sha256"] == "82dbc2383196703442c7bed3725f76b9485a1c1dbe8cef8a058dfd564b899c3d"
    assert result["scaler"] == {
        "path": str(scaler),
        "sha256": hashlib.sha256(b"scaler bytes").hexdigest(),
        "parameters": selected["scaler"],
    }
    assert "test_evaluated" not in result
    with pytest.raises(FileExistsError):
        api.persist_frozen_candidate(selected, tmp_path, allocation)


def test_runner_rejects_test_before_constructing_sequences(epochs, allocation, tmp_path, monkeypatch):
    api = experiment_api()
    frame = pd.concat([epochs, pd.DataFrame([{"subject_id": "Bidslab03", "split": "test"}])], ignore_index=True)
    def prohibited(*args, **kwargs):
        pytest.fail("sequence construction preceded test-row rejection")
    monkeypatch.setattr(api, "build_validation_sequences", prohibited)
    with pytest.raises(ValueError, match="test rows"):
        api.run_validation_ladder(frame, allocation, tmp_path / "absent_mesa", tmp_path / "out")


def test_native_sequences_preserve_feature_order_and_gap_boundaries(epochs):
    api = experiment_api()
    epochs.loc[1, "epoch_start_s"] = 90
    epochs.loc[3, FEATURE_COLUMNS[0]] = 7
    values, labels, subjects, splits = api.build_validation_sequences(epochs, FEATURE_COLUMNS, 2)
    assert values.shape == (1, 2, len(FEATURE_COLUMNS))
    assert values[0, -1, 0] == 7
    assert labels.tolist() == [0]
    assert subjects.tolist() == ["Bidslab02"]
    assert splits.tolist() == ["validation"]


def test_small_ladder_saves_reproducible_validation_only_records(tmp_path, allocation):
    import torch
    from ml.transfer_features import adapt_bidsleep_common
    api = experiment_api()
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    rows = []
    for subject, split in (("Bidslab01", "train"), ("Bidslab02", "validation")):
        for i in range(24):
            rows.append({"subject_id": subject, "split": split, "epoch_start_s": i * 30,
                         "label": i % 2, **{column: float(i % 2) for column in FEATURE_COLUMNS}})
    frame = pd.DataFrame(rows)
    mesa = adapt_bidsleep_common(frame).rename(columns={"activity_availability": "activity_observed", "heart_rate_availability": "heart_rate_valid_ratio"})
    mesa.subject_id = mesa.subject_id.str.replace("Bidslab", "mesa-")
    mesa_path = tmp_path / "mesa.parquet"
    mesa.to_parquet(mesa_path, index=False)
    output = tmp_path / "artifacts" / "run"
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_threads = torch.get_num_threads()
    report = api.run_validation_ladder(frame, allocation, mesa_path, output,
                                       max_epochs=1, pretrain_epochs=1, sequence_epochs=3, threads=1)
    assert torch.are_deterministic_algorithms_enabled() == original_deterministic
    assert torch.get_num_threads() == original_threads
    assert report["baseline"] == "native_logistic"
    assert len(report["candidates"]) == 3
    assert report["selection_split"] == "validation"
    assert report["test_evaluation_policy"] == {
        "performed_by_validation_runner": False,
        "completion_record": "test_evaluation_completed.json",
    }
    assert "test_evaluated" not in report
    for row in report["candidates"]:
        assert row["participants"] == {"train": ["Bidslab01"], "validation": ["Bidslab02"]}
        assert row["scaler"]["fit_split"] == "train"
        assert Path(row["checkpoint_path"]).is_file()
        assert 0 <= row["validation_threshold"] <= 1
        assert row["model_config"]["sequence_epochs"] == 3
        assert row["feature_columns"]
    assert json.loads((output / "validation_report.json").read_text())["ranking"] == report["ranking"]
    if report["frozen_candidate"] is None:
        assert not (output / "frozen_candidate.json").exists()


def frozen_evaluation_fixture(tmp_path, allocation):
    import torch
    from sklearn.preprocessing import RobustScaler
    from ml.models import CnnGru
    from ml.transfer_features import COMMON_FEATURE_COLUMNS

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
    output = tmp_path / "artifacts" / "frozen"
    output.mkdir(parents=True)
    checkpoint = output / "mesa_transfer.pt"
    torch.save(CnnGru(len(COMMON_FEATURE_COLUMNS), hidden_size=32).state_dict(), checkpoint)
    scaler = RobustScaler().fit(np.array([[0.0] * 8, [1.0] * 8], dtype=np.float32))
    joblib.dump(scaler, output / "mesa_transfer_scaler.joblib")
    candidate = {
        **candidate_metrics("mesa_transfer", .7, .7, .7),
        "checkpoint_path": str(checkpoint),
        "validation_threshold": .42,
        "feature_columns": COMMON_FEATURE_COLUMNS,
        "model_config": {"name": "causal_cnn_gru", "hidden_size": 32, "sequence_epochs": 10},
        "scaler": {"type": "RobustScaler", "fit_split": "train",
                   "center": scaler.center_.tolist(), "scale": scaler.scale_.tolist()},
        "participants": {"train": allocation["train"], "validation": allocation["validation"]},
    }
    experiment_api().persist_frozen_candidate(candidate, output, allocation)
    validation_sentinel = output / "validation_report.json"
    validation_sentinel.write_text('{"untouched": true}\n', encoding="utf-8")

    raw = tmp_path / "raw"
    night = raw / allocation["test"][0] / "1"
    night.mkdir(parents=True)
    starts = np.arange(12) * 30
    motion_rows = []
    for start in starts:
        motion_rows.extend([
            {"Timestamp": start, "x": 0, "y": 0, "z": 0},
            {"Timestamp": start + 1, "x": 2, "y": 0, "z": 0},
        ])
    pd.DataFrame(motion_rows).to_csv(night / "motion.csv", index=False)
    pd.DataFrame({"Timestamp": starts, "hr": 60 + np.arange(12) % 3}).to_csv(night / "hr.csv", index=False)
    savemat(night / "labels.mat", {
        "recStart": "1969-12-31 19:00:00",
        "dreem_label": np.array([1, 0] * 6),
    })
    splits = tmp_path / "frozen_splits.json"
    splits.write_text(json.dumps(allocation), encoding="utf-8")
    return raw, splits, output, validation_sentinel


def candidate_metrics(name, macro, pooled, balanced):
    return {"name": name, "participant_macro_f1": macro, "pooled": {"f1": pooled},
            "participant_macro_balanced_accuracy": balanced}


def test_frozen_evaluator_refuses_missing_candidate_before_raw_access(tmp_path, allocation, monkeypatch):
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(FileNotFoundError, match="frozen candidate"):
        api.evaluate_frozen_candidate(tmp_path / "missing.json", tmp_path / "raw", allocation)


def test_frozen_evaluator_refuses_changed_checkpoint_before_raw_access(tmp_path, allocation, monkeypatch):
    raw, splits, output, _ = frozen_evaluation_fixture(tmp_path, allocation)
    (output / "mesa_transfer.pt").write_bytes(b"changed after freezing")
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(ValueError, match="SHA256"):
        api.evaluate_frozen_candidate(output / "frozen_candidate.json", raw, splits)
    assert not (output / "test_evaluation_started.json").exists()


def test_frozen_evaluator_refuses_absent_checkpoint_before_raw_access(tmp_path, allocation, monkeypatch):
    raw, splits, output, _ = frozen_evaluation_fixture(tmp_path, allocation)
    (output / "mesa_transfer.pt").unlink()
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(FileNotFoundError, match="frozen checkpoint"):
        api.evaluate_frozen_candidate(output / "frozen_candidate.json", raw, splits)
    assert not (output / "test_evaluation_started.json").exists()


def test_frozen_evaluator_refuses_changed_allocation_before_raw_access(tmp_path, allocation, monkeypatch):
    raw, _, output, _ = frozen_evaluation_fixture(tmp_path, allocation)
    changed = {**allocation, "test": ["Bidslab04"]}
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(ValueError, match="allocation"):
        api.evaluate_frozen_candidate(output / "frozen_candidate.json", raw, changed)
    assert not (output / "test_evaluation_started.json").exists()


def test_frozen_evaluator_refuses_changed_scaler_before_raw_access(tmp_path, allocation, monkeypatch):
    raw, splits, output, _ = frozen_evaluation_fixture(tmp_path, allocation)
    (output / "mesa_transfer_scaler.joblib").write_bytes(b"changed after freezing")
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(ValueError, match="scaler SHA256"):
        api.evaluate_frozen_candidate(output / "frozen_candidate.json", raw, splits)
    assert not (output / "test_evaluation_started.json").exists()


def test_frozen_evaluator_refuses_scaler_parameter_mismatch_before_raw_access(tmp_path, allocation, monkeypatch):
    raw, splits, output, _ = frozen_evaluation_fixture(tmp_path, allocation)
    path = output / "frozen_candidate.json"
    frozen = json.loads(path.read_text())
    frozen["scaler"]["parameters"]["center"][0] = 99.0
    path.write_text(json.dumps(frozen), encoding="utf-8")
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(ValueError, match="scaler parameters"):
        api.evaluate_frozen_candidate(path, raw, splits)
    assert not (output / "test_evaluation_started.json").exists()


def test_frozen_evaluator_verifies_loaded_scaler_parameters_before_raw_access(tmp_path, allocation, monkeypatch):
    from sklearn.preprocessing import RobustScaler
    raw, splits, output, _ = frozen_evaluation_fixture(tmp_path, allocation)
    scaler_path = output / "mesa_transfer_scaler.joblib"
    changed = RobustScaler().fit(np.array([[10.0] * 8, [11.0] * 8], dtype=np.float32))
    joblib.dump(changed, scaler_path)
    frozen_path = output / "frozen_candidate.json"
    frozen = json.loads(frozen_path.read_text())
    frozen["scaler"]["sha256"] = hashlib.sha256(scaler_path.read_bytes()).hexdigest()
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    api = experiment_api()
    monkeypatch.setattr(api, "read_official_bidsleep_epochs",
                        lambda *args, **kwargs: pytest.fail("raw test data was accessed"))
    with pytest.raises(ValueError, match="loaded scaler parameters"):
        api.evaluate_frozen_candidate(frozen_path, raw, splits)
    assert not (output / "test_evaluation_started.json").exists()


def test_frozen_evaluator_evaluates_named_test_subjects_once_without_changing_validation(tmp_path, allocation):
    raw, splits, output, validation_sentinel = frozen_evaluation_fixture(tmp_path, allocation)
    validation_before = validation_sentinel.read_bytes()
    frozen_path = output / "frozen_candidate.json"
    frozen_before = frozen_path.read_bytes()

    report = experiment_api().evaluate_frozen_candidate(output / "frozen_candidate.json", raw, splits)

    assert report["candidate"] == "mesa_transfer"
    assert report["test_subjects"] == allocation["test"]
    assert report["threshold_source"] == "frozen_validation"
    assert report["threshold"] == .42
    assert report["test_epoch_count"] == 12
    assert report["test_sequence_count"] == 3
    assert report["participant_macro_f1"] == report["per_subject_summary"]["macro_f1"]
    assert len(report["calibration"]) > 0
    assert pd.read_parquet(output / "corrected_test_epochs.parquet").subject_id.unique().tolist() == allocation["test"]
    assert pd.read_csv(output / "frozen_test_participants.csv").subject_id.tolist() == allocation["test"]
    assert validation_sentinel.read_bytes() == validation_before
    assert frozen_path.read_bytes() == frozen_before
    completion = json.loads((output / "test_evaluation_completed.json").read_text())
    assert completion["frozen_candidate_sha256"] == hashlib.sha256(frozen_before).hexdigest()
    assert completion["test_evaluation_sha256"] == hashlib.sha256(
        (output / "frozen_test_evaluation.json").read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="already (started|evaluated)"):
        experiment_api().evaluate_frozen_candidate(output / "frozen_candidate.json", raw, splits)
