import sys
import types

import numpy as np
import pytest

from ml.diagnose_raw import (
    build_artifact_windows,
    best_validation_f1_threshold,
    build_validation_windows,
    diagnostic_report,
    deterministic_training_batches,
    run_raw_diagnostic,
    validate_raw_artifact_metadata,
    validation_probability_report,
    split_raw_sequences,
)


def test_split_raw_sequences_keeps_each_split_and_only_fits_train_inputs():
    sequences = np.arange(4 * 2 * 1, dtype=np.float32).reshape(4, 2, 1)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    splits = np.array(["train", "validation", "test", "train"])

    result = split_raw_sequences(sequences, labels, splits)

    assert result["train"][0].shape == (2, 2, 1)
    assert result["train"][1].tolist() == [0, 1]
    assert result["validation"][0].reshape(-1).tolist() == [2.0, 3.0]
    assert result["test"][1].tolist() == [0]


def test_best_validation_f1_threshold_selects_best_score_from_supplied_validation_only():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.array([0.10, 0.40, 0.60, 0.90], dtype=np.float32)

    threshold, f1 = best_validation_f1_threshold(labels, probabilities)

    assert threshold == 0.41
    assert f1 == 1.0


def test_build_validation_windows_uses_supplied_epoch_starts_and_omits_gaps():
    sequences = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    splits = np.array(["validation"] * 5)
    # The third epoch begins after a five-minute recording break, not after
    # the expected thirty-second neighbour interval.
    epoch_starts = np.array([0.0, 30.0, 330.0, 360.0, 390.0])

    windows, window_labels = build_validation_windows(
        sequences, labels, splits, epoch_starts, window_epochs=2, max_gap_s=31.0
    )

    assert windows[:, :, 0, 0].tolist() == [[0.0, 1.0], [2.0, 3.0], [3.0, 4.0]]
    assert window_labels.tolist() == [1, 1, 0]


def test_validation_probability_report_includes_truthful_metric_keys():
    report = validation_probability_report(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([0.10, 0.40, 0.60, 0.90], dtype=np.float32),
    )

    assert {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "prediction_rate",
        "always_light",
        "always_rest",
        "best_f1_threshold",
    }.issubset(report)
    assert report["best_f1_threshold"] == 0.41


def test_build_artifact_windows_blocks_a_window_at_an_explicit_session_boundary():
    sequences = np.arange(5, dtype=np.float32).reshape(5, 1, 1)
    labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    splits = np.array(["validation"] * 5)
    # Timestamps alone look contiguous, so this proves the diagnostic uses the
    # artifact's explicit recording identity rather than trusting timestamps.
    epoch_starts = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
    session_ids = np.array(["person/night-1"] * 3 + ["person/night-2"] * 2)

    windows, window_labels, window_splits = build_artifact_windows(
        sequences, labels, splits, epoch_starts, session_ids, window_epochs=2, max_gap_s=31.0
    )

    assert windows[:, :, 0, 0].tolist() == [[0.0, 1.0], [1.0, 2.0], [3.0, 4.0]]
    assert window_labels.tolist() == [1, 0, 0]
    assert window_splits.tolist() == ["validation"] * 3


def test_deterministic_training_batches_are_seeded_permutations():
    first = deterministic_training_batches(count=8, batch_size=3, seed=9, epoch=2)
    second = deterministic_training_batches(count=8, batch_size=3, seed=9, epoch=2)

    assert [batch.tolist() for batch in first] == [batch.tolist() for batch in second]
    assert np.sort(np.concatenate(first)).tolist() == list(range(8))
    assert np.concatenate(first).tolist() != list(range(8))


def test_diagnostic_report_includes_both_raw_recurrent_variants_without_test_metrics():
    report = diagnostic_report(
        input_shape=(2, 30, 6),
        counts={"train": 8, "validation": 4, "test": 2},
        subject_counts={"train": 2, "validation": 1, "test": 1},
        baseline_report={"f1": 0.5},
        neural_reports={
            "raw_cnn_gru": {"epochs": [{"f1": 0.6}]},
            "raw_cnn_lstm": {"epochs": [{"f1": 0.7}]},
        },
        seed=7,
    )

    assert {"raw_logistic_validation", "raw_cnn_gru", "raw_cnn_lstm"}.issubset(report)
    assert report["subject_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert "test_metrics" not in report


@pytest.mark.parametrize(
    ("subject_ids", "session_ids"),
    [
        (np.array(["person", "person"]), np.array(["person/night-1", "person/night-2"])),
        (np.array(["person-a", "person-b"]), np.array(["shared-night", "shared-night"])),
    ],
)
def test_validate_raw_artifact_metadata_rejects_identity_that_crosses_splits(subject_ids, session_ids):
    with pytest.raises(ValueError, match="exactly one split"):
        validate_raw_artifact_metadata(
            np.zeros((2, 2, 6), dtype=np.float32),
            np.array([0, 1]),
            np.array(["train", "validation"]),
            np.array([0.0, 30.0]),
            subject_ids,
            session_ids,
        )


def test_validate_raw_artifact_metadata_rejects_mismatched_metadata_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        validate_raw_artifact_metadata(
            np.zeros((2, 2, 6), dtype=np.float32),
            np.array([0, 1]),
            np.array(["train", "validation"]),
            np.array([0.0]),
            np.array(["person-a", "person-b"]),
            np.array(["person-a/night", "person-b/night"]),
        )


def test_run_raw_diagnostic_orchestrates_4d_windows_without_neural_training(tmp_path, monkeypatch):
    legacy_path = tmp_path / "legacy.npz"
    np.savez(legacy_path, sequences=np.zeros((2, 2, 6)), labels=np.array([0, 1]), splits=np.array(["train", "validation"]))
    with pytest.raises(ValueError, match="missing required metadata"):
        run_raw_diagnostic(legacy_path, tmp_path / "unused.json", window_epochs=2)

    split_rows = []
    for split, subject in (("train", "person-a"), ("validation", "person-b"), ("test", "person-c")):
        for epoch, label in enumerate((0, 1, 0, 1)):
            split_rows.append((split, subject, f"{subject}/night-1", epoch * 30.0, label))
    artifact_path = tmp_path / "raw.npz"
    np.savez(
        artifact_path,
        sequences=np.arange(len(split_rows) * 2 * 6, dtype=np.float32).reshape(len(split_rows), 2, 6),
        labels=np.array([row[4] for row in split_rows]),
        splits=np.array([row[0] for row in split_rows]),
        epoch_starts=np.array([row[3] for row in split_rows]),
        subject_ids=np.array([row[1] for row in split_rows]),
        session_ids=np.array([row[2] for row in split_rows]),
    )
    observed = {"standardizer_shapes": [], "baseline_shapes": [], "recurrent": []}

    class Scaler:
        def transform(self, values):
            return values

    class Baseline:
        def predict_proba(self, values):
            return np.tile(np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]), (len(values) // 3 + 1, 1))[: len(values)]

    fake_train = types.ModuleType("ml.train")
    def fit_standardizer(values):
        observed["standardizer_shapes"].append(values.shape)
        return Scaler()
    def fit_logistic_baseline(values, labels):
        observed["baseline_shapes"].append(values.shape)
        return Baseline()
    fake_train.fit_standardizer = fit_standardizer
    fake_train.fit_logistic_baseline = fit_logistic_baseline
    monkeypatch.setitem(sys.modules, "ml.train", fake_train)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

    def fake_train_cnn(train_x, train_y, validation_x, validation_y, *args):
        observed["recurrent"].append(args[-2])
        assert train_x.ndim == validation_x.ndim == 4
        return [{"f1": 0.5}], "cpu"
    monkeypatch.setattr("ml.diagnose_raw._train_cnn_diagnostic", fake_train_cnn)

    report = run_raw_diagnostic(artifact_path, tmp_path / "report.json", epochs=1, batch_size=2, window_epochs=2)

    assert observed["standardizer_shapes"] == [(3, 2, 2, 6)]
    assert observed["baseline_shapes"] == [(3, 2, 2, 6)]
    assert observed["recurrent"] == ["gru", "lstm"]
    assert {"raw_cnn_gru", "raw_cnn_lstm"}.issubset(report)
