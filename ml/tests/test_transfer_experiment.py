import numpy as np
import pandas as pd
import pytest
import torch
import ml.transfer_experiment as experiment

from ml.transfer_experiment import fit_dataset_scaler, paired_subject_bootstrap, train_common_cnn_gru


def test_equal_subject_loss_weights_balance_total_contribution_even_with_class_imbalance():
    labels = np.array([0, 1, 0, 0, 0, 0, 0, 1])
    subjects = np.array(["short"] * 2 + ["long"] * 6)
    weights = experiment.equal_subject_loss_weights(labels, subjects)
    np.testing.assert_allclose(weights[:2].sum(), 4.0)
    np.testing.assert_allclose(weights[2:].sum(), 4.0)
    assert weights.mean() == pytest.approx(1.0)
    np.testing.assert_allclose(weights, [1.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5])


def test_macro_threshold_sweep_matches_direct_subject_metrics_with_ties():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(14)
    labels = np.r_[np.zeros(3, dtype=int), rng.integers(0, 2, 37)]
    probabilities = rng.choice([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], size=40)
    subjects = np.array(["zero_positive"] * 3 + ["short"] * 5 + ["long"] * 32)
    scores = {
        float(threshold): float(np.mean([
            f1_score(labels[subjects == subject], probabilities[subjects == subject] >= threshold, zero_division=0)
            for subject in np.unique(subjects)
        ])) for threshold in np.unique(probabilities)
    }
    expected = min(scores, key=lambda threshold: (-scores[threshold], abs(threshold - 0.5), threshold))
    threshold, score = experiment.select_participant_macro_threshold(labels, probabilities, subjects)
    assert threshold == expected
    assert score == pytest.approx(scores[expected])


def test_macro_threshold_prioritizes_participants_instead_of_long_recordings():
    labels = np.array([1] * 10 + [0] * 10 + [1, 0])
    probabilities = np.array([0.8] * 10 + [0.6] * 10 + [0.5, 0.1])
    subjects = np.array(["long"] * 20 + ["short"] * 2)
    threshold, score = experiment.select_participant_macro_threshold(labels, probabilities, subjects)
    assert threshold == 0.5
    assert score == pytest.approx(5 / 6)


def test_checkpoint_selects_thresholded_macro_f1_and_restores_that_epoch(monkeypatch):
    # First epoch: perfect long subject, inverted short subject (macro best=5/6).
    # Second epoch: one long false positive, perfect short subject (macro=41/42).
    labels = np.array([1] * 10 + [0] * 10 + [1, 0])
    subjects = np.array(["long"] * 20 + ["short"] * 2)
    predictions = iter([
        np.array([0.8] * 10 + [0.1] * 10 + [0.2, 0.9]),
        np.array([0.8] * 10 + [0.9] + [0.1] * 9 + [0.4, 0.1]),
    ])
    snapshots = []
    def epoch_predictions(model, values):
        snapshots.append({name: value.clone() for name, value in model.state_dict().items()})
        return next(predictions)
    monkeypatch.setattr(experiment, "_predict", epoch_predictions)
    values = np.random.default_rng(3).normal(size=(22, 3, 2)).astype(np.float32)
    model, history = train_common_cnn_gru(
        values, labels, values, labels, train_subjects=subjects,
        validation_subjects=subjects, seed=3, max_epochs=2, patience=2,
    )
    assert history["best_epoch"] == 2
    assert history["best_validation_participant_macro_f1"] == pytest.approx(41 / 42)
    assert history["best_validation_threshold"] == 0.4
    assert all(torch.equal(value, snapshots[1][name]) for name, value in model.state_dict().items())


def test_equal_subject_training_loss_does_not_depend_on_recording_duplication():
    values = np.random.default_rng(3).normal(size=(4, 3, 2)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    subjects = np.array(["a", "a", "b", "b"])
    _, original = train_common_cnn_gru(values, labels, values, labels,
        train_subjects=subjects, validation_subjects=subjects,
        seed=8, max_epochs=1, learning_rate=0, batch_size=32)
    repeated = np.array([0, 1, 2, 3, 2, 3, 2, 3])
    _, duplicated = train_common_cnn_gru(values[repeated], labels[repeated], values, labels,
        train_subjects=subjects[repeated], validation_subjects=subjects,
        seed=8, max_epochs=1, learning_rate=0, batch_size=32)
    assert duplicated["epochs"][0]["train_loss"] == pytest.approx(original["epochs"][0]["train_loss"], abs=1e-6)


def test_dataset_scaler_uses_feature_axis_and_only_supplied_training_sequences():
    train = np.array([[[0.0, 100.0], [2.0, 200.0]], [[4.0, 300.0], [6.0, 400.0]]])
    scaler = fit_dataset_scaler(train)

    assert scaler.center_.tolist() == [3.0, 250.0]
    assert scaler.n_features_in_ == 2


def test_training_can_start_from_frozen_pretrained_state_without_mutating_it():
    rng = np.random.default_rng(4)
    train_x = rng.normal(size=(24, 3, 2)).astype(np.float32)
    train_y = np.asarray([0, 1] * 12)
    validation_x = rng.normal(size=(8, 3, 2)).astype(np.float32)
    validation_y = np.asarray([0, 1] * 4)
    torch.manual_seed(9)
    initial_model, _ = train_common_cnn_gru(
        train_x, train_y, validation_x, validation_y, seed=9, max_epochs=1, patience=1
    )
    frozen = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}

    fine_tuned, history = train_common_cnn_gru(
        train_x, train_y, validation_x, validation_y, seed=10, max_epochs=1, patience=1,
        initial_state=frozen,
    )

    assert history["initial_state_loaded"] is True
    assert all(torch.equal(frozen[name], initial_model.state_dict()[name]) for name in frozen)
    assert any(not torch.equal(frozen[name], fine_tuned.state_dict()[name]) for name in frozen)


def test_paired_subject_bootstrap_reports_paired_delta_and_interval():
    control = pd.DataFrame({"subject_id": ["a", "b", "c"], "f1": [0.4, 0.5, 0.6]})
    transfer = pd.DataFrame({"subject_id": ["a", "b", "c"], "f1": [0.5, 0.7, 0.9]})

    report = paired_subject_bootstrap(control, transfer, iterations=200, seed=3)

    assert report["subject_count"] == 3
    assert report["mean_delta"] == pytest.approx(0.2)
    assert report["ci95_low"] <= report["mean_delta"] <= report["ci95_high"]


def test_paired_subject_bootstrap_rejects_mismatched_subjects():
    control = pd.DataFrame({"subject_id": ["a"], "f1": [0.5]})
    transfer = pd.DataFrame({"subject_id": ["b"], "f1": [0.6]})
    with pytest.raises(ValueError, match="same subjects"):
        paired_subject_bootstrap(control, transfer)
