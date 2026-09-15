import numpy as np
import pandas as pd
import pytest
import torch

from ml.transfer_experiment import fit_dataset_scaler, paired_subject_bootstrap, train_common_cnn_gru


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
