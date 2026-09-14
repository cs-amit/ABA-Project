import numpy as np
import pandas as pd
import torch

from ml.data_contract import FEATURE_COLUMNS
from ml.models import CnnGru
from ml.train import build_sequences, fit_logistic_baseline, fit_standardizer, select_training_device, train_neural_candidate


def test_sequence_builder_keeps_subjects_separate_and_rejects_time_gaps():
    def row(subject_id, epoch_start_s, label):
        return {
            "subject_id": subject_id,
            "epoch_start_s": epoch_start_s,
            "label": label,
            **{feature: float(epoch_start_s) for feature in FEATURE_COLUMNS},
        }

    frame = pd.DataFrame([
        row("a", 0, 0), row("a", 30, 1), row("a", 60, 0),
        row("b", 0, 1), row("b", 30, 0), row("b", 90, 1),
    ])

    sequences, labels = build_sequences(frame, sequence_epochs=3)

    assert sequences.shape == (1, 3, len(FEATURE_COLUMNS))
    assert sequences[0, :, 0].tolist() == [0.0, 30.0, 60.0]
    assert labels.tolist() == [0]


def test_standardizer_fits_training_sequences_only():
    training = __import__("numpy").array([[[0.0] * len(FEATURE_COLUMNS)], [[2.0] * len(FEATURE_COLUMNS)]], dtype="float32")
    held_out = __import__("numpy").array([[[100.0] * len(FEATURE_COLUMNS)]], dtype="float32")

    standardizer = fit_standardizer(training)

    assert standardizer.transform(held_out)[0, 0, 0] == 99.0


def test_candidate_trainers_return_probabilistic_models():
    sequences = np.zeros((8, 10, len(FEATURE_COLUMNS)), dtype=np.float32)
    sequences[4:, :, 0] = 1.0
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    baseline = fit_logistic_baseline(sequences, labels)
    neural, validation_f1 = train_neural_candidate(
        CnnGru(feature_count=len(FEATURE_COLUMNS), hidden_size=4),
        sequences,
        labels,
        sequences,
        labels,
        max_epochs=2,
        patience=1,
    )

    assert baseline.predict_proba(sequences.reshape(8, -1)).shape == (8, 2)
    assert torch.all((neural(torch.from_numpy(sequences)) >= 0.0) & (neural(torch.from_numpy(sequences)) <= 1.0))
    assert 0.0 <= validation_f1 <= 1.0


def test_candidate_trainer_accepts_mini_batches():
    sequences = np.zeros((8, 10, len(FEATURE_COLUMNS)), dtype=np.float32)
    sequences[4:, :, 0] = 1.0
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    _, validation_f1 = train_neural_candidate(
        CnnGru(feature_count=len(FEATURE_COLUMNS), hidden_size=4),
        sequences,
        labels,
        sequences,
        labels,
        max_epochs=1,
        patience=1,
        batch_size=2,
        learning_rate=1e-4,
    )

    assert 0.0 <= validation_f1 <= 1.0


def test_device_selection_uses_cuda_only_when_available():
    assert select_training_device(cuda_available=False).type == "cpu"
    assert select_training_device(cuda_available=True).type == "cuda"
