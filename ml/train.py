"""Reproducible training helpers for the causal sleep candidates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .data_contract import FEATURE_COLUMNS, validate_epoch_frame


def select_training_device(cuda_available: bool | None = None) -> torch.device:
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    return torch.device("cuda" if available else "cpu")


def load_prepared_splits(artifacts_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load only Task 7 artifacts and enforce its subject-held-out allocation."""
    required = [artifacts_dir / name for name in ("epochs.parquet", "splits.json", "dataset_manifest.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prepared Task 7 artifacts: {missing}")
    frame = pd.read_parquet(required[0])
    validate_epoch_frame(frame)
    allocation = json.loads(required[1].read_text(encoding="utf-8"))
    manifest = json.loads(required[2].read_text(encoding="utf-8"))
    if manifest.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("dataset manifest feature order does not match Android contract")
    expected = {subject: split for split, subjects in allocation.items() for subject in subjects}
    actual = frame.groupby("subject_id")["split"].first().to_dict()
    if actual != {subject: expected[subject] for subject in actual if subject in expected} or set(actual) != set(expected):
        raise ValueError("epoch split assignments do not match the subject split manifest")
    return {split: frame[frame["split"] == split].copy() for split in ("train", "validation", "test")}, manifest


class SequenceStandardizer:
    def __init__(self, scaler: StandardScaler) -> None:
        self._scaler = scaler

    def transform(self, sequences: np.ndarray) -> np.ndarray:
        if not len(sequences):
            return sequences.astype(np.float32, copy=True)
        shape = sequences.shape
        return self._scaler.transform(sequences.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def fit_standardizer(training_sequences: np.ndarray) -> SequenceStandardizer:
    if not len(training_sequences):
        raise ValueError("training sequences are required to fit normalization")
    return SequenceStandardizer(StandardScaler().fit(training_sequences.reshape(-1, training_sequences.shape[-1])))


def fit_logistic_baseline(training_sequences: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit the scaled-input baseline with class balancing derived from train only."""
    if len(np.unique(labels)) < 2:
        raise ValueError("training labels must contain both classes")
    return LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=20260821).fit(
        training_sequences.reshape(len(training_sequences), -1), labels
    )


def train_neural_candidate(
    model: torch.nn.Module,
    training_sequences: np.ndarray,
    training_labels: np.ndarray,
    validation_sequences: np.ndarray,
    validation_labels: np.ndarray,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
) -> tuple[torch.nn.Module, float]:
    """Train one causal candidate and restore its best validation-F1 state."""
    if not len(training_sequences) or not len(validation_sequences):
        raise ValueError("training and validation sequences are required")
    positives = int(training_labels.sum())
    negatives = len(training_labels) - positives
    if not positives or not negatives:
        raise ValueError("training labels must contain both classes")
    device = select_training_device()
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    )
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x = torch.from_numpy(training_sequences)
    train_y = torch.from_numpy(training_labels.astype(np.float32)).reshape(-1, 1)
    validation_x = torch.from_numpy(validation_sequences).to(device)
    effective_batch_size = batch_size or len(train_x)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_f1, stale_epochs = -1.0, 0
    for _ in range(max_epochs):
        model.train()
        for start in range(0, len(train_x), effective_batch_size):
            end = start + effective_batch_size
            optimizer.zero_grad()
            logits = torch.logit(model(train_x[start:end].to(device)).clamp(1e-6, 1 - 1e-6))
            loss = criterion(logits, train_y[start:end].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = (model(validation_x).squeeze(1).cpu().numpy() >= 0.5).astype(np.int64)
        score = float(f1_score(validation_labels, predicted, zero_division=0))
        if score > best_f1:
            best_f1, stale_epochs = score, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model.cpu(), best_f1


def build_sequences(frame: pd.DataFrame, sequence_epochs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Build causal, contiguous sequences without crossing subject boundaries."""
    sequences: list[np.ndarray] = []
    labels: list[int] = []
    for _, subject in frame.sort_values(["subject_id", "epoch_start_s"]).groupby("subject_id", sort=False):
        subject = subject.reset_index(drop=True)
        for end in range(sequence_epochs - 1, len(subject)):
            window = subject.iloc[end - sequence_epochs + 1 : end + 1]
            starts = window["epoch_start_s"].to_numpy()
            if not np.all(np.diff(starts) == 30):
                continue
            sequences.append(window[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            labels.append(int(window.iloc[-1]["label"]))
    if not sequences:
        return (
            np.empty((0, sequence_epochs, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(sequences), np.asarray(labels, dtype=np.int64)
