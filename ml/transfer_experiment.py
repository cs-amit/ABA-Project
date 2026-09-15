"""Exploratory MESA-pretraining experiment on held-out BIDSleep subjects."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .mesa_baseline import select_validation_threshold
from .metrics import evaluate_binary_probabilities
from .models import CnnGru
from .transfer_features import (
    COMMON_FEATURE_COLUMNS,
    adapt_bidsleep_common,
    adapt_mesa_common,
    build_common_sequences,
)


def fit_dataset_scaler(train_sequences: np.ndarray) -> RobustScaler:
    """Fit one robust feature-wise scaler using training sequences only."""
    values = np.asarray(train_sequences)
    if values.ndim != 3 or not len(values) or not np.isfinite(values).all():
        raise ValueError("training sequences must be a finite non-empty 3D array")
    return RobustScaler().fit(values.reshape(-1, values.shape[-1]))


def _transform(scaler: RobustScaler, sequences: np.ndarray) -> np.ndarray:
    shape = sequences.shape
    return scaler.transform(sequences.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _predict(model: CnnGru, values: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).float()
            outputs.append(model(batch).squeeze(1).numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def train_common_cnn_gru(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    seed: int,
    max_epochs: int,
    patience: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[CnnGru, dict]:
    """Train with validation-only early stopping and optional pretrained weights."""
    train_x = np.asarray(train_x, dtype=np.float32)
    validation_x = np.asarray(validation_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.int64)
    validation_y = np.asarray(validation_y, dtype=np.int64)
    if train_x.ndim != 3 or validation_x.ndim != 3 or train_x.shape[2] != validation_x.shape[2]:
        raise ValueError("train and validation inputs must be compatible 3D arrays")
    if not len(train_x) or not len(validation_x) or set(np.unique(train_y)) != {0, 1}:
        raise ValueError("training and validation data must be non-empty with two training classes")
    if max_epochs <= 0 or patience <= 0:
        raise ValueError("max_epochs and patience must be positive")
    _seed_everything(seed)
    model = CnnGru(train_x.shape[2], hidden_size=32)
    if initial_state is not None:
        model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weight = float(np.sum(train_y == 0) / max(np.sum(train_y == 1), 1))
    criterion = nn.BCELoss(reduction="none")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs = []
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, labels in loader:
            optimizer.zero_grad()
            probabilities = model(features).squeeze(1)
            weights = torch.where(labels == 1, positive_weight, 1.0)
            loss = (criterion(probabilities, labels) * weights).mean()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(features)
            seen += len(features)
        validation_probabilities = _predict(model, validation_x)
        validation_f1 = float(f1_score(validation_y, validation_probabilities >= 0.5, zero_division=0))
        epochs.append({"epoch": epoch + 1, "train_loss": running_loss / seen, "validation_f1_at_0_5": validation_f1})
        if validation_f1 > best_f1 + 1e-12:
            best_f1 = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "seed": seed,
        "initial_state_loaded": initial_state is not None,
        "epochs": epochs,
        "best_validation_f1_at_0_5": best_f1,
    }


def paired_subject_bootstrap(
    control: pd.DataFrame,
    transfer: pd.DataFrame,
    iterations: int = 10000,
    seed: int = 20260916,
) -> dict:
    """Bootstrap the paired mean subject-level F1 difference."""
    required = {"subject_id", "f1"}
    if not required.issubset(control.columns) or not required.issubset(transfer.columns):
        raise ValueError("paired reports require subject_id and f1")
    left = control.loc[:, ["subject_id", "f1"]].sort_values("subject_id").reset_index(drop=True)
    right = transfer.loc[:, ["subject_id", "f1"]].sort_values("subject_id").reset_index(drop=True)
    if left["subject_id"].tolist() != right["subject_id"].tolist():
        raise ValueError("paired reports must contain the same subjects")
    if not len(left) or iterations <= 0:
        raise ValueError("paired bootstrap requires subjects and positive iterations")
    differences = right["f1"].to_numpy(dtype=float) - left["f1"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, len(differences), size=(iterations, len(differences)))].mean(axis=1)
    return {
        "metric": "subject_f1",
        "subject_count": int(len(differences)),
        "mean_delta": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "iterations": iterations,
    }


def _calibration(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict]:
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    rows = []
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            rows.append({
                "bin": index,
                "count": int(selected.sum()),
                "mean_probability": float(probabilities[selected].mean()),
                "observed_rate": float(labels[selected].mean()),
            })
    return rows


def _per_subject(
    subjects: np.ndarray, labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> pd.DataFrame:
    rows = []
    predictions = probabilities >= threshold
    for subject in sorted(set(subjects)):
        selected = subjects == subject
        rows.append({
            "subject_id": str(subject),
            "epoch_count": int(selected.sum()),
            "prevalence": float(labels[selected].mean()),
            "accuracy": float(accuracy_score(labels[selected], predictions[selected])),
            "balanced_accuracy": float(balanced_accuracy_score(labels[selected], predictions[selected])),
            "precision": float(precision_score(labels[selected], predictions[selected], zero_division=0)),
            "recall": float(recall_score(labels[selected], predictions[selected], zero_division=0)),
            "f1": float(f1_score(labels[selected], predictions[selected], zero_division=0)),
        })
    return pd.DataFrame(rows)


def _condition_report(
    model: CnnGru,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_subjects: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    validation_probabilities = _predict(model, validation_x)
    threshold, validation_f1 = select_validation_threshold(validation_y, validation_probabilities)
    test_probabilities = _predict(model, test_x)
    metrics = evaluate_binary_probabilities(test_y, test_probabilities, threshold)
    per_subject = _per_subject(test_subjects, test_y, test_probabilities, threshold)
    report = {
        "validation_threshold": threshold,
        "validation_f1": validation_f1,
        "test": metrics,
        "participant_macro_f1": float(per_subject["f1"].mean()),
        "participant_macro_balanced_accuracy": float(per_subject["balanced_accuracy"].mean()),
        "calibration": _calibration(test_y, test_probabilities),
    }
    return report, per_subject


def run_transfer_experiment(
    mesa_artifacts: Path,
    bidsleep_artifacts: Path,
    output_dir: Path,
    seed: int = 20260916,
) -> dict:
    """Run the two predeclared conditions and evaluate BIDSleep test once each."""
    mesa = adapt_mesa_common(pd.read_parquet(Path(mesa_artifacts) / "epochs.parquet"))
    bidsleep = adapt_bidsleep_common(pd.read_parquet(Path(bidsleep_artifacts) / "epochs.parquet"))
    mesa_x, mesa_y, mesa_subjects, mesa_splits = build_common_sequences(mesa)
    target_x, target_y, target_subjects, target_splits = build_common_sequences(bidsleep)
    for name, labels, splits in (("MESA", mesa_y, mesa_splits), ("BIDSleep", target_y, target_splits)):
        for split in ("train", "validation", "test"):
            selected = splits == split
            if not selected.any() or np.unique(labels[selected]).size != 2:
                raise ValueError(f"{name} {split} sequences must contain both classes")

    mesa_scaler = fit_dataset_scaler(mesa_x[mesa_splits == "train"])
    target_scaler = fit_dataset_scaler(target_x[target_splits == "train"])
    mesa_scaled = _transform(mesa_scaler, mesa_x)
    target_scaled = _transform(target_scaler, target_x)

    pretrained, pretrain_history = train_common_cnn_gru(
        mesa_scaled[mesa_splits == "train"], mesa_y[mesa_splits == "train"],
        mesa_scaled[mesa_splits == "validation"], mesa_y[mesa_splits == "validation"],
        seed=seed, max_epochs=30, patience=5,
    )
    control, control_history = train_common_cnn_gru(
        target_scaled[target_splits == "train"], target_y[target_splits == "train"],
        target_scaled[target_splits == "validation"], target_y[target_splits == "validation"],
        seed=seed + 1, max_epochs=20, patience=5,
    )
    transfer, transfer_history = train_common_cnn_gru(
        target_scaled[target_splits == "train"], target_y[target_splits == "train"],
        target_scaled[target_splits == "validation"], target_y[target_splits == "validation"],
        seed=seed + 2, max_epochs=20, patience=5, initial_state=pretrained.state_dict(),
    )
    validation_mask = target_splits == "validation"
    test_mask = target_splits == "test"
    control_report, control_subjects = _condition_report(
        control, target_scaled[validation_mask], target_y[validation_mask],
        target_scaled[test_mask], target_y[test_mask], target_subjects[test_mask],
    )
    transfer_report, transfer_subjects = _condition_report(
        transfer, target_scaled[validation_mask], target_y[validation_mask],
        target_scaled[test_mask], target_y[test_mask], target_subjects[test_mask],
    )
    bootstrap = paired_subject_bootstrap(control_subjects, transfer_subjects, seed=seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pretrained.state_dict(), output_dir / "mesa_pretrained.pt")
    torch.save(control.state_dict(), output_dir / "bidsleep_control.pt")
    torch.save(transfer.state_dict(), output_dir / "mesa_transfer.pt")
    control_subjects.to_csv(output_dir / "control_per_subject.csv", index=False)
    transfer_subjects.to_csv(output_dir / "transfer_per_subject.csv", index=False)
    report = {
        "study_type": "exploratory",
        "seed": seed,
        "feature_columns": COMMON_FEATURE_COLUMNS,
        "architecture": {"name": "causal_cnn_gru", "hidden_size": 32, "sequence_epochs": 10},
        "subject_counts": {
            "mesa": {split: int(len(set(mesa_subjects[mesa_splits == split]))) for split in ("train", "validation", "test")},
            "bidsleep": {split: int(len(set(target_subjects[target_splits == split]))) for split in ("train", "validation", "test")},
        },
        "scalers": {
            "mesa_fit_split": "train", "bidsleep_fit_split": "train",
            "mesa_center": mesa_scaler.center_.tolist(), "mesa_scale": mesa_scaler.scale_.tolist(),
            "bidsleep_center": target_scaler.center_.tolist(), "bidsleep_scale": target_scaler.scale_.tolist(),
        },
        "training": {"mesa_pretraining": pretrain_history, "bidsleep_control": control_history, "bidsleep_transfer": transfer_history},
        "bidsleep_control": control_report,
        "mesa_pretrained_bidsleep_finetuned": transfer_report,
        "paired_subject_bootstrap": bootstrap,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
