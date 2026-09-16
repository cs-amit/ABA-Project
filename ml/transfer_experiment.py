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


def _subject_array(subjects: np.ndarray | None, length: int) -> np.ndarray:
    values = np.zeros(length, dtype=np.int64) if subjects is None else np.asarray(subjects)
    if values.ndim != 1 or len(values) != length or pd.isna(values).any():
        raise ValueError("subjects must contain one nonmissing ID per sequence")
    return values.astype(str)


def equal_subject_loss_weights(labels: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Class-weighted loss coefficients with equal total mass per participant.

    Normalize after class weighting, so different participant prevalences cannot
    undo equal participant mass. The returned weights have global mean one.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("training labels must be a non-empty binary array with both classes")
    subjects = _subject_array(subjects, len(labels))
    positive_weight = np.sum(labels == 0) / np.sum(labels == 1)
    weights = np.where(labels == 1, positive_weight, 1.0)
    _, inverse = np.unique(subjects, return_inverse=True)
    totals = np.bincount(inverse, weights=weights)
    return (weights / totals[inverse] * len(labels) / len(totals)).astype(np.float32)


def select_participant_macro_threshold(
    labels: np.ndarray, probabilities: np.ndarray, subjects: np.ndarray
) -> tuple[float, float]:
    """Maximize validation participant-macro F1, resolving ties near 0.5.

    Sweep distinct predicted probabilities in descending order. Each subject's
    F1 increments are accumulated once, avoiding a threshold-by-row matrix.
    """
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (labels.ndim != 1 or probabilities.ndim != 1 or not len(labels)
            or len(labels) != len(probabilities) or not np.isin(labels, [0, 1]).all()
            or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("validation requires aligned binary labels and finite probabilities in [0, 1]")
    subjects = _subject_array(subjects, len(labels))
    order = np.argsort(-probabilities, kind="stable")
    ordered_labels, ordered_subjects = labels[order], subjects[order]
    deltas = np.zeros(len(labels), dtype=np.float64)
    unique_subjects = np.unique(subjects)
    for subject in unique_subjects:
        positions = np.flatnonzero(ordered_subjects == subject)
        subject_labels = ordered_labels[positions]
        true_positive = np.cumsum(subject_labels)
        denominator = np.arange(1, len(positions) + 1) + subject_labels.sum()
        scores = 2 * true_positive / denominator
        deltas[positions] = np.diff(np.r_[0.0, scores]) / len(unique_subjects)
    macro_scores = np.cumsum(deltas)
    sorted_probabilities = probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_probabilities) != 0), len(labels) - 1]
    best = float(macro_scores[ends].max())
    tied = ends[np.isclose(macro_scores[ends], best, rtol=0, atol=1e-12)]
    selected = min(tied, key=lambda index: (abs(sorted_probabilities[index] - 0.5), sorted_probabilities[index]))
    return float(sorted_probabilities[selected]), float(macro_scores[selected])


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
    train_subjects: np.ndarray | None = None,
    validation_subjects: np.ndarray | None = None,
    equal_subject_weighting: bool = True,
    checkpoint_metric: str = "participant_macro_f1",
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
    if (train_y.ndim != 1 or validation_y.ndim != 1 or len(train_y) != len(train_x)
            or len(validation_y) != len(validation_x) or not np.isin(validation_y, [0, 1]).all()
            or not np.isfinite(train_x).all() or not np.isfinite(validation_x).all()):
        raise ValueError("inputs must be finite with one binary label per sequence")
    train_subjects = _subject_array(train_subjects, len(train_x))
    validation_subjects = _subject_array(validation_subjects, len(validation_x))
    if checkpoint_metric not in {"participant_macro_f1", "pooled_f1_at_0_5"}:
        raise ValueError("unsupported checkpoint_metric")
    _seed_everything(seed)
    model = CnnGru(train_x.shape[2], hidden_size=32)
    if initial_state is not None:
        model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_weight = float(np.sum(train_y == 0) / max(np.sum(train_y == 1), 1))
    loss_weights = (equal_subject_loss_weights(train_y, train_subjects) if equal_subject_weighting
                    else np.where(train_y == 1, positive_weight, 1.0).astype(np.float32))
    criterion = nn.BCELoss(reduction="none")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)),
                      torch.from_numpy(loss_weights)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_macro_f1 = -1.0
    best_pooled_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    epochs = []
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        seen = 0
        for features, labels, weights in loader:
            optimizer.zero_grad()
            probabilities = model(features).squeeze(1)
            loss = (criterion(probabilities, labels) * weights).mean()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(features)
            seen += len(features)
        validation_probabilities = _predict(model, validation_x)
        validation_f1 = float(f1_score(validation_y, validation_probabilities >= 0.5, zero_division=0))
        threshold, macro_f1 = select_participant_macro_threshold(
            validation_y, validation_probabilities, validation_subjects)
        score = macro_f1 if checkpoint_metric == "participant_macro_f1" else validation_f1
        epochs.append({"epoch": epoch + 1, "train_loss": running_loss / seen,
                       "validation_f1_at_0_5": validation_f1,
                       "validation_participant_macro_f1": macro_f1, "validation_threshold": threshold})
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_epoch = epoch + 1
            best_threshold = threshold
            best_macro_f1 = macro_f1
            best_pooled_f1 = validation_f1
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
        "checkpoint_metric": checkpoint_metric,
        "equal_subject_weighting": equal_subject_weighting,
        "train_subject_count": len(np.unique(train_subjects)),
        "validation_subject_count": len(np.unique(validation_subjects)),
        "best_epoch": best_epoch,
        "best_validation_threshold": best_threshold,
        "best_validation_participant_macro_f1": best_macro_f1,
        "best_validation_f1_at_0_5": best_pooled_f1,
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
    validation_subjects: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    validation_probabilities = _predict(model, validation_x)
    threshold, validation_macro_f1 = select_participant_macro_threshold(
        validation_y, validation_probabilities, validation_subjects)
    validation_f1 = float(f1_score(validation_y, validation_probabilities >= threshold, zero_division=0))
    test_probabilities = _predict(model, test_x)
    metrics = evaluate_binary_probabilities(test_y, test_probabilities, threshold)
    per_subject = _per_subject(test_subjects, test_y, test_probabilities, threshold)
    report = {
        "validation_threshold": threshold,
        "validation_f1": validation_f1,
        "validation_participant_macro_f1": validation_macro_f1,
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
        train_subjects=mesa_subjects[mesa_splits == "train"],
        validation_subjects=mesa_subjects[mesa_splits == "validation"],
    )
    control, control_history = train_common_cnn_gru(
        target_scaled[target_splits == "train"], target_y[target_splits == "train"],
        target_scaled[target_splits == "validation"], target_y[target_splits == "validation"],
        seed=seed + 1, max_epochs=20, patience=5,
        train_subjects=target_subjects[target_splits == "train"],
        validation_subjects=target_subjects[target_splits == "validation"],
    )
    transfer, transfer_history = train_common_cnn_gru(
        target_scaled[target_splits == "train"], target_y[target_splits == "train"],
        target_scaled[target_splits == "validation"], target_y[target_splits == "validation"],
        # Hold BIDSleep shuffling and optimizer randomness constant between the
        # control and fine-tuning conditions; the only intended difference is
        # the pretrained MESA initialization.
        seed=seed + 1, max_epochs=20, patience=5, initial_state=pretrained.state_dict(),
        train_subjects=target_subjects[target_splits == "train"],
        validation_subjects=target_subjects[target_splits == "validation"],
    )
    validation_mask = target_splits == "validation"
    test_mask = target_splits == "test"
    control_report, control_subjects = _condition_report(
        control, target_scaled[validation_mask], target_y[validation_mask],
        target_scaled[test_mask], target_y[test_mask], target_subjects[test_mask],
        target_subjects[validation_mask],
    )
    transfer_report, transfer_subjects = _condition_report(
        transfer, target_scaled[validation_mask], target_y[validation_mask],
        target_scaled[test_mask], target_y[test_mask], target_subjects[test_mask],
        target_subjects[validation_mask],
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
