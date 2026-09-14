"""Causal hybrid raw-signal and engineered-context validation candidate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from .data_contract import FEATURE_COLUMNS
from .diagnose_raw import best_validation_f1_threshold
from .train import fit_standardizer


def build_hybrid_features(
    frame: pd.DataFrame,
    raw_sequences: np.ndarray,
    feature_columns: list[str] = FEATURE_COLUMNS,
    context_epochs: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use current raw data plus only contiguous current-and-past engineered epochs."""
    if len(frame) != len(raw_sequences):
        raise ValueError("raw artifact and epoch frame must have equal row counts")
    if context_epochs <= 0:
        raise ValueError("context_epochs must be positive")
    required = {"subject_id", "epoch_start_s", "label", "split", *feature_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"epoch frame is missing columns: {sorted(missing)}")
    ordered = frame.copy()
    ordered["_raw_index"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["subject_id", "epoch_start_s"], kind="stable")
    windows = []
    for _, subject in ordered.groupby("subject_id", sort=False):
        starts = subject["epoch_start_s"].to_numpy()
        indices = subject["_raw_index"].to_numpy()
        for end in range(context_epochs - 1, len(subject)):
            window_starts = starts[end - context_epochs + 1 : end + 1]
            if np.all(np.diff(window_starts) == 30):
                windows.append(indices[end - context_epochs + 1 : end + 1])
    if not windows:
        return (
            np.empty((0, raw_sequences.shape[1] * raw_sequences.shape[2] + context_epochs * len(feature_columns) + 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=str),
        )
    window_indices = np.stack(windows)
    end_indices = window_indices[:, -1]
    raw_current = raw_sequences[end_indices].reshape(len(end_indices), -1).astype(np.float32, copy=False)
    engineered = frame[feature_columns].to_numpy(dtype=np.float32)[window_indices].reshape(len(end_indices), -1)
    starts = frame["epoch_start_s"].to_numpy(dtype=np.float64)
    subjects = frame["subject_id"].to_numpy()
    first_start = {subject: starts[np.flatnonzero(subjects == subject)].min() for subject in np.unique(subjects)}
    current_starts = starts[end_indices]
    elapsed_hours = np.asarray([(start - first_start[subject]) / 3600 for start, subject in zip(current_starts, subjects[end_indices])], dtype=np.float32)
    clock_angle = (current_starts % 86_400) * (2 * np.pi / 86_400)
    time_features = np.column_stack((elapsed_hours, np.sin(clock_angle), np.cos(clock_angle))).astype(np.float32)
    return (
        np.concatenate((raw_current, engineered, time_features), axis=1),
        frame["label"].to_numpy(dtype=np.int64)[end_indices],
        frame["split"].astype(str).to_numpy()[end_indices],
    )


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    threshold, best_f1 = best_validation_f1_threshold(labels, probabilities)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "best_f1": best_f1,
        "best_f1_threshold": threshold,
        "f1_at_0_5": float(f1_score(labels, probabilities >= 0.5, zero_division=0)),
    }


def run_hybrid_validation(
    epochs_path: Path,
    raw_path: Path,
    output_path: Path,
    context_epochs: int = 10,
    regularization_values: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
) -> dict:
    """Fit and select regularization exclusively on the existing validation split."""
    frame = pd.read_parquet(epochs_path)
    raw = np.load(raw_path)
    if not np.array_equal(frame["label"].to_numpy(), raw["labels"]):
        raise ValueError("raw labels do not match the epoch artifact")
    if not np.array_equal(frame["split"].astype(str).to_numpy(), raw["splits"].astype(str)):
        raise ValueError("raw split assignments do not match the epoch artifact")
    features, labels, splits = build_hybrid_features(frame, raw["sequences"], context_epochs=context_epochs)
    train_mask, validation_mask = splits == "train", splits == "validation"
    standardizer = fit_standardizer(features[train_mask, None, :])
    scaled = standardizer.transform(features[:, None, :])[:, 0, :]
    trials = []
    for c_value in regularization_values:
        model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=1_000, random_state=20260821)
        model.fit(scaled[train_mask], labels[train_mask])
        probabilities = model.predict_proba(scaled[validation_mask])[:, 1]
        trials.append({"C": c_value, **_metrics(labels[validation_mask], probabilities)})
    selected = max(trials, key=lambda trial: (trial["best_f1"], trial["auc"], -trial["C"]))
    report = {
        "purpose": "validation-only hybrid candidate selection; held-out test was not evaluated",
        "seed": 20260821,
        "context_epochs": context_epochs,
        "feature_description": "current 30-second raw signal plus ten-epoch causal engineered context and current elapsed-session/clock encodings",
        "feature_count": int(features.shape[1]),
        "counts": {split: int((splits == split).sum()) for split in ("train", "validation", "test")},
        "selected_validation_trial": selected,
        "validation_trials": sorted(trials, key=lambda trial: (-trial["best_f1"], -trial["auc"], trial["C"])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=Path, default=Path("ml/artifacts/epochs.parquet"))
    parser.add_argument("--raw", type=Path, default=Path("ml/artifacts/raw_sequences.npz"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/hybrid_raw_engineered_validation.json"))
    parser.add_argument("--context-epochs", type=int, default=10)
    args = parser.parse_args()
    run_hybrid_validation(args.epochs, args.raw, args.output, args.context_epochs)


if __name__ == "__main__":
    main()
