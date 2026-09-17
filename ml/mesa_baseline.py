"""Exploratory participant-held-out baseline for prepared MESA epochs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from .mesa import MESA_FEATURE_COLUMNS
from .metrics import evaluate_binary_probabilities


def build_mesa_sequences(
    frame: pd.DataFrame, sequence_epochs: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal sequences without crossing subjects, splits, or time gaps."""
    required = {"subject_id", "epoch_start_s", "label", "split", *MESA_FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MESA epochs missing required columns: {missing}")
    if sequence_epochs <= 0:
        raise ValueError("sequence_epochs must be positive")
    sequences = []
    labels = []
    subjects = []
    for (_, _), group in frame.sort_values(["split", "subject_id", "epoch_start_s"]).groupby(
        ["split", "subject_id"], sort=False
    ):
        group = group.reset_index(drop=True)
        for end in range(sequence_epochs - 1, len(group)):
            window = group.iloc[end - sequence_epochs + 1 : end + 1]
            if not np.all(np.diff(window["epoch_start_s"].to_numpy()) == 30):
                continue
            values = window[MESA_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                continue
            sequences.append(values)
            labels.append(int(window.iloc[-1]["label"]))
            subjects.append(str(window.iloc[-1]["subject_id"]))
    if not sequences:
        return (
            np.empty((0, sequence_epochs, len(MESA_FEATURE_COLUMNS)), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=str),
        )
    return np.stack(sequences), np.asarray(labels, dtype=np.int64), np.asarray(subjects)


def select_validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select the validation-F1 threshold deterministically without test data."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities) or not len(labels):
        raise ValueError("validation labels and probabilities must have the same non-zero length")
    if np.unique(labels).size != 2:
        raise ValueError("validation labels must contain both classes")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best = float(np.max(scores))
    indices = np.flatnonzero(np.isclose(scores, best, rtol=0, atol=1e-12))
    index = min(indices, key=lambda value: (abs(float(thresholds[value]) - 0.5), float(thresholds[value])))
    return float(thresholds[index]), best


def _confusion_counts(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_mesa_baseline(artifacts_dir: Path, seed: int = 20260915) -> dict:
    """Fit on train, select on validation, and evaluate the frozen test split."""
    artifacts_dir = Path(artifacts_dir)
    required = [artifacts_dir / name for name in ("epochs.parquet", "splits.json", "dataset_manifest.json")]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing prepared MESA artifacts: {missing}")
    frame = pd.read_parquet(artifacts_dir / "epochs.parquet")
    splits = json.loads((artifacts_dir / "splits.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifacts_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("feature_columns") != MESA_FEATURE_COLUMNS:
        raise ValueError("MESA manifest feature order does not match the evaluator")
    expected = {subject: split for split, values in splits.items() for subject in values}
    actual = frame.groupby("subject_id")["split"].first().to_dict()
    if actual != expected:
        raise ValueError("MESA frame split assignments do not match splits.json")

    prepared = {}
    for split_name in ("train", "validation", "test"):
        prepared[split_name] = build_mesa_sequences(frame[frame["split"] == split_name])
        if not len(prepared[split_name][0]) or np.unique(prepared[split_name][1]).size != 2:
            raise ValueError(f"{split_name} sequences must contain both labels")
    train_x, train_y, _ = prepared["train"]
    validation_x, validation_y, _ = prepared["validation"]
    test_x, test_y, test_subjects = prepared["test"]
    scaler = StandardScaler().fit(train_x.reshape(len(train_x), -1))
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed).fit(
        scaler.transform(train_x.reshape(len(train_x), -1)), train_y
    )
    validation_probabilities = model.predict_proba(
        scaler.transform(validation_x.reshape(len(validation_x), -1))
    )[:, 1]
    threshold, validation_f1 = select_validation_threshold(validation_y, validation_probabilities)
    test_probabilities = model.predict_proba(scaler.transform(test_x.reshape(len(test_x), -1)))[:, 1]
    test_report = evaluate_binary_probabilities(test_y, test_probabilities, threshold)
    test_report["confusion_counts"] = _confusion_counts(test_y, test_probabilities, threshold)
    test_report["prevalence"] = float(np.mean(test_y))
    report = {
        "dataset": "MESA Sleep",
        "seed": seed,
        "sequence_epochs": 10,
        "feature_columns": MESA_FEATURE_COLUMNS,
        "validation_threshold": threshold,
        "validation_f1": validation_f1,
        "scaler_mean": scaler.mean_.tolist(),
        "test": test_report,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    per_subject_rows = []
    for subject_id in sorted(set(test_subjects)):
        selected = test_subjects == subject_id
        subject_metrics = evaluate_binary_probabilities(test_y[selected], test_probabilities[selected], threshold)
        row = {"subject_id": subject_id, "prevalence": float(np.mean(test_y[selected]))}
        row.update({key: value for key, value in subject_metrics.items() if not isinstance(value, dict)})
        row.update(_confusion_counts(test_y[selected], test_probabilities[selected], threshold))
        per_subject_rows.append(row)
    pd.DataFrame(per_subject_rows).to_csv(artifacts_dir / "per_subject_metrics.csv", index=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the exploratory MESA pilot baseline")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    run_mesa_baseline(args.artifacts_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
