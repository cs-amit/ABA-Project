"""Truthful binary probability metrics for subject-held-out evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _thresholded_metrics(
    labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "prediction_rate": float(np.mean(predictions)),
    }


def evaluate_binary_probabilities(
    labels, probabilities, threshold: float
) -> dict[str, float | dict[str, float]]:
    """Evaluate binary probabilities and constant-prediction baselines.

    Label ``1`` is the positive/light class and label ``0`` is rest. ROC-AUC
    and PR-AUC require both classes, so one-class inputs are rejected instead
    of returning misleading scores.
    """
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    if labels_array.ndim != 1 or probabilities_array.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional")
    if labels_array.size == 0 or labels_array.size != probabilities_array.size:
        raise ValueError("labels and probabilities must have the same non-zero length")

    try:
        labels_as_float = labels_array.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("labels must be binary values 0 and 1") from exc
    if not np.all(np.isfinite(labels_as_float)) or not np.all(
        np.isin(labels_as_float, [0.0, 1.0])
    ):
        raise ValueError("labels must be binary values 0 and 1")
    labels_numeric = labels_as_float.astype(np.int64)
    if np.unique(labels_numeric).size != 2:
        raise ValueError("labels must contain two classes for ROC-AUC and PR-AUC")

    try:
        probabilities_numeric = probabilities_array.astype(np.float64)
        threshold_numeric = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("probabilities and threshold must be finite numbers") from exc
    if not np.all(np.isfinite(probabilities_numeric)):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if np.any((probabilities_numeric < 0) | (probabilities_numeric > 1)):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if not np.isfinite(threshold_numeric) or not 0 <= threshold_numeric <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")

    predictions = (probabilities_numeric >= threshold_numeric).astype(np.int64)
    report = _thresholded_metrics(labels_numeric, probabilities_numeric, predictions)
    report["always_light"] = _thresholded_metrics(
        labels_numeric,
        np.ones_like(probabilities_numeric),
        np.ones_like(labels_numeric),
    )
    report["always_rest"] = _thresholded_metrics(
        labels_numeric,
        np.zeros_like(probabilities_numeric),
        np.zeros_like(labels_numeric),
    )
    return report
