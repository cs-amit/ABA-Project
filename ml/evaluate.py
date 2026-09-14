"""Evaluation records and deployment-model selection policy."""

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CandidateMetrics:
    name: str
    f1: float
    latency_ms: float


def select_model(candidates: list[CandidateMetrics], latency_limit_ms: float = 250.0) -> CandidateMetrics:
    """Apply Task 8's test-F1 and CPU-latency deployment rule."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    best_f1 = max(candidate.f1 for candidate in candidates)
    best = [candidate for candidate in candidates if candidate.f1 == best_f1]
    fast_best = [candidate for candidate in best if candidate.latency_ms < latency_limit_ms]
    if fast_best:
        return min(fast_best, key=lambda candidate: candidate.latency_ms)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.latency_ms < latency_limit_ms and candidate.f1 >= best_f1 - 0.03
    ]
    if not eligible:
        raise ValueError("no candidate meets the CPU latency/F1 deployment rule")
    return min(eligible, key=lambda candidate: candidate.latency_ms)


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Calculate Task 8's thresholded and ranking metrics at a 0.5 threshold."""
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def write_candidate_report(output_dir: Path, metrics: dict) -> None:
    """Write a candidate's machine-readable metrics and human-readable matrix image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figure, axis = plt.subplots(figsize=(3, 3))
    axis.imshow(metrics["confusion_matrix"], cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["rest", "light"], yticklabels=["rest", "light"], xlabel="Predicted", ylabel="Actual")
    for row, values in enumerate(metrics["confusion_matrix"]):
        for column, value in enumerate(values):
            axis.text(column, row, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
