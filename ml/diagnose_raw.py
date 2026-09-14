"""Validation-only diagnostics for the BIDSleep raw-signal experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .causal_windows import build_causal_windows
from .metrics import evaluate_binary_probabilities


def split_raw_sequences(
    sequences: np.ndarray, labels: np.ndarray, splits: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Separate a raw artifact into its existing participant-held-out splits."""
    if len(sequences) != len(labels) or len(labels) != len(splits):
        raise ValueError("raw sequences, labels, and splits must have equal lengths")
    result = {}
    split_names = splits.astype(str)
    for split in ("train", "validation", "test"):
        mask = split_names == split
        if not mask.any():
            raise ValueError(f"raw artifact has no {split} rows")
        result[split] = (sequences[mask], labels[mask])
    return result


def validate_raw_artifact_metadata(
    sequences: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    epoch_starts: np.ndarray,
    subject_ids: np.ndarray,
    session_ids: np.ndarray,
) -> None:
    """Reject malformed artifact identities before causal window construction."""
    arrays = {
        "labels": np.asarray(labels),
        "splits": np.asarray(splits),
        "epoch_starts": np.asarray(epoch_starts),
        "subject_ids": np.asarray(subject_ids),
        "session_ids": np.asarray(session_ids),
    }
    count = len(np.asarray(sequences))
    if any(values.ndim != 1 for values in arrays.values()) or any(
        len(values) != count for values in arrays.values()
    ):
        raise ValueError("raw sequences and per-epoch metadata must have equal lengths")
    split_values = arrays["splits"].astype(str)
    for name in ("subject_ids", "session_ids"):
        identities = arrays[name].astype(str)
        if np.any(identities == ""):
            raise ValueError(f"{name} must not contain empty values")
        for identity in np.unique(identities):
            if np.unique(split_values[identities == identity]).size != 1:
                raise ValueError(f"each {name[:-1]} must belong to exactly one split")


def best_validation_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select an F1 threshold exclusively from the supplied validation values."""
    candidates = np.arange(0.01, 1.00, 0.01)
    scores = [float(f1_score(labels, probabilities >= threshold, zero_division=0)) for threshold in candidates]
    index = int(np.argmax(scores))
    return round(float(candidates[index]), 2), scores[index]


def validation_probability_report(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Return validation-selected threshold metrics without using test data."""
    threshold, best_f1 = best_validation_f1_threshold(labels, probabilities)
    return {
        **evaluate_binary_probabilities(labels, probabilities, threshold),
        # Retain legacy diagnostic fields while adding the full, imbalance-aware
        # validation report above.
        "auc": float(roc_auc_score(labels, probabilities)),
        "f1_at_0_5": float(f1_score(labels, probabilities >= 0.5, zero_division=0)),
        "best_f1": best_f1,
        "best_f1_threshold": threshold,
        "probability_mean": float(probabilities.mean()),
        "probability_standard_deviation": float(probabilities.std()),
    }


def _validation_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    """Backward-compatible name for the validation-only probability report."""
    return validation_probability_report(labels, probabilities)


def build_validation_windows(
    sequences: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    epoch_starts: np.ndarray,
    window_epochs: int,
    max_gap_s: float,
    session_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build gap-safe causal windows whose endpoint belongs to validation."""
    if session_ids is None:
        windows, window_labels, window_splits = build_causal_windows(
            sequences, labels, splits, epoch_starts, window_epochs, max_gap_s
        )
    else:
        windows, window_labels, window_splits = build_artifact_windows(
            sequences, labels, splits, epoch_starts, session_ids, window_epochs, max_gap_s
        )
    validation_mask = window_splits.astype(str) == "validation"
    return windows[validation_mask], window_labels[validation_mask]


def build_artifact_windows(
    sequences: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    epoch_starts: np.ndarray,
    session_ids: np.ndarray,
    window_epochs: int,
    max_gap_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build causal windows without crossing explicit recording boundaries.

    Sessions are processed as consecutive runs in artifact order.  This makes
    a night boundary safe even if its timestamps happen to look contiguous
    with a neighbouring recording; it also preserves the existing split on
    every emitted endpoint.
    """
    sequence_array = np.asarray(sequences)
    label_array = np.asarray(labels)
    split_array = np.asarray(splits)
    start_array = np.asarray(epoch_starts)
    session_array = np.asarray(session_ids)
    if session_array.ndim != 1 or len(session_array) != len(sequence_array):
        raise ValueError("session_ids must be one-dimensional and match sequences")
    if not len(session_array):
        raise ValueError("raw artifact must contain at least one epoch")
    if np.any(session_array.astype(str) == ""):
        raise ValueError("session_ids must not be empty")

    window_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    run_start = 0
    for index in range(1, len(session_array) + 1):
        if index != len(session_array) and session_array[index] == session_array[run_start]:
            continue
        windows, endpoint_labels, endpoint_splits = build_causal_windows(
            sequence_array[run_start:index],
            label_array[run_start:index],
            split_array[run_start:index],
            start_array[run_start:index],
            window_epochs,
            max_gap_s,
        )
        if len(windows):
            window_parts.append(windows)
            label_parts.append(endpoint_labels)
            split_parts.append(endpoint_splits)
        run_start = index

    if not window_parts:
        return (
            np.empty((0, window_epochs, *sequence_array.shape[1:]), dtype=sequence_array.dtype),
            np.empty((0,), dtype=label_array.dtype),
            np.empty((0,), dtype=split_array.dtype),
        )
    return np.concatenate(window_parts), np.concatenate(label_parts), np.concatenate(split_parts)


def deterministic_training_batches(
    count: int, batch_size: int, seed: int, epoch: int
) -> list[np.ndarray]:
    """Return a reproducible shuffled minibatch permutation for one epoch."""
    if count <= 0 or batch_size <= 0:
        raise ValueError("count and batch_size must be positive")
    indices = np.random.default_rng(seed + epoch).permutation(count)
    return [indices[start : start + batch_size] for start in range(0, count, batch_size)]


def _train_cnn_diagnostic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    recurrent: str,
    seed: int,
) -> tuple[list[dict[str, float]], str]:
    import torch

    from .models import RawCnnGru, RawCnnLstm
    from .train import select_training_device

    if np.unique(train_y).size != 2:
        raise ValueError("training labels must contain both classes before calculating pos_weight")
    device = select_training_device()
    torch.manual_seed(seed)
    model_type = {"gru": RawCnnGru, "lstm": RawCnnLstm}[recurrent]
    model = model_type(
        samples_per_epoch=train_x.shape[-2], channels=train_x.shape[-1], hidden_size=hidden_size
    ).to(device)
    positives = int(train_y.sum())
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((len(train_y) - positives) / positives, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_inputs = torch.from_numpy(train_x)
    train_labels = torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1)
    validation_inputs = torch.from_numpy(validation_x).to(device)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_indices in deterministic_training_batches(len(train_inputs), batch_size, seed, epoch):
            index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
            optimizer.zero_grad()
            probabilities = model(train_inputs[index_tensor].to(device))
            logits = torch.logit(probabilities.clamp(1e-6, 1 - 1e-6))
            loss = criterion(logits, train_labels[index_tensor].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_probabilities = model(validation_inputs).squeeze(1).cpu().numpy()
        history.append({"epoch": epoch, "mean_train_loss": float(np.mean(losses)), **_validation_metrics(validation_y, validation_probabilities)})
    return history, str(device)


def diagnostic_report(
    *,
    input_shape: tuple[int, ...],
    counts: dict[str, int],
    subject_counts: dict[str, int],
    baseline_report: dict,
    neural_reports: dict[str, dict],
    seed: int,
) -> dict:
    """Assemble the validation-only report without fabricating test metrics."""
    required_variants = {"raw_cnn_gru", "raw_cnn_lstm"}
    if set(neural_reports) != required_variants:
        raise ValueError("neural_reports must contain CNN-GRU and CNN-LSTM results")
    return {
        "purpose": "diagnostic only; no held-out test evaluation or model selection",
        "seed": seed,
        "input_shape": list(input_shape),
        "counts": counts,
        "subject_counts": subject_counts,
        "standardization": "StandardScaler fit on train raw-window samples only",
        "raw_logistic_validation": baseline_report,
        **neural_reports,
    }


def run_raw_diagnostic(
    raw_artifact: Path,
    output_path: Path,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-4,
    hidden_size: int = 64,
    seed: int = 20260821,
    window_epochs: int = 10,
    max_gap_s: float = 31.0,
) -> dict:
    """Run a train/validation-only scaling, baseline, and loss diagnostic."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    artifact = np.load(raw_artifact)
    required_metadata = {"sequences", "labels", "splits", "epoch_starts", "subject_ids", "session_ids"}
    missing_metadata = sorted(required_metadata.difference(artifact.files))
    if missing_metadata:
        raise ValueError(f"raw artifact is missing required metadata: {missing_metadata}; rebuild it")
    validate_raw_artifact_metadata(
        artifact["sequences"],
        artifact["labels"],
        artifact["splits"],
        artifact["epoch_starts"],
        artifact["subject_ids"],
        artifact["session_ids"],
    )
    windows, window_labels, window_splits = build_artifact_windows(
        artifact["sequences"],
        artifact["labels"],
        artifact["splits"],
        artifact["epoch_starts"],
        artifact["session_ids"],
        window_epochs,
        max_gap_s,
    )
    partitions = split_raw_sequences(windows, window_labels, window_splits)
    import torch

    from .train import fit_logistic_baseline, fit_standardizer

    train_x, train_y = partitions["train"]
    validation_x, validation_y = partitions["validation"]
    scaler = fit_standardizer(train_x)
    train_x, validation_x = scaler.transform(train_x), scaler.transform(validation_x)
    baseline = fit_logistic_baseline(train_x, train_y)
    baseline_probabilities = baseline.predict_proba(validation_x.reshape(len(validation_x), -1))[:, 1]
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    neural_reports = {}
    for recurrent in ("gru", "lstm"):
        history, device = _train_cnn_diagnostic(
            train_x, train_y, validation_x, validation_y, epochs, batch_size, learning_rate,
            hidden_size, recurrent, seed,
        )
        neural_reports[f"raw_cnn_{recurrent}"] = {
            "hidden_size": hidden_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": history,
            "device": device,
        }
    report = diagnostic_report(
        input_shape=tuple(windows.shape[1:]),
        counts={split: int(len(labels)) for split, (_, labels) in partitions.items()},
        subject_counts={
            split: int(np.unique(artifact["subject_ids"][artifact["splits"].astype(str) == split]).size)
            for split in ("train", "validation", "test")
        },
        baseline_report=_validation_metrics(validation_y, baseline_probabilities),
        neural_reports=neural_reports,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path("ml/artifacts/raw_sequences.npz"))
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/raw_cnn_gru_diagnostic.json"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    args = parser.parse_args()
    run_raw_diagnostic(args.artifact, args.output, args.epochs, args.batch_size, args.learning_rate, args.hidden_size)


if __name__ == "__main__":
    main()
