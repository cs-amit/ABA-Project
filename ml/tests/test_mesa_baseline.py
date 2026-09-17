import json

import numpy as np
import pandas as pd

from ml.mesa import MESA_FEATURE_COLUMNS
from ml.mesa_baseline import build_mesa_sequences, run_mesa_baseline, select_validation_threshold


def _epoch_rows(subject: str, split: str, starts, offset: float = 0.0):
    rows = []
    for index, start in enumerate(starts):
        label = index % 2
        features = {name: 0.0 for name in MESA_FEATURE_COLUMNS}
        features["activity_count"] = offset + label
        features["heart_rate_mean"] = 60.0 + label
        rows.append({"subject_id": subject, "epoch_start_s": start, "label": label, **features, "split": split})
    return rows


def test_build_mesa_sequences_is_causal_and_does_not_cross_gaps_or_subjects():
    frame = pd.DataFrame(
        _epoch_rows("p1", "train", range(0, 330, 30))
        + _epoch_rows("p2", "train", [*range(0, 300, 30), 330, 360])
    )

    sequences, labels, subjects = build_mesa_sequences(frame, sequence_epochs=10)

    assert sequences.shape == (3, 10, len(MESA_FEATURE_COLUMNS))
    assert labels.tolist() == [1, 0, 1]
    assert subjects.tolist() == ["p1", "p1", "p2"]
    assert sequences[0, -1, MESA_FEATURE_COLUMNS.index("activity_count")] == 1.0


def test_select_validation_threshold_maximizes_validation_f1():
    threshold, score = select_validation_threshold(np.array([0, 1, 1]), np.array([0.2, 0.6, 0.8]))

    assert threshold == 0.6
    assert score == 1.0


def test_run_mesa_baseline_fits_train_only_and_writes_test_reports(tmp_path):
    rows = []
    for subject in ("train-a", "train-b"):
        rows.extend(_epoch_rows(subject, "train", range(0, 900, 30)))
    for subject in ("validation-a", "validation-b"):
        rows.extend(_epoch_rows(subject, "validation", range(0, 900, 30)))
    for subject in ("test-a", "test-b"):
        rows.extend(_epoch_rows(subject, "test", range(0, 900, 30), offset=100.0))
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "epochs.parquet", index=False)
    splits = {
        "train": ["train-a", "train-b"],
        "validation": ["validation-a", "validation-b"],
        "test": ["test-a", "test-b"],
    }
    (tmp_path / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps({"dataset": "MESA Sleep", "feature_columns": MESA_FEATURE_COLUMNS, "splits": splits}),
        encoding="utf-8",
    )

    report = run_mesa_baseline(tmp_path)

    assert max(report["scaler_mean"]) <= 61.0
    assert report["test"]["f1"] >= 0.0
    assert set(report["test"]["confusion_counts"]) == {"true_negative", "false_positive", "false_negative", "true_positive"}
    assert "always_light" in report["test"]
    assert "always_rest" in report["test"]
    assert (tmp_path / "metrics.json").is_file()
    per_subject = pd.read_csv(tmp_path / "per_subject_metrics.csv")
    assert set(per_subject["subject_id"]) == {"test-a", "test-b"}
