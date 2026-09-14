import math

import pytest

from ml.metrics import evaluate_binary_probabilities


def test_evaluate_binary_probabilities_reports_metrics_and_dummy_baselines():
    labels = [0, 0, 1, 1]
    probabilities = [0.1, 0.4, 0.6, 0.9]

    report = evaluate_binary_probabilities(labels, probabilities, threshold=0.5)

    assert report["accuracy"] == pytest.approx(1.0)
    assert report["balanced_accuracy"] == pytest.approx(1.0)
    assert report["precision"] == pytest.approx(1.0)
    assert report["recall"] == pytest.approx(1.0)
    assert report["f1"] == pytest.approx(1.0)
    assert report["roc_auc"] == pytest.approx(1.0)
    assert report["pr_auc"] == pytest.approx(1.0)
    assert report["brier_score"] == pytest.approx(0.085)
    assert report["prediction_rate"] == pytest.approx(0.5)

    assert report["always_light"]["accuracy"] == pytest.approx(0.5)
    assert report["always_rest"]["accuracy"] == pytest.approx(0.5)
    assert report["always_light"]["recall"] == pytest.approx(1.0)
    assert report["always_rest"]["recall"] == pytest.approx(0.0)
    assert report["always_light"]["prediction_rate"] == pytest.approx(1.0)
    assert report["always_rest"]["prediction_rate"] == pytest.approx(0.0)
    assert report["always_light"]["brier_score"] == pytest.approx(0.5)
    assert report["always_rest"]["brier_score"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("labels", "probabilities", "threshold"),
    [
        ([0, 1], [0.2], 0.5),
        ([0, 2], [0.2, 0.8], 0.5),
        ([0, 1], [float("nan"), 0.8], 0.5),
        ([0, 1], [-0.1, 0.8], 0.5),
        ([0, 1], [0.1, 1.1], 0.5),
        ([0, 1], [0.1, 0.8], -0.1),
        ([0, 1], [0.1, 0.8], 1.1),
        ([0, 0.5, 1, 1], [0.1, 0.4, 0.6, 0.9], 0.5),
    ],
)
def test_evaluate_binary_probabilities_rejects_invalid_inputs(
    labels, probabilities, threshold
):
    with pytest.raises(ValueError):
        evaluate_binary_probabilities(labels, probabilities, threshold)


@pytest.mark.parametrize("labels", ([0, 0], [1, 1]))
def test_evaluate_binary_probabilities_rejects_one_class_labels(labels):
    with pytest.raises(ValueError, match="two classes"):
        evaluate_binary_probabilities(labels, [0.2, 0.8], threshold=0.5)
