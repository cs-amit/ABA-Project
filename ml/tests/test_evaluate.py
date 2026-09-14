import numpy as np

from ml.evaluate import CandidateMetrics, evaluate_probabilities, select_model, write_candidate_report


def test_selection_uses_fast_near_best_candidate_when_top_f1_is_too_slow():
    candidates = [
        CandidateMetrics("cnn_gru", f1=0.82, latency_ms=300.0),
        CandidateMetrics("baseline", f1=0.80, latency_ms=90.0),
        CandidateMetrics("cnn_lstm", f1=0.75, latency_ms=100.0),
    ]

    selected = select_model(candidates)

    assert selected.name == "baseline"


def test_selection_prefers_highest_f1_when_it_meets_latency_budget():
    candidates = [
        CandidateMetrics("cnn_gru", f1=0.82, latency_ms=125.0),
        CandidateMetrics("baseline", f1=0.80, latency_ms=90.0),
    ]

    selected = select_model(candidates)

    assert selected.name == "cnn_gru"


def test_probability_evaluation_reports_required_binary_metrics():
    metrics = evaluate_probabilities(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.4, 0.2]))

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 2 / 3
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_candidate_report_writes_metrics_and_confusion_matrix_image(tmp_path):
    write_candidate_report(tmp_path, {"f1": 0.8, "confusion_matrix": [[2, 0], [1, 1]], "parameter_count": 10})

    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "confusion_matrix.png").read_bytes().startswith(b"\x89PNG")
