import numpy as np
import pandas as pd

from ml.hybrid_candidate import build_hybrid_features


def test_build_hybrid_features_uses_current_raw_epoch_and_causal_contiguous_engineered_context():
    frame = pd.DataFrame([
        {"subject_id": "a", "epoch_start_s": 0, "label": 0, "split": "train", "f": 1.0},
        {"subject_id": "a", "epoch_start_s": 30, "label": 1, "split": "train", "f": 2.0},
        {"subject_id": "a", "epoch_start_s": 60, "label": 0, "split": "train", "f": 3.0},
        {"subject_id": "b", "epoch_start_s": 0, "label": 1, "split": "validation", "f": 4.0},
        {"subject_id": "b", "epoch_start_s": 90, "label": 0, "split": "validation", "f": 5.0},
    ])
    raw = np.arange(5 * 2 * 6, dtype=np.float32).reshape(5, 2, 6)

    features, labels, splits = build_hybrid_features(frame, raw, ["f"], context_epochs=3)

    assert features.shape == (1, 12 + 3 + 3)
    assert features[0, :12].tolist() == raw[2].reshape(-1).tolist()
    assert features[0, 12:15].tolist() == [1.0, 2.0, 3.0]
    assert labels.tolist() == [0]
    assert splits.tolist() == ["train"]
