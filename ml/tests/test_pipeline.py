import json

import pandas as pd

from ml.data_contract import FEATURE_COLUMNS
from ml.train import load_prepared_splits


def test_loader_returns_only_the_manifest_subject_held_out_splits(tmp_path):
    rows = []
    for subject, split in (("p1", "train"), ("p2", "validation"), ("p3", "test")):
        for epoch in (0, 30, 60):
            rows.append({"subject_id": subject, "epoch_start_s": epoch, "label": epoch == 30, "split": split, **{feature: 0.0 for feature in FEATURE_COLUMNS}})
    pd.DataFrame(rows).to_parquet(tmp_path / "epochs.parquet", index=False)
    (tmp_path / "splits.json").write_text(json.dumps({"train": ["p1"], "validation": ["p2"], "test": ["p3"]}))
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"dataset": "BIDSleep", "feature_columns": FEATURE_COLUMNS}))

    splits, manifest = load_prepared_splits(tmp_path)

    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"].subject_id.unique().tolist() == ["p1"]
    assert splits["test"].subject_id.unique().tolist() == ["p3"]
    assert manifest["dataset"] == "BIDSleep"
