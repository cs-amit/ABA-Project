import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.io import savemat


def test_prepare_dataset_requires_environment_variable(tmp_path):
    env = os.environ.copy()
    env.pop("SMART_SLEEP_DATA_DIR", None)
    result = subprocess.run([sys.executable, "-m", "ml.prepare_dataset", "--output-dir", str(tmp_path)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "SMART_SLEEP_DATA_DIR" in result.stderr


def test_prepare_dataset_reads_official_raw_bidsleep_layout(tmp_path):
    raw_root = tmp_path / "raw"
    for subject in ("Bidslab00", "Bidslab01", "Bidslab02"):
        night = raw_root / subject / "1"
        night.mkdir(parents=True)
        pd.DataFrame({"Timestamp": [0.0, 30.0, 60.0], "x": [0.0] * 3, "y": [0.0] * 3, "z": [1.0] * 3}).to_csv(night / "motion.csv", index=False)
        pd.DataFrame({"Timestamp": [0.0, 30.0, 60.0], "hr": [60.0] * 3}).to_csv(night / "hr.csv", index=False)
        savemat(night / "labels.mat", {"recStart": "1969-12-31 19:00:00", "dreem_label": np.array([1, 0, 2], dtype=np.uint8)})

    output_dir = tmp_path / "artifacts"
    env = os.environ.copy()
    env["SMART_SLEEP_DATA_DIR"] = str(raw_root)
    result = subprocess.run(
        [sys.executable, "-m", "ml.prepare_dataset", "--output-dir", str(output_dir), "--seed", "20260821"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "epochs.parquet").is_file()
    assert (output_dir / "splits.json").is_file()
    assert (output_dir / "dataset_manifest.json").is_file()
