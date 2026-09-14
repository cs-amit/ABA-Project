# Public dataset preparation protocol

The primary training source is the open-access BIDSleep dataset: 47 healthy
participants and 253 nights with Apple Watch accelerometer and heart-rate data
and 30-second sleep-stage labels. Raw files are kept outside this repository in
`SMART_SLEEP_DATA_DIR`; no raw data or generated artifacts are committed.

The adapter expects normalized `accelerometer.csv`, `heart_rate.csv`, and
`labels.csv` exports (see `ml/features.py`), creates causal 30-second epochs,
and writes `epochs.parquet`, `splits.json`, and `dataset_manifest.json` under
`ml/artifacts`. Splits are deterministic and subject-disjoint. Unknown labels
are dropped and counted. `LIGHT` is label 1; `AWAKE`, `DEEP`, and `REM` are 0.

MESA access is optional future comparison work only. No MESA data has been
downloaded or used. Samsung Health stages are not ground-truth training labels.
