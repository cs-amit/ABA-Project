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

MESA access is approved for this project. A checksum-verified, stratified
24-participant pilot from release 0.8.0 is stored locally and remains excluded
from Git. It contains actigraphy epochs, NSRR PSG stage annotations, ECG
R-points, and the official PSG/actigraphy overlap mapping; EDF files were not
downloaded.

The MESA adapter (`ml/prepare_mesa.py`) requires `match5`, `havepsg5`, and
`haveact5` to equal 1. It treats the official overlap line as PSG epoch zero,
maps stages 1 and 2 to light sleep, maps wake/stages 3-4/REM to not-light, and
excludes unknown or active stages. Its MESA-native features use activity count,
off-wrist/availability values, ECG-derived HR and IBI statistics, elapsed time,
and clock encodings. It does not fabricate the raw XYZ-derived features used by
the BIDSleep path.

The pilot uses seed 20260915 and a deterministic 16/4/4 participant split.
Training-only standardisation and model fitting are followed by validation-only
threshold selection and one frozen test evaluation. Samsung Health stages are
not ground-truth training labels.
