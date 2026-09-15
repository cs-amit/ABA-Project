# MESA Pilot Model Design

## Objective

Add a reproducible MESA Sleep preprocessing and evaluation path that determines whether MESA can improve light-sleep classification beyond the existing BIDSleep benchmark. The first deliverable is a valid 24-participant pilot benchmark, not a claim that MESA improves F1.

## Scope

The pilot uses only the selectively downloaded MESA files:

- 30-second actigraphy epoch CSVs;
- NSRR PSG annotation XML files;
- ECG R-point CSVs;
- the official PSG/actigraphy overlap mapping;
- the MESA 0.8.0 phenotype dataset and pilot manifest.

EDF files remain out of scope. The pipeline must not require or download them. Expanding beyond 24 participants is a separate step after the pilot passes its data-quality and evaluation gates.

## Approaches Considered

### Recommended: separate MESA adapter and common-feature benchmark

Build a dedicated adapter that emits the existing tabular epoch contract where features are genuinely compatible, and explicitly records unavailable raw-accelerometer features. Train a MESA-specific baseline and compare its participant-held-out metrics with the existing BIDSleep results. This gives the cleanest evidence without pretending activity counts are raw XYZ acceleration.

### Alternative: pool MESA and BIDSleep immediately

This offers more participants but is rejected for the pilot. MESA activity counts and BIDSleep raw acceleration features have different meanings and scales; naive pooling could create a dataset/device classifier and make any F1 change misleading.

### Alternative: train only from MESA R-points

This avoids motion incompatibility but discards the wearable movement signal and is less representative of the deployed Watch model. It may be retained later as an ablation, not the primary pilot.

## Inputs and Alignment

Each participant must satisfy `match5 == 1`, `havepsg5 == 1`, and `haveact5 == 1` and must have all required pilot files.

The official overlap CSV provides the actigraphy `line` corresponding to PSG time zero. PSG stage events and R-point `seconds` are relative to that start. For epoch `e`, the adapter joins:

- actigraphy row `overlap_line + e`;
- PSG interval `[30e, 30(e + 1))`;
- normal ECG beats (`Type == 1`) whose `seconds` fall in the same interval.

The adapter must fail closed for a missing participant mapping, duplicate mapping, non-30-second actigraphy cadence, discontinuous actigraphy lines, malformed stage duration, or conflicting labels. It must never guess an alignment from clock time alone.

## Labels

PSG stages map to the binary target as follows:

- Stage 1 and Stage 2: `light = 1`;
- Wake, Stage 3/4, and REM: `light = 0`;
- Active/unknown/unscored values: excluded and counted in the manifest.

Stage event durations must be exact multiples of 30 seconds. Contiguous XML stage events are expanded into one label per 30-second epoch.

## Features

The primary MESA pilot feature set is deliberately dataset-native and causal:

- current and prior activity count;
- activity availability/on-wrist indicator;
- heart-rate mean and standard deviation derived from R-point intervals;
- mean IBI and RMSSD derived from consecutive normal R-points;
- heart-rate availability ratio;
- causal elapsed-session and clock-time encodings where available without future information.

Raw XYZ acceleration magnitude, magnitude dispersion, median absolute deviation, and zero-crossing rate are unavailable in MESA actigraphy and must not be fabricated. The MESA feature manifest records the exact feature order and formulas independently of the Android/BIDSleep nine-feature contract.

## Outputs

The adapter writes ignored local artifacts beneath `ml/artifacts/mesa_pilot/`:

- `epochs.parquet`, one row per aligned labelled epoch;
- `splits.json`, deterministic participant assignments;
- `dataset_manifest.json`, versions, source files, checksums, mapping, exclusions, feature definitions, seed, and class counts;
- `metrics.json`, validation-selected threshold and frozen test metrics;
- `per_subject_metrics.csv`, test metrics by participant.

No token, controlled raw data, participant-level raw record, or derived artifact is committed to Git.

## Evaluation

Use deterministic stratified participant-level train/validation/test splits. No participant may occur in multiple splits. Standardisation, class weights, hyperparameter selection, and threshold selection use training/validation participants only.

The first model is scaled class-weighted logistic regression. Report accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, prediction rate, confusion counts, class prevalence, and always-light/always-rest baselines. The held-out test split is evaluated once after the pilot configuration is frozen.

The pilot answers whether the adapter is viable and whether MESA contains predictive signal. With only 24 participants, its result is exploratory and cannot establish a reliable improvement over BIDSleep test F1 `0.6409`. A larger MESA comparison is authorized only after the pilot produces valid alignment, reasonable coverage, and metrics above the dummy baselines.

## Testing and Verification

Fixture tests must cover:

- overlap-line alignment and 30-second offset arithmetic;
- expansion and mapping of compressed XML stage events;
- HR, IBI, and RMSSD calculations from known R-points;
- off-wrist and missing-data handling;
- rejection of missing/duplicate mappings, line gaps, cadence errors, malformed durations, and unknown stages;
- subject split isolation and deterministic output;
- manifest contents and exclusion counts.

After unit tests pass, run the adapter on all 24 pilot participants and verify row counts, participant counts, label prevalence, feature missingness, checksum-backed source coverage, and zero cross-split participant leakage before training.

## Documentation

Update the MESA data documentation and model card to distinguish proposed, downloaded, prepared, and evaluated states. Any reported metric must identify the dataset release, pilot participant count, split seed, feature set, and exploratory limitation.
