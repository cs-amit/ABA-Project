# Causal Deep Sleep Model Design

## Goal

Improve the project’s wearable-only light-sleep estimate without changing the existing BIDSleep source data. The deliverable is a reproducible causal CNN-GRU/LSTM training path and an evaluation protocol that prevents misleading F1-only results.

## Constraints

- Preserve BIDSleep raw files and its subject-held-out allocation.
- Inputs at inference time are only current or past wearable motion and heart-rate values plus availability masks; no PSG, future samples, or derived clinical labels.
- Keep every night for a subject in that subject’s existing split.
- Fit normalization, threshold selection, and architecture selection only with train/validation data.
- Treat Samsung Health stages as weak calibration labels, never ground truth.
- Do not claim clinical sleep staging or promise an accuracy improvement.

## Architecture

The preparation layer creates fixed, causal raw windows ending at a labelled 30-second epoch. Each epoch contains resampled accelerometer XYZ, HR, and availability channels. A window builder concatenates only contiguous current-and-past epochs, breaks at a configurable time gap, and emits the label and existing split of the final epoch.

The model path is a small temporal CNN followed by either a unidirectional GRU or LSTM. Raw samples are encoded within each epoch, then the recurrent layer models several prior epochs. Availability and sample-age information remain model inputs rather than becoming zero-filled sensor values. Classical logistic regression remains a required baseline.

## Evaluation

Reports must include always-rest and always-light baselines, accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, prediction rate, and per-subject summaries. Thresholds are selected on validation only. A candidate advances only if subject-held-out validation evidence improves balanced accuracy without a material precision or calibration regression; the held-out test is reserved for a frozen candidate.

## Data-quality safeguards

Derived artifacts carry a feature-schema version and record exactly which features were available. Session elapsed time resets at each contiguous night/session, never at a participant’s first-ever recording. Constant or unavailable signals are reported rather than silently interpreted as physiology.

## External data

DREAMT 2.2.0 is an optional separate adapter, not a replacement for BIDSleep. Its wearable-only 64 Hz data may be downloaded and used only after its restricted-use agreement is accepted. The first implementation works with the local BIDSleep artifact and does not require downloading or mixing external data.

## Verification

Unit tests prove causal windows never cross a gap, sessions reset at a gap, and model inputs/shapes are valid. The Python ML test suite is run in an environment satisfying `ml/requirements.txt`; dependency absence is reported rather than worked around.
