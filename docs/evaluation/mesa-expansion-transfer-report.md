# MESA expansion and exploratory transfer report

## Scope

This is a research benchmark. No MESA or transfer model is exported to ONNX or
bundled in the Android application. Scores are not deployment accuracy or a
clinical sleep-stage claim.

## MESA-500 cohort

Using MESA Sleep release 0.8.0, the deterministic seed `20260916` selected 500
participants after excluding the 24 pilot IDs. Selection used only race/ethnicity,
gender, and age band, plus availability of concurrent-study flags, all three
permitted source files, and the official actigraphy/PSG overlap mapping. The
frozen split is 350 train, 75 validation, and 75 test participants.

Only actigraphy CSV, NSRR PSG-event XML, and ECG R-point CSV files were
retrieved: 1,500 files and 2,629,389,426 server-reported bytes. Every selected
file passed its server MD5 and size check; no EDF files or partial downloads were
used. Preparation yielded 629,646 labelled 30-second epochs. All splits contain
both labels, with zero duplicate participant/time keys, cross-split subjects, or
non-finite features. Stage 1/2 is light; wake, stages 3/4, and REM are not-light.

The adapter uses the official overlap line, activity count/availability and
off-wrist state, R-point-derived HR/IBI features, elapsed time, and clock
encodings. Nine source R-point exports had unordered rows; they were stable-sorted
by their recording-relative timestamps before adjacent-beat interval calculation,
after checking every timestamp belongs to its declared right-closed 30-second
epoch. This prevents artificial negative intervals.

## Frozen MESA-native baseline

The predeclared ten-epoch, class-balanced logistic baseline used training-only
standardisation and selected its threshold on the 75 validation participants
only. Threshold: 0.417156; validation F1: 0.707998.

| Metric | Held-out MESA test |
| --- | ---: |
| Accuracy | 0.6645 |
| Balanced accuracy | 0.7081 |
| Precision | 0.5499 |
| Recall | 0.9350 |
| F1 | 0.6925 |
| ROC-AUC | 0.7612 |
| PR-AUC | 0.5896 |
| Brier score | 0.1948 |

Test light-sleep prevalence was 0.4040. Per-participant F1 ranged from 0.1493
to 0.8883 (median 0.7076). This measures MESA adapter stability, not a Watch
improvement.

## Exploratory BIDSleep transfer comparison

Both conditions use the same causal CNN-GRU (hidden size 32) over ten epochs and
the same BIDSleep participant splits. The shared feature order is activity count,
activity availability, HR mean, HR standard deviation, HR availability,
elapsed-session time, clock sine, and clock cosine. Each dataset has a separate
robust scaler fitted on its training participants only. MESA XYZ statistics and
BIDSleep IBI values were not fabricated, and dataset identity is not a feature.

The control trains on BIDSleep only. The transfer condition pretrains on MESA
train participants, then fine-tunes on the identical BIDSleep train participants.
Early stopping and thresholds use BIDSleep validation only; each frozen condition
was evaluated once on the existing BIDSleep test participants.

| Metric | BIDSleep-only | MESA pretrain + BIDSleep fine-tune |
| --- | ---: | ---: |
| Validation threshold | 0.3611 | 0.3965 |
| Test F1 | 0.6393 | 0.6435 |
| Test balanced accuracy | 0.5844 | 0.6195 |
| Test ROC-AUC | 0.6385 | 0.6637 |
| Test PR-AUC | 0.5606 | 0.5831 |
| Test Brier score | 0.2435 | 0.2352 |
| Participant-macro F1 | 0.6348 | 0.6374 |
| Participant-macro balanced accuracy | 0.5841 | 0.6170 |

The paired nine-participant F1 delta (transfer minus control) was +0.0026, with
a 10,000-resample paired bootstrap 95% interval of -0.0097 to 0.0152. Although
the point estimates for macro F1, balanced accuracy, and calibration improved,
the F1 interval includes harm and no demographic-subgroup conclusion is possible.
Therefore transfer is **not promising** under the predeclared gate, and MESA
remains a separate research benchmark.

This comparison is exploratory because BIDSleep test results were previously
known. MESA actigraphy is not Watch raw XYZ acceleration, MESA ECG R-points are
not Watch PPG, and the MESA cohort/device/population differs from intended users.
The nine-person BIDSleep test set also cannot establish robust demographic
subgroup conclusions.
