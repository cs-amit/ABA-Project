# Smart Sleep Alarm model card

## Status

No trained model is bundled in the Android application. Two experimental
dataset-specific results are available: the completed BIDSleep causal-time
logistic benchmark (test F1 0.6409) and the MESA pilot described below. Their
scores are not a head-to-head model comparison because the cohorts, sensors,
feature schemas, and held-out participants differ.

## Intended use

The experimental models estimate likely light sleep from the preceding ten causal
30-second wearable feature epochs. It is a wellness estimate only, not a
sleep-stage diagnosis or a disorder assessment. The phone fallback alarm is
independent of inference and always remains available.

## MESA pilot data

- Dataset: MESA Sleep release 0.8.0.
- Pilot: 24 stratified participants and 31,274 labelled epochs.
- Split: 16 train, 4 validation, and 4 test participants; seed 20260915.
- Inputs: 30-second activity count and availability, off-wrist state,
  ECG-R-point-derived HR/IBI features, elapsed time, and clock encodings.
- Labels: PSG stages 1/2 are light; wake, stages 3/4, and REM are not-light.
- Alignment: official MESA PSG/actigraphy overlap line; no clock-time guessing.
- Model: train-standardised, class-balanced logistic regression over ten causal
  epochs; threshold 0.506699 selected on validation participants only.

## Frozen pilot result

| Metric | Test value |
| --- | ---: |
| Accuracy | 0.7686 |
| Balanced accuracy | 0.7695 |
| Precision | 0.7062 |
| Recall | 0.9145 |
| F1 | 0.7970 |
| ROC-AUC | 0.8160 |
| PR-AUC | 0.7254 |
| Brier score | 0.1634 |

Test prevalence was 0.4967. The always-light baseline had F1 0.6637 and
balanced accuracy 0.5000; always-rest had F1 0 and balanced accuracy 0.5000.
The four test participants had individual F1 values from 0.7368 to 0.8466.

## Limitations

This is an exploratory 24-participant result and does not establish an F1
improvement for the deployed Galaxy Watch model. MESA actigraphy provides
activity counts rather than raw XYZ acceleration, its ECG differs from Watch
PPG, and its older cardiovascular cohort may not represent intended users.
The strong MESA result justifies a larger selectively downloaded comparison;
it does not justify pooling MESA and BIDSleep without harmonisation. No MESA
model has been exported to ONNX or bundled in the app.

## MESA expansion and transfer outcome

The completed 500-participant MESA benchmark and exploratory shared-feature
transfer comparison are documented in
[`mesa-expansion-transfer-report.md`](mesa-expansion-transfer-report.md).
The MESA-native test F1 was 0.6925. MESA pretraining's participant-macro F1
point estimate was 0.6374 versus 0.6348 for BIDSleep-only, but its paired 95%
interval included harm. MESA is therefore retained as a separate research
benchmark; there is no claim of Galaxy Watch or deployment improvement.
