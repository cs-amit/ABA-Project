# Smart Sleep Alarm model card

## Status

No trained model is bundled in the Android application. The current BIDSleep
research candidate is the frozen MESA-pretrained CNN-GRU from the performance
recovery study (test F1 0.6497; participant-macro F1 0.6434). Separate
BIDSleep causal-time logistic and MESA-native benchmarks remain useful research
references. Their scores are not deployment accuracy, and cross-dataset scores
are not head-to-head comparisons because cohorts, sensors, feature schemas, and
held-out participants differ.

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

## BIDSleep performance recovery result

The validation-only recovery ladder regenerated corrected BIDSleep features,
weighted training participants equally, used local-time/log-activity shared
features, and selected checkpoints by participant-macro validation F1. The
MESA-pretrained candidate passed the predeclared validation advancement gate
and was frozen at threshold 0.371485. Its checkpoint SHA-256 was verified before
the one permitted evaluation on the nine locked BIDSleep test participants.

| Metric | Frozen test value |
| --- | ---: |
| Accuracy | 0.5999 |
| Balanced accuracy | 0.6160 |
| Precision | 0.5426 |
| Recall | 0.8093 |
| F1 | 0.6497 |
| ROC-AUC | 0.6736 |
| PR-AUC | 0.6043 |
| Brier score | 0.2307 |
| Participant-macro F1 | 0.6434 |
| Participant-macro balanced accuracy | 0.6140 |

Compared with the earlier MESA-transfer run on these same participants, pooled
F1 increased by 0.0062 and participant-macro F1 by 0.0060. The paired
participant bootstrap 95% interval was -0.0061 to 0.0182, so this is a small
exploratory point improvement, not evidence of a reliable population gain.
Participant-macro balanced accuracy decreased by 0.0030.

The frozen model exported to ONNX and matched PyTorch on the smoke sample with
maximum absolute probability difference 5.96e-08 (tolerance 1e-05). Android,
core, and Wear JVM tests passed, but the ONNX file remains an ignored research
artifact and is not bundled in the application. See
[`bidsleep-performance-recovery-report.md`](bidsleep-performance-recovery-report.md)
for the full protocol, calibration summary, and limitations.
