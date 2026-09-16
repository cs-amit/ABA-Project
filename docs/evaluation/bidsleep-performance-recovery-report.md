# BIDSleep performance recovery report

## Outcome

The predeclared validation ladder advanced `mesa_transfer`, a MESA-pretrained
causal CNN-GRU fine-tuned on the BIDSleep training participants. Its frozen
threshold was 0.371485 and its checkpoint SHA-256 was
`0dab6ddc31331a21a17011607b2bc510f684f8187fff924525104baae58932d5`.
The hash was verified before any test signal or label was opened.

The separate evaluator then created corrected features for exactly the nine
locked test participants and evaluated the checkpoint once. Its completion
record binds the candidate digest and test-result digest. The evaluator code
now fails closed on any second run and, for future freezes, binds the complete
train/validation/test allocation plus checkpoint and scaler bytes and
parameters before opening test data. No test row was used for feature/scaler
fitting, checkpoint selection, threshold selection, candidate ranking, or
retry decisions in this run.

## Validation selection

All candidates used the same 29 training and nine validation participants.
Ranking used participant-macro validation F1; advancement also required pooled
F1 non-regression and participant-macro balanced-accuracy loss no worse than
0.005 against the corrected native-logistic baseline.

| Candidate | Macro F1 | Pooled F1 | Macro balanced accuracy |
| --- | ---: | ---: | ---: |
| Native logistic | 0.6307 | 0.6261 | 0.5498 |
| Shared-feature control | 0.6545 | 0.6501 | 0.6223 |
| MESA transfer (frozen) | 0.6562 | 0.6530 | 0.6385 |

The shared models use ten causal 30-second epochs with activity count and
availability, heart-rate mean/standard deviation and availability, contiguous
session elapsed time, and America/New_York clock sine/cosine. Activity is
log1p-transformed and scaled with BIDSleep-training-only robust statistics.
Training gives each participant equal total loss mass, and checkpointing uses a
validation-selected participant-macro-F1 objective.

## Frozen test result

The corrected test artifact contained 36,365 labelled epochs and produced
35,740 causal sequences. Light-sleep prevalence across those evaluated causal
sequences was 0.4583.

| Metric | Test value |
| --- | ---: |
| Accuracy | 0.5999 |
| Balanced accuracy | 0.6160 |
| Precision | 0.5426 |
| Recall | 0.8093 |
| F1 | 0.6497 |
| ROC-AUC | 0.6736 |
| PR-AUC | 0.6043 |
| Brier score | 0.2307 |
| Prediction rate | 0.6836 |
| Participant-macro F1 | 0.6434 |
| Participant-macro balanced accuracy | 0.6140 |

The always-light reference had F1 0.6286 and balanced accuracy 0.5000;
always-rest had F1 0 and balanced accuracy 0.5000. Per-participant F1 ranged
from 0.5295 to 0.7152.

| Participant | Epoch sequences | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Bidslab07 | 6,079 | 0.6880 | 0.6455 |
| Bidslab15 | 2,178 | 0.6055 | 0.5878 |
| Bidslab20 | 4,398 | 0.7152 | 0.6727 |
| Bidslab22 | 4,059 | 0.5783 | 0.6258 |
| Bidslab31 | 4,794 | 0.6293 | 0.5952 |
| Bidslab42 | 4,032 | 0.5295 | 0.5555 |
| Bidslab51 | 3,336 | 0.6394 | 0.5904 |
| Bidslab65 | 2,558 | 0.6958 | 0.6153 |
| Bidslab66 | 4,306 | 0.7095 | 0.6376 |

## Comparison with the prior transfer run

This post-evaluation comparison is descriptive because the BIDSleep test set
was already known from earlier studies. Against the earlier MESA-transfer
checkpoint on the identical participants, pooled F1 increased from 0.6435 to
0.6497 (+0.0062), participant-macro F1 from 0.6374 to 0.6434 (+0.0060),
ROC-AUC from 0.6637 to 0.6736, PR-AUC from 0.5831 to 0.6043, and Brier score
improved from 0.2352 to 0.2307. Participant-macro balanced accuracy decreased
from 0.6170 to 0.6140 (-0.0030).

Six participants improved in F1 and three declined. The paired nine-participant
F1 delta was +0.0060; a 10,000-resample paired bootstrap 95% interval was
-0.0061 to 0.0182. The interval includes no improvement, so the measured bump
is not a reliable population-level claim.

## Calibration and deployment smoke

The Brier score was 0.2307. Calibration was closest in the middle probability
range: the 0.3-0.4 bin averaged 0.3511 probability with 0.3416 observed light
sleep, and the 0.4-0.5 bin averaged 0.4498 with 0.4316 observed. The model was
overconfident in higher bins; for example, the 0.8-0.9 bin averaged 0.8408 with
0.6773 observed. Calibration therefore remains a deployment risk.

The frozen PyTorch model exported to ONNX successfully. On a 32-sequence smoke
sample, maximum absolute PyTorch/ONNX probability difference was 5.96e-08,
well below the 1e-05 tolerance. The full Gradle JVM suite completed successfully
for the app, core, and Wear modules. No app behavior changed, and no model was
bundled: the generated checkpoint, corrected test features, predictions, and
ONNX file remain Git-ignored research artifacts.

## Limitations

- The nine-person test set is too small for a stable population or demographic
  conclusion, and prior access makes this exploratory rather than confirmatory.
- Validation advancement favored MESA transfer over the native baseline, but
  its macro-F1 margin over the shared-feature control was only 0.0017.
- MESA activity counts and ECG-derived heart rate differ from Watch raw motion
  and PPG; pretraining does not eliminate device and cohort shift.
- The corrected zero-crossing feature repairs the native artifact, but the
  selected shared-feature candidate does not consume raw XYZ-derived features.
- High-probability calibration is weak. A future calibration change would need
  a new validation-only protocol and a new untouched confirmatory cohort.
- ONNX numerical parity and JVM tests establish conversion/build compatibility,
  not on-device latency, battery behavior, sensor-domain accuracy, or safety.

The result supports retaining this checkpoint as the strongest current research
candidate. It does not support claiming clinical sleep staging or shipping the
model without prospective Watch validation and an explicit deployment review.
