# BIDSleep Performance Recovery Design

## Objective

Improve BIDSleep light-sleep classification over the existing exploratory
baseline without selecting, tuning, or inspecting new BIDSleep test results.

## Evidence

The saved BIDSleep epochs artifact has an all-zero
`accel_zero_crossing_rate`, while recomputation from the same raw signal using
the current feature implementation produces nonzero values. Transfer training
also checkpoints pooled validation F1 at threshold 0.5 although the decision
criterion is participant-macro F1 after validation threshold selection. The
MESA clock features use local recording time, whereas BIDSleep transfer clock
features currently use UTC. Finally, raw activity counts have a long tail that
is not controlled by the current robust scaling when its median/IQR are zero.

## Design

Create a validation-only experiment runner that accepts frozen BIDSleep subject
splits and never loads the test rows into feature fitting, model fitting,
checkpoint selection, threshold selection, or experiment ranking. Regenerate a
new ignored BIDSleep epoch artifact from the approved raw source using current
feature code and preserve exact subject assignments.

The runner evaluates a small declared ladder: the corrected native feature
artifact, participant-balanced loss, session-local time features, log1p
activity, and objective-aligned checkpoint selection. It ranks candidates by
validation participant-macro F1; pooled F1 cannot regress and macro balanced
accuracy cannot regress by more than 0.005. Every candidate uses the same train
and validation participants. Test evaluation is a separate command permitted
only after a stored candidate configuration is frozen.

For shared MESA/BIDSleep transfer, BIDSleep clock sine/cosine is derived in
America/New_York local recording time, matching the MESA local actigraphy
semantics. Activity uses log1p before its training-only robust scaling. The
control and transfer training loops select checkpoints by validation
participant-macro F1 at a validation-selected threshold and weight subjects
equally rather than weighting them by recording length.

## Guardrails

- The BIDSleep test split remains inaccessible to experiment fitting and rank
  selection.
- No test prediction is generated until a candidate config and checkpoint are
  frozen in a validation artifact.
- New artifacts, raw data, checkpoints, predictions, and token files remain
  ignored by Git.
- The final result remains exploratory because prior BIDSleep test results are
  known.
