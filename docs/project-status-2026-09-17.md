# Smart Sleep Alarm — current handoff status

**Branch:** `feat/mesa-pilot-model`  
**Remote:** `origin/feat/mesa-pilot-model` at `323f39d`

## Ready

- Watch capture, ordered transfer, Room persistence, causal feature epochs,
  fallback alarm, and failure-safe transport are implemented and covered by
  the JVM/Android test suites.
- The validation-only MESA transfer study is complete. The current research
  candidate reached validation participant-macro F1 `0.6562`; its one permitted
  exploratory BIDSleep test evaluation reached participant-macro F1 `0.6434`.
- The checkpoint exported to ONNX with maximum PyTorch/ONNX probability delta
  `5.96e-08`. A 31 KB copy is bundled in the phone APK for the classroom MVP;
  it remains exploratory and is not a clinical/deployment claim.
- Verification completed: 153 ML tests and the full Gradle app/core/Wear suite.

## Owner hardware checks still required

1. Install the debug app/Wear builds on the owner-controlled Watch4 and phone.
2. Capture data, disable Bluetooth for two minutes, re-enable it, and verify
   queued batches arrive in sequence.
3. Restart the phone app and verify durable replay without duplicate batches.
4. Test unavailable sensors and confirm the app stays safe and does not guess.
5. Confirm the exact-time fallback alarm fires when Watch data or inference is
   unavailable.

Record screenshots, logcat, batch sequence numbers, and pass/fail outcomes.
These physical checks are the remaining evidence gap; further tuning on the
already-used BIDSleep test set would not be a reliable improvement.
