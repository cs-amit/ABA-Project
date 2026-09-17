# Final demo runbook

## Before the demo

Use the owner-controlled Watch4 only. Follow
[`watch4-setup.md`](watch4-setup.md), then build and install the phone and Wear
debug apps:

```powershell
.\gradlew.bat :app:installDebug :wear:installDebug
```

The app demo covers local Watch capture, ordered transfer, persistence, and the
exact-time fallback alarm. It does not load the offline research ONNX model.

## Demo sequence

1. Start a session and show capture active on the Watch.
2. Let several batches transfer to the phone.
3. Disable phone Bluetooth for two minutes while capture continues.
4. Re-enable Bluetooth and show queued batches arriving in sequence.
5. Restart the phone app and show durable replay without duplicate acceptance.
6. Demonstrate the exact-time fallback alarm with Watch data unavailable.

Record screenshots, logcat, batch sequence numbers, and pass/fail results.

## Evidence to present

- Android behavior: this runbook and the JVM test result.
- Offline ML evidence: [model card](../evaluation/model-card.md), [MESA report](../evaluation/mesa-expansion-transfer-report.md), and [BIDSleep recovery report](../evaluation/bidsleep-performance-recovery-report.md).
- Do not present ignored checkpoints, ONNX files, raw datasets, tokens, or
  nine-person test scores as deployment accuracy.

## ML verification

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests -q
.\gradlew.bat test
```

The frozen-test evaluator is fail-closed and one-shot. Do not rerun it for a
demo or to search for a better score.
