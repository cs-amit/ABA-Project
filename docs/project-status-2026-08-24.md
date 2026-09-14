# Smart Sleep Alarm — Team Project Status

**Status date:** 24 August 2026  
**Current branch:** `feat/initial-smart-sleep-prototype`

## Finalized project definition

Smart Sleep Alarm is an Android and Wear OS wellness prototype for Samsung Galaxy Watch4 and later Samsung Wear OS watches. It uses watch motion, heart-rate, and inter-beat-interval signals to identify a *likely* light-sleep opportunity inside a user-selected wake window. It is not a medical device and does not diagnose sleep disorders or claim certainty about sleep stages.

The safety rule is non-negotiable: the phone's exact-time fallback alarm must fire even if the Watch, Bluetooth, transfer, storage, feature extraction, or inference fails.

## Finalized technical workflow

```text
Samsung Watch4 accelerometer + heart rate/IBI
  -> Samsung Health Sensor Service capture on the Watch
  -> ordered, versioned sensor batches staged in Watch-private storage
  -> Wear OS Data Layer transfer and acknowledgement/retry
  -> durable, sequence-keyed phone handoff
  -> future Room persistence and causal 30-second feature epochs
  -> future on-device sleep-probability inference and wake-window decision
  -> Watch haptic signal + phone audio, with exact-time phone fallback
```

All health and session data stay local by default. The prototype deliberately excludes cloud accounts, sync, analytics, advertising, microphone/audio analysis, substance/context logging, and schedule-advisor features.

## Completed work

### 1. Project foundation and core contracts

- Multi-module Gradle project created: phone app, Wear app, and pure Kotlin core module.
- Core alarm/session/sensor contracts implemented, including ordered `SensorBatch` validation and supported sensor capabilities.
- Samsung Health Sensor SDK AAR is used locally from `wear/libs/samsung-health-sensor-api.aar` and remains outside source control.

### 2. Safe phone fallback alarm

- Exact-time fallback is scheduled through Android `AlarmManager.setAlarmClock()`.
- Per-session one-shot alarm claiming prevents duplicate light-sleep/fallback triggers.
- Alarm notification, dismissal, and safe audio behaviour are implemented.
- Physical phone validation succeeded: fallback audio and notification fired, and dismissal stopped the alarm.

### 3. Watch4 sensor capture

- Samsung continuous accelerometer and heart-rate trackers are detected and used only when supported.
- Runtime `ACTIVITY_RECOGNITION` and `BODY_SENSORS` permissions are requested; no extra health-data permissions were added.
- Accelerometer axes, heart rate, IBI, off-body state, tracker/service failure cleanup, and bounded haptics are implemented.
- Watch controls are circular-safe and accessible.
- Physical Watch4 verification succeeded after enabling Health Platform/Health Sensor Service developer mode: both required tracker types are available and `SensorCaptureService` is running as a foreground health service.

### 4. Watch-to-phone ordered transfer (Task 5)

- Versioned binary `SensorBatchCodec` implemented with strict validation for malformed payloads, payload size, sample ordering, timestamps, nullable values, and sensor quality.
- Watch batches are split before the 30-second or 90 KiB limit, saved in Watch-private storage before upload, transferred through urgent Data Items, and retried after interruption.
- Watch acknowledgement markers survive sender restarts; malformed acknowledgements are ignored safely.
- Phone transport validates the Data Layer path and session ID, uses sequence-keyed ordered handoff, and acknowledges only after durable acceptance.
- The phone handoff is intentionally **at least once** until Task 6 persists each `{sessionId, sequence}` delivery transactionally and then calls `confirmPersisted()`.
- Focused Task 5 validation: 22 tests passed with zero failures/errors; phone and Wear production Kotlin compilation passed.

## In progress / pending

### Immediate hardware acceptance check

- Complete the Task 5 Bluetooth reconnect test: while Watch capture is active, turn phone Bluetooth off for two minutes, turn it back on, then confirm queued batches reach the phone in sequence.
- This test has started: the Watch correctly staged four durable batch files during the Bluetooth outage. Phone-side delivery confirmation is still pending because the phone's ADB transport became unstable during reconnection.
- No audio test is required for this check; phone fallback audio was validated separately.

### Next development tasks

1. **Task 6:** Room persistence and causal 30-second feature epochs. This task must transactionally deduplicate `{sessionId, sequence}` and call `confirmPersisted()` only after the successful database write.
2. **Task 7:** Public-dataset access/protocol and reproducible preprocessing.
3. **Task 8:** Train and compare baseline, CNN-GRU, and CNN-LSTM models; export the selected ONNX model.
4. **Task 9:** On-device inference and deterministic wake-window decisions.
5. **Task 10:** Complete phone session/history/settings screens and remaining Watch status UX.
6. **Task 11:** End-to-end failure matrix, overnight/accelerated tests, demo material, and final scope review.

## Known limitations and team decisions

- Current Watch UI has Start/Stop/Dismiss controls but does not yet display a clear live “capture active” state; this should be improved with Task 10 UX work.
- Samsung developer mode is for the owner's test Watch only. Public distribution requires Samsung partner registration and package/signature approval.
- Transport is deliberately designed for safe recovery rather than pretending to provide impossible effect-level exactly-once delivery before Room exists. Task 6 owns the final transactional idempotency boundary.
- Android Gradle Plugin warns that version 8.8.2 was tested through `compileSdk 35` while this project uses `compileSdk 36`; builds/tests currently pass, but the warning remains.

## Future / extended scope

The following ideas are part of the wider product direction but are deliberately outside the current core prototype. They should be attempted only after the sensor-to-alarm workflow and the full validation checklist are reliable.

### Personal calibration and model improvement

- Collect seven to ten consented personal nights to establish baseline normalization values.
- Compare the prototype output with Samsung Health sleep stages only as weak-label calibration analysis, never as independent clinical ground truth.
- Use a public, licence-approved dataset with compatible actigraphy, heart-rate/ECG-derived signals, and sleep-stage labels to train the baseline, CNN-GRU, and CNN-LSTM candidates.

### Sleep schedule advisor

- Let a user select a target sleep duration and receive a bedtime goal based on the target wake time and an adjustable wind-down buffer.
- After enough valid sessions, show a confidence-qualified likely favourable wake range from the user's own history.
- Never guarantee a future light-sleep time or present sleep advice as medical guidance.

### Optional context logging

- Offer an explicit, local-only pre-sleep check-in for broad timing/amount categories such as caffeine, alcohol, cannabis, nicotine, medication, exercise, and stress.
- Keep this data separate from the live light-sleep classifier until sufficient representative personal data and an evaluation plan exist.
- Present only cautious within-person descriptions with uncertainty; do not make causal or diagnostic claims.

### Optional local sound analysis

- Add only with explicit session-start consent.
- Process short audio chunks on-device and discard raw recordings.
- Classify only coarse events such as quiet, environmental noise, possible snore, or other sound.
- Never diagnose sleep apnea or present the feature as a medical assessment.

### Future device support

- Keep Samsung Watch4 and later Samsung Wear OS as the only supported hardware for this submission.
- Future Apple Watch, Fitbit, Garmin, or other Wear OS support must be implemented as separate `WearableDataSource` adapters that explicitly report their sensor capabilities before model use.

## GitHub visibility

The repository remote is [ABA-Project on GitHub](https://github.com/cs-amit/ABA-Project), but it currently exposes only the initial project commit. The completed Task 3–5 source work, build-artifact cleanup, and this report are local and uncommitted. Teammates will not see the current implementation through a GitHub link until the work is reviewed, committed, and pushed.

Do not commit or push until the project owner approves the exact files to include—especially because the original commit accidentally tracked build artifacts that are currently being removed from Git tracking.
