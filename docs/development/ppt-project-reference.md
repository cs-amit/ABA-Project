# Smart Sleep Alarm — presentation reference

This file is a practical source of truth for preparing the project presentation. It combines the story, methods, experiment results, final Android/Wear implementation, verification evidence, and pointers to the detailed artifacts.

## 1. One-minute project summary

Smart Sleep Alarm is a classroom MVP that captures accelerometer and heart-rate data from a Samsung Galaxy Watch, transfers it reliably to an Android phone, converts the stream into causal 30-second feature epochs, and runs a frozen ONNX sleep/light-sleep classifier. The phone also schedules a deterministic fallback alarm, so the alarm path remains available if live sensing or inference is unavailable.

The current research candidate is a MESA-pretrained CNN-GRU fine-tuned on BIDSleep. It is exploratory, not a clinical sleep-stage system. The positive class is N1/N2 light sleep versus all other stages grouped as rest/not-light sleep.

## 2. Suggested slide order

1. Problem and classroom objective
2. User journey and final demo
3. System architecture
4. Watch sensing and Samsung Health integration
5. Reliable watch-to-phone transport
6. Phone persistence and causal feature engineering
7. ML data sources and label definition
8. MESA preparation and native benchmark
9. BIDSleep recovery and MESA-transfer experiment
10. Candidate selection, threshold, and limitations
11. ONNX export and Android inference contract
12. Live dashboard, alarm controls, and demo mode
13. Hardware verification and debugging fixes
14. Results, risks, and next steps

## 3. User journey / final working flow

1. Pair the Watch4 and phone using Samsung Wearable/ADB as needed.
2. Grant Samsung sensor permissions and notification permission once.
3. Start a capture session from the Wear app/service.
4. Watch collects continuous accelerometer and heart-rate/IBI samples.
5. Watch batches samples, stores pending batches locally, and transfers them through the Wear Data Layer.
6. Phone receives batches, deduplicates by `(session_id, sequence)`, persists them in Room, and acknowledges the watch.
7. The phone feature pipeline emits completed 30-second epochs.
8. Each epoch is summarized in the Live tab; valid epochs fill a 10-epoch inference window.
9. After 10 valid epochs, the phone invokes the bundled ONNX model and displays the probability with two decimal places.
10. The Alarm tab lets the user select a time, choose a 15/30/45-minute wake window, enable/disable the fallback alarm, and cancel it.
11. The Live-tab demo button runs a deterministic synthetic 10-epoch scenario through the same inference path. It is visibly marked as demo data and never sounds or schedules a real alarm.

## 4. Architecture

```text
Samsung Watch sensors
        │
        ▼
SamsungHealthSensorDataSource
  (accelerometer + continuous HR/IBI)
        │
        ▼
SensorBatchAccumulator → durable pending batch files
        │ Wear Data Layer
        ▼
WearableReceiver / PhoneBatchInbox
        │
        ▼
RoomSessionRepository
  ├─ persisted batches + acknowledgements
  ├─ feature epochs + pipeline checkpoint
  └─ prediction events
        │
        ▼
FeaturePipeline (causal 30-second epochs)
        │
        ▼
LiveInferenceProcessor (10 contiguous valid epochs)
        │
        ▼
SleepInferenceAdapter → ONNX Runtime → probability
        │
        ▼
DashboardState → Live / Alarm / History UI

AlarmManager.setAlarmClock → exact-time phone fallback
```

Main modules:

- `core/`: shared models, feature extraction, inference contract, and tests.
- `wear/`: Samsung Health Sensor SDK integration, capture service, batching, resend/acknowledgement logic.
- `app/`: Data Layer receiver, Room database/repository, ONNX bridge, alarm receiver, and phone UI.
- `ml/`: MESA/BIDSleep preparation, transfer training, validation-only selection, evaluation, and export tooling.

## 5. Watch sensing

The Watch implementation uses Samsung Health Sensor SDK continuous trackers:

- `HealthTrackerType.ACCELEROMETER_CONTINUOUS`
- `HealthTrackerType.HEART_RATE_CONTINUOUS`

Accelerometer points preserve XYZ values. Heart-rate points preserve BPM, optional IBI, timestamp, and mapped sensor quality. Samsung status `1` means a successful measurement; status `0` is initial measuring, and `-3` is off-body. The corrected mapper is tested in `wear/src/test/java/com/aba/smartsleep/wear/sensor/SamsungSampleMapperTest.kt`.

Important device behavior discovered during testing: Samsung may buffer heart-rate points while the watch screen is off. `ActiveTrackerFlusher` now calls the active tracker’s `flush()` every five seconds during capture, and stops on capture stop/failure/close. This keeps HR timestamps close enough to the live accelerometer stream for epoch construction.

Relevant files:

- `wear/src/main/java/com/aba/smartsleep/wear/sensor/SamsungHealthSensorDataSource.kt`
- `wear/src/main/java/com/aba/smartsleep/wear/sensor/ActiveTrackerFlusher.kt`
- `wear/src/main/java/com/aba/smartsleep/wear/sensor/SensorCaptureService.kt`
- `wear/src/main/java/com/aba/smartsleep/wear/transport/WearBatchSender.kt`

## 6. Reliable transport and persistence

Each watch batch has a safe session ID and monotonic sequence number. Before upload, the watch writes the encoded payload to private pending storage. The phone inserts `(session_id, sequence)` with `insertIgnore`, so retransmissions do not duplicate data. The phone writes durable Room state before acknowledging the watch. A retry coordinator replays pending acknowledgements after transient failures.

The transport is designed to survive temporary Bluetooth/Data Layer loss:

- watch pending files provide local durability;
- the phone inbox/recovery path replays persisted Data Items;
- sequence-key deduplication prevents duplicate Room rows;
- acknowledgements are retried until durable delivery is confirmed.

Relevant files:

- `wear/src/main/java/com/aba/smartsleep/wear/transport/BatchTransferPolicy.kt`
- `wear/src/main/java/com/aba/smartsleep/wear/transport/WearBatchSender.kt`
- `app/src/main/java/com/aba/smartsleep/app/transport/WearableReceiver.kt`
- `app/src/main/java/com/aba/smartsleep/app/transport/PhoneBatchDataLayerRecovery.kt`
- `app/src/main/java/com/aba/smartsleep/app/data/RoomSessionRepository.kt`
- `app/src/main/java/com/aba/smartsleep/app/data/AppDatabase.kt`

## 7. Causal feature engineering

The phone groups samples into fixed 30-second epochs. It never uses future samples when completing an epoch. The pipeline stores a Room checkpoint containing the open epoch, last timestamp, pending samples, and prior valid raw values, allowing process restart/replay without losing causal state.

An epoch is inference-valid only when it has no off-body flag, at least 80% valid modality ratio, at least 24 occupied accelerometer seconds, at least 12 occupied heart-rate seconds, at least 24 seconds of coverage, and no gap over five seconds. The first sufficient epoch establishes the normalization baseline; the inference window then requires ten contiguous valid epochs.

The raw extraction contract contains:

- accelerometer magnitude mean, standard deviation, median absolute deviation;
- activity count and zero-crossing rate;
- heart-rate mean and standard deviation;
- IBI mean and RMSSD;
- accelerometer and heart-rate valid-sample ratios;
- off-body flag.

The deployed shared model consumes eight values per epoch:

1. activity count;
2. activity availability;
3. heart-rate mean;
4. heart-rate standard deviation;
5. heart-rate availability;
6. elapsed session hours;
7. India-local clock sine;
8. India-local clock cosine.

The Android live model clock is now `Asia/Kolkata`, matching the owner’s deployment context. The timezone regression is in `app/src/test/java/com/aba/smartsleep/app/inference/LiveInferenceProcessorTest.kt`.

Relevant files:

- `core/src/main/kotlin/com/aba/smartsleep/core/features/FeaturePipeline.kt`
- `core/src/main/kotlin/com/aba/smartsleep/core/inference/SleepInferenceAdapter.kt`
- `app/src/main/java/com/aba/smartsleep/app/inference/LiveInferenceProcessor.kt`

## 8. ML data and labels

The target is wearable-only light-sleep estimation. In BIDSleep, EEG-scored N1/N2 are grouped as the positive/light-sleep class. Wake, N3/deep sleep, REM, and other stages are grouped as not-light/rest.

BIDSleep provides Apple Watch accelerometer and instantaneous heart-rate data aligned to 30-second EEG-derived labels. MESA provides a separate actigraphy/ECG-based research source. The project does not mix raw datasets naively: MESA activity counts and ECG-derived R-point features have different semantics from Watch XYZ/PPG features.

Dataset protocols and label explanation:

- `docs/data/public-dataset-protocol.md`
- `docs/data/BIDSleep-Dataset-and-ML-Implementation-Plan.md`
- `docs/evaluation/sleep-model-problem-brief-2026-08-31.md`

## 9. MESA preparation and native benchmark

MESA Sleep release `0.8.0` was selectively expanded to 500 participants using deterministic seed `20260916`. Selection excluded the 24 pilot IDs and used only declared demographic/availability criteria plus the official actigraphy/PSG overlap mapping.

Downloaded source scope was deliberately selective rather than the approximately 360 GB EDF collection:

- actigraphy CSV;
- NSRR PSG-event XML;
- ECG R-point CSV.

Exactly 1,500 files and 2,629,389,426 server-reported bytes were retrieved. Every selected file passed size/MD5 checks. No EDF files or uncontrolled partial downloads were used. Preparation yielded 629,646 labelled 30-second epochs with both labels in every split, no participant leakage, and no duplicate participant/time keys.

The MESA-native class-balanced logistic baseline used training-only standardization and selected its threshold on the 75-person validation split only.

Held-out MESA test result:

| Metric | Value |
|---|---:|
| Accuracy | 0.6645 |
| Balanced accuracy | 0.7081 |
| Precision | 0.5499 |
| Recall | 0.9350 |
| F1 | 0.6925 |
| ROC-AUC | 0.7612 |
| PR-AUC | 0.5896 |
| Brier score | 0.1948 |

This is a research benchmark, not proof of Watch performance.

Detailed artifact: `docs/evaluation/mesa-expansion-transfer-report.md`.

## 10. BIDSleep recovery and MESA transfer experiment

The predeclared validation ladder compared:

- native logistic baseline;
- shared-feature BIDSleep control;
- MESA-pretrained CNN-GRU fine-tuned on BIDSleep training participants.

All candidates used participant-level isolation, ten causal 30-second epochs, training-only scaling, equal participant loss weighting, and validation-only threshold/checkpoint selection. The BIDSleep test split was opened only once for the frozen evaluation; later comparisons are explicitly exploratory because prior test results were already known.

Validation selection:

| Candidate | Participant-macro F1 | Pooled F1 | Macro balanced accuracy |
|---|---:|---:|---:|
| Native logistic | 0.6307 | 0.6261 | 0.5498 |
| Shared-feature control | 0.6545 | 0.6501 | 0.6223 |
| MESA transfer | **0.6562** | **0.6530** | **0.6385** |

Frozen exploratory BIDSleep test result for the selected transfer candidate:

| Metric | Value |
|---|---:|
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

The transfer improvement over the earlier BIDSleep-only condition was small: participant-macro F1 increased from 0.6374 to 0.6434, with a paired bootstrap 95% interval of `-0.0061` to `0.0182`. The interval includes no improvement, so this is not a reliable population-level claim.

Detailed artifacts:

- `docs/evaluation/bidsleep-performance-recovery-report.md`
- `docs/evaluation/mesa-expansion-transfer-report.md`
- `docs/evaluation/model-card.md`
- `ml/README.md`
- `ml/performance_experiment.py`
- `ml/transfer_experiment.py`
- `ml/transfer_features.py`
- `ml/mesa_baseline.py`

## 11. ONNX deployment

The selected frozen checkpoint was exported to ONNX. The Android adapter enforces the exact eight-feature, ten-epoch tensor contract, applies the frozen log1p/robust-scaling parameters, rejects invalid/non-finite input, and fails closed when the model/runtime is unavailable.

Deployment constants:

- sequence length: 10 epochs;
- epoch length: 30 seconds;
- feature count: 8;
- validation-selected alarm threshold: `0.37148505` (37.1485%);
- bundled asset: `app/src/main/assets/models/mesa_transfer.onnx`;
- bundled model size: approximately 31 KB;
- checkpoint SHA-256: `0dab6ddc31331a21a17011607b2bc510f684f8187fff924525104baae58932d5`;
- maximum PyTorch/ONNX probability difference: `5.96e-08` on the smoke sample.

The threshold is a classification threshold, not a clinically calibrated confidence level. The model is optimized for F1, and calibration remains a deployment risk.

Relevant files:

- `core/src/main/kotlin/com/aba/smartsleep/core/inference/SleepInferenceAdapter.kt`
- `app/src/main/java/com/aba/smartsleep/app/inference/OnnxProbabilityModel.kt`
- `app/src/main/java/com/aba/smartsleep/app/inference/LiveInferenceProcessor.kt`
- `app/src/main/assets/models/mesa_transfer.onnx`

## 12. Phone UI and alarm behavior

The phone app has three stable tabs:

### Live

- watch connection and transport counts;
- latest formatted 30-second epoch;
- movement and heart-rate summaries;
- motion/heart coverage percentages;
- valid inference progress from `0/10` to `10/10`;
- latest ONNX probability shown with two decimal places;
- conditional notification-permission action (hidden once granted);
- safe synthetic 10-epoch demo button.

### Alarm

- target time picker;
- 15/30/45-minute wake-window selector;
- persistent enable/disable switch;
- real `AlarmManager` cancellation when switched off;
- exact-time phone fallback scheduling.

### History

- current session identifier;
- samples, batches, and epochs captured;
- latest prediction;
- local-storage explanation.

The current exact-time alarm path is implemented through `AlarmManager.setAlarmClock`, `AlarmReceiver`, `AlarmCoordinator`, and phone audio/notification output. The live ONNX probability is currently informational; the demo explicitly does not schedule or sound an alarm. A future change would be required to connect a live `LIGHT_SLEEP` result to wake-window alarm triggering.

Relevant files:

- `app/src/main/java/com/aba/smartsleep/app/MainActivity.kt`
- `app/src/main/java/com/aba/smartsleep/app/DashboardUiLogic.kt`
- `app/src/main/java/com/aba/smartsleep/app/SmartSleepApplication.kt`
- `app/src/main/java/com/aba/smartsleep/app/alarm/AlarmScheduler.kt`
- `app/src/main/java/com/aba/smartsleep/app/alarm/AlarmReceiver.kt`
- `app/src/main/java/com/aba/smartsleep/app/alarm/AlarmCoordinator.kt`

## 13. Demo mode

The Live tab’s **Run 10-epoch demo** button creates ten contiguous, valid synthetic epochs with a sleep-like profile: zero activity count, HR 60 bpm, HR SD 1 bpm, and full sensor availability. These epochs go through `DemoInferenceRunner → LiveInferenceProcessor → SleepInferenceAdapter → bundled ONNX`.

The UI labels the result `DEMO DATA — NOT FROM WATCH`, shows `10/10`, reports whether the result crossed the 37.15% threshold, and says `no real alarm scheduled`. This is suitable for a classroom presentation because it proves the end-to-end inference path without waiting five minutes or making a real alarm ring.

Files/tests:

- `app/src/main/java/com/aba/smartsleep/app/inference/DemoInferenceScenario.kt`
- `app/src/test/java/com/aba/smartsleep/app/inference/DemoInferenceScenarioTest.kt`
- `app/src/androidTest/java/com/aba/smartsleep/app/inference/DemoOnnxIntegrationTest.kt`

The connected-phone instrumentation test verified that the bundled model produced an above-threshold result for this scenario.

## 14. Debugging findings that are useful to explain

### No prediction after many batches

Sixteen batches did not guarantee ten valid epochs. The investigation found a deeper issue: HR points were arriving minutes late while accelerometer batches continued. The phone had already finalized accel-only epochs and correctly rejected late samples whose timestamps belonged to finalized epochs. The fix was to flush the Samsung HR tracker every five seconds.

### HR missing and progress stuck at 0/10

The Samsung HR status mapping was initially inverted. Status `1` (successful measurement) was marked degraded, so HR validity was zero even when BPM values existed. The mapper was corrected and covered by tests.

### Predictions displayed as 0%

The ONNX model was returning nonzero probabilities such as `0.00627`, but the UI converted them to integer percentages. `0.00627 × 100 = 0.627%`, which truncated to `0%`. The shared formatter now displays two decimals, for example `0.63%`.

### India-time mismatch

The live model initially encoded clock features in `America/New_York`, which distorted time-of-day features for an India deployment. The model bridge now uses `Asia/Kolkata`, with a regression test.

## 15. Verification evidence

Latest feature branch: `feat/mesa-pilot-model`.

Recent commits:

| Commit | Purpose |
|---|---|
| `1b6c771` | Verify demo scenario with bundled ONNX on phone |
| `4baac33` | Add safe 10-epoch demo scenario/button |
| `c1a2b5c` | Use India timezone for live model clock |
| `5708a5c` | Display precise prediction percentages |
| `55dbd39` | Flush batched heart rate during watch capture |
| `acc691a` | Add epoch-driven dashboard, alarm toggle/cancel, cleaner UI |
| `be77a34` | Correct Samsung HR status mapping |
| `cdb54a1` | Connect live epochs to optional ONNX inference |
| `7161e66` | Add frozen-model inference adapter |

Verification completed across the work:

- ML suite: 153 tests passed in the performance-recovery handoff.
- Core JVM tests: passing.
- App unit tests: passing, including timezone, dashboard, alarm, demo, and inference tests.
- Wear unit tests: passing, including HR mapper and periodic flush tests.
- App and Wear debug APK builds: successful.
- ONNX parity: maximum probability delta `5.96e-08`.
- Connected phone instrumentation: demo model crossed the alarm threshold.
- Phone/Watch hardware: ADB-connected, APKs installed, sensor permissions granted, batches received and persisted.

Typical verification commands from `E:\ABA-Project\.worktrees\mesa-pilot`:

```powershell
$env:GRADLE_OPTS='-Xmx512m -XX:MaxMetaspaceSize=384m'
.\gradlew.bat --no-daemon --max-workers=1 :core:test :app:testDebugUnitTest :wear:testDebugUnitTest
.\gradlew.bat --no-daemon --max-workers=1 :app:assembleDebug :wear:assembleDebug
.\gradlew.bat --no-daemon --max-workers=1 :app:compileDebugAndroidTestKotlin
```

APK paths:

- `app/build/outputs/apk/debug/app-debug.apk`
- `wear/build/outputs/apk/debug/wear-debug.apk`

Hardware identifiers used during verification:

- phone: `RZCX21YKBDW` (`SM-S928B`);
- watch: `SM-R895F`, ADB TLS endpoint varies when re-paired.

## 16. Current limitations / honest claims

- This is a classroom MVP and exploratory research candidate, not a clinical diagnostic device.
- MESA transfer improved point estimates only slightly; the paired interval includes no improvement.
- The BIDSleep test set is small and had been seen in prior work, so the final comparison is descriptive/exploratory.
- Apple Watch/MESA devices and Samsung Watch4 differ in sensor hardware, sampling, placement, cohort, and feature semantics.
- Probability calibration is weak; a numeric probability should not be interpreted as a medical certainty.
- The current positive label is specifically N1/N2 light sleep, not “all sleep.”
- The live ONNX result is displayed/persisted but does not yet replace the exact-time fallback alarm trigger.
- Prospective overnight Watch validation, battery measurements, Bluetooth outage testing, and explicit deployment review remain future work.

## 17. Artifact lookup map

| Need | Start here |
|---|---|
| Current project status | `docs/project-status-2026-09-17.md` |
| Final demo steps | `docs/development/final-demo-runbook.md` |
| Model card | `docs/evaluation/model-card.md` |
| BIDSleep recovery metrics | `docs/evaluation/bidsleep-performance-recovery-report.md` |
| MESA cohort/transfer metrics | `docs/evaluation/mesa-expansion-transfer-report.md` |
| Dataset access/download protocol | `docs/data/public-dataset-protocol.md` |
| BIDSleep plan and labels | `docs/data/BIDSleep-Dataset-and-ML-Implementation-Plan.md` |
| Shared feature definitions | `ml/transfer_features.py`, `core/.../FeaturePipeline.kt` |
| Training/selection | `ml/performance_experiment.py`, `ml/transfer_experiment.py` |
| ONNX contract | `core/.../SleepInferenceAdapter.kt` |
| Live inference | `app/.../LiveInferenceProcessor.kt` |
| Android dashboard | `app/.../MainActivity.kt` |
| Demo scenario | `app/.../DemoInferenceScenario.kt` |
| Watch sensor integration | `wear/.../SamsungHealthSensorDataSource.kt` |
| Watch HR flush | `wear/.../ActiveTrackerFlusher.kt` |
| Alarm scheduling/cancellation | `app/.../alarm/AlarmScheduler.kt` |
| Test inventory | `ml/tests`, `core/src/test`, `app/src/test`, `wear/src/test`, `app/src/androidTest` |

## 18. Presentation closing message

The project demonstrates a complete, failure-aware wearable ML pipeline: real watch sensing, durable transfer, causal feature construction, an experimentally selected and ONNX-exported model, a usable phone dashboard, and a safe fallback alarm. The strongest conclusion is not that the model is clinically accurate; it is that the end-to-end system is working and reproducible enough for a classroom demonstration, while its calibration, device-domain generalization, and live smart-alarm wiring remain clearly identified next steps.

## 19. Historical experimentation timeline

This section is useful for a “how we improved the model” slide.

### Initial BIDSleep baselines

BIDSleep v1.0.0 contains 47 participants, 253 nights, and 213,387 labelled 30-second epochs. The participant split used seed `20260821`: 29 train, 9 validation, and 9 test participants. The initial class counts were 116,952 rest and 96,435 light-sleep epochs.

The early experiments established the baseline and rejected weak approaches:

- fixed-threshold scaled logistic: test F1 `0.552`;
- engineered CNN-GRU: test F1 `0.475`;
- engineered CNN-LSTM: test F1 `0.515`;
- tuned logistic: test F1 `0.626`;
- XGBoost/ExtraTrees sweeps: did not beat the validation baseline;
- causal-time logistic (`C=0.3`, threshold `0.345`): test F1 `0.641`, ROC-AUC `0.607`;
- later raw/hybrid study: hybrid validation F1 `0.6350`, raw CNN-GRU test F1 `0.6234`, rolling engineered test F1 `0.6399`.

The raw/hybrid study also exposed a deployment concern: raw HR availability was approximately 19% of one-second bins versus approximately 96% for accelerometry. The hybrid representation had 303 standardized features (180 raw, 120 engineered, and 3 time features), but its small validation gain was not confirmed on the locked test split.

### MESA pilot and expansion

The 24-participant MESA pilot produced 31,274 epochs using seed `20260915` and a 16/4/4 participant split. Its exploratory held-out test results were accuracy `0.7686`, balanced accuracy `0.7695`, precision `0.7062`, recall `0.9145`, F1 `0.7970`, ROC-AUC `0.8160`, PR-AUC `0.7254`, and Brier score `0.1634`. Because the test contained only four participants, this was a viability benchmark, not a deployment claim.

The later MESA-500 expansion selected 500 participants with a 350/75/75 split, produced 629,646 labelled epochs, and recorded class counts of 373,553 rest and 256,093 light. It had 5,262 missing-activity epochs and 19 unlabelled rows. The selective download was approximately 2.45 GiB: 1,500 CSV/XML files and 2,629,389,426 server-reported bytes, with no EDF files.

### Transfer and recovery

The final selection was not a MESA-only model. The selected architecture was pretrained on MESA, then fine-tuned on the identical BIDSleep training participants. The validation ladder advanced this candidate because it had the strongest participant-macro F1 and balanced accuracy under the predeclared gate. The final test comparison remained exploratory because the BIDSleep test results had been seen in prior work.

## 20. Exact final model architecture and training recipe

The final neural model is a small causal sequence classifier:

```text
[batch, 10 epochs, 8 features]
        │ transpose to channels-first
Conv1d(8 → 32, kernel size 3, no future padding)
        │ ReLU
Unidirectional GRU(32 hidden units)
        │ last time step
Linear(32 → 1)
        │ sigmoid
Light-sleep probability
```

The convolution reduces the sequence to causal local patterns, the GRU summarizes the temporal progression, and the final sigmoid produces the positive-class probability. Training used Adam with learning rate `0.001`, batch size `256`, early-stopping patience `5`, hidden size `32`, ten epochs per sequence, participant-balanced loss contribution, and validation participant-macro-F1 checkpointing. Activity count receives `log1p`; the robust scaler is fit on BIDSleep training participants only. The MESA pretrained weights are loaded before BIDSleep fine-tuning.

Implementation: `ml/models.py`, `ml/transfer_experiment.py`, and `ml/performance_experiment.py`.

## 21. Reproducible demo output

The deterministic synthetic scenario in `DemoInferenceScenario.kt` uses ten contiguous 30-second epochs with full coverage, activity count `0`, heart rate `60 bpm`, and HR standard deviation `1 bpm`. With the current `Asia/Kolkata` clock mapping and the bundled ONNX asset, the demo returns approximately `0.5672374`, displayed as `56.72%`, above the frozen threshold `0.37148505`.

This value is a controlled software demonstration, not a measured sleep event. The connected instrumentation test verifies the more important invariant—that the bundled model returns a valid above-threshold result—without hard-coding a calibration claim.

## 22. Current implementation caveats to state explicitly

- The live ONNX probability is displayed and persisted, but it does not currently trigger `AlarmReason.LIGHT_SLEEP`; only the exact-time fallback alarm is wired to a real receiver/audio path.
- `AlarmSettings.probabilityThreshold` defaults to `0.70`, while the frozen model threshold is `0.37148505`. This mismatch has no current runtime effect because smart-trigger wiring is not active, but it must be resolved before integrating live wake decisions.
- The phone fallback’s `startWatchHaptics()` is currently a no-op. The Watch has a local alarm controller, but there is no completed phone-to-watch alarm signaling path.
- The 15/30/45-minute wake-window choice is stored/displayed but does not yet modify the exact-time fallback or drive a smart wake decision.
- `watchLinked=true` currently means that at least one batch was received; it is not a continuous connectivity probe.
- IBI is captured and extracted but is not one of the eight deployed model inputs.
- The training reports describe New York clock semantics, while the live Android bridge now uses India Standard Time as a deployment-context correction. The India-time change is validated by a unit test but is not a retrained model/calibration study.

## 23. Ignored artifact locations

These artifacts are intentionally ignored because they contain raw/licensed data, generated research outputs, or controlled evaluation material. They are useful for local reproduction but should not be copied into a public presentation repository:

- corrected BIDSleep validation artifact: `ml/artifacts/bidsleep_corrected_validation/`;
- performance-recovery outputs: `ml/artifacts/performance_recovery/`;
- MESA pilot outputs: `ml/artifacts/mesa_pilot/`;
- MESA-500 outputs: `ml/artifacts/mesa_500/`;
- early transfer outputs: `ml/artifacts/mesa_transfer/`.

The intentionally tracked deployment artifact is the small ONNX copy at `app/src/main/assets/models/mesa_transfer.onnx`.

## 24. Existing visual/document assets

Useful pre-existing artifacts for presentation preparation:

- `docs/data/Smart-Sleep-Alarm-Training-Data-MESA.pptx`;
- `docs/Smart-Sleep-Alarm-ML-Progress-Report-2026-09-04.docx`;
- `docs/Smart-Sleep-Alarm-Technical-Findings-and-Microphone-Decision.docx`;
- detailed specifications and implementation plans under `docs/superpowers/specs/` and `docs/superpowers/plans/`.

Older status documents contain stale wording such as “no model bundled” or obsolete branch hashes. For the final presentation, treat the current code, latest feature-branch commits, this reference file, and the explicit evaluation reports as authoritative.
