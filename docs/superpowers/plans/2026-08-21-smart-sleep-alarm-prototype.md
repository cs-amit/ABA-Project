# Smart Sleep Alarm Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Samsung Galaxy Watch4 and Android phone wellness prototype that detects likely light sleep in a chosen wake window and always delivers a safe fallback alarm.

**Architecture:** A Wear OS app collects Samsung accelerometer and heart-rate/IBI data, batches it to the paired phone through the Wear OS Data Layer, and provides watch haptics. The phone persists derived session records locally, builds causal 30-second feature epochs, runs an exported ONNX CNN-GRU or CNN-LSTM model, and owns all alarm decisions and phone audio.

**Tech Stack:** Kotlin, Gradle Kotlin DSL, Jetpack Compose, Wear OS Compose, Room, Kotlin coroutines/Flow, Samsung Health Sensor SDK 1.4.1, Google Play Services Wearable 20.0.1, ONNX Runtime Android 1.20.1, Python 3.11, PyTorch 2.5.1, scikit-learn 1.6.1, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-smart-sleep-alarm-design.md`

## Global Constraints

- Use package namespace `com.aba.smartsleep`; use `compileSdk = 36`, `targetSdk = 35`, phone `minSdk = 29`, and Wear `minSdk = 30`.
- Support Galaxy Watch4 and later Wear OS powered by Samsung watches only; show unsupported capabilities rather than guessing them.
- Use Samsung Health Sensor SDK 1.4.1 from `wear/libs/samsung-health-sensor-api.aar`; its binary is obtained manually from Samsung and never placed in source control.
- Request only `ACTIVITY_RECOGNITION` and `BODY_SENSORS` permissions for the first prototype. Do not add microphone, Samsung Health Data SDK, context logging, or schedule-advisor code to the core build.
- Store health/session data locally by default. Do not add accounts, network APIs, cloud sync, analytics SDKs, or advertising SDKs.
- Present all predictions as wellness estimates. Never state that the user is definitively in light sleep or that the app diagnoses a disorder.
- The exact-time phone alarm must fire even if watch collection, transfer, storage, feature extraction, or model inference fails.
- Start Samsung Health Sensor Service developer mode only on the project owner's test watch. Public distribution requires Samsung partner registration and app-signature approval.

---

## File Structure

```text
ABA-Project/
├── settings.gradle.kts
├── build.gradle.kts
├── gradle/libs.versions.toml
├── app/                                 # Android phone application
│   ├── build.gradle.kts
│   └── src/main/java/com/aba/smartsleep/app/
│       ├── SmartSleepApplication.kt
│       ├── MainActivity.kt
│       ├── alarm/AlarmCoordinator.kt
│       ├── alarm/AlarmReceiver.kt
│       ├── data/AppDatabase.kt
│       ├── data/RoomSessionRepository.kt
│       ├── transport/PhoneDataLayerListener.kt
│       ├── transport/WearableReceiver.kt
│       ├── inference/OnnxSleepInferenceEngine.kt
│       └── ui/
│           ├── SmartSleepRoot.kt
│           ├── HomeScreen.kt
│           ├── ActiveSessionScreen.kt
│           ├── SummaryScreen.kt
│           └── SettingsScreen.kt
├── wear/                                # Wear OS watch application
│   ├── libs/.gitkeep
│   ├── build.gradle.kts
│   └── src/main/java/com/aba/smartsleep/wear/
│       ├── WearMainActivity.kt
│       ├── sensor/SamsungHealthSensorDataSource.kt
│       ├── sensor/SensorCaptureService.kt
│       ├── transport/WearBatchSender.kt
│       └── haptics/WatchAlarmController.kt
├── core/                                # Pure Kotlin domain and algorithms
│   ├── build.gradle.kts
│   └── src/main/kotlin/com/aba/smartsleep/core/
│       ├── model/SessionModels.kt
│       ├── model/WearableDataSource.kt
│       ├── transport/SensorBatchCodec.kt
│       ├── features/FeaturePipeline.kt
│       ├── inference/SleepInferenceEngine.kt
│       └── alarm/AlarmDecisionEngine.kt
├── ml/
│   ├── requirements.txt
│   ├── README.md
│   ├── data_contract.py
│   ├── prepare_dataset.py
│   ├── features.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── tests/
├── docs/
│   ├── development/watch4-setup.md
│   ├── data/public-dataset-protocol.md
│   ├── evaluation/model-card.md
│   └── superpowers/
└── README.md
```

## Task 1: Create the Multi-Module Android and ML Skeleton

**Files:**
- Create: `settings.gradle.kts`, `build.gradle.kts`, `gradle/libs.versions.toml`, `app/build.gradle.kts`, `wear/build.gradle.kts`, `core/build.gradle.kts`, `README.md`, `docs/development/watch4-setup.md`, `ml/requirements.txt`, `ml/README.md`, `wear/libs/.gitkeep`
- Create: `core/src/test/kotlin/com/aba/smartsleep/core/BuildSmokeTest.kt`

**Interfaces:**
- Produces Gradle modules `:app`, `:wear`, and `:core` and Python environment requirements used by all later tasks.

- [ ] **Step 1: Generate the Gradle wrapper and root settings**

Create the root settings with the exact modules and plugin repositories:

```kotlin
pluginManagement { repositories { google(); mavenCentral(); gradlePluginPortal() } }
dependencyResolutionManagement { repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS); repositories { google(); mavenCentral() } }
rootProject.name = "SmartSleep"
include(":app", ":wear", ":core")
```

- [ ] **Step 2: Configure version catalog and module SDK floors**

Declare Android Gradle Plugin, Kotlin, Compose, Room, coroutines, wearable, ONNX Runtime, and test dependencies in `gradle/libs.versions.toml`. Configure `app` as an Android application, `wear` as a Wear OS Android application, and `core` as a Kotlin JVM library with Java 17 toolchains. Put the Samsung AAR dependency only in `wear/build.gradle.kts`:

```kotlin
implementation(files("libs/samsung-health-sensor-api.aar"))
implementation("com.google.android.gms:play-services-wearable:20.0.1")
```

- [ ] **Step 3: Add the first failing core test**

Create `BuildSmokeTest.kt`:

```kotlin
class BuildSmokeTest {
    @Test fun `core module runs junit tests`() = assertTrue(true)
}
```

- [ ] **Step 4: Run the unit-test target and Android build**

Run: `./gradlew.bat :core:test :app:assembleDebug :wear:assembleDebug`

Expected: the command succeeds after the Samsung AAR is downloaded into `wear/libs/`. If the AAR is absent, Gradle reports its explicit missing-file path and no source code is changed to bypass the SDK.

- [ ] **Step 5: Document physical watch preparation**

Write `docs/development/watch4-setup.md` with the exact local-development steps: enable ADB debugging on the watch, pair with Android Studio/ADB, update Health Sensor Service to the version required by SDK 1.4.1, and enable its developer mode. Include a warning that this is for the owner’s test device only and that public release requires Samsung partner registration.

## Task 2: Define Core Domain Contracts and Alarm State

**Files:**
- Create: `core/src/main/kotlin/com/aba/smartsleep/core/model/SessionModels.kt`
- Create: `core/src/main/kotlin/com/aba/smartsleep/core/model/WearableDataSource.kt`
- Create: `core/src/test/kotlin/com/aba/smartsleep/core/model/SessionModelsTest.kt`

**Interfaces:**
- Produces `AlarmSettings`, `SleepSession`, `SensorSample`, `SensorBatch`, `SessionStatus`, `WearableCapability`, and `WearableDataSource` for Tasks 3–10.

- [ ] **Step 1: Write failing invariant tests**

```kotlin
@Test fun `wake window must end at target alarm`() {
    val settings = AlarmSettings(targetEpochMillis = 1_000_000, wakeWindowMinutes = 30)
    assertEquals(-1_800_000, settings.wakeWindowStartEpochMillis)
}

@Test fun `sensor batch rejects unordered samples`() {
    assertFailsWith<IllegalArgumentException> {
        SensorBatch("session", listOf(sampleAt(20), sampleAt(10)))
    }
}
```

- [ ] **Step 2: Run the tests to confirm missing contracts**

Run: `./gradlew.bat :core:test --tests '*SessionModelsTest'`

Expected: compilation fails because `AlarmSettings`, `SensorBatch`, and `sampleAt` do not yet exist.

- [ ] **Step 3: Implement immutable domain types**

```kotlin
data class AlarmSettings(val targetEpochMillis: Long, val wakeWindowMinutes: Int, val probabilityThreshold: Float = 0.70f) {
    init { require(wakeWindowMinutes in setOf(15, 30, 45)); require(probabilityThreshold in 0f..1f) }
    val wakeWindowStartEpochMillis get() = targetEpochMillis - wakeWindowMinutes * 60_000L
}

data class SensorSample(val timestampEpochMillis: Long, val accelX: Float?, val accelY: Float?, val accelZ: Float?, val heartRateBpm: Float?, val ibiMillis: Int?, val quality: SensorQuality)
data class SensorBatch(val sessionId: String, val samples: List<SensorSample>) { init { require(samples.zipWithNext().all { it.first.timestampEpochMillis <= it.second.timestampEpochMillis }) } }
enum class SensorQuality { VALID, DEGRADED, OFF_BODY, UNAVAILABLE }
enum class WearableCapability { ACCELEROMETER, HEART_RATE_WITH_IBI, HAPTICS }
```

Define `WearableDataSource` with `suspend fun capabilities(): Set<WearableCapability>`, `fun samples(): Flow<SensorSample>`, `suspend fun start(sessionId: String)`, and `suspend fun stop()`.

- [ ] **Step 4: Re-run model tests**

Run: `./gradlew.bat :core:test --tests '*SessionModelsTest'`

Expected: PASS.

## Task 3: Implement Exact-Time Fallback Alarm First

**Files:**
- Create: `app/src/main/java/com/aba/smartsleep/app/alarm/AlarmCoordinator.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/alarm/AlarmReceiver.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/alarm/AlarmScheduler.kt`
- Modify: `app/src/main/AndroidManifest.xml`
- Create: `app/src/test/java/com/aba/smartsleep/app/alarm/AlarmCoordinatorTest.kt`

**Interfaces:**
- Consumes: `AlarmSettings` from Task 2.
- Produces: `AlarmScheduler.scheduleFallback(sessionId: String, targetEpochMillis: Long)` and `AlarmCoordinator.trigger(reason: AlarmReason)` for Tasks 9–10.

- [ ] **Step 1: Write failing tests with a fake clock and alarm gateway**

```kotlin
@Test fun `schedules fallback at configured target`() {
    val gateway = FakeAlarmGateway()
    AlarmScheduler(gateway, FixedClock(100)).scheduleFallback("s1", 10_000)
    assertEquals(10_000, gateway.scheduled.single().triggerAtMillis)
}

@Test fun `first trigger wins over all later triggers`() {
    val coordinator = AlarmCoordinator(FakeAlarmOutput())
    assertTrue(coordinator.trigger(AlarmReason.LIGHT_SLEEP))
    assertFalse(coordinator.trigger(AlarmReason.FALLBACK))
}
```

- [ ] **Step 2: Run the alarm tests to confirm failure**

Run: `./gradlew.bat :app:testDebugUnitTest --tests '*AlarmCoordinatorTest'`

Expected: compilation fails because the scheduler and coordinator are absent.

- [ ] **Step 3: Implement exact alarm and one-shot trigger behavior**

Use `AlarmManager.setAlarmClock()` for the user-facing target-time fallback. Declare `USE_EXACT_ALARM`, `POST_NOTIFICATIONS`, and `USE_FULL_SCREEN_INTENT`; issue a high-priority alarm notification from `AlarmReceiver`; and route it to `AlarmCoordinator.trigger(AlarmReason.FALLBACK)`. `AlarmCoordinator` uses an `AtomicBoolean` and starts watch haptics, then phone audio after the configured delay.

```kotlin
enum class AlarmReason { LIGHT_SLEEP, FALLBACK }
interface AlarmScheduler { fun scheduleFallback(sessionId: String, targetEpochMillis: Long) }
```

- [ ] **Step 4: Re-run unit tests and perform a five-minute physical alarm check**

Run: `./gradlew.bat :app:testDebugUnitTest --tests '*AlarmCoordinatorTest'`

Expected: PASS. Install the phone app, create an alarm five minutes ahead, lock the phone, and verify audio still fires at the target time.

## Task 4: Implement Samsung Watch Capture and Capability Checks

**Files:**
- Create: `wear/src/main/java/com/aba/smartsleep/wear/sensor/SamsungHealthSensorDataSource.kt`
- Create: `wear/src/main/java/com/aba/smartsleep/wear/sensor/SensorCaptureService.kt`
- Create: `wear/src/main/java/com/aba/smartsleep/wear/haptics/WatchAlarmController.kt`
- Create: `wear/src/main/java/com/aba/smartsleep/wear/WearMainActivity.kt`
- Modify: `wear/src/main/AndroidManifest.xml`
- Create: `wear/src/test/java/com/aba/smartsleep/wear/sensor/SamsungSampleMapperTest.kt`

**Interfaces:**
- Consumes: `WearableDataSource`, `SensorSample`, `SensorQuality`, and `WearableCapability` from Task 2.
- Produces: `SamsungHealthSensorDataSource` and `SensorCaptureService` for Task 5.

- [ ] **Step 1: Write mapping tests for vendor values**

```kotlin
@Test fun `maps off body heart rate status to invalid quality`() {
    val sample = SamsungSampleMapper.heartRate(timestamp = 1000, bpm = 0f, ibi = null, status = -3)
    assertEquals(SensorQuality.OFF_BODY, sample.quality)
}
```

- [ ] **Step 2: Run the mapping test to confirm failure**

Run: `./gradlew.bat :wear:testDebugUnitTest --tests '*SamsungSampleMapperTest'`

Expected: compilation fails because `SamsungSampleMapper` is absent.

- [ ] **Step 3: Implement continuous tracker setup**

Connect to Samsung Health Sensor Service, query supported tracker types, request `ACTIVITY_RECOGNITION` and `BODY_SENSORS` before collection, and start only `ACCELEROMETER_CONTINUOUS` and `HEART_RATE_CONTINUOUS` when available. Convert their events to `SensorSample`: retain 25 Hz raw accelerometer values, retain 1 Hz heart rate, unpack IBI values from the callback, and map off-body/error states to `SensorQuality`.

- [ ] **Step 4: Implement lifecycle-safe watch service and haptics**

Run collection in `SensorCaptureService` only after the user starts a session on the phone/watch. Stop all tracker listeners on service stop, connection failure, and explicit session stop. Implement `WatchAlarmController.vibrate()` using a bounded vibration pattern and expose an accessible dismissal action in `WearMainActivity`.

- [ ] **Step 5: Run tests and execute hardware capability gate**

Run: `./gradlew.bat :wear:testDebugUnitTest --tests '*SamsungSampleMapperTest'`

Expected: PASS. Install on the Watch4, enable Samsung developer mode, start a 60-second capture, and record which of the two required continuous trackers are available. Stop implementation and resolve SDK/service/permission errors before moving on.

## Task 5: Transfer Ordered Sensor Batches to the Phone

**Files:**
- Create: `core/src/main/kotlin/com/aba/smartsleep/core/transport/SensorBatchCodec.kt`
- Create: `core/src/test/kotlin/com/aba/smartsleep/core/transport/SensorBatchCodecTest.kt`
- Create: `wear/src/main/java/com/aba/smartsleep/wear/transport/WearBatchSender.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/transport/PhoneDataLayerListener.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/transport/WearableReceiver.kt`
- Modify: `app/src/main/AndroidManifest.xml`

**Interfaces:**
- Consumes: `SensorBatch` from Task 2 and `SamsungHealthSensorDataSource.samples()` from Task 4.
- Produces: `SensorBatchCodec.encode/decode`, `WearBatchSender.send(batch)`, and `WearableReceiver.receivedBatches: Flow<SensorBatch>` for Task 6.

- [ ] **Step 1: Write codec round-trip and corruption tests**

```kotlin
@Test fun `round trip preserves nullable IBI and timestamps`() {
    val original = SensorBatch("s1", listOf(sampleAt(1000, ibiMillis = null), sampleAt(1040, ibiMillis = 812)))
    assertEquals(original, SensorBatchCodec.decode(SensorBatchCodec.encode(original)))
}

@Test fun `decode rejects unsupported payload version`() {
    assertFailsWith<IllegalArgumentException> { SensorBatchCodec.decode(byteArrayOf(99)) }
}
```

- [ ] **Step 2: Run codec tests to confirm failure**

Run: `./gradlew.bat :core:test --tests '*SensorBatchCodecTest'`

Expected: compilation fails because `SensorBatchCodec` is absent.

- [ ] **Step 3: Implement a versioned binary codec**

Encode `version`, `sessionId`, sample count, and delta timestamps with `DataOutputStream`; use explicit presence bytes for nullable values. Limit each batch to at most 30 seconds of samples and reject encoded payloads at or above 90 KB.

- [ ] **Step 4: Implement durable Data Layer batch sync**

Write each encoded batch to a unique urgent `DataItem` path `/sessions/{sessionId}/batches/{sequence}` using `PutDataRequest.setUrgent()`. This permits buffering while disconnected. The phone’s exported `WearableListenerService` decodes only that path shape, persists valid batches through `WearableReceiver`, and writes a small acknowledged-sequence `DataItem` back to the watch. Delete acknowledged local batch files from watch-private storage.

- [ ] **Step 5: Re-run codec tests and perform disconnect transfer test**

Run: `./gradlew.bat :core:test --tests '*SensorBatchCodecTest'`

Expected: PASS. Start capture, disable Bluetooth for two minutes, re-enable it, and verify ordered batches arrive on the phone without blocking the phone fallback alarm.

## Task 6: Persist Sessions and Build Causal Feature Epochs

**Files:**
- Create: `app/src/main/java/com/aba/smartsleep/app/data/AppDatabase.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/data/RoomSessionRepository.kt`
- Create: `core/src/main/kotlin/com/aba/smartsleep/core/features/FeaturePipeline.kt`
- Create: `core/src/test/kotlin/com/aba/smartsleep/core/features/FeaturePipelineTest.kt`
- Create: `app/src/androidTest/java/com/aba/smartsleep/app/data/RoomSessionRepositoryTest.kt`

**Interfaces:**
- Consumes: `SensorBatch` from Task 5.
- Produces: `SessionRepository.append(batch)`, `SessionRepository.session(sessionId)`, and `FeaturePipeline.append(sample): List<FeatureEpoch>` for Tasks 7–10.

- [ ] **Step 1: Write failing feature-window tests**

```kotlin
@Test fun `emits one completed 30 second epoch`() {
    val pipeline = FeaturePipeline(epochMillis = 30_000)
    samplesFrom(0, 30_000, everyMillis = 40).flatMap { pipeline.append(it) }
    assertEquals(1, pipeline.completedEpochs.size)
}

@Test fun `does not use samples after epoch end`() {
    val epoch = FeaturePipeline().consume(samplesAroundBoundary()).single()
    assertEquals(30_000, epoch.endEpochMillis)
}
```

- [ ] **Step 2: Run feature tests to confirm failure**

Run: `./gradlew.bat :core:test --tests '*FeaturePipelineTest'`

Expected: compilation fails because `FeaturePipeline` is absent.

- [ ] **Step 3: Implement Room entities and repository**

Persist session metadata, alarm settings, derived 30-second feature epochs, prediction events, alarm events, and refreshed-feeling feedback. Do not persist continuous raw samples after each epoch has been successfully transformed unless the session has the explicit research-retention flag. Use transactionally increasing batch sequence numbers to make duplicate Data Layer delivery idempotent.

- [ ] **Step 4: Implement causal feature extraction**

For each 30-second epoch, calculate accelerometer magnitude mean, standard deviation, median absolute deviation, activity counts, zero-crossing rate, heart-rate mean, heart-rate standard deviation, mean IBI, RMSSD, valid-sample ratios, and an off-body flag. Emit no inference-ready epoch when required signal quality is insufficient. Normalize each numeric feature using baseline median and interquartile range computed from prior valid epochs only.

```kotlin
data class FeatureEpoch(
    val sessionId: String,
    val startEpochMillis: Long,
    val endEpochMillis: Long,
    val values: FloatArray,
    val validForInference: Boolean,
)

interface SessionRepository {
    suspend fun append(batch: SensorBatch)
    fun session(sessionId: String): Flow<SleepSession?>
}
```

- [ ] **Step 5: Run feature and database tests**

Run: `./gradlew.bat :core:test --tests '*FeaturePipelineTest' :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.aba.smartsleep.app.data.RoomSessionRepositoryTest`

Expected: all tests PASS; duplicate batches do not create duplicate persisted epochs.

## Task 7: Create Reproducible Public-Dataset Preparation

**Files:**
- Create: `ml/requirements.txt`, `ml/data_contract.py`, `ml/prepare_dataset.py`, `ml/features.py`, `ml/tests/test_data_contract.py`, `ml/tests/test_features.py`, `docs/data/public-dataset-protocol.md`

**Interfaces:**
- Consumes: raw files placed outside the repository at the path in `SMART_SLEEP_DATA_DIR`.
- Produces: `ml/artifacts/epochs.parquet`, `ml/artifacts/splits.json`, and `ml/artifacts/dataset_manifest.json` for Task 8.

- [ ] **Step 1: Write the dataset protocol before downloading records**

Document the selection gate in `docs/data/public-dataset-protocol.md`: use the MESA Sleep dataset only after confirming approved access to wrist actigraphy, PSG stage labels, and ECG/heart-rate derivation for the chosen participants. Record the MESA release, access date, licence/terms, selected signal files, epoch-label mapping, and exclusions in `dataset_manifest.json`. Do not substitute EEG-only data.

- [ ] **Step 2: Write failing data-contract tests**

```python
def test_rejects_duplicate_subject_epoch_rows():
    frame = pd.DataFrame({"subject_id": ["a", "a"], "epoch_start_s": [0, 0], "label": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_epoch_frame(frame)

def test_light_mapping_is_binary():
    assert map_stage("LIGHT") == 1
    assert map_stage("DEEP") == 0
```

- [ ] **Step 3: Run Python tests to confirm failure**

Run: `python -m pytest ml/tests/test_data_contract.py -q`

Expected: import fails because `data_contract.py` is absent.

- [ ] **Step 4: Implement the canonical epoch contract**

Implement `validate_epoch_frame(frame)` requiring `subject_id`, `epoch_start_s`, `label`, the exact Android feature columns from Task 6, and `split`. Implement `map_stage()` so only the documented reference `LIGHT` stage maps to `1`; `AWAKE`, `DEEP`, and `REM` map to `0`; unknown states are dropped and counted in the manifest. Generate subject-level train/validation/test splits with no subject present in more than one split.

- [ ] **Step 5: Implement preparation and feature parity tests**

Implement `prepare_dataset.py` to read MESA-derived source records through one explicit adapter function, create 30-second epochs, calculate the same feature names/formulas as Task 6, validate the frame, and write Parquet plus manifest. Add fixture-record tests that compare expected activity-count, RMSSD, and label values. Run: `python -m pytest ml/tests -q`.

Expected: PASS. If MESA access is not approved by the end of Day 2, preserve this tested pipeline and document the blocked data-access condition in the project risk log; do not claim model-validation results from unmatched data.

## Task 8: Train, Compare, and Export the Causal Models

**Files:**
- Create: `ml/models.py`, `ml/train.py`, `ml/evaluate.py`, `ml/export_onnx.py`, `ml/tests/test_models.py`, `ml/tests/test_export.py`, `docs/evaluation/model-card.md`
- Create: `app/src/main/assets/models/.gitkeep`

**Interfaces:**
- Consumes: prepared artifacts from Task 7.
- Produces: `ml/artifacts/{baseline,cnn_gru,cnn_lstm}/metrics.json`, confusion-matrix images, and one selected `app/src/main/assets/models/sleep_model.onnx` with `model_metadata.json`.

- [ ] **Step 1: Write failing model shape tests**

```python
def test_cnn_gru_outputs_one_probability_per_sequence():
    model = CnnGru(feature_count=12, hidden_size=32)
    assert model(torch.zeros(4, 10, 12)).shape == (4, 1)

def test_cnn_lstm_is_causal():
    assert CnnLstm(feature_count=12, hidden_size=32).lstm.bidirectional is False
```

- [ ] **Step 2: Run model tests to confirm failure**

Run: `python -m pytest ml/tests/test_models.py -q`

Expected: import fails because model classes are absent.

- [ ] **Step 3: Implement baseline and neural candidates**

Implement a scaled logistic-regression baseline, `CnnGru`, and `CnnLstm`. Both neural models accept tensors shaped `[batch, sequence_epochs, feature_count]`, apply a 1D convolution over feature epochs, then a unidirectional recurrent layer, and return a sigmoid probability. Train with binary cross entropy, early stopping on validation F1, fixed seed `20260821`, and class weighting calculated from the training split only.

- [ ] **Step 4: Evaluate with subject-held-out splits**

Implement `evaluate.py` to load only the saved test split and write precision, recall, F1, ROC-AUC, confusion matrix, latency on CPU, parameter count, and model size. Select the model with the highest test-set F1 only if one 10-epoch CPU prediction completes under 250 ms; otherwise select the fastest model that completes under that limit and has F1 no more than 0.03 below the best score.

- [ ] **Step 5: Export and verify ONNX**

Export the selected model with opset 17, dynamic batch axis, fixed sequence length 10, and feature dimension taken from `model_metadata.json`. Verify PyTorch and ONNX Runtime probabilities differ by less than `1e-4` for a deterministic test tensor. Copy only the verified `.onnx` and metadata to `app/src/main/assets/models/` and write the exact dataset, split, seed, metrics, limitation, and selection rationale to `docs/evaluation/model-card.md`.

## Task 9: Add Phone Inference and Wake-Window Decisions

**Files:**
- Create: `core/src/main/kotlin/com/aba/smartsleep/core/inference/SleepInferenceEngine.kt`
- Create: `core/src/main/kotlin/com/aba/smartsleep/core/alarm/AlarmDecisionEngine.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/inference/OnnxSleepInferenceEngine.kt`
- Create: `core/src/test/kotlin/com/aba/smartsleep/core/alarm/AlarmDecisionEngineTest.kt`
- Create: `app/src/test/java/com/aba/smartsleep/app/inference/OnnxSleepInferenceEngineTest.kt`

**Interfaces:**
- Consumes: `FeatureEpoch` from Task 6, model assets from Task 8, and `AlarmCoordinator` from Task 3.
- Produces: `SleepInferenceEngine.predict(sequence): Float` and `AlarmDecisionEngine.evaluate(now, probability): AlarmDecision` for Task 10.

- [ ] **Step 1: Write failing decision tests**

```kotlin
@Test fun `triggers only after threshold inside wake window`() {
    val engine = AlarmDecisionEngine(settings(target = 100_000, window = 15, threshold = 0.7f), requiredConsecutiveEpochs = 2)
    assertEquals(AlarmDecision.WAIT, engine.evaluate(90_000, 0.8f))
    assertEquals(AlarmDecision.TRIGGER_LIGHT_SLEEP, engine.evaluate(91_000, 0.8f))
}

@Test fun `returns fallback at target without valid samples`() {
    assertEquals(AlarmDecision.TRIGGER_FALLBACK, engine.evaluate(100_000, null))
}
```

- [ ] **Step 2: Run decision tests to confirm failure**

Run: `./gradlew.bat :core:test --tests '*AlarmDecisionEngineTest'`

Expected: compilation fails because the engine is absent.

- [ ] **Step 3: Implement causal sequence buffering and ONNX wrapper**

Define `SleepInferenceEngine` as `suspend fun predict(epochs: List<FeatureEpoch>): Float?`. `OnnxSleepInferenceEngine` loads `sleep_model.onnx`, verifies metadata feature order and 10-epoch sequence length, returns `null` for missing/invalid epochs, and closes all ONNX tensors/results with `use` blocks.

- [ ] **Step 4: Implement deterministic alarm decision rules**

Keep a rolling two-probability average. Before `wakeWindowStart`, return `WAIT`; from the window start through one millisecond before target, return `TRIGGER_LIGHT_SLEEP` only when two consecutive valid averaged values meet threshold; at or after target, return `TRIGGER_FALLBACK`; after any trigger, return `ALREADY_TRIGGERED`. Persist every decision input and result.

- [ ] **Step 5: Run unit tests and fake-model integration test**

Run: `./gradlew.bat :core:test --tests '*AlarmDecisionEngineTest' :app:testDebugUnitTest --tests '*OnnxSleepInferenceEngineTest'`

Expected: PASS. The fake model test must prove that invalid input cannot suppress the fallback path.

## Task 10: Build the Phone and Watch User Flows

**Files:**
- Create: `app/src/main/java/com/aba/smartsleep/app/SmartSleepApplication.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/MainActivity.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/ui/SmartSleepRoot.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/ui/HomeScreen.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/ui/ActiveSessionScreen.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/ui/SummaryScreen.kt`
- Create: `app/src/main/java/com/aba/smartsleep/app/ui/SettingsScreen.kt`
- Create: `app/src/androidTest/java/com/aba/smartsleep/app/ui/AlarmFlowTest.kt`
- Modify: `wear/src/main/java/com/aba/smartsleep/wear/WearMainActivity.kt`

**Interfaces:**
- Consumes: session repository, capability state, inference/decision state, and alarm coordinator from Tasks 3–9.
- Produces: user-driven session start/stop, target settings, feedback, session history, and data deletion behavior.

- [ ] **Step 1: Write failing Compose navigation tests**

```kotlin
@Test fun startSleepSessionShowsActiveState() {
    composeRule.setContent { SmartSleepRoot(FakeAppViewModel()) }
    composeRule.onNodeWithText("Start sleep session").performClick()
    composeRule.onNodeWithText("Session active").assertIsDisplayed()
}

@Test fun unsupportedSensorBlocksSessionStart() {
    composeRule.setContent { SmartSleepRoot(FakeAppViewModel(capabilities = emptySet())) }
    composeRule.onNodeWithText("Galaxy Watch sensor access is unavailable").assertIsDisplayed()
}
```

- [ ] **Step 2: Run UI tests to confirm failure**

Run: `./gradlew.bat :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.aba.smartsleep.app.ui.AlarmFlowTest`

Expected: compilation fails because the UI root and view model do not exist.

- [ ] **Step 3: Implement onboarding and home settings**

Build onboarding with explicit wellness wording, permission rationale, and Galaxy Watch capability check. Build the home screen with target time picker, exactly three wake-window choices (15/30/45), threshold setting hidden under advanced settings, haptic/audio toggles, and the `Start sleep session` button. Do not show any microphone or substance questions.

- [ ] **Step 4: Implement active, wake, summary, history, and deletion UI**

Show connection status, battery, sensor quality, session duration, and a low-stimulation dark surface during active sessions. After alarm dismissal, offer one refreshed-feeling rating. Summary displays actual wake time, trigger reason, confidence, and data-quality note. History lists sessions. Settings includes `Delete all local sleep data`, which requires a confirmation and deletes Room records plus retained research raw files.

- [ ] **Step 5: Complete watch controls and run UI tests**

On watch, show capture state, connection quality, and alarm-dismiss button only. Run the instrumentation command from Step 2. Expected: PASS on the paired physical phone; hardware Samsung sensor checks remain manual because the vendor SDK does not support an emulator.

## Task 11: Validate the Complete Prototype and Prepare Handoff Material

**Files:**
- Create: `docs/evaluation/end-to-end-test-checklist.md`
- Create: `docs/evaluation/demo-script.md`
- Modify: `README.md`, `docs/evaluation/model-card.md`

**Interfaces:**
- Consumes: complete implementation from Tasks 1–10.
- Produces: reproducible verification evidence, documented limitations, and a ready demonstration path.

- [ ] **Step 1: Create the end-to-end failure matrix**

Add one checklist row each for: healthy sensor session, watch disconnect, Bluetooth disconnect/reconnect, off-body signal, denied sensor permission, Samsung Health Sensor Service unavailable, malformed sensor batch, missing ONNX asset, inference exception, no light-sleep match, duplicate trigger attempt, and deletion confirmation. Each row states expected UI message, stored event, and exact-time fallback result.

- [ ] **Step 2: Execute automated checks**

Run: `./gradlew.bat :core:test :app:testDebugUnitTest :wear:testDebugUnitTest :app:assembleDebug :wear:assembleDebug` and `python -m pytest ml/tests -q`.

Expected: every command exits `0`. Record command output date, model artifact hash, and APK version in `docs/evaluation/model-card.md`.

- [ ] **Step 3: Execute hardware overnight and accelerated alarm checks**

Run one real overnight collection session and one accelerated daytime test with target alarms 5–10 minutes ahead. Confirm batched data arrives, valid epochs are persisted, watch haptics occur for a forced light-sleep test probability, and phone audio fires at the target when the forced inference source returns `null`.

- [ ] **Step 4: Create the academic demonstration script**

Write a six-minute script: problem and wellness disclaimer; Watch4/phone architecture; live capability and session start; feature/model comparison metrics; forced wake-window trigger; forced fallback trigger; session summary; limitations and deferred audio/schedule/context features. The script must call Samsung Health stages weak calibration labels rather than ground truth.

- [ ] **Step 5: Final scope review**

Confirm the build contains no cloud APIs, microphone capture, substance-context collection, or diagnostic claims. Confirm README includes setup, device prerequisites, tests, data licence responsibility, and the Samsung distribution constraint. Keep stretch features outside the release build until the core checklist passes.

## Plan Self-Review

- Spec coverage: Tasks 1–6 cover project setup, sensors, transfer, local storage, features, exact fallback, and safety; Tasks 7–9 cover dataset discipline, CNN-GRU/CNN-LSTM comparison, export, inference, and calibration boundaries; Task 10 covers all approved UI flows; Task 11 covers validation and demonstration. Deferred audio, schedule, and context features are intentionally excluded from code by the approved scope.
- Placeholder scan: this plan contains concrete modules, paths, interfaces, commands, test names, expected outcomes, SDK versions, thresholds, and hardware gates.
- Type consistency: `SensorSample` feeds `SensorBatch`, `FeaturePipeline` emits `FeatureEpoch`, `SleepInferenceEngine` consumes sequences of `FeatureEpoch`, and `AlarmDecisionEngine` directs `AlarmCoordinator` through the same one-shot decision flow.
