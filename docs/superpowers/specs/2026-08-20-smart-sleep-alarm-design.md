# Smart Sleep Alarm Prototype Design

## Purpose

Build an academic, consumer-oriented wellness prototype that uses a Samsung Galaxy Watch4 Classic and a paired Android phone to identify likely light sleep within a user-selected wake window. The app should wake the user with watch haptics and phone audio when the probability is high enough, and always fall back to the target alarm time.

The prototype is not a medical device and must not diagnose sleep disorders or promise clinical sleep-stage accuracy.

## Goals

- Deliver a working Android phone and Wear OS prototype within three weeks.
- Collect live Galaxy Watch4 motion, heart-rate, and inter-beat-interval data.
- Train and compare compact CNN-GRU and CNN-LSTM light-sleep classifiers.
- Demonstrate a complete sensor-to-inference-to-alarm workflow with safe fallback behavior.
- Establish a future-ready wearable integration boundary without implementing other watch brands for the submission.

## Non-goals for the Submission

- Clinical validation or sleep-disorder diagnosis.
- A guarantee that a user is in light sleep at any future time.
- General support for Apple Watch, Fitbit, Garmin, or all Wear OS devices.
- Cloud accounts, cloud health-data storage, social features, or payments.
- Shipping audio analysis, schedule advice, or substance-context analytics as required functionality.

## Product Scope

### Core Workflow

1. The user pairs a Galaxy Watch4 with an Android phone and grants requested sensor and Samsung Health permissions.
2. The user chooses a target alarm time, a 15-, 30-, or 45-minute wake window, and haptic/audio settings.
3. The user starts a sleep session while the phone app is visible.
4. The watch collects supported continuous sensor data and transfers batched samples to the phone.
5. The phone creates 30-second feature epochs, normalizes them to the user's baseline, and obtains a causal light-sleep probability from the active model.
6. During the wake window, the phone wakes the user after a smoothed, threshold-qualified light-sleep decision.
7. If no qualified decision occurs, or data, connectivity, or inference fails, the target-time fallback alarm fires.
8. After waking, the user may rate how refreshed they feel. The session summary shows the actual wake time, confidence, and any data-quality issues.

### Supported Hardware

The first implementation supports Samsung Galaxy Watch4 and later Wear OS powered by Samsung devices only. A `WearableDataSource` interface protects the rest of the app from vendor APIs. `SamsungHealthSensorDataSource` is the sole implementation in the prototype. Future adapters must announce their supported signal capabilities before they can be used by the model.

## Architecture

The Android project contains two applications/modules:

- **Phone app:** Jetpack Compose UI, permissions, session orchestration, local storage, feature construction, inference, alarm audio, history, and settings.
- **Wear OS app:** Samsung sensor connection, batched collection, watch status UI, transfer to the phone, and haptic alarm control.

Core phone-side boundaries are:

- `WearableDataSource`: starts/stops capture and reports capability, sensor, connection, and battery state.
- `SessionRepository`: persists session metadata, derived features, prediction events, alarm events, and feedback locally.
- `FeaturePipeline`: validates timestamps, computes rolling sensor features, normalizes values, and emits fixed 30-second epochs.
- `SleepInferenceEngine`: runs a versioned, causal CNN-GRU or CNN-LSTM model and returns light-sleep probabilities.
- `AlarmDecisionEngine`: applies confidence smoothing, threshold, wake-window, and one-shot alarm rules.
- `AlarmCoordinator`: sends watch haptics first and activates phone audio after its configured delay or immediately upon fallback.

Data flow is `watch sensors -> batched phone transfer -> local session store -> feature epochs -> inference -> decision engine -> watch haptics + phone audio -> session summary`.

## Data and Modeling

### Signals and Labels

The live feature set is restricted to signals that are available from the Galaxy Watch4 implementation: three-axis accelerometer data, heart rate, and inter-beat intervals/derived HRV. Sensor-quality and off-body state are stored alongside each sample.

The public training dataset must contain sleep-stage reference labels and modalities that overlap with the deployed signals. EEG-only datasets are not acceptable training sources for this model. Dataset choice is an early delivery gate: verify licence, downloadable access, label definitions, sampling rate, and subject-level splits before building model code.

The classifier label is binary:

- `light`: reference light sleep.
- `not_light`: awake, deep, REM, unavailable, or excluded reference states as documented in the selected dataset's label mapping.

The final mapping must be recorded with the dataset version and training run.

### Model Candidates

Both models share the same input and preprocessing pipeline:

- A short rolling sequence of 30-second sensor epochs enters the model.
- A 1D CNN learns within-epoch motion and cardiovascular patterns.
- A unidirectional GRU or LSTM models temporal transitions using only prior epochs, matching live inference constraints.
- The output is a calibrated `P(light sleep)` probability.

The CNN-GRU is the initial deployment candidate because it has fewer recurrent parameters and lower mobile inference cost. CNN-LSTM is a required comparison, not an excluded option. The selected model is the one that meets the latency budget and performs better on subject-held-out validation.

### Personalization

Seven to ten personal nights establish robust-scaling and baseline values for the wearer. With consent, post-session Samsung Health sleep stages can be retrospectively aligned as weak labels for calibration analysis. They cannot be reported as clinical ground truth or as independent validation of the same vendor's sleep estimate.

## Alarm Rules and Failure Handling

- Inference begins before or at the start of the configured wake window.
- Each new epoch updates a smoothed probability score.
- The app triggers at most once during a session when the smoothed score crosses the selected threshold within the wake window.
- Watch haptics are the preferred first signal; phone audio follows a short user-configurable delay.
- The phone raises the exact-time fallback alarm if no qualifying epoch occurs.
- A disconnected watch, low sensor quality, denied permission, off-body state, transfer failure, model error, or missed inference must never prevent that fallback alarm.

## User Experience

### Phone Screens

- **Onboarding:** wellness disclaimer, watch connection, data permissions, and separate opt-in explanations for future sensitive features.
- **Home:** next target alarm, wake-window selector, haptic/audio controls, and start-session action.
- **Active session:** a dark, low-stimulation view of watch connection, watch battery, sensor quality, and duration. It does not claim a live clinical sleep stage.
- **Wake:** dismissal/snooze controls and a one-tap refreshed-feeling rating.
- **Summary and history:** actual wake time, trigger reason, confidence, data-quality notes, feedback, export, and deletion controls.
- **Settings:** alarm defaults, threshold, data settings, and wearable capability status.

### Watch Screens

The Wear OS app remains minimal: capture status, connection/sensor state, and alarm-dismiss controls. It does not host the main settings or model UI.

## Privacy and Safety

- The product is labelled as a wellness estimate, never a diagnosis or treatment recommendation.
- Health and session data are kept locally by default. Users can delete or export their records.
- Permissions are requested only for the enabled features and explained before the platform prompt.
- Raw data retention is limited to consented research sessions; long-term history stores derived features, prediction events, decisions, and feedback.
- Copy avoids definitive claims such as "you are in light sleep" or "this substance caused poor sleep."

## Deferred Personalization Features

### Sleep Schedule Advisor

The user may set a target sleep duration and receive an initial bedtime goal based on the target wake time and an adjustable wind-down buffer. After a minimum number of valid sessions, the app can report a confidence-qualified, likely favorable wake range from the user's history. It must not guarantee a future light-sleep time.

### Context Logging

An optional pre-sleep check-in may record timing and broad amount categories for caffeine, alcohol, cannabis, nicotine, medication, exercise, and stress. This sensitive data is local-only by default. It remains separate from the live light-sleep classifier until sufficient, representative personal data and an explicit evaluation plan exist. Future insights are descriptive, within-person comparisons with uncertainty, not causal or medical claims.

### Sound Analysis

An opt-in, local-only audio module may classify coarse sound events such as quiet, environmental noise, possible snore, and other sound. It must run only after an explicit session-start consent, process short chunks on-device, and discard raw recordings. It does not diagnose sleep apnea. It is a stretch goal after the core alarm has passed end-to-end validation.

## Validation

### Model Evaluation

- Use subject-held-out train/validation/test splits to prevent leakage.
- Compare CNN-GRU, CNN-LSTM, and a simple logistic-regression or random-forest baseline.
- Report F1, precision, recall, confusion matrix, inference latency, model size, and the final threshold-selection method.
- Preserve dataset version, feature configuration, random seed, and training configuration for reproducibility.

### Prototype Validation

- Verify real Watch4 sensor collection, transfer, timestamp handling, and session persistence.
- Exercise simulated light and non-light prediction sequences to verify trigger timing and no duplicate alarms.
- Verify every failure path still reaches the exact-time fallback alarm.
- Test onboarding, permission denial, reconnect, alarm dismissal, feedback, history, and data deletion on the target phone and watch.
- Compare personal-session output to consented Samsung Health records only as weak-label calibration analysis.

## Three-Week Delivery Order

1. Establish Android/Wear project, pair the Watch4, verify live collection and phone transfer, and implement a reliable exact-time alarm.
2. Validate and prepare the selected public dataset, build reproducible preprocessing, train baseline, CNN-GRU, and CNN-LSTM candidates, then freeze the selected exportable model.
3. Integrate inference and decision rules, complete session UI/history/privacy controls, run end-to-end tests, collect personal calibration sessions, and prepare the demonstration.

Audio analysis, schedule advice, and context-log insights are attempted only after all core scope items above work reliably.
