# Smart Sleep Alarm: Model Problem Brief

**Purpose:** A plain-language record of the machine-learning problem, the work completed so far, what the results mean, and the most useful directions to investigate next.

## The product question

The app is trying to answer one limited question during a user's chosen wake-up window:

> Is this a moment when the person's watch signals make waking them *more likely* to be comfortable than waiting?

It is **not** trying to diagnose a sleep disorder or claim that it knows the user's exact clinical sleep stage. The phone must always sound a normal alarm at the selected target time, even when the watch disconnects, data is missing, or the model is uncertain.

## Why this is difficult

The sleep stage that is most useful for the project is broadly called “light sleep.” In the BIDSleep reference data, this means EEG-scored N1 or N2. The other usable stages (wake, N3/deep sleep, and REM) are grouped as “rest.”

EEG measures brain activity, which is the standard signal for distinguishing sleep stages. A watch does not have EEG. It mainly has motion and cardiovascular signals. A quiet person in REM, a quiet person in light sleep, and a person lying awake can have very similar movement and heart-rate readings. This means the project is attempting to estimate an EEG-derived label from weaker, indirect signals.

Large consumer-watch companies partly handle this with huge private datasets, more sensor types, years of tuning, and personal calibration across many nights. Our project starts with a public dataset and only two compatible signal types, so it cannot reasonably expect the same accuracy immediately.

## Data used

The source is BIDSleep v1.0.0, stored outside the repository. It contains Apple Watch accelerometer data and instantaneous heart-rate values aligned to 30-second EEG-derived stage labels.

Preparation was run with fixed seed `20260821` and produced ignored local artifacts under `ml/artifacts/`:

| Item | Result |
|---|---:|
| Participants | 47 |
| Prepared 30-second epochs | 213,387 |
| Rest-labelled epochs | 116,952 |
| Light-labelled epochs | 96,435 |
| Train / validation / test participants | 29 / 9 / 9 |
| Train / validation / test epochs | 134,645 / 42,377 / 36,365 before sequence filtering |

The split is by participant, not by individual rows. This is important: otherwise the model could learn a person's personal baseline in training and appear artificially good on the same person's test-night data.

## Inputs given to the first models

Each 30-second epoch was converted to 12 wearable-compatible summary features:

- movement magnitude: mean, standard deviation, median absolute deviation, activity count, and zero-crossing rate;
- heart rate: mean and standard deviation;
- IBI statistics where present: mean and RMSSD;
- motion and heart-rate coverage ratios; and
- an off-body flag.

For a prediction, the models received ten consecutive epochs (five minutes) of these summaries. They did **not** receive raw high-frequency accelerometer traces, a PPG waveform, EEG, oxygen saturation, skin temperature, or a user's personalised history.

## What was tried

All reported choices used validation participants for tuning. The test participants were not used to choose a threshold or candidate within an experiment.

| Experiment | Main choice | Held-out test F1 |
|---|---|---:|
| Initial scaled logistic baseline | Fixed 0.5 threshold | 0.552 |
| CNN-GRU | Ten engineered epochs, 32 hidden units | 0.475 |
| CNN-LSTM | Ten engineered epochs, 32 hidden units | 0.515 |
| Tuned logistic regression | Validation-selected regularisation and threshold | 0.626 |
| XGBoost and extra-trees sweeps | Validation-only nonlinear candidates | Did not beat tuned logistic on validation |
| Causal time-context logistic model | Adds elapsed-session and clock-time encodings | **0.641** |

The neural models performed worse than the simple logistic model. That does not mean neural networks are useless; it means a neural network cannot invent sleep information that is absent from the 12 summary features. With only a small sequence of compressed values, a simpler regularised model generalised better.

## Best measured result

The current best experiment uses causal information available on a phone at prediction time:

- 12 existing features for each of ten epochs;
- elapsed time since the current contiguous sleep session; and
- clock-time sine/cosine encodings.

It used regularised logistic regression with `C = 0.3`. Its probability threshold (`0.345`) was chosen on validation participants to optimise F1, then applied unchanged to held-out test participants.

| Metric on held-out test epochs | Result |
|---|---:|
| Accuracy | 52.4% |
| Precision for light sleep | 49.0% |
| Recall for light sleep | 92.7% |
| F1 for light sleep | 64.1% |
| ROC-AUC | 60.7% |

Confusion matrix (rows are actual class; columns are predicted class):

|  | Predicted rest | Predicted light |
|---|---:|---:|
| Actual rest | 3,540 | 15,819 |
| Actual light | 1,196 | 15,185 |

## What those numbers mean in simple terms

The model is tuned to miss as few light-sleep epochs as possible. It finds about 93 out of 100 light epochs, but it also calls many rest epochs “light.” That produces a better F1 score but poor precision and weak accuracy.

For an alarm, those false positives matter. A user might be woken during REM, deep sleep, or quiet wakefulness while the model says “likely light.” Therefore this experiment is useful research evidence, but it is not safe to present as a reliable sleep-stage detector.

The F1 improvement from 0.552 to 0.641 is real under the recorded procedure, but 0.641 is still moderate—not a breakthrough.

## The real bottleneck

The main problem is not that we have not tried enough model brands or libraries.

The main problem is **signal information**:

1. The target labels come from EEG, but the inputs are only watch motion and instantaneous heart rate.
2. The pipeline reduces each 30-second period to a handful of averages and counts, discarding fine-grained movement and heart-rate patterns.
3. N1/N2 overlap substantially with other states when observed only through watch-style signals.
4. The dataset has many epochs but only 47 independent people, which limits how confidently a model can generalise to new users.
5. The source device is an Apple Watch while the app target is a Samsung Watch4, creating an additional device-domain mismatch.

Changing from logistic regression to XGBoost or another model can make incremental differences. It cannot remove this information limit.

## Plausible ways to improve the project

### 1. Use richer causal raw-signal inputs — recommended next research step

Instead of feeding only 12 summaries, create a model input from each 30-second accelerometer segment and its aligned heart-rate representation. A causal CNN/GRU could learn motion bursts, motion rhythm, short-term heart-rate change, and relationships across several minutes.

Requirements:

- document a fixed resampling/masking method for irregular heart-rate observations;
- never use sensor values from after the predicted epoch;
- preserve subject-held-out splits;
- calculate all normalisation statistics from train participants only; and
- compare against the current logistic baseline using the same test subjects.

This is the most credible route to a meaningful improvement, but it is a larger data-and-model change, not a quick library swap.

### 2. Add honest causal wearable features

Possible features include rolling one-, two-, and five-minute movement statistics; movement-burst duration; HR trend; HR recovery; HR sample spacing; per-session baseline deviation; and robust missing-data features. Each feature must also be implementable from Watch4/phone data at inference time.

### 3. Personal calibration

After several consented nights, the app can calculate each user's own normal motion and heart-rate baseline. This may improve a *smart wake* decision, but it does not create reliable clinical stage labels. Any personal calibration must stay local and be explicitly described as wellness estimation.

### 4. Change the product objective, not the evidence

The app's decision can be stricter than the model's stage label. For example, it can trigger early only when there is high confidence, good sensor coverage, and a stable low-motion pattern. At all other times it waits for the exact-time fallback alarm. This can reduce risky early alarms even if epoch-level F1 stays moderate.

### 5. Use more or better-matched training data

A larger licence-compatible dataset with watch-like motion and cardiovascular data plus reference labels could help. It must be evaluated separately; data from a different population/device should not be mixed casually or used to make clinical claims.

## Things we should not do

- Do not select a model by repeatedly checking the test split until the number looks good. That creates an inflated score that will fail on new people.
- Do not claim the model knows the exact sleep stage.
- Do not copy a PSG/EEG sleep-staging model and pretend it works from watch motion and HR alone.
- Do not use future sensor data to predict the current epoch; the live alarm cannot see the future.
- Do not remove the exact-time fallback alarm.

## Questions to bring back

1. Is the main presentation goal a strong academic classification metric, or a safe smart-alarm demonstration?
2. Are we allowed to add a licence-compatible dataset with more watch-like signals or participants?
3. Can we collect several consented personal nights for local calibration analysis?
4. Is using clock time acceptable for the product, given it may learn people’s typical sleep schedules?
5. Should the next effort focus on raw-signal CNN/GRU inputs, richer engineered features, or a stricter alarm-decision policy?

## Current status

The preparation artifacts and exploratory reports are local, ignored derived files. The causal-time feature experiment has not been made into a deployment model, an ONNX asset, or an Android app feature. No model is currently selected for release.
