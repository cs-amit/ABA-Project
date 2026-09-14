# Smart Sleep Alarm

## BIDSleep Dataset and Machine-Learning Implementation Plan

**Purpose:** Class presentation and course implementation plan  
**Primary dataset:** BIDSleep — *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels* (PhysioNet, 2026)  
**Dataset page:** https://physionet.org/content/bidsleep-dataset/1.0.0/  
**Project status:** BIDSleep is open access and is the dataset we will use to train the course models. MESA Sleep access has been requested separately. It is not part of current training or reported results.

---

## 1. Project goal

The Smart Sleep Alarm is an Android phone-and-watch prototype. The Watch collects motion and heart-rate data while the user sleeps. The phone will eventually use a trained model to estimate whether the user is in a light-sleep state during an alarm window, then choose a suitable time to trigger the alarm.

The project is an educational prototype. It is **not** a medical device and will not diagnose sleep disorders or make clinical claims.

## 2. Why BIDSleep is our primary dataset

BIDSleep is the best dataset for the current course requirement because it can be used immediately and its inputs closely match the signals available from our Samsung Watch4:

| Requirement | BIDSleep provides | Why it matters |
|---|---|---|
| Wearable movement | 3-axis accelerometer data | Matches Watch accelerometer data. |
| Wearable heart rate | Instantaneous heart-rate readings derived from PPG | Matches the Watch heart-rate signal type. |
| Training labels | Expert-reviewed EEG-based sleep stages every 30 seconds | Gives a target value for supervised learning. |
| Enough data | 47 participants and 253 nights | Produces far more than 1,000 labelled 30-second windows. |
| Immediate use | Open access under ODC Attribution licence | No access-review delay for the course model. |

The dataset was recorded with an Apple Watch and a Dreem 2 EEG headband. The device is not identical to the Samsung Watch4, but the model will use device-independent window features such as movement magnitude, mean heart rate, and heart-rate change. This makes BIDSleep an appropriate training source for our prototype while retaining an important cross-device limitation.

## 3. Dataset overview

- **Participants:** 47 healthy adults.
- **Recording duration:** up to seven nights per person, for **253 nights** in total.
- **Signals:** Apple Watch three-axis acceleration and instantaneous heart rate.
- **Reference labels:** sleep stages from the Dreem 2 EEG headband, reviewed by a trained sleep expert.
- **Label frequency:** one label for every 30-second epoch.
- **Classes:** Wake, N1, N2, N3, REM, and Unknown.

One night is stored in a participant folder using three files:

| File | What it contains | How we use it |
|---|---|---|
| `motion.csv` | Timestamp plus x, y, z acceleration | Creates motion features for each 30-second window. |
| `hr.csv` | Timestamp plus heart rate in beats per minute | Creates heart-rate features for each 30-second window. |
| `labels.mat` | Recording start time and sleep-stage labels | Supplies the correct answer (target) for supervised learning. |

### What counts as one observation?

One observation is **one 30-second epoch**, not one participant. A 6-hour night alone has 720 epochs. Across 253 nights, the dataset should provide roughly hundreds of thousands of labelled windows before cleaning; the exact total will be reported only after we download and validate the files.

There are only 47 independent people. Therefore, model evaluation must keep a person entirely in either the training set or the test set. Randomly splitting individual windows would leak a person's normal movement and heart-rate pattern into both groups and give an unrealistically high score.

## 4. Label plan

The original labels are:

| Original label | Meaning | Initial binary target |
|---|---|---|
| Wake | Awake | Not-light sleep |
| N1 | Light non-REM sleep | Light sleep |
| N2 | Light non-REM sleep | Light sleep |
| N3 | Deep / slow-wave sleep | Not-light sleep |
| REM | Rapid eye movement sleep | Not-light sleep |
| Unknown | Unusable or unscored | Exclude from training and evaluation |

### Initial model target

For the first version, we will train a **binary classifier**:

`Light sleep = N1 or N2`  
`Not-light sleep = Wake, N3, or REM`

This target directly supports the alarm decision: during the configurable wake-up window, the system looks for a window predicted as light sleep. Later, we may also train a five-class model (Wake/N1/N2/N3/REM) to compare performance, but that is a secondary experiment because multi-class classification is harder and some classes are less common.

## 5. Step-by-step data preparation

### Step 1 — Download and record the dataset version

We will download BIDSleep from the official PhysioNet page and store a dataset manifest containing the version, download date, file checksums where available, and licence/citation information. Raw source data will not be edited.

### Step 2 — Validate each participant-night

For every night, the preparation program will check that all three required files exist and can be read. It will check timestamps for invalid values, duplicates, missing labels, and out-of-range heart-rate values. A report will show included and excluded nights with reasons.

### Step 3 — Align sensor samples to labels

The recording start time, `recStart`, is the start of epoch 1. Every motion or heart-rate timestamp is placed into its matching 30-second epoch:

`epoch_index = floor((timestamp - recStart) / 30)`

Only sensor samples that fall inside the recording interval will be used. This creates one aligned sensor window per available sleep label.

### Step 4 — Clean the sensor data

For each 30-second epoch we will:

- remove rows with missing timestamps or values;
- discard `Unknown` labels;
- remove impossible or clearly corrupt heart-rate readings according to documented, pre-set limits;
- identify epochs with too little sensor coverage and exclude them rather than guessing values;
- preserve an audit column that records why an epoch was excluded.

We will not fill a long missing sequence with invented measurements. For a short heart-rate gap, a carefully documented interpolation rule may be evaluated only if it is applied consistently within the training data.

### Step 5 — Derive wearable-compatible features

The raw signals are converted into a compact feature row per epoch. Initial features will be deliberately simple and reproducible:

| Signal | Example features per 30-second epoch |
|---|---|
| Acceleration | Vector magnitude, mean, standard deviation, minimum, maximum, signal energy, and number of movement peaks. |
| Heart rate | Mean BPM, minimum, maximum, standard deviation, rate of change, and available-sample count. |
| Quality/context | Motion-sample count, heart-rate-sample count, and missing-data flags. |

Acceleration magnitude will be calculated from all three axes so that feature meaning is less dependent on how the person wore the watch:

`magnitude = sqrt(x^2 + y^2 + z^2)`

### Step 6 — Build two model-ready forms

We will create two different representations from the same cleaned data:

1. **Feature table for baseline ML:** one row per epoch, with engineered features and the target label.
2. **Time-series tensors for deep learning:** a fixed-size sequence of acceleration and heart-rate values for each epoch. We will record the resampling method, padding/masking rules, and sensor-coverage features so that irregular heart-rate sampling does not silently distort the data.

## 6. Training and evaluation plan

### Dataset split

We will split by participant, not by individual rows. A practical first split is approximately 70% of participants for training, 15% for validation, and 15% for testing. The held-out test participants will not influence feature normalisation, model selection, or tuning.

All normalisation values—such as mean and standard deviation—will be calculated from the training participants only, then applied unchanged to validation and test participants.

### Model 1 — baseline machine learning

The first model will be a **Random Forest classifier** using the engineered 30-second features. This model is quick to train and helps us establish a transparent baseline. Feature importance can show whether movement, heart rate, or data quality is driving a prediction.

### Model 2 — deep learning

The main deep-learning experiment will be a **1D CNN plus GRU**:

`sensor sequence -> 1D CNN -> GRU -> dense classifier -> light / not-light probability`

- The **1D CNN** learns short movement and heart-rate patterns within a window.
- The **GRU** learns the time relationship between those patterns while using only sensor data already collected by the alarm.
- The final layer outputs the predicted probability of light sleep.

We chose a GRU rather than a bidirectional LSTM because the alarm must work in real time. A bidirectional LSTM uses both past and future sequence context, which is useful for offline scoring of a completed night but would make a live prediction depend on sensor data that has not happened yet. A GRU is causal when used in the normal forward direction, so it can make the same kind of prediction during deployment that it made during evaluation. It also has fewer gates and parameters than a standard LSTM, which usually means faster training and lower phone-side inference cost while still modelling temporal patterns.

If the CNN-GRU model is too large for the data volume or time budget, we will train a smaller 1D CNN or GRU-only model. We will document the exact final architecture, parameter count, training epochs, early-stopping rule, and random seed.

### Evaluation metrics

We will report:

- class counts before and after cleaning;
- accuracy;
- precision, recall, and F1 score for light sleep;
- confusion matrix;
- ROC-AUC for the binary target when appropriate;
- participant-level test performance, not only pooled epoch-level performance.

Because classes may be unbalanced, F1 score and confusion matrices are more informative than accuracy alone.

## 7. How the model connects to our Android prototype

1. The Watch4 captures accelerometer and heart-rate samples.
2. The Watch sends ordered sensor batches to the Android phone.
3. The phone groups the received samples into 30-second windows.
4. The phone applies the same feature calculations and data-quality checks used during training.
5. The saved model estimates a light-sleep probability for each available window.
6. During the configured alarm window, the phone chooses a suitable predicted-light-sleep moment; at the hard deadline it triggers the normal phone alarm.

The current prototype already captures Watch sensor data and transfers ordered batches to the phone. Model inference, persistence of labelled training data, and the final alarm-decision policy remain future implementation tasks.

## 8. MESA Sleep: secondary comparison only

MESA Sleep access has been requested. If approval is granted, we will first inspect the permitted files, data dictionary, labels, and actigraphy/PSG overlap before using it. It will not replace BIDSleep automatically.

Possible uses of MESA after approval are:

- compare the BIDSleep-trained pipeline against an independent dataset;
- assess whether results change with a larger, older population;
- retrain or fine-tune only after confirming comparable labels and signal representations;
- document any age/device/domain mismatch rather than mixing incompatible records without justification.

MESA is an older cohort and was designed to study subclinical cardiovascular disease. It is valuable research data, but it may not represent young smartwatch users as closely as BIDSleep. Therefore, BIDSleep remains the primary dataset for this course project.

## 9. Limitations and ethical use

- BIDSleep used an Apple Watch, while our prototype runs on a Samsung Watch4; sensor sampling and algorithms may differ.
- The dataset contains 47 people, so the number of independent participants is modest despite having many 30-second windows.
- Sleep labels are based on EEG-headband scoring, not full clinical PSG; they are still suitable as supervised reference labels for a course prototype.
- A result on this dataset is not proof of clinical accuracy or safety.
- We will keep the dataset de-identified, follow its licence and citation requirements, and will not make medical claims.

## 10. Presentation-ready summary

> We are training our course ML and deep-learning models primarily on BIDSleep, an open dataset with 253 nights of Apple Watch accelerometer and heart-rate data aligned with expert-reviewed 30-second sleep-stage labels. We convert each 30-second period into wearable-compatible features and predict light sleep versus not-light sleep. We will evaluate using participant-level splits to prevent data leakage. MESA access has been requested; if it is approved, we will use it as a separate comparison dataset rather than claim it is already part of our training.

## References

1. Song, T.-A. *A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels*, PhysioNet, version 1.0.0, 2026. https://physionet.org/content/bidsleep-dataset/1.0.0/
2. National Sleep Research Resource. *MESA Sleep.* https://sleepdata.org/datasets/mesa
