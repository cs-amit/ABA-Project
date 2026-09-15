% Smart Sleep Alarm: Proposed Training Data
% ABA Project Team
% 24 August 2026

# The training-data question

## What must the model learn?

- Input: wearable-compatible movement and cardiovascular features in 30-second epochs.
- Target: `light` versus `not_light` sleep.
- Deployment target: Samsung Galaxy Watch4 accelerometer, heart rate, and inter-beat interval (IBI) data.
- Important: this is a wellness estimate, not a clinical diagnosis.

# Evaluated pilot dataset: MESA Sleep

## Why MESA Sleep?

- National Sleep Research Resource (NSRR) dataset from the NHLBI-supported Multi-Ethnic Study of Atherosclerosis.
- MESA Sleep enrolled **2,237** participants in a dedicated sleep exam.
- It includes full overnight polysomnography (PSG), seven-day wrist actigraphy, and sleep questionnaires.
- It offers the signal overlap we need: wrist movement, PSG sleep-stage reference labels, and ECG/heart-rate information.

Source: [NSRR MESA Sleep overview](https://sleepdata.org/datasets/mesa)

# Available data for the model

## Movement: wrist actigraphy

- Actiwatch Spectrum worn on the non-dominant wrist for one week.
- Epoch-by-epoch files are available for **2,159 participants**.
- Each row summarizes **30 seconds** of activity data.
- Fields include activity count, off-wrist indicator, light values, clock time, and wake indicator.

This aligns naturally with our planned 30-second feature epochs.

Source: [MESA actigraphy documentation](https://sleepdata.org/datasets/mesa/pages/actigraphy-introduction.md)

# Reference labels and cardiovascular signals

## PSG + ECG

- Raw PSG is available for **2,056 participants** with signal files and epoch-staging annotations.
- PSG annotations provide the reference sleep-stage labels.
- PSG includes ECG sampled at **256 Hz**; the dataset also documents heart-rate and HRV-related analysis resources.
- We will derive HR/IBI-compatible features, not use EEG as the deployed model input.

Source: [MESA PSG documentation](https://sleepdata.org/datasets/mesa/pages/polysomnography-introduction.md) and [equipment/sampling information](https://sleepdata.org/datasets/mesa/pages/equipment)

# Building the aligned training set

## Inclusion and alignment rules

1. Keep only participants with valid concurrent PSG and actigraphy.
2. Use the MESA `match5` alignment flag to identify concurrent recordings.
3. Align PSG stage annotations to 30-second actigraphy epochs.
4. Extract a MESA-native common feature set without fabricating raw XYZ values.
5. Exclude invalid/off-wrist epochs and record all exclusions.

The documented concurrent PSG/actigraphy subset contains **1,798** participants.

Source: [MESA `match5` variable](https://sleepdata.org/datasets/mesa/variables/match5)

# Labels, features, and evaluation

## Planned data contract

| Item | Decision |
| --- | --- |
| Positive label | PSG reference `LIGHT` sleep |
| Negative label | Awake, deep, or REM |
| Epoch length | 30 seconds |
| Feature families | Activity count, HR, IBI/HRV, availability, off-wrist state, causal time |
| Split rule | Subject-level train/validation/test split; no participant appears in more than one split |
| Pilot model | Class-balanced logistic regression over ten causal epochs |

Unknown, active, and unscored stages are excluded and counted rather than
treated as negative examples.

# Important limitations

## What we will and will not claim

- MESA devices are not identical to Galaxy Watch4 sensors: actigraphy is not raw three-axis Watch accelerometer data, and PSG ECG is not Watch PPG.
- Therefore this is a feature-compatibility training approach, not sensor-equivalence proof.
- PSG staging supplies reference labels; the deployed app still reports only a likely light-sleep wellness estimate.
- The 24-participant pilot is exploratory; its result is not deployment accuracy.

# Current status and next step

## Honest project status

- Access is approved and the release 0.8.0 pilot is checksum-verified.
- The selected files total about 127 MiB; no EDF files were downloaded.
- The adapter aligned 31,274 epochs from 24 participants with zero duplicate keys or split leakage.
- Activity was observed in 98.7% of aligned epochs and cardiac data in 88.6%.
- The deterministic split contains 16 train, 4 validation, and 4 test participants (seed 20260915).
- Frozen test F1 is **0.7932** and balanced accuracy is **0.7638**, compared with always-light F1 0.6637 and balanced accuracy 0.5000.
- Next: selectively expand the same three small modalities and overlap mapping to a larger participant cohort, then test harmonisation or transfer to the Watch-compatible BIDSleep feature path.

# References

- [NSRR MESA Sleep dataset](https://sleepdata.org/datasets/mesa)
- [MESA actigraphy introduction](https://sleepdata.org/datasets/mesa/pages/actigraphy-introduction.md)
- [MESA polysomnography introduction](https://sleepdata.org/datasets/mesa/pages/polysomnography-introduction.md)
- [MESA equipment and sampling rates](https://sleepdata.org/datasets/mesa/pages/equipment)
- [MESA PSG/actigraphy overlap variable](https://sleepdata.org/datasets/mesa/variables/match5)
