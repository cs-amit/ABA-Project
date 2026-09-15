# MESA Expansion and BIDSleep Transfer Design

## Objective

Determine whether MESA pretraining improves light-sleep classification on the existing Watch-compatible BIDSleep target domain, while preserving subject isolation and avoiding the approximately 360 GB MESA EDF collection.

## Cohort

Select 500 MESA participants from the 1,719 locally verified eligible IDs that have actigraphy, NSRR PSG annotations, ECG R-points, and valid concurrent-study flags. Exclude all 24 pilot participants because their labels and frozen test performance have already influenced the decision to expand.

Selection uses demographics only: race/ethnicity, gender, and age band. Within available strata, sample deterministically with seed `20260916` and allocate 350 train, 75 validation, and 75 test participants. The selection process must not inspect sleep-stage labels, features, or model performance.

## Selective Download

Download only three participant files:

- `actigraphy/mesa-sleep-####.csv`;
- `polysomnography/annotations-events-nsrr/mesa-sleep-####-nsrr.xml`;
- `polysomnography/annotations-rpoints/mesa-sleep-####-rpoint.csv`.

Reuse the single official overlap mapping and release 0.8.0 phenotype files. Never request EDF/EDFZ files. Stream each file through a `.part` path, verify the NSRR-provided byte size and MD5, and atomically rename it. The downloader resumes by checksum and records a manifest without exposing the access token or signed URLs.

Expected storage is a few GB. Before downloading, calculate the exact server-reported byte total for the selected files and confirm sufficient free space.

## Dataset Preparation

Use the verified MESA adapter and corrected adjacent-normal-beat rule. Produce ignored expansion artifacts separately from the 24-participant pilot. The manifest records cohort selection, checksums, split IDs, alignment source, feature definitions, exclusions, coverage, label prevalence, and the exact code commit.

Preparation stops on a missing/duplicate overlap row, checksum mismatch, invalid concurrency flag, non-30-second actigraphy cadence, malformed stage duration, duplicate epoch key, cross-split subject, or a one-class split.

## Experiments

### MESA-500 native baseline

Run the frozen MESA-native ten-epoch logistic baseline on the 350/75/75 split. This estimates stability relative to the 24-person pilot but does not measure Watch-domain improvement.

### Common feature representation

Create an explicit shared sequence schema available from both datasets:

- activity count after training-only robust scaling within each dataset;
- activity availability;
- heart-rate mean and standard deviation;
- heart-rate availability;
- elapsed-session time;
- clock sine and cosine.

Do not fabricate XYZ statistics for MESA or IBI for BIDSleep. Dataset identity is not a model input.

### Target-domain comparison

Train the same small causal CNN-GRU architecture in two predeclared conditions:

1. BIDSleep-only: initialize randomly and train on BIDSleep training participants.
2. MESA transfer: pretrain on MESA training participants, then fine-tune on the identical BIDSleep training participants.

Use the same architecture, optimizer search space, seeds, early stopping rule, and BIDSleep validation participants. Select thresholds on BIDSleep validation only. Evaluate each frozen condition once on the existing BIDSleep test participants.

Because prior BIDSleep test results are already known, this comparison is exploratory rather than a pristine confirmatory test. Report paired per-subject changes and bootstrap confidence intervals; do not claim deployment improvement from a point estimate alone.

## Success Criteria

MESA expansion is operationally successful when 500 selected participants pass checksum/alignment preparation with documented coverage and no split leakage.

Transfer is promising when the pretrained condition improves BIDSleep participant-level macro F1 and balanced accuracy over the same-architecture BIDSleep-only condition, the bootstrap interval is reported, calibration does not materially regress, and gains are not caused by failure on a demographic subgroup. Otherwise retain MESA as a separate research benchmark.

## Outputs

Ignored local artifacts live under:

- `data/mesa/expansion_500_manifest.csv`;
- `ml/artifacts/mesa_500/`;
- `ml/artifacts/mesa_transfer/`.

Committed code includes the resumable selective downloader, cohort selector, shared-feature adapter, transfer experiment, tests, and documentation. Controlled raw data, tokens, trained weights, and participant-level artifact tables remain uncommitted.

## Verification

Tests cover demographic-only deterministic selection, exact file filtering, resume/checksum behavior, disk-space calculation, pilot exclusion, split isolation, common-feature equivalence, training-only normalization, pretrained weight loading, BIDSleep-only control parity, validation-only selection, and paired participant reporting.

Before reporting results, run the full ML and Gradle suites, reconcile every manifest/artifact count, and verify Git contains no token, controlled source file, signed URL, generated metric artifact, or model weight.
