# Smart Sleep Alarm model card

## Status

No trained model is bundled yet. BIDSleep has not been downloaded or used.
Task 8 training requires Task 7's externally stored, permitted preparation
artifacts: `epochs.parquet`, `splits.json`, and `dataset_manifest.json`.

## Intended use

The eventual model estimates likely light sleep from the preceding ten causal
30-second wearable feature epochs. It is a wellness estimate only, not a
sleep-stage diagnosis or a disorder assessment. The phone fallback alarm is
independent of inference and always remains available.

## Data and limitations

The planned primary source is BIDSleep. Its Apple Watch signals and healthy
participant population do not establish performance for Galaxy Watch users or
people with sleep disorders. MESA is not used by this project and remains an
optional future comparison only. The final exported-model section is written
only after a reproducible permitted-data run records its exact manifest,
subject-held-out split, seed, metrics, selection rationale, and ONNX parity.
