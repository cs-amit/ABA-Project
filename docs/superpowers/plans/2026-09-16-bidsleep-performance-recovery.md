# BIDSleep Performance Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover lost BIDSleep motion information and run a validation-only,
participant-balanced performance ladder before a single frozen test evaluation.

**Architecture:** A corrected ignored epoch artifact preserves subject splits.
A validation runner accepts train/validation rows only and saves each candidate's
configuration, threshold, per-subject validation metrics, and frozen checkpoint.
Shared-feature transfer preprocessing gains local-time clocks and log-scaled
activity; its trainer uses the same participant-macro validation objective as
the experiment gate.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-bidsleep-performance-recovery-design.md`

## Global Constraints

- Keep the BIDSleep test split out of feature/scaler/model fitting and candidate ranking.
- Rank on validation participant-macro F1; require pooled F1 non-regression and macro balanced-accuracy loss no worse than 0.005.
- Preserve frozen subject assignments and causal 30-second sequencing.
- Keep generated artifacts ignored by Git.

---

### Task 1: Corrected BIDSleep artifact and validation-only loader

**Files:** `ml/features.py`, `ml/performance_experiment.py`, `ml/tests/test_performance_experiment.py`

- [x] Write tests proving regenerated ZCR is nonzero for a changing acceleration
  signal and that the experiment loader rejects a frame containing `test` rows.
- [x] Run the tests and observe failure because the loader does not exist.
- [x] Implement an ignored-artifact regeneration command and a train/validation
  loader that checks frozen subject allocation, finite values, and no test rows.
- [x] Run focused and full ML tests; commit.

### Task 2: Participant-aligned training and feature semantics

**Files:** `ml/transfer_features.py`, `ml/transfer_experiment.py`,
`ml/tests/test_transfer_features.py`, `ml/tests/test_transfer_experiment.py`

- [x] Write failing tests for America/New_York clock encoding, log1p activity,
  equal-subject loss weights, and macro-F1 checkpoint selection.
- [x] Run focused tests and observe failure.
- [x] Implement the minimal preprocessing/training changes and preserve separate
  training-only dataset scalers.
- [x] Run focused and full ML tests; commit.

### Task 3: Validation ladder and frozen candidate

**Files:** `ml/performance_experiment.py`, `ml/tests/test_performance_experiment.py`, ignored `ml/artifacts/performance_recovery/`

- [x] Write failing tests for candidate ranking and frozen-candidate persistence.
- [x] Implement the declared validation ladder and report artifacts.
- [x] Regenerate corrected epochs and run the validation ladder without test data.
- [x] Freeze only a candidate that meets the global advancement gate; commit code and documentation.

### Task 4: One frozen evaluation and deployment smoke test

**Files:** `docs/evaluation/model-card.md`, `docs/evaluation/bidsleep-performance-recovery-report.md`, tests as needed.

- [x] Run the separate frozen evaluation command once if Task 3 advances a candidate.
- [x] Record paired participant metrics, calibration, limitations, and app/ONNX parity checks.
- [x] Run `pytest ml/tests -q`, `gradlew.bat test`, `git diff --check`, and status audit.
- [ ] Commit and push only after verification.
