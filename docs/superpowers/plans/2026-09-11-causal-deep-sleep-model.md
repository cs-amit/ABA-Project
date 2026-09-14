# Causal Deep Sleep Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a causal raw-window CNN-GRU/LSTM experiment path with integrity checks and truthful subject-held-out evaluation.

**Architecture:** Raw wearable epochs are resampled then assembled into contiguous current-and-past windows. A temporal CNN encodes each epoch and a unidirectional recurrent layer aggregates epoch embeddings. Evaluation reports calibration and class-imbalance-aware metrics alongside F1.

**Tech Stack:** Python 3.11, NumPy, pandas, PyTorch, scikit-learn, pytest.

**Spec:** `docs/superpowers/specs/2026-09-11-causal-deep-sleep-model-design.md`

## Global Constraints

- Do not modify raw datasets or subject split allocation.
- Never feed future epochs or PSG channels to a deployable candidate.
- Fit transforms and select thresholds from train/validation only.
- Use test-first development and run focused tests before the full ML suite.

---

### Task 1: Add causal raw-window construction

**Files:**
- Create: `ml/causal_windows.py`
- Create: `ml/tests/test_causal_windows.py`

**Interfaces:**
- Produces: `build_causal_windows(sequences, labels, splits, epoch_starts, window_epochs, max_gap_s) -> (windows, labels, splits)`.

- [ ] **Step 1: Write failing tests** proving a window contains only past/current epochs, skips a gap, and preserves the final epoch label/split.
- [ ] **Step 2: Run `python -m pytest ml/tests/test_causal_windows.py -q`** and observe import failure.
- [ ] **Step 3: Implement the smallest NumPy window builder** with strict contiguous/gap checks and argument validation.
- [ ] **Step 4: Re-run the focused test** and then `python -m pytest ml/tests -q`.

### Task 2: Add a hierarchical causal CNN-recurrent model

**Files:**
- Modify: `ml/models.py`
- Modify: `ml/tests/test_models.py`

**Interfaces:**
- Produces: `RawCnnRecurrent(window_epochs, samples_per_epoch, channels, hidden_size, recurrent)` accepting `[batch, epochs, samples, channels]` and returning probabilities `[batch, 1]`.

- [ ] **Step 1: Write failing shape and causality-compatible forward-pass tests** for GRU and LSTM variants.
- [ ] **Step 2: Run `python -m pytest ml/tests/test_models.py -q`** and observe the missing-model failure.
- [ ] **Step 3: Implement per-epoch 1D convolution, pooling, and unidirectional recurrence** without bidirectional layers or future padding.
- [ ] **Step 4: Re-run focused and full ML tests.**

### Task 3: Add integrity/evaluation reporting

**Files:**
- Create: `ml/metrics.py`
- Create: `ml/tests/test_metrics.py`

**Interfaces:**
- Produces: `evaluate_binary_probabilities(labels, probabilities, threshold) -> dict` including accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, prediction rate, and dummy baselines.

- [ ] **Step 1: Write failing tests for metric values and invalid one-class inputs.**
- [ ] **Step 2: Run the focused test and observe failure.**
- [ ] **Step 3: Implement scikit-learn-backed metrics with explicit zero-division behavior.**
- [ ] **Step 4: Run focused and full ML tests.**

### Task 4: Integrate a validation-only deep-learning diagnostic

**Files:**
- Modify: `ml/diagnose_raw.py`
- Modify: `ml/tests/test_diagnose_raw.py`

**Interfaces:**
- Consumes: raw artifact and Task 1 windows; emits a validation-only JSON report for baseline, CNN-GRU, and CNN-LSTM.

- [ ] **Step 1: Write failing tests for gap-safe window construction and report metric keys.**
- [ ] **Step 2: Run the focused test and observe failure.**
- [ ] **Step 3: Integrate window construction, shuffled minibatches, train-only standardization, and validation-only threshold selection.**
- [ ] **Step 4: Run focused and full ML tests; do not launch expensive training without an explicit dataset-ready command.**
