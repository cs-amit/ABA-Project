# MESA Pilot Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, participant-held-out MESA pilot benchmark for binary light-sleep classification without EDF files or fabricated accelerometer features.

**Architecture:** A MESA adapter aligns actigraphy, PSG stages, and ECG R-points from the official overlap line and emits a dataset-native epoch schema. A separate evaluator builds causal ten-epoch sequences, fits preprocessing and a balanced logistic model on training participants, selects its threshold on validation participants, and evaluates test participants once.

**Tech Stack:** Python 3.11, pandas, NumPy, scikit-learn, PyArrow, pytest, XML ElementTree.

**Spec:** `docs/superpowers/specs/2026-09-15-mesa-pilot-model-design.md`

## Global Constraints

- Use MESA release `0.8.0` and seed `20260915`.
- Never download EDF files or guess PSG alignment from clock time.
- Never represent activity counts as raw XYZ acceleration.
- Keep tokens, source data, and derived participant artifacts ignored by Git.
- Keep participants isolated across train, validation, and test.
- Fit all preprocessing on training data and select thresholds on validation data.
- Treat results from 24 participants as exploratory.

---

### Task 1: Stage, cardiac, and actigraphy alignment

**Files:**
- Create: `ml/mesa.py`
- Create: `ml/tests/test_mesa.py`

**Interfaces:**
- Produces: `MESA_FEATURE_COLUMNS: list[str]`
- Produces: `expand_stage_events(xml_path: Path) -> pd.DataFrame`
- Produces: `derive_cardiac_epochs(rpoints: pd.DataFrame, epoch_count: int) -> pd.DataFrame`
- Produces: `align_participant(actigraphy: pd.DataFrame, stages: pd.DataFrame, rpoints: pd.DataFrame, overlap_line: int, subject_id: str) -> tuple[pd.DataFrame, dict]`

- [ ] **Step 1: Write failing stage tests**

Create an NSRR XML fixture with compressed `Stages|Stages` events. Assert exact 30-second expansion and mappings `0/3/4/5 -> 0`, `1/2 -> 1`. Assert rejection of fractional epochs and overlapping events; stage `9` must be excluded and counted.

```python
stages = expand_stage_events(xml_path)
assert stages[["epoch_index", "stage", "label"]].to_dict("records") == [
    {"epoch_index": 0, "stage": 0, "label": 0},
    {"epoch_index": 1, "stage": 1, "label": 1},
    {"epoch_index": 2, "stage": 1, "label": 1},
    {"epoch_index": 3, "stage": 5, "label": 0},
]
```

- [ ] **Step 2: Run and verify RED**

Run: `C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests/test_mesa.py -q`

Expected: collection fails because `ml.mesa` does not exist.

- [ ] **Step 3: Implement stage expansion and verify GREEN**

Parse only `Stages|Stages`, require finite starts/durations divisible by 30, reject duplicate epoch indices, and return `epoch_index`, `epoch_start_s`, `stage`, and nullable `label`. Run the focused tests until they pass.

- [ ] **Step 4: Write failing cardiac tests**

Assert IBI is assigned to the later normal beat, HR is `60000 / ibi_ms`, RMSSD uses consecutive valid IBIs, and availability is normal beats divided by all beats in the epoch. Reject unsorted/nonpositive seconds and epoch indices outside `1..epoch_count`.

```python
cardiac = derive_cardiac_epochs(
    pd.DataFrame({"epoch": [1, 1, 1, 1], "seconds": [1.0, 2.0, 3.1, 4.3], "Type": [1, 1, 1, 1]}), 1
)
assert cardiac.loc[0, "ibi_mean_ms"] == pytest.approx(1100.0)
assert cardiac.loc[0, "ibi_rmssd_ms"] == pytest.approx(100.0)
```

- [ ] **Step 5: Run cardiac tests and verify RED**

Expected: failure because `derive_cardiac_epochs` is missing.

- [ ] **Step 6: Implement cardiac features and verify GREEN**

Keep `Type == 1`; derive adjacent-beat intervals; retain 300--2000 ms intervals; aggregate HR mean/std, IBI mean/RMSSD, and valid-beat ratio per PSG epoch. Use zeros plus availability for missing measurements.

- [ ] **Step 7: Write failing alignment tests**

Assert PSG epoch zero joins `line == overlap_line`, later epochs increment by one, missing/off-wrist activity sets `activity_observed = 0`, and time encodings come from `linetime`. Reject missing/duplicate/gapped lines and non-30-second cadence.

- [ ] **Step 8: Implement alignment and verify GREEN**

Use this exact feature order:

```python
MESA_FEATURE_COLUMNS = [
    "activity_count", "activity_observed", "off_wrist",
    "heart_rate_mean", "heart_rate_standard_deviation",
    "ibi_mean_ms", "ibi_rmssd_ms", "heart_rate_valid_ratio",
    "elapsed_hours", "clock_sin", "clock_cos",
]
```

Return `subject_id`, `epoch_start_s`, `label`, the features, and quality counts. Run the focused file and full `ml/tests` suite.

- [ ] **Step 9: Commit**

```powershell
git add ml/mesa.py ml/tests/test_mesa.py
git commit -m "feat: add validated MESA epoch alignment"
```

---

### Task 2: Pilot preparation and manifest

**Files:**
- Create: `ml/prepare_mesa.py`
- Create: `ml/tests/test_prepare_mesa.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `stratified_pilot_split(pilot: pd.DataFrame, seed: int) -> dict[str, list[str]]`
- Produces: `prepare_mesa_pilot(mesa_root: Path, pilot_manifest: Path, overlap_csv: Path, output_dir: Path, seed: int = 20260915) -> tuple[pd.DataFrame, dict]`
- Produces CLI arguments `--mesa-root`, `--pilot-manifest`, `--overlap-csv`, `--output-dir`, and `--seed`.

- [ ] **Step 1: Write failing split tests**

Create 24 subjects spanning race, gender, and age bands. Assert deterministic `16/4/4` allocation, full coverage, no intersection, and demographic representation in validation and test.

- [ ] **Step 2: Run and verify RED**

Run: `C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests/test_prepare_mesa.py -q`

Expected: collection fails because `ml.prepare_mesa` does not exist.

- [ ] **Step 3: Implement split assignment and verify GREEN**

Within each race/gender block, seed-shuffle age bands, assign two subjects to training, and alternate the third between validation and test. Require exactly `16/4/4` IDs.

- [ ] **Step 4: Write failing preparation tests**

Build fixture files and assert column order, checksum validation, flag validation, unique overlap rows, split isolation, label counts, exclusions, and manifest feature formulas. Add failure cases for `match5 != 1`, missing files, wrong MD5, and duplicate overlap mappings.

- [ ] **Step 5: Run tests and verify RED**

Expected: failures because `prepare_mesa_pilot` is missing.

- [ ] **Step 6: Implement preparation and artifact writing**

Read checksums from `pilot_subjects.csv`; validate `match5/havepsg5/haveact5` in `datasets/mesa-sleep-dataset-0.8.0.csv`; require overlap columns `mesaid,line,linetime,starttime_psg`; process IDs in numeric order; then write `epochs.parquet`, `splits.json`, and `dataset_manifest.json`. Record release, seed, feature order/formulas, checksums, inclusions, exclusions, classes, splits, and alignment method.

- [ ] **Step 7: Verify tests and ignore rules**

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests -q
git check-ignore .nsrr-token data/mesa/pilot_subjects.csv ml/artifacts/mesa_pilot/epochs.parquet
```

Expected: all tests pass and all three paths are ignored.

- [ ] **Step 8: Commit**

```powershell
git add .gitignore ml/prepare_mesa.py ml/tests/test_prepare_mesa.py
git commit -m "feat: prepare aligned MESA pilot epochs"
```

---

### Task 3: Exploratory baseline evaluation

**Files:**
- Create: `ml/mesa_baseline.py`
- Create: `ml/tests/test_mesa_baseline.py`

**Interfaces:**
- Produces: `build_mesa_sequences(frame: pd.DataFrame, sequence_epochs: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Produces: `select_validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]`
- Produces: `run_mesa_baseline(artifacts_dir: Path, seed: int = 20260915) -> dict`

- [ ] **Step 1: Write failing sequence tests**

Assert ten-epoch windows never cross participant, split, or time gaps, output subject IDs come from final epochs, and tensors use `MESA_FEATURE_COLUMNS` order.

- [ ] **Step 2: Run and verify RED**

Run: `C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests/test_mesa_baseline.py -q`

Expected: collection fails because `ml.mesa_baseline` does not exist.

- [ ] **Step 3: Implement sequence construction and verify GREEN**

Return shapes `[windows, 10, 11]`, `[windows]`, and `[windows]`; skip any noncontiguous window.

- [ ] **Step 4: Write failing evaluation tests**

Use separable synthetic participants. Assert training-only scaling, validation-only threshold selection, complete metrics from `evaluate_binary_probabilities`, confusion counts, prevalence, and per-subject rows limited to test IDs.

- [ ] **Step 5: Implement evaluation and verify GREEN**

Fit `StandardScaler` on flattened training sequences and `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)`. Maximize validation F1, tie-breaking by distance to `0.5` then lower threshold. Freeze it for test evaluation and write `metrics.json` plus `per_subject_metrics.csv`.

- [ ] **Step 6: Run focused and full tests**

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests/test_mesa_baseline.py -q
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests -q
```

- [ ] **Step 7: Commit**

```powershell
git add ml/mesa_baseline.py ml/tests/test_mesa_baseline.py
git commit -m "feat: evaluate MESA pilot baseline"
```

---

### Task 4: Acquire mapping and run the pilot

**Files:**
- Ignored input: `data/mesa/mesa/actigraphy/mesa-actigraphy-psg-overlap.csv`
- Ignored outputs: `ml/artifacts/mesa_pilot/`

- [ ] **Step 1: Download only the overlap CSV**

Use `.nsrr-token` with the authenticated NSRR metadata/download endpoint already used for the pilot. Select only the path ending `actigraphy/mesa-actigraphy-psg-overlap.csv`, stream to `.part`, validate server byte size and MD5, and atomically rename. Print only status codes and exception classes, never tokens, headers, signed URLs, or exception strings.

- [ ] **Step 2: Verify mapping coverage**

Require columns `mesaid,line,linetime,starttime_psg`, one row for each of 24 IDs, positive integer lines, and equal rounded PSG/actigraphy times.

- [ ] **Step 3: Prepare the pilot**

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m ml.prepare_mesa --mesa-root data/mesa/mesa --pilot-manifest data/mesa/pilot_subjects.csv --overlap-csv data/mesa/mesa/actigraphy/mesa-actigraphy-psg-overlap.csv --output-dir ml/artifacts/mesa_pilot --seed 20260915
```

Expected: 24 participants, nonzero rows and both labels in every split, and zero checksum/alignment failures.

- [ ] **Step 4: Audit before training**

Reconcile participant/split/row/class counts with the manifest; require no duplicate subject/epoch keys or cross-split IDs; report feature missingness, on-wrist and cardiac coverage, and time gaps. Stop if reconciliation fails or a split has one class.

- [ ] **Step 5: Evaluate test once**

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m ml.mesa_baseline --artifacts-dir ml/artifacts/mesa_pilot --seed 20260915
```

Do not tune features or thresholds from test results.

---

### Task 5: Documentation and expansion decision

**Files:**
- Modify: `docs/data/mesa-training-data-presentation.md`
- Modify: `docs/data/public-dataset-protocol.md`
- Modify: `docs/evaluation/model-card.md`
- Create: `ml/tests/test_mesa_documentation.py`

- [ ] **Step 1: Write and run a failing documentation test**

Assert stale claims that MESA is unapproved/not downloaded are absent and the model card contains release `0.8.0`, 24 participants, seed `20260915`, F1, balanced accuracy, dummy comparison, and the exploratory limitation. Confirm failure before edits.

- [ ] **Step 2: Update documentation from artifacts**

Record access, verified pilot download, preparation, and evaluation as distinct states. Copy exact metrics from `metrics.json`; claim improvement only if frozen test evidence supports it.

- [ ] **Step 3: Run final verification**

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests -q
.\gradlew.bat test
git diff --check
git status --short --untracked-files=all
```

Expected: Python and Gradle tests pass, no whitespace errors exist, and no token, controlled source file, or generated artifact appears in status.

- [ ] **Step 4: Commit**

```powershell
git add docs/data/mesa-training-data-presentation.md docs/data/public-dataset-protocol.md docs/evaluation/model-card.md ml/tests/test_mesa_documentation.py
git commit -m "docs: report MESA pilot benchmark"
```

- [ ] **Step 5: Apply the expansion gate**

Recommend selective expansion only when checksum/alignment failures are zero, activity/cardiac coverage is documented, validation and frozen-test F1 and balanced accuracy beat dummy baselines, and per-subject results do not totally fail a demographic group. Otherwise retain the pilot as a diagnostic and revise only with training/validation data.
