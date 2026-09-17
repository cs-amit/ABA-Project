# MESA Expansion and Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select, download, prepare, and evaluate 500 independent MESA participants, then compare BIDSleep-only training with MESA-pretrained/BIDSleep-fine-tuned training on an identical common-feature CNN-GRU.

**Architecture:** A deterministic cohort catalog selects participants from server metadata and phenotype flags without reading labels. A resumable checksum downloader retrieves only actigraphy, NSRR XML, and R-point CSV files. Existing MESA alignment is generalized to manifest-defined splits, while a separate shared-feature experiment preserves dataset-specific training-only scaling and evaluates two frozen target-domain candidates once.

**Tech Stack:** Python 3.11, pandas, NumPy, requests, scikit-learn, PyTorch, PyArrow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-mesa-expansion-transfer-design.md`

## Global Constraints

- Select 500 MESA participants with seed `20260916`; exclude all 24 pilot IDs.
- Allocate 350 train, 75 validation, and 75 test subjects using demographic strata only.
- Download exactly actigraphy CSV, NSRR event XML, and R-point CSV files; never EDF/EDFZ.
- Verify server size and MD5 before atomic rename; never print tokens or authenticated URLs.
- Preserve separate MESA and BIDSleep scalers fitted on their training participants only.
- Use the identical CNN-GRU architecture and BIDSleep train/validation/test subjects for both transfer conditions.
- Evaluate each frozen BIDSleep test candidate once and label the comparison exploratory.
- Keep raw data, manifests with controlled paths, artifacts, and model weights ignored by Git.

---

### Task 1: Deterministic expansion cohort

**Files:**
- Create: `ml/mesa_expansion.py`
- Create: `ml/tests/test_mesa_expansion.py`

**Interfaces:**
- `build_file_catalog(entries: list[dict]) -> dict[str, dict[str, dict]]`
- `select_expansion_cohort(phenotype: pd.DataFrame, catalogs: dict, excluded_ids: set[str], count: int = 500, seed: int = 20260916) -> pd.DataFrame`
- `write_expansion_manifest(frame: pd.DataFrame, path: Path) -> None`

- [ ] **Step 1: Write failing catalog and selection tests**

Create literal metadata entries for all three permitted folders plus EDF decoys. Assert one exact three-file record per subject, no EDF path, exclusion of pilot IDs and invalid flags, deterministic selection, complete demographic strata, unique IDs, and exact 350/75/75 splits.

- [ ] **Step 2: Verify RED**

Run: `C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests/test_mesa_expansion.py -q`

Expected: import failure because `ml.mesa_expansion` does not exist.

- [ ] **Step 3: Implement catalog parsing and demographic-only selection**

Accept only these regular expressions:

```text
actigraphy/mesa-sleep-(\d{4}).csv
polysomnography/annotations-events-nsrr/mesa-sleep-(\d{4})-nsrr.xml
polysomnography/annotations-rpoints/mesa-sleep-(\d{4})-rpoint.csv
```

Intersect catalogs with `match5/havepsg5/haveact5 == 1`, remove excluded IDs, derive age bands `<65`, `65-74`, and `75+`, and use two deterministic `StratifiedShuffleSplit` operations on race/gender/age-band strata: first select 500, then split 350/150 and 75/75.

- [ ] **Step 4: Verify GREEN and commit**

Run focused and full ML tests, then commit:

```powershell
git add ml/mesa_expansion.py ml/tests/test_mesa_expansion.py
git commit -m "feat: select independent MESA expansion cohort"
```

---

### Task 2: Secure resumable selective downloader

**Files:**
- Modify: `ml/mesa_expansion.py`
- Modify: `ml/tests/test_mesa_expansion.py`

**Interfaces:**
- `required_download_bytes(manifest: pd.DataFrame, root: Path) -> int`
- `download_expansion(manifest: pd.DataFrame, root: Path, token_path: Path, workers: int = 8) -> dict`
- CLI subcommands: `catalog`, `download`, and `verify`.

- [ ] **Step 1: Write failing download-boundary tests**

Use a local fake HTTP server response object, not a mocked downloader. Assert valid files stream via `.part`, final size/MD5 match, valid existing files skip, corrupt existing files redownload, failed checksum removes only the `.part` file, path traversal is rejected, and logs/results contain no supplied token.

- [ ] **Step 2: Verify RED**

Run the focused download tests and confirm missing-function failures.

- [ ] **Step 3: Implement downloader and disk gate**

Compute missing bytes before network activity and require free space of missing bytes plus 10%. Use eight threads, per-file retries limited to two, 60-second connect/read timeouts, exact manifest paths, and the maintained NSRR download route. Return aggregate counts and bytes only.

- [ ] **Step 4: Verify GREEN and commit**

Run focused/full tests and commit the same two files with message `feat: download MESA expansion selectively`.

---

### Task 3: Generalized manifest preparation

**Files:**
- Modify: `ml/prepare_mesa.py`
- Create: `ml/tests/test_prepare_mesa_expansion.py`

**Interfaces:**
- `prepare_mesa_manifest(mesa_root: Path, source_manifest: Path, overlap_csv: Path, output_dir: Path, release: str, seed: int) -> tuple[pd.DataFrame, dict]`

- [ ] **Step 1: Write failing manifest-preparation tests**

Construct a six-subject fixture with explicit split values. Assert the function preserves assignments, validates all source checksums and phenotype flags, emits the existing `MESA_FEATURE_COLUMNS`, reconciles counts, and rejects cross-split duplicate IDs, unknown splits, missing classes, and pilot IDs when an exclusion file is supplied.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm the new interface is missing.

- [ ] **Step 3: Refactor preparation without changing pilot behavior**

Extract common per-subject preparation. Keep `prepare_mesa_pilot` and its tests unchanged, while the new function consumes manifest `split` values and writes release/seed/source-selection metadata.

- [ ] **Step 4: Verify GREEN and commit**

Run all MESA tests and the full ML suite; commit with `feat: prepare manifest-defined MESA cohorts`.

---

### Task 4: Select, download, prepare, and benchmark MESA-500

**Files:**
- Ignored: `data/mesa/expansion_500_manifest.csv`
- Ignored: `data/mesa/mesa/actigraphy/*.csv`
- Ignored: `data/mesa/mesa/polysomnography/annotations-events-nsrr/*.xml`
- Ignored: `data/mesa/mesa/polysomnography/annotations-rpoints/*.csv`
- Ignored: `ml/artifacts/mesa_500/`

- [ ] **Step 1: Fetch server metadata and select 500**

Read `.nsrr-token`, request only the three permitted folder catalogs, select the cohort without labels, calculate exact bytes, verify free space, and write the ignored manifest.

- [ ] **Step 2: Download and checksum all 1,500 files**

Run the resumable downloader. Require 1,500 verified files, zero `.part` files, zero failures, and no EDF paths.

- [ ] **Step 3: Prepare and audit epochs**

Run `prepare_mesa_manifest`; require 500 subjects, 350/75/75 assignments, both classes per split, zero duplicate keys/leakage/NaNs, and manifest reconciliation.

- [ ] **Step 4: Run MESA-native baseline once**

Run `ml.mesa_baseline` on `ml/artifacts/mesa_500`, preserve the validation threshold, frozen test metrics, and per-subject summaries, and do not tune from its test result.

---

### Task 5: Common BIDSleep/MESA feature sequences

**Files:**
- Create: `ml/transfer_features.py`
- Create: `ml/tests/test_transfer_features.py`

**Interfaces:**
- `COMMON_FEATURE_COLUMNS: list[str]`
- `adapt_mesa_common(frame: pd.DataFrame) -> pd.DataFrame`
- `adapt_bidsleep_common(frame: pd.DataFrame) -> pd.DataFrame`
- `build_common_sequences(frame: pd.DataFrame, sequence_epochs: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

- [ ] **Step 1: Write failing adapters tests**

Assert both adapters emit activity, activity availability, HR mean/std, HR availability, elapsed time, clock sine, and clock cosine in identical order. Assert no MESA XYZ or BIDSleep IBI value is fabricated and sequences do not cross subject, split, or time gaps.

- [ ] **Step 2: Verify RED**

Run the focused file and confirm import failure.

- [ ] **Step 3: Implement common adapters and sequences**

Derive BIDSleep elapsed time per contiguous session and clock encodings from epoch timestamps. Use MESA's aligned values directly. Preserve labels, subject IDs, and splits.

- [ ] **Step 4: Verify GREEN and commit**

Run focused/full tests and commit with `feat: harmonize MESA and BIDSleep features`.

---

### Task 6: Pretraining and paired target-domain evaluation

**Files:**
- Create: `ml/transfer_experiment.py`
- Create: `ml/tests/test_transfer_experiment.py`

**Interfaces:**
- `fit_dataset_scaler(train_sequences: np.ndarray) -> RobustScaler`
- `train_common_cnn_gru(...) -> tuple[CnnGru, dict]`
- `run_transfer_experiment(mesa_artifacts: Path, bidsleep_artifacts: Path, output_dir: Path, seed: int = 20260916) -> dict`
- `paired_subject_bootstrap(control: pd.DataFrame, transfer: pd.DataFrame, iterations: int = 10000, seed: int = 20260916) -> dict`

- [ ] **Step 1: Write failing isolation and weight-transfer tests**

Assert scalers fit only corresponding training sequences, both candidates start with identical architecture, control starts from the same seeded random weights, transfer loads MESA-pretrained weights before fine-tuning, early stopping and thresholds use BIDSleep validation only, and no test tensor reaches training or selection functions.

- [ ] **Step 2: Verify RED**

Run the focused file and confirm missing interfaces.

- [ ] **Step 3: Implement deterministic training**

Use hidden size 32, Adam learning rate `0.001`, batch size 256, at most 30 pretraining epochs and 20 fine-tuning epochs, patience 5, class weights from the active training set, and seeds `20260916`, `20260917`, and `20260918`. Freeze both candidates before a single BIDSleep test evaluation.

- [ ] **Step 4: Implement paired reporting**

Write validation choices, test accuracy/balanced accuracy/precision/recall/F1/AUC/PR-AUC/Brier, calibration, per-subject metrics, paired deltas, and 10,000-subject bootstrap intervals. Store weights only in ignored artifacts.

- [ ] **Step 5: Verify GREEN and commit**

Run focused/full tests and commit with `feat: compare MESA pretraining on BIDSleep`.

---

### Task 7: Run transfer study and document evidence

**Files:**
- Modify: `docs/evaluation/model-card.md`
- Modify: `docs/data/public-dataset-protocol.md`
- Create: `docs/evaluation/mesa-expansion-transfer-report.md`

- [ ] **Step 1: Audit inputs before training**

Verify MESA-500 and BIDSleep manifests, exact subject isolation, both-class splits, feature order, finite tensors, scaler fit sources, and the absence of pilot IDs in MESA-500.

- [ ] **Step 2: Run the two predeclared conditions once**

Execute BIDSleep-only and MESA-pretrained/BIDSleep-fine-tuned training. Do not add conditions or tune from BIDSleep test results.

- [ ] **Step 3: Report exact results and limitations**

Document MESA-500 stability, both target-domain candidates, paired subject deltas/intervals, calibration, demographic limitations, and whether transfer met the promising-result gate. Do not call the result confirmatory or deployment accuracy.

- [ ] **Step 4: Final verification**

```powershell
C:\Users\AMIT\anaconda3\envs\fashion-trends\python.exe -m pytest ml/tests -q
.\gradlew.bat test
git diff --check
git status --short --untracked-files=all
```

Require all tests/builds to pass and no token, controlled data, artifact, or model weight in Git status.

- [ ] **Step 5: Commit and push**

Commit the report/documentation, push `feat/mesa-pilot-model`, and update PR #1 with the expansion and transfer evidence.
