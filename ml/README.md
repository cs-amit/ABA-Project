# Smart Sleep ML environment

This directory contains local, reproducible preprocessing, model training,
evaluation, and ONNX export tools for wellness estimates. It is not a
diagnostic workflow.

Create a Python 3.11 environment and install `ml/requirements.txt`, then run
`python -m pytest ml/tests -q` from the repository root. The checked-in code
does not contain raw BIDSleep/MESA data, access tokens, checkpoints, or ONNX
artifacts; those remain authorized, local, Git-ignored files.

## Authorized reproduction

Set `SMART_SLEEP_DATA_DIR` to a local BIDSleep raw-data directory only on an
authorized machine. The frozen participant split is supplied as a local JSON
file. The validation-only recovery workflow is:

```powershell
python -m ml.performance_experiment regenerate `
  --raw-root $env:SMART_SLEEP_DATA_DIR `
  --frozen-splits ml/artifacts/bidsleep_frozen_splits.json
python -m ml.performance_experiment run `
  --bidsleep-artifacts ml/artifacts/bidsleep_corrected_validation `
  --frozen-splits ml/artifacts/bidsleep_frozen_splits.json `
  --mesa-artifacts ml/artifacts/mesa_500
```

These commands write only ignored artifacts. `evaluate-frozen` is a separate,
one-shot operation for the already frozen candidate and must not be rerun or
used for tuning:

```powershell
python -m ml.performance_experiment evaluate-frozen `
  --frozen-candidate ml/artifacts/performance_recovery/frozen_candidate.json `
  --raw-root $env:SMART_SLEEP_DATA_DIR `
  --frozen-splits ml/artifacts/bidsleep_frozen_splits.json
```

For the MESA pipeline, use the documented `ml.prepare_mesa`,
`ml.mesa_baseline`, and `ml.mesa_expansion` CLIs with separately authorized
local files. Never put a token, signed URL, raw data, or generated model into
Git or share a token with teammates.

Training data must have compatible motion and cardiovascular signals, documented sleep-stage reference labels, a confirmed licence, and subject-level train/validation/test splits. Samsung Health stages may be used only as weak calibration labels, never as ground truth.
