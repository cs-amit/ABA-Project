# Smart Sleep ML environment

This directory will hold local, reproducible preprocessing, model training, evaluation, and ONNX export tools for wellness estimates. It is not a diagnostic workflow.

Create a Python 3.11 virtual environment, install `requirements.txt`, and run future tests with `python -m pytest ml/tests -q` from the repository root.

Training data must have compatible motion and cardiovascular signals, documented sleep-stage reference labels, a confirmed licence, and subject-level train/validation/test splits. Samsung Health stages may be used only as weak calibration labels, never as ground truth.
