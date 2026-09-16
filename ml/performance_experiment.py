"""Validation-only BIDSleep performance experiments and corrected artifacts.

The regeneration command opens train and validation raw subjects only. The
loader rejects test identities before accessing labels or feature values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .data_contract import FEATURE_COLUMNS, validate_epoch_frame, write_manifest
from .features import read_official_bidsleep_epochs


def _read_frozen_splits(source: dict | Path | str) -> dict[str, list[str]]:
    allocation = json.loads(Path(source).read_text(encoding="utf-8")) if not isinstance(source, dict) else source
    if set(allocation) != {"train", "validation", "test"}:
        raise ValueError("frozen allocation must specify train, validation, and test")
    seen = set()
    for split, subjects in allocation.items():
        if not isinstance(subjects, list) or any(not isinstance(subject, str) or not subject for subject in subjects):
            raise ValueError("frozen allocation must contain lists of subject strings")
        if len(set(subjects)) != len(subjects) or seen.intersection(subjects):
            raise ValueError("frozen subject appears more than once or in multiple splits")
        if split != "test" and not subjects:
            raise ValueError("frozen train and validation allocations must be non-empty")
        seen.update(subjects)
    return {split: list(subjects) for split, subjects in allocation.items()}


def _check_validation_identities(frame: pd.DataFrame, allocation: dict) -> None:
    if not {"subject_id", "split"}.issubset(frame.columns):
        raise ValueError("experiment epochs require subject_id and split columns")
    if frame["split"].eq("test").any():
        raise ValueError("test rows are prohibited in validation experiments")
    expected = {subject: split for split, subjects in allocation.items() for subject in subjects}
    assigned = frame["subject_id"].map(expected)
    if assigned.isna().any() or assigned.eq("test").any() or not assigned.eq(frame["split"]).all():
        raise ValueError("epochs violate the frozen subject allocation")
    required = set(allocation["train"]) | set(allocation["validation"])
    missing = required - set(frame["subject_id"])
    if missing:
        raise ValueError(f"missing subjects from frozen train/validation allocation: {sorted(missing)}")


def load_validation_epochs(
    source: pd.DataFrame | Path | str, frozen_splits: dict | Path | str
) -> pd.DataFrame:
    """Load only an already separated train/validation artifact, failing closed.

    For Parquet input, read identity columns first. A mixed artifact is rejected
    without loading its test label or feature columns; callers must regenerate
    a separate validation artifact rather than silently filtering test rows.
    """
    allocation = _read_frozen_splits(frozen_splits)
    if isinstance(source, pd.DataFrame):
        _check_validation_identities(source, allocation)
        frame = source.copy()
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "epochs.parquet"
        identities = pd.read_parquet(path, columns=["subject_id", "split"])
        _check_validation_identities(identities, allocation)
        frame = pd.read_parquet(path)
        _check_validation_identities(frame, allocation)
    validate_epoch_frame(frame)
    try:
        values = frame[["epoch_start_s", *FEATURE_COLUMNS]].to_numpy(dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("epoch times and features must be finite numeric values") from error
    if not np.isfinite(values).all():
        raise ValueError("epoch times and features must be finite numeric values")
    return frame.sort_values(["subject_id", "epoch_start_s"], kind="stable").reset_index(drop=True)


def _require_ignored_output(output_dir: Path) -> None:
    ancestor = output_dir
    while not ancestor.exists():
        ancestor = ancestor.parent
    for name in ("epochs.parquet", "splits.json", "dataset_manifest.json"):
        result = subprocess.run(
            ["git", "-C", str(ancestor), "check-ignore", "-q", "--", str(output_dir / name)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise ValueError("corrected artifact output must be Git-ignored (use ml/artifacts/)")


def regenerate_corrected_bidsleep(
    raw_root: Path | str, output_dir: Path | str, frozen_splits: dict | Path | str
) -> tuple[pd.DataFrame, dict]:
    """Regenerate corrected train/validation epochs without changing old outputs."""
    allocation = _read_frozen_splits(frozen_splits)
    raw_root, output_dir = Path(raw_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"use a new corrected artifact directory: {output_dir}")
    _require_ignored_output(output_dir)
    subjects = allocation["train"] + allocation["validation"]
    for subject in subjects:
        subject_path = (raw_root / subject).resolve()
        if subject_path.parent != raw_root or not subject_path.is_dir():
            raise ValueError(f"frozen subject has no direct raw directory: {subject}")
    frame, unknown = read_official_bidsleep_epochs(raw_root, subjects=subjects)
    reverse = {subject: split for split, names in allocation.items() for subject in names}
    frame["split"] = frame["subject_id"].map(reverse)
    frame = load_validation_epochs(frame, allocation)
    manifest = {
        "dataset": "BIDSleep", "release_version": "1.0.0",
        "artifact_kind": "corrected_validation_only",
        "feature_revision": "centered-magnitude-adjacent-zcr-v1",
        "feature_columns": FEATURE_COLUMNS, "epoch_seconds": 30,
        "splits": allocation, "included_splits": ["train", "validation"],
        "unknown_stage_count": unknown,
        "source_layout": "Bidslab*/<night>/{motion.csv,hr.csv,labels.mat}",
        "label_mapping": {"N1": 1, "N2": 1, "Wake": 0, "N3": 0, "REM": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output_dir / "epochs.parquet", index=False)
    write_manifest(output_dir / "splits.json", allocation)
    write_manifest(output_dir / "dataset_manifest.json", manifest)
    return frame, manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    regenerate = commands.add_parser("regenerate", help="regenerate corrected train/validation epochs only")
    regenerate.add_argument("--raw-root", type=Path, required=True)
    regenerate.add_argument("--frozen-splits", type=Path, required=True)
    regenerate.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/bidsleep_corrected_validation"))
    args = parser.parse_args(argv)
    frame, _ = regenerate_corrected_bidsleep(args.raw_root, args.output_dir, args.frozen_splits)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "epoch_count": len(frame), "included_splits": ["train", "validation"]}))


if __name__ == "__main__":
    main()
