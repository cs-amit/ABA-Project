"""Validation-only BIDSleep performance experiments and corrected artifacts.

The regeneration command opens train and validation raw subjects only. The
loader rejects test identities before accessing labels or feature values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .data_contract import FEATURE_COLUMNS, validate_epoch_frame, write_manifest
from .features import read_official_bidsleep_epochs


def build_validation_sequences(frame: pd.DataFrame, feature_columns: list[str], sequence_epochs: int = 10):
    """Causal windows over a specified schema, preserving IDs and session gaps."""
    if frame["split"].eq("test").any():
        raise ValueError("test rows are prohibited in validation experiments")
    if sequence_epochs <= 0:
        raise ValueError("sequence_epochs must be positive")
    arrays, labels, subjects, splits = [], [], [], []
    ordered = frame.sort_values(["subject_id", "split", "epoch_start_s"], kind="stable")
    for (split, subject), group in ordered.groupby(["split", "subject_id"], sort=False):
        starts = group["epoch_start_s"].to_numpy()
        values = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group["label"].to_numpy(dtype=np.int64)
        if not np.isfinite(values).all():
            raise ValueError("sequence features must be finite")
        boundaries = np.flatnonzero(np.diff(starts) != 30) + 1
        for indices in np.split(np.arange(len(group)), boundaries):
            if len(indices) < sequence_epochs:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(values[indices], sequence_epochs, axis=0).transpose(0, 2, 1).copy()
            arrays.append(windows)
            labels.append(targets[indices[sequence_epochs - 1:]])
            subjects.append(np.repeat(str(subject), len(windows)))
            splits.append(np.repeat(str(split), len(windows)))
    return (np.concatenate(arrays) if arrays else np.empty((0, sequence_epochs, len(feature_columns)), dtype=np.float32),
            np.concatenate(labels) if labels else np.empty(0, dtype=np.int64),
            np.concatenate(subjects) if subjects else np.empty(0, dtype=str),
            np.concatenate(splits) if splits else np.empty(0, dtype=str))


def rank_validation_candidates(candidates: list[dict], baseline_name: str) -> tuple[list[str], str | None]:
    """Rank by validation macro F1; freeze the highest-ranked eligible candidate."""
    by_name = {row["name"]: row for row in candidates}
    if len(by_name) != len(candidates) or baseline_name not in by_name:
        raise ValueError("unique candidates and a declared baseline are required")
    for row in candidates:
        scores = [row["participant_macro_f1"], row["pooled"]["f1"], row["participant_macro_balanced_accuracy"]]
        if not np.isfinite(scores).all() or not all(0 <= value <= 1 for value in scores):
            raise ValueError("candidate validation metrics must be finite and in [0, 1]")
    baseline = by_name[baseline_name]
    ranking = sorted(by_name, key=lambda name: (-by_name[name]["participant_macro_f1"], name))
    for name in ranking:
        row = by_name[name]
        if (name != baseline_name
                and row["participant_macro_f1"] >= baseline["participant_macro_f1"] + .01 - 1e-12
                and row["pooled"]["f1"] >= baseline["pooled"]["f1"] - 1e-12
                and row["participant_macro_balanced_accuracy"] >= baseline["participant_macro_balanced_accuracy"] - .005 - 1e-12):
            return ranking, name
    return ranking, None


def persist_frozen_candidate(candidate: dict, output_dir: Path | str) -> Path:
    """Persist a one-time immutable selection record bound to checkpoint bytes."""
    path = Path(output_dir) / "frozen_candidate.json"
    checkpoint = Path(candidate["checkpoint_path"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"candidate": candidate, "checkpoint_sha256": digest,
                   "selection_split": "validation", "test_evaluated": False}, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def run_validation_ladder(
    source: pd.DataFrame | Path | str, frozen_splits: dict | Path | str,
    mesa_source: Path | str, output_dir: Path | str, *, seed: int = 20260916,
    max_epochs: int = 20, pretrain_epochs: int = 30, patience: int = 5,
    sequence_epochs: int = 10, threads: int = 4,
) -> dict:
    """Run the ladder without leaking Torch runtime settings to callers."""
    import torch
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_threads = torch.get_num_threads()
    try:
        return _run_validation_ladder(source, frozen_splits, mesa_source, output_dir,
                                      seed=seed, max_epochs=max_epochs, pretrain_epochs=pretrain_epochs,
                                      patience=patience, sequence_epochs=sequence_epochs, threads=threads)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(original_threads)


def _run_validation_ladder(
    source, frozen_splits, mesa_source, output_dir, *, seed, max_epochs,
    pretrain_epochs, patience, sequence_epochs, threads,
) -> dict:
    """Execute the predeclared three-condition ladder with no BIDSleep test reads.

    The corrected native logistic is the same-artifact baseline. Historical
    results are deliberately excluded because preprocessing has changed.
    """
    frame = load_validation_epochs(source, frozen_splits)  # Must precede any sequence construction.
    allocation = _read_frozen_splits(frozen_splits)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"use a new experiment directory: {output_dir}")
    _require_ignored_output(output_dir, _VALIDATION_LADDER_OUTPUT_NAMES)
    if min(max_epochs, pretrain_epochs, patience, sequence_epochs, threads) <= 0:
        raise ValueError("epoch, patience, sequence and thread settings must be positive")
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import torch
    from .metrics import evaluate_binary_probabilities
    from .transfer_features import COMMON_FEATURE_COLUMNS, adapt_bidsleep_common, adapt_mesa_common
    from .transfer_experiment import (
        _per_subject, _predict, _transform, equal_subject_loss_weights,
        fit_dataset_scaler, select_participant_macro_threshold, train_common_cnn_gru,
    )
    torch.set_num_threads(threads)
    output_dir.mkdir(parents=True, exist_ok=False)
    participants = {split: sorted(allocation[split]) for split in ("train", "validation")}
    candidates = []
    configuration = {"seed": seed, "max_epochs": max_epochs, "pretrain_epochs": pretrain_epochs,
                     "patience": patience, "sequence_epochs": sequence_epochs, "threads": threads,
                     "baseline": "native_logistic", "declared_ladder": ["native_logistic", "shared_control", "mesa_transfer"],
                     "advancement_gate": {"macro_f1_min_delta": .01, "pooled_f1_min_delta": 0, "macro_balanced_accuracy_min_delta": -.005}}
    write_manifest(output_dir / "declared_configuration.json", configuration)

    def prepare(data, columns, scaling):
        x, y, subjects, splits = build_validation_sequences(data, columns, sequence_epochs)
        for split in ("train", "validation"):
            mask = splits == split
            if not mask.any() or set(np.unique(y[mask])) != {0, 1}:
                raise ValueError(f"{split} sequences must contain both classes")
            expected = set(data.loc[data.split == split, "subject_id"])
            if set(subjects[mask]) != expected:
                raise ValueError(f"{split} participants lost during sequence construction")
        train = splits == "train"
        scaler = (StandardScaler().fit(x[train].reshape(-1, len(columns)))
                  if scaling == "standard" else fit_dataset_scaler(x[train]))
        scaled = _transform(scaler, x)
        return scaled, y, subjects, splits, scaler

    def record(name, model, prepared, columns, config, history=None):
        x, y, subjects, splits, scaler = prepared
        validation = splits == "validation"
        if name == "native_logistic":
            probabilities = model.predict_proba(x[validation].reshape(validation.sum(), -1))[:, 1]
            checkpoint = output_dir / f"{name}.joblib"
            joblib.dump({"model": model, "scaler": scaler}, checkpoint)
        else:
            probabilities = _predict(model, x[validation])
            checkpoint = output_dir / f"{name}.pt"
            torch.save(model.state_dict(), checkpoint)
            joblib.dump(scaler, output_dir / f"{name}_scaler.joblib")
        threshold, macro = select_participant_macro_threshold(y[validation], probabilities, subjects[validation])
        per_subject = _per_subject(subjects[validation], y[validation], probabilities, threshold)
        per_subject.to_csv(output_dir / f"{name}_validation_participants.csv", index=False)
        np.savez_compressed(output_dir / f"{name}_validation_predictions.npz", probabilities=probabilities,
                            labels=y[validation], subjects=subjects[validation])
        center = scaler.mean_ if hasattr(scaler, "mean_") else scaler.center_
        row = {"name": name, "participants": participants, "feature_columns": columns,
               "feature_revision": "centered-magnitude-adjacent-zcr-v1",
               "model_config": {**config, "sequence_epochs": sequence_epochs},
               "scaler": {"type": type(scaler).__name__, "fit_split": "train", "fit_dataset": "BIDSleep",
                          "center": center.tolist(), "scale": scaler.scale_.tolist(), "fit_subjects": participants["train"]},
               "validation_threshold": threshold, "pooled": evaluate_binary_probabilities(y[validation], probabilities, threshold),
               "participant_macro_f1": macro,
               "participant_macro_balanced_accuracy": float(per_subject.balanced_accuracy.mean()),
               "checkpoint_path": str(checkpoint), "training": history,
               "sequence_counts": {split: int((splits == split).sum()) for split in ("train", "validation")}}
        candidates.append(row)
        write_manifest(output_dir / f"{name}_validation.json", row)
        print(json.dumps({"candidate": name, "validation_macro_f1": macro, "pooled_f1": row["pooled"]["f1"]}), flush=True)

    native = prepare(frame, FEATURE_COLUMNS, "standard")
    x, y, subjects, splits, _ = native
    train = splits == "train"
    logistic = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    logistic.fit(x[train].reshape(train.sum(), -1), y[train], sample_weight=equal_subject_loss_weights(y[train], subjects[train]))
    if logistic.n_iter_.max() >= 1000:
        raise RuntimeError("native logistic failed to converge within the declared iteration limit")
    record("native_logistic", logistic, native, FEATURE_COLUMNS,
           {"name": "logistic", "C": 1.0, "penalty": "l2", "max_iter": 1000, "seed": seed, "equal_subject_weighting": True})
    del native, x, logistic
    shared = prepare(adapt_bidsleep_common(frame), COMMON_FEATURE_COLUMNS, "robust")
    neural_config = {"name": "causal_cnn_gru", "hidden_size": 32, "seed": seed + 1,
                     "max_epochs": max_epochs, "patience": patience, "batch_size": 256,
                     "learning_rate": .001, "equal_subject_weighting": True,
                     "checkpoint_metric": "participant_macro_f1", "activity_transform": "log1p_nonnegative",
                     "clock_timezone": "America/New_York", "elapsed_time": "reset_at_non_30_second_gap"}

    def neural_fit(prepared, fit_seed, epochs, initial_state=None):
        x, y, subjects, splits, _ = prepared
        train, validation = splits == "train", splits == "validation"
        return train_common_cnn_gru(x[train], y[train], x[validation], y[validation],
                                   train_subjects=subjects[train], validation_subjects=subjects[validation],
                                   seed=fit_seed, max_epochs=epochs, patience=patience, initial_state=initial_state)

    control, history = neural_fit(shared, seed + 1, max_epochs)
    record("shared_control", control, shared, COMMON_FEATURE_COLUMNS, neural_config, history)
    del control
    mesa_path = Path(mesa_source)
    if mesa_path.is_dir():
        mesa_path /= "epochs.parquet"
    # Predicate pushdown ensures even source-dataset test rows stay out of memory.
    mesa_frame = pd.read_parquet(mesa_path, filters=[("split", "in", ["train", "validation"])])
    mesa = prepare(adapt_mesa_common(mesa_frame), COMMON_FEATURE_COLUMNS, "robust")
    pretrained, pretrain_history = neural_fit(mesa, seed, pretrain_epochs)
    torch.save(pretrained.state_dict(), output_dir / "mesa_pretrained.pt")
    joblib.dump(mesa[-1], output_dir / "mesa_scaler.joblib")
    write_manifest(output_dir / "mesa_pretraining.json", {
        "source": str(mesa_path.resolve()), "feature_columns": COMMON_FEATURE_COLUMNS,
        "participants": {split: sorted(set(mesa[2][mesa[3] == split])) for split in ("train", "validation")},
        "scaler": {"fit_split": "train", "fit_dataset": "MESA", "center": mesa[-1].center_.tolist(), "scale": mesa[-1].scale_.tolist()},
        "training": pretrain_history})
    del mesa, mesa_frame
    transfer, history = neural_fit(shared, seed + 1, max_epochs, pretrained.state_dict())
    record("mesa_transfer", transfer, shared, COMMON_FEATURE_COLUMNS,
           {**neural_config, "initial_checkpoint": str(output_dir / "mesa_pretrained.pt")}, history)
    ranking, selected = rank_validation_candidates(candidates, "native_logistic")
    if selected is not None:
        persist_frozen_candidate(next(row for row in candidates if row["name"] == selected), output_dir)
    report = {**configuration, "study_type": "exploratory", "selection_split": "validation",
              "baseline_rationale": "corrected native logistic on the same artifact; historical preprocessing differs",
              "test_evaluated": False, "candidates": candidates, "ranking": ranking, "frozen_candidate": selected}
    write_manifest(output_dir / "validation_report.json", report)
    return report


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


_REGENERATION_OUTPUT_NAMES = ("epochs.parquet", "splits.json", "dataset_manifest.json")
_VALIDATION_LADDER_OUTPUT_NAMES = (
    "declared_configuration.json", "native_logistic.joblib",
    "native_logistic_validation_participants.csv", "native_logistic_validation_predictions.npz",
    "native_logistic_validation.json", "shared_control.pt", "shared_control_scaler.joblib",
    "shared_control_validation_participants.csv", "shared_control_validation_predictions.npz",
    "shared_control_validation.json", "mesa_pretrained.pt", "mesa_scaler.joblib",
    "mesa_pretraining.json", "mesa_transfer.pt", "mesa_transfer_scaler.joblib",
    "mesa_transfer_validation_participants.csv", "mesa_transfer_validation_predictions.npz",
    "mesa_transfer_validation.json", "frozen_candidate.json", "validation_report.json",
)


def _require_ignored_output(output_dir: Path, output_names: tuple[str, ...]) -> None:
    ancestor = output_dir
    while not ancestor.exists():
        ancestor = ancestor.parent
    for name in output_names:
        result = subprocess.run(
            ["git", "-C", str(ancestor), "check-ignore", "-q", "--", str(output_dir / name)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            raise ValueError("artifact output must be Git-ignored (use ml/artifacts/)")


def regenerate_corrected_bidsleep(
    raw_root: Path | str, output_dir: Path | str, frozen_splits: dict | Path | str
) -> tuple[pd.DataFrame, dict]:
    """Regenerate corrected train/validation epochs without changing old outputs."""
    allocation = _read_frozen_splits(frozen_splits)
    raw_root, output_dir = Path(raw_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"use a new corrected artifact directory: {output_dir}")
    _require_ignored_output(output_dir, _REGENERATION_OUTPUT_NAMES)
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
    run = commands.add_parser("run", help="run and freeze the declared validation-only ladder")
    run.add_argument("--bidsleep-artifacts", type=Path, required=True)
    run.add_argument("--frozen-splits", type=Path, required=True)
    run.add_argument("--mesa-artifacts", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, default=Path("ml/artifacts/performance_recovery"))
    run.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "regenerate":
        frame, _ = regenerate_corrected_bidsleep(args.raw_root, args.output_dir, args.frozen_splits)
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), "epoch_count": len(frame), "included_splits": ["train", "validation"]}))
    else:
        report = run_validation_ladder(args.bidsleep_artifacts, args.frozen_splits, args.mesa_artifacts,
                                       args.output_dir, threads=args.threads)
        print(json.dumps({"output_dir": str(args.output_dir.resolve()), "ranking": report["ranking"],
                          "frozen_candidate": report["frozen_candidate"]}))


if __name__ == "__main__":
    main()
