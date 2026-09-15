"""Prepare an aligned, participant-held-out MESA pilot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import pandas as pd

from .mesa import MESA_FEATURE_COLUMNS, align_participant, expand_stage_events


def stratified_pilot_split(pilot: pd.DataFrame, seed: int) -> dict[str, list[str]]:
    """Allocate two of each race/gender age trio to train and balance holdouts."""
    required = {"mesaid", "race1c", "gender1", "age_band"}
    missing = sorted(required - set(pilot.columns))
    if missing:
        raise ValueError(f"pilot manifest missing stratification columns: {missing}")
    if pilot["mesaid"].duplicated().any():
        raise ValueError("pilot manifest contains duplicate mesaid values")
    frame = pilot.copy()
    frame["subject_id"] = frame["mesaid"].map(lambda value: f"{int(value):04d}")
    allocation = {"train": [], "validation": [], "test": []}
    races = sorted(frame["race1c"].unique())
    genders = sorted(frame["gender1"].unique())
    for race_index, race in enumerate(races):
        for gender_index, gender in enumerate(genders):
            block = frame[(frame["race1c"] == race) & (frame["gender1"] == gender)]
            if len(block) != 3 or block["age_band"].nunique() != 3:
                raise ValueError("each race/gender block must contain three distinct age bands")
            subjects = sorted(block["subject_id"].tolist())
            random.Random(seed + race_index * 10 + gender_index).shuffle(subjects)
            held_out = "validation" if gender_index == race_index % 2 else "test"
            allocation["train"].extend(subjects[:2])
            allocation[held_out].append(subjects[2])
    for values in allocation.values():
        values.sort()
    counts = {split: len(subjects) for split, subjects in allocation.items()}
    if counts != {"train": 16, "validation": 4, "test": 4}:
        raise ValueError(f"pilot split must contain 16/4/4 subjects, got {counts}")
    return allocation


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_path(mesa_root: Path, relative: str) -> Path:
    root = mesa_root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("pilot source path must stay within mesa_root")
    if not path.is_file():
        raise FileNotFoundError(f"missing MESA pilot source: {relative}")
    return path


def _seconds(value: str) -> int:
    try:
        hours, minutes, seconds = map(int, str(value).split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("overlap times must use HH:MM:SS") from exc
    if not 0 <= hours < 24 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError("overlap times must use HH:MM:SS")
    return hours * 3600 + minutes * 60 + seconds


def _validate_overlap_time(row) -> None:
    line_seconds = _seconds(row.linetime)
    start_seconds = _seconds(row.starttime_psg)
    rounded = ((start_seconds + 15) // 30 * 30) % 86400
    if line_seconds != rounded:
        raise ValueError("overlap linetime does not match rounded PSG start time")


def prepare_mesa_pilot(
    mesa_root: Path,
    pilot_manifest: Path,
    overlap_csv: Path,
    output_dir: Path,
    seed: int = 20260915,
) -> tuple[pd.DataFrame, dict]:
    """Validate and prepare the checksum-backed 24-participant MESA pilot."""
    mesa_root = Path(mesa_root)
    pilot = pd.read_csv(pilot_manifest)
    allocation = stratified_pilot_split(pilot, seed)
    pilot["subject_id"] = pilot["mesaid"].map(lambda value: f"{int(value):04d}")

    overlap = pd.read_csv(overlap_csv)
    overlap_required = {"mesaid", "line", "linetime", "starttime_psg"}
    overlap_missing = sorted(overlap_required - set(overlap.columns))
    if overlap_missing:
        raise ValueError(f"overlap mapping missing required columns: {overlap_missing}")
    if overlap["mesaid"].duplicated().any():
        raise ValueError("duplicate participant in overlap mapping")
    overlap = overlap.set_index(overlap["mesaid"].map(lambda value: f"{int(value):04d}"))

    phenotype_path = mesa_root / "datasets" / "mesa-sleep-dataset-0.8.0.csv"
    phenotype = pd.read_csv(phenotype_path)
    required_flags = {"mesaid", "match5", "havepsg5", "haveact5"}
    flag_missing = sorted(required_flags - set(phenotype.columns))
    if flag_missing:
        raise ValueError(f"MESA phenotype dataset missing flags: {flag_missing}")
    phenotype = phenotype.set_index(phenotype["mesaid"].map(lambda value: f"{int(value):04d}"))

    reverse_split = {subject: split_name for split_name, subjects in allocation.items() for subject in subjects}
    frames = []
    quality_totals = {"unlabelled_epochs": 0, "missing_activity_epochs": 0}
    sources = []
    for pilot_row in pilot.sort_values("mesaid").itertuples(index=False):
        subject_id = pilot_row.subject_id
        if subject_id not in phenotype.index:
            raise ValueError(f"participant {subject_id} missing from phenotype dataset")
        flags = phenotype.loc[subject_id]
        if isinstance(flags, pd.DataFrame):
            raise ValueError(f"duplicate participant {subject_id} in phenotype dataset")
        if not all(int(flags[name]) == 1 for name in ("match5", "havepsg5", "haveact5")):
            raise ValueError(f"participant {subject_id} does not have valid concurrent PSG and actigraphy")
        if subject_id not in overlap.index:
            raise ValueError(f"participant {subject_id} missing official overlap mapping")
        overlap_row = overlap.loc[subject_id]
        _validate_overlap_time(overlap_row)

        file_specs = [
            ("actigraphy", pilot_row.actigraphy_file, pilot_row.actigraphy_md5),
            ("events", pilot_row.events_file, pilot_row.events_md5),
            ("rpoints", pilot_row.rpoints_file, pilot_row.rpoints_md5),
        ]
        paths = {}
        for kind, relative, expected_md5 in file_specs:
            path = _source_path(mesa_root, str(relative))
            actual_md5 = _md5(path)
            if actual_md5.lower() != str(expected_md5).lower():
                raise ValueError(f"checksum mismatch for participant {subject_id} {kind}")
            paths[kind] = path
            sources.append({"subject_id": subject_id, "kind": kind, "path": str(relative), "md5": actual_md5})

        actigraphy = pd.read_csv(paths["actigraphy"])
        stages = expand_stage_events(paths["events"])
        rpoints = pd.read_csv(paths["rpoints"])
        frame, quality = align_participant(
            actigraphy,
            stages,
            rpoints,
            overlap_line=int(overlap_row["line"]),
            subject_id=subject_id,
        )
        frame["split"] = reverse_split[subject_id]
        frames.append(frame)
        for name in quality_totals:
            quality_totals[name] += int(quality[name])

    epochs = pd.concat(frames, ignore_index=True)
    expected_columns = ["subject_id", "epoch_start_s", "label", *MESA_FEATURE_COLUMNS, "split"]
    epochs = epochs.loc[:, expected_columns]
    if epochs.duplicated(["subject_id", "epoch_start_s"]).any():
        raise ValueError("prepared MESA epochs contain duplicate subject/time keys")
    for split_name in allocation:
        labels = epochs.loc[epochs["split"] == split_name, "label"]
        if labels.empty or labels.nunique() != 2:
            raise ValueError(f"prepared {split_name} split must contain both labels")

    class_counts = {str(int(label)): int(count) for label, count in epochs["label"].value_counts().sort_index().items()}
    manifest = {
        "dataset": "MESA Sleep",
        "release_version": "0.8.0",
        "seed": seed,
        "participant_count": int(epochs["subject_id"].nunique()),
        "row_count": int(len(epochs)),
        "class_counts": class_counts,
        "feature_columns": MESA_FEATURE_COLUMNS,
        "feature_formulas": {
            "activity_count": "MESA Actiwatch 30-second activity count; zero only when activity_observed is zero",
            "heart_rate_mean": "mean 60000/IBI for 300-2000 ms consecutive normal R-point intervals",
            "ibi_rmssd_ms": "root mean square of successive valid IBI differences within the epoch",
            "clock_sin_clock_cos": "sine and cosine of actigraphy clock seconds on a 24-hour period",
        },
        "alignment": "official mesa-actigraphy-psg-overlap.csv line is PSG epoch zero",
        "splits": allocation,
        "quality_counts": quality_totals,
        "source_files": sources,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs.to_parquet(output_dir / "epochs.parquet", index=False)
    (output_dir / "splits.json").write_text(json.dumps(allocation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return epochs, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare aligned MESA pilot epochs")
    parser.add_argument("--mesa-root", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--overlap-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    prepare_mesa_pilot(args.mesa_root, args.pilot_manifest, args.overlap_csv, args.output_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
