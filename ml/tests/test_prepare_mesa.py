import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from ml.mesa import MESA_FEATURE_COLUMNS
from ml.prepare_mesa import prepare_mesa_manifest, prepare_mesa_pilot, stratified_pilot_split


def _pilot_frame() -> pd.DataFrame:
    rows = []
    subject = 1
    for race in range(1, 5):
        for gender in (0, 1):
            for age_band in ("under65", "65to74", "75plus"):
                rows.append(
                    {
                        "mesaid": subject,
                        "race1c": race,
                        "gender1": gender,
                        "age_band": age_band,
                    }
                )
                subject += 1
    return pd.DataFrame(rows)


def test_stratified_pilot_split_is_deterministic_complete_and_disjoint():
    pilot = _pilot_frame()

    first = stratified_pilot_split(pilot, seed=20260915)
    second = stratified_pilot_split(pilot, seed=20260915)

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {"train": 16, "validation": 4, "test": 4}
    allocated = [subject for ids in first.values() for subject in ids]
    assert len(allocated) == len(set(allocated)) == 24
    assert set(allocated) == {f"{value:04d}" for value in pilot["mesaid"]}


def test_stratified_pilot_split_represents_all_races_and_genders_in_held_out_sets():
    pilot = _pilot_frame()
    lookup = pilot.assign(subject_id=pilot.mesaid.map(lambda value: f"{value:04d}")).set_index("subject_id")

    allocation = stratified_pilot_split(pilot, seed=20260915)

    for split in ("validation", "test"):
        held_out = lookup.loc[allocation[split]]
        assert set(held_out["race1c"]) == {1, 2, 3, 4}
        assert set(held_out["gender1"]) == {0, 1}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write_pilot_source(tmp_path: Path):
    mesa_root = tmp_path / "mesa"
    (mesa_root / "actigraphy").mkdir(parents=True)
    (mesa_root / "polysomnography" / "annotations-events-nsrr").mkdir(parents=True)
    (mesa_root / "polysomnography" / "annotations-rpoints").mkdir(parents=True)
    (mesa_root / "datasets").mkdir(parents=True)
    pilot_rows = []
    overlap_rows = []
    phenotype_rows = []
    for row in _pilot_frame().itertuples(index=False):
        sid = f"{row.mesaid:04d}"
        act_rel = Path("actigraphy") / f"mesa-sleep-{sid}.csv"
        event_rel = Path("polysomnography/annotations-events-nsrr") / f"mesa-sleep-{sid}-nsrr.xml"
        rpoint_rel = Path("polysomnography/annotations-rpoints") / f"mesa-sleep-{sid}-rpoint.csv"
        pd.DataFrame(
            {
                "line": [100, 101, 102],
                "linetime": ["22:00:00", "22:00:30", "22:01:00"],
                "offwrist": [0, 0, 0],
                "activity": [1.0, 2.0, 3.0],
            }
        ).to_csv(mesa_root / act_rel, index=False)
        (mesa_root / event_rel).write_text(
            "<PSGAnnotation><ScoredEvents>"
            "<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>30</Duration></ScoredEvent>"
            "<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2 sleep|2</EventConcept><Start>30</Start><Duration>60</Duration></ScoredEvent>"
            "</ScoredEvents></PSGAnnotation>",
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "epoch": [1, 1, 2, 2, 3, 3],
                "seconds": [1.0, 2.0, 31.0, 32.0, 61.0, 62.0],
                "Type": [1, 1, 1, 1, 1, 1],
            }
        ).to_csv(mesa_root / rpoint_rel, index=False)
        pilot_rows.append(
            {
                **row._asdict(),
                "actigraphy_file": act_rel.as_posix(),
                "actigraphy_md5": _md5(mesa_root / act_rel),
                "events_file": event_rel.as_posix(),
                "events_md5": _md5(mesa_root / event_rel),
                "rpoints_file": rpoint_rel.as_posix(),
                "rpoints_md5": _md5(mesa_root / rpoint_rel),
            }
        )
        overlap_rows.append({"mesaid": row.mesaid, "line": 100, "linetime": "22:00:00", "starttime_psg": "22:00:00"})
        phenotype_rows.append({"mesaid": row.mesaid, "match5": 1, "havepsg5": 1, "haveact5": 1})
    pilot_path = tmp_path / "pilot.csv"
    overlap_path = tmp_path / "overlap.csv"
    pd.DataFrame(pilot_rows).to_csv(pilot_path, index=False)
    pd.DataFrame(overlap_rows).to_csv(overlap_path, index=False)
    pd.DataFrame(phenotype_rows).to_csv(mesa_root / "datasets" / "mesa-sleep-dataset-0.8.0.csv", index=False)
    return mesa_root, pilot_path, overlap_path


def test_prepare_mesa_pilot_validates_sources_and_writes_reconciled_artifacts(tmp_path):
    mesa_root, pilot_path, overlap_path = _write_pilot_source(tmp_path)
    output = tmp_path / "artifacts"

    epochs, manifest = prepare_mesa_pilot(mesa_root, pilot_path, overlap_path, output)

    assert epochs.columns.tolist() == ["subject_id", "epoch_start_s", "label", *MESA_FEATURE_COLUMNS, "split"]
    assert len(epochs) == 72
    assert epochs.groupby("split")["subject_id"].nunique().to_dict() == {"test": 4, "train": 16, "validation": 4}
    assert manifest["dataset"] == "MESA Sleep"
    assert manifest["release_version"] == "0.8.0"
    assert manifest["feature_columns"] == MESA_FEATURE_COLUMNS
    assert manifest["class_counts"] == {"0": 24, "1": 48}
    assert pd.read_parquet(output / "epochs.parquet").equals(epochs)
    assert json.loads((output / "splits.json").read_text(encoding="utf-8")) == manifest["splits"]
    assert json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))["row_count"] == 72


def test_prepare_mesa_pilot_rejects_invalid_concurrency_flag(tmp_path):
    mesa_root, pilot_path, overlap_path = _write_pilot_source(tmp_path)
    phenotype_path = mesa_root / "datasets" / "mesa-sleep-dataset-0.8.0.csv"
    phenotype = pd.read_csv(phenotype_path)
    phenotype.loc[0, "match5"] = 0
    phenotype.to_csv(phenotype_path, index=False)

    with pytest.raises(ValueError, match="concurrent"):
        prepare_mesa_pilot(mesa_root, pilot_path, overlap_path, tmp_path / "artifacts")


def test_prepare_mesa_pilot_rejects_checksum_mismatch_and_duplicate_mapping(tmp_path):
    mesa_root, pilot_path, overlap_path = _write_pilot_source(tmp_path)
    pilot = pd.read_csv(pilot_path)
    pilot.loc[0, "actigraphy_md5"] = "0" * 32
    pilot.to_csv(pilot_path, index=False)

    with pytest.raises(ValueError, match="checksum"):
        prepare_mesa_pilot(mesa_root, pilot_path, overlap_path, tmp_path / "bad-checksum")

    mesa_root, pilot_path, overlap_path = _write_pilot_source(tmp_path / "duplicate")
    overlap = pd.read_csv(overlap_path)
    pd.concat([overlap, overlap.iloc[[0]]], ignore_index=True).to_csv(overlap_path, index=False)

    with pytest.raises(ValueError, match="duplicate"):
        prepare_mesa_pilot(mesa_root, pilot_path, overlap_path, tmp_path / "bad-overlap")


def test_prepare_mesa_manifest_preserves_frozen_subject_splits(tmp_path):
    mesa_root, pilot_path, overlap_path = _write_pilot_source(tmp_path)
    cohort = pd.read_csv(pilot_path)
    allocation = stratified_pilot_split(cohort, seed=20260915)
    reverse_split = {subject: split for split, subjects in allocation.items() for subject in subjects}
    cohort["subject_id"] = cohort["mesaid"].map(lambda value: f"{int(value):04d}")
    cohort["split"] = cohort["subject_id"].map(reverse_split)
    cohort = cohort.rename(
        columns={
            "actigraphy_file": "actigraphy_path",
            "events_file": "events_path",
            "rpoints_file": "rpoints_path",
        }
    )
    manifest_path = tmp_path / "expansion.csv"
    cohort.to_csv(manifest_path, index=False)

    epochs, manifest = prepare_mesa_manifest(mesa_root, manifest_path, overlap_path, tmp_path / "expanded")

    observed = epochs.groupby("split")["subject_id"].unique().apply(lambda values: sorted(values)).to_dict()
    assert observed == allocation
    assert manifest["splits"] == allocation


def test_prepare_mesa_manifest_rejects_invalid_or_duplicate_assignments(tmp_path):
    mesa_root, pilot_path, overlap_path = _write_pilot_source(tmp_path)
    cohort = pd.read_csv(pilot_path).rename(
        columns={
            "actigraphy_file": "actigraphy_path",
            "events_file": "events_path",
            "rpoints_file": "rpoints_path",
        }
    )
    cohort["subject_id"] = cohort["mesaid"].map(lambda value: f"{int(value):04d}")
    cohort["split"] = "train"
    cohort.loc[0, "split"] = "future"
    invalid_path = tmp_path / "invalid.csv"
    cohort.to_csv(invalid_path, index=False)

    with pytest.raises(ValueError, match="split"):
        prepare_mesa_manifest(mesa_root, invalid_path, overlap_path, tmp_path / "invalid")

    cohort.loc[0, "split"] = "train"
    duplicate_path = tmp_path / "duplicate.csv"
    pd.concat([cohort, cohort.iloc[[0]]], ignore_index=True).to_csv(duplicate_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_mesa_manifest(mesa_root, duplicate_path, overlap_path, tmp_path / "duplicate")
