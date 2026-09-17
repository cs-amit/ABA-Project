from pathlib import Path

import pandas as pd
import pytest

from ml.mesa import align_participant, derive_cardiac_epochs, expand_stage_events


def _write_xml(path: Path, events: list[tuple[str, float, float]]) -> None:
    body = "".join(
        f"<ScoredEvent><EventType>Stages|Stages</EventType>"
        f"<EventConcept>{concept}</EventConcept><Start>{start}</Start>"
        f"<Duration>{duration}</Duration></ScoredEvent>"
        for concept, start, duration in events
    )
    path.write_text(f"<PSGAnnotation><ScoredEvents>{body}</ScoredEvents></PSGAnnotation>", encoding="utf-8")


def test_expand_stage_events_maps_and_expands_thirty_second_epochs(tmp_path):
    xml_path = tmp_path / "stages.xml"
    _write_xml(
        xml_path,
        [
            ("Wake|0", 0, 30),
            ("Stage 1 sleep|1", 30, 60),
            ("REM sleep|5", 90, 30),
        ],
    )

    stages = expand_stage_events(xml_path)

    assert stages[["epoch_index", "epoch_start_s", "stage", "label"]].to_dict("records") == [
        {"epoch_index": 0, "epoch_start_s": 0, "stage": 0, "label": 0},
        {"epoch_index": 1, "epoch_start_s": 30, "stage": 1, "label": 1},
        {"epoch_index": 2, "epoch_start_s": 60, "stage": 1, "label": 1},
        {"epoch_index": 3, "epoch_start_s": 90, "stage": 5, "label": 0},
    ]


@pytest.mark.parametrize(
    "events, message",
    [
        ([("Wake|0", 0, 31)], "multiple of 30"),
        ([("Wake|0", 0, 60), ("Stage 1 sleep|1", 30, 30)], "duplicate"),
    ],
)
def test_expand_stage_events_rejects_malformed_epoch_geometry(tmp_path, events, message):
    xml_path = tmp_path / "invalid.xml"
    _write_xml(xml_path, events)

    with pytest.raises(ValueError, match=message):
        expand_stage_events(xml_path)


def test_expand_stage_events_keeps_unknown_stage_unlabelled(tmp_path):
    xml_path = tmp_path / "unknown.xml"
    _write_xml(xml_path, [("Unscored|9", 0, 30)])

    stages = expand_stage_events(xml_path)

    assert stages.loc[0, "stage"] == 9
    assert pd.isna(stages.loc[0, "label"])


def test_derive_cardiac_epochs_uses_normal_consecutive_beats():
    rpoints = pd.DataFrame(
        {
            "epoch": [1, 1, 1, 1],
            "seconds": [1.0, 2.0, 3.1, 4.3],
            "Type": [1, 1, 1, 1],
        }
    )

    cardiac = derive_cardiac_epochs(rpoints, epoch_count=1)

    assert cardiac.loc[0, "ibi_mean_ms"] == pytest.approx(1100.0)
    assert cardiac.loc[0, "ibi_rmssd_ms"] == pytest.approx(100.0)
    assert cardiac.loc[0, "heart_rate_mean"] == pytest.approx((60 + 60000 / 1100 + 50) / 3)
    assert cardiac.loc[0, "heart_rate_valid_ratio"] == 1.0


def test_derive_cardiac_epochs_excludes_non_normal_beats_from_intervals():
    rpoints = pd.DataFrame(
        {
            "epoch": [1, 1, 1, 1],
            "seconds": [1.0, 2.0, 2.5, 3.0],
            "Type": [1, 1, 2, 1],
        }
    )

    cardiac = derive_cardiac_epochs(rpoints, epoch_count=1)

    assert cardiac.loc[0, "ibi_mean_ms"] == 1000.0
    assert cardiac.loc[0, "heart_rate_valid_ratio"] == 0.75


def test_derive_cardiac_epochs_does_not_bridge_across_an_abnormal_beat():
    rpoints = pd.DataFrame(
        {
            "epoch": [1, 1, 1],
            "seconds": [1.0, 1.6, 2.2],
            "Type": [1, 2, 1],
        }
    )

    cardiac = derive_cardiac_epochs(rpoints, epoch_count=1)

    assert cardiac.loc[0, "ibi_mean_ms"] == 0.0
    assert cardiac.loc[0, "heart_rate_mean"] == 0.0


def test_derive_cardiac_epochs_sorts_unordered_source_rows_by_recording_time():
    ordered = pd.DataFrame({"epoch": [1, 1, 1, 1], "seconds": [1.0, 2.0, 3.1, 4.3], "Type": [1, 1, 1, 1]})
    unordered = ordered.iloc[[2, 0, 3, 1]].reset_index(drop=True)

    assert derive_cardiac_epochs(unordered, epoch_count=1).equals(derive_cardiac_epochs(ordered, epoch_count=1))


@pytest.mark.parametrize(
    "rpoints, epoch_count, message",
    [
        (pd.DataFrame({"epoch": [1], "seconds": [0.0], "Type": [1]}), 1, "positive"),
        (pd.DataFrame({"epoch": [2], "seconds": [30.0], "Type": [1]}), 2, "declared epoch"),
        (pd.DataFrame({"epoch": [2], "seconds": [1.0], "Type": [1]}), 1, "epoch"),
    ],
)
def test_derive_cardiac_epochs_rejects_invalid_positions(rpoints, epoch_count, message):
    with pytest.raises(ValueError, match=message):
        derive_cardiac_epochs(rpoints, epoch_count=epoch_count)


def _alignment_inputs():
    actigraphy = pd.DataFrame(
        {
            "line": [9, 10, 11, 12],
            "linetime": ["23:59:30", "00:00:00", "00:00:30", "00:01:00"],
            "offwrist": [0, 0, 1, 0],
            "activity": [99.0, 12.0, None, 8.0],
        }
    )
    stages = pd.DataFrame(
        {
            "epoch_index": [0, 1, 2],
            "epoch_start_s": [0, 30, 60],
            "stage": [0, 1, 2],
            "label": [0, 1, 1],
        }
    )
    rpoints = pd.DataFrame(
        {
            "epoch": [1, 1, 2, 2, 3, 3],
            "seconds": [1.0, 2.0, 31.0, 32.0, 61.0, 62.0],
            "Type": [1, 1, 1, 1, 1, 1],
        }
    )
    return actigraphy, stages, rpoints


def test_align_participant_starts_at_official_overlap_line_and_marks_missing_activity():
    actigraphy, stages, rpoints = _alignment_inputs()

    aligned, quality = align_participant(actigraphy, stages, rpoints, overlap_line=10, subject_id="0648")

    assert aligned[["subject_id", "epoch_start_s", "label", "activity_count"]].to_dict("records") == [
        {"subject_id": "0648", "epoch_start_s": 0, "label": 0, "activity_count": 12.0},
        {"subject_id": "0648", "epoch_start_s": 30, "label": 1, "activity_count": 0.0},
        {"subject_id": "0648", "epoch_start_s": 60, "label": 1, "activity_count": 8.0},
    ]
    assert aligned["activity_observed"].tolist() == [1.0, 0.0, 1.0]
    assert aligned["off_wrist"].tolist() == [0.0, 1.0, 0.0]
    assert aligned.loc[0, "clock_sin"] == pytest.approx(0.0, abs=1e-12)
    assert aligned.loc[0, "clock_cos"] == pytest.approx(1.0)
    assert aligned["elapsed_hours"].tolist() == pytest.approx([0.0, 30 / 3600, 60 / 3600])
    assert quality == {"unlabelled_epochs": 0, "missing_activity_epochs": 1}


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda frame: frame[frame.line != 10], "overlap"),
        (lambda frame: pd.concat([frame, frame[frame.line == 10]], ignore_index=True), "duplicate"),
        (lambda frame: frame.assign(line=[9, 10, 12, 13]), "consecutive"),
        (lambda frame: frame.assign(linetime=["23:59:30", "00:00:00", "00:00:31", "00:01:00"]), "30-second"),
    ],
)
def test_align_participant_rejects_unsafe_actigraphy_alignment(mutation, message):
    actigraphy, stages, rpoints = _alignment_inputs()

    with pytest.raises(ValueError, match=message):
        align_participant(mutation(actigraphy), stages, rpoints, overlap_line=10, subject_id="0648")
