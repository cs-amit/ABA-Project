"""Validated adapters for aligned MESA actigraphy, PSG, and ECG epochs."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


STAGE_LABELS = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0}
MESA_FEATURE_COLUMNS = [
    "activity_count",
    "activity_observed",
    "off_wrist",
    "heart_rate_mean",
    "heart_rate_standard_deviation",
    "ibi_mean_ms",
    "ibi_rmssd_ms",
    "heart_rate_valid_ratio",
    "elapsed_hours",
    "clock_sin",
    "clock_cos",
]


def _epoch_number(value: str | None, field: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stage {field} must be finite") from exc
    if not math.isfinite(numeric) or numeric < 0 or numeric % 30 != 0:
        raise ValueError(f"stage {field} must be a non-negative multiple of 30 seconds")
    return int(numeric // 30)


def expand_stage_events(xml_path: Path) -> pd.DataFrame:
    """Expand compressed NSRR stage events into one row per 30-second epoch."""
    rows: list[dict] = []
    used: set[int] = set()
    for event in ET.parse(xml_path).getroot().findall(".//ScoredEvent"):
        values = {child.tag: child.text for child in event}
        if values.get("EventType") != "Stages|Stages":
            continue
        start_epoch = _epoch_number(values.get("Start"), "start")
        duration_epochs = _epoch_number(values.get("Duration"), "duration")
        if duration_epochs == 0:
            raise ValueError("stage duration must be a positive multiple of 30 seconds")
        try:
            stage = int(str(values.get("EventConcept")).rsplit("|", 1)[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("stage event concept must end with a numeric code") from exc
        for epoch_index in range(start_epoch, start_epoch + duration_epochs):
            if epoch_index in used:
                raise ValueError("duplicate PSG stage epoch")
            used.add(epoch_index)
            rows.append(
                {
                    "epoch_index": epoch_index,
                    "epoch_start_s": epoch_index * 30,
                    "stage": stage,
                    "label": STAGE_LABELS.get(stage, pd.NA),
                }
            )
    return pd.DataFrame(rows, columns=["epoch_index", "epoch_start_s", "stage", "label"]).sort_values(
        "epoch_index", ignore_index=True
    )


def derive_cardiac_epochs(rpoints: pd.DataFrame, epoch_count: int) -> pd.DataFrame:
    """Aggregate normal R-point intervals into PSG-relative 30-second epochs."""
    required = {"epoch", "seconds", "Type"}
    missing = sorted(required - set(rpoints.columns))
    if missing:
        raise ValueError(f"R-points missing required columns: {missing}")
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    frame = rpoints.loc[:, ["epoch", "seconds", "Type"]].copy()
    frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
    frame["seconds"] = pd.to_numeric(frame["seconds"], errors="coerce")
    if frame[["epoch", "seconds"]].isna().any().any():
        raise ValueError("R-point epoch and seconds must be numeric")
    if (frame["seconds"] <= 0).any():
        raise ValueError("R-point seconds must be positive")
    if not frame["seconds"].is_monotonic_increasing:
        raise ValueError("R-point seconds must be sorted")
    if (frame["epoch"] % 1 != 0).any() or not frame["epoch"].between(1, epoch_count).all():
        raise ValueError("R-point epoch must be an integer within the PSG epoch range")
    frame["epoch"] = frame["epoch"].astype(int)

    totals = frame.groupby("epoch").size()
    normal = frame[frame["Type"] == 1].copy()
    normal_counts = normal.groupby("epoch").size()
    normal["ibi_ms"] = normal["seconds"].diff() * 1000.0
    normal.loc[~normal["ibi_ms"].between(300.0, 2000.0), "ibi_ms"] = np.nan
    normal["heart_rate_bpm"] = 60000.0 / normal["ibi_ms"]

    rows = []
    for epoch in range(1, epoch_count + 1):
        values = normal.loc[normal["epoch"] == epoch, "ibi_ms"].dropna()
        heart_rates = normal.loc[normal["epoch"] == epoch, "heart_rate_bpm"].dropna()
        interval_differences = values.diff().dropna()
        total = int(totals.get(epoch, 0))
        rows.append(
            {
                "epoch_index": epoch - 1,
                "heart_rate_mean": float(heart_rates.mean()) if len(heart_rates) else 0.0,
                "heart_rate_standard_deviation": float(heart_rates.std(ddof=0)) if len(heart_rates) > 1 else 0.0,
                "ibi_mean_ms": float(values.mean()) if len(values) else 0.0,
                "ibi_rmssd_ms": float(np.sqrt(np.mean(np.square(interval_differences)))) if len(interval_differences) else 0.0,
                "heart_rate_valid_ratio": float(normal_counts.get(epoch, 0) / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _clock_seconds(values: pd.Series) -> pd.Series:
    try:
        seconds = pd.to_timedelta(values.astype(str)).dt.total_seconds()
    except (TypeError, ValueError) as exc:
        raise ValueError("actigraphy linetime must use HH:MM:SS") from exc
    if seconds.isna().any() or not seconds.between(0, 86399).all():
        raise ValueError("actigraphy linetime must use HH:MM:SS")
    return seconds


def align_participant(
    actigraphy: pd.DataFrame,
    stages: pd.DataFrame,
    rpoints: pd.DataFrame,
    overlap_line: int,
    subject_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Align one participant without inferring the PSG start from clock time."""
    required_actigraphy = {"line", "linetime", "offwrist", "activity"}
    missing = sorted(required_actigraphy - set(actigraphy.columns))
    if missing:
        raise ValueError(f"actigraphy missing required columns: {missing}")
    if actigraphy["line"].duplicated().any():
        raise ValueError("duplicate actigraphy line")
    if int((actigraphy["line"] == overlap_line).sum()) != 1:
        raise ValueError("official overlap line is missing")
    if stages.empty or stages["epoch_index"].duplicated().any():
        raise ValueError("PSG stages must contain unique epochs")

    expected_lines = overlap_line + stages["epoch_index"].astype(int)
    indexed = actigraphy.set_index("line", drop=False)
    if not set(expected_lines).issubset(indexed.index):
        raise ValueError("actigraphy lines are not consecutive across the PSG recording")
    selected = indexed.loc[expected_lines].reset_index(drop=True).copy()
    if not np.array_equal(selected["line"].to_numpy(), expected_lines.to_numpy()):
        raise ValueError("actigraphy lines are not consecutive across the PSG recording")
    seconds = _clock_seconds(selected["linetime"])
    if len(seconds) > 1 and not np.all(np.diff(seconds.to_numpy()) % 86400 == 30):
        raise ValueError("actigraphy linetime must have 30-second cadence")

    epoch_count = int(stages["epoch_index"].max()) + 1
    cardiac = derive_cardiac_epochs(rpoints, epoch_count).set_index("epoch_index")
    result = stages.reset_index(drop=True).copy()
    result = result.join(cardiac, on="epoch_index")
    result.insert(0, "subject_id", str(subject_id))
    off_wrist = selected["offwrist"].fillna(1).ne(0)
    observed = ~off_wrist & pd.to_numeric(selected["activity"], errors="coerce").notna()
    result["activity_count"] = pd.to_numeric(selected["activity"], errors="coerce").where(observed, 0.0)
    result["activity_observed"] = observed.astype(float)
    result["off_wrist"] = off_wrist.astype(float)
    result["elapsed_hours"] = result["epoch_start_s"] / 3600.0
    angle = 2 * math.pi * seconds / 86400.0
    result["clock_sin"] = np.sin(angle)
    result["clock_cos"] = np.cos(angle)

    quality = {
        "unlabelled_epochs": int(result["label"].isna().sum()),
        "missing_activity_epochs": int((~observed).sum()),
    }
    result = result[result["label"].notna()].copy()
    result["label"] = result["label"].astype(int)
    columns = ["subject_id", "epoch_start_s", "label", *MESA_FEATURE_COLUMNS]
    return result.loc[:, columns].reset_index(drop=True), quality
