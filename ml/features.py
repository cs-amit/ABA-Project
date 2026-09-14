"""BIDSleep adapter and Android-compatible causal epoch features."""
import math
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from scipy.io import loadmat

from .data_contract import FEATURE_COLUMNS, map_stage, split_subjects, validate_epoch_frame, write_manifest

ACTIVITY_DELTA_THRESHOLD = 0.10


def load_bidsleep_night(night_dir: Path, subject_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Adapt one official BIDSleep night without copying raw data into the repo."""
    motion = pd.read_csv(night_dir / "motion.csv")
    motion = motion.rename(columns={"Timestamp": "timestamp_s"})
    accel = motion[["timestamp_s", "x", "y", "z"]].copy()
    accel.insert(0, "subject_id", subject_id)
    heart_rate = pd.read_csv(night_dir / "hr.csv")
    if "Timestamp" not in heart_rate.columns:
        heart_rate = pd.read_csv(night_dir / "hr.csv", header=None, names=["Timestamp", "heart_rate_bpm"])
    else:
        heart_rate = heart_rate.rename(columns={"hr": "heart_rate_bpm", "HR": "heart_rate_bpm"})
    heart_rate = heart_rate.rename(columns={"Timestamp": "timestamp_s"})[["timestamp_s", "heart_rate_bpm"]]
    heart_rate["ibi_ms"] = pd.NA
    heart_rate.insert(0, "subject_id", subject_id)
    source = loadmat(night_dir / "labels.mat", squeeze_me=True)
    recording_start = datetime.strptime(str(source["recStart"]), "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("America/New_York")).timestamp()
    stages = {0: "AWAKE", 1: "LIGHT", 2: "LIGHT", 3: "DEEP", 4: "REM"}
    labels = [
        {"subject_id": subject_id, "epoch_start_s": int(recording_start + index * 30), "stage": stages[int(code)]}
        for index, code in enumerate(source["dreem_label"])
        if int(code) in stages
    ]
    return accel, heart_rate, pd.DataFrame(labels, columns=["subject_id", "epoch_start_s", "stage"])


def prepare_official_bidsleep(raw_root: Path, output_dir: Path, seed: int = 42):
    """Prepare official per-night BIDSleep files without storing raw inputs in the repository."""
    rows = []
    unknown = 0
    for subject_dir in sorted(path for path in raw_root.iterdir() if path.is_dir() and path.name.startswith("Bidslab")):
        for night_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            accel, heart_rate, labels = load_bidsleep_night(night_dir, subject_dir.name)
            accel_times = accel["timestamp_s"].to_numpy()
            heart_times = heart_rate["timestamp_s"].to_numpy()
            for label in labels.itertuples(index=False):
                start, end = label.epoch_start_s, label.epoch_start_s + 30
                a_start, a_end = accel_times.searchsorted([start, end])
                h_start, h_end = heart_times.searchsorted([start, end])
                mapped, dropped = map_stage(pd.Series([label.stage]))
                unknown += dropped
                if pd.isna(mapped.iloc[0]):
                    continue
                row = {"subject_id": subject_dir.name, "epoch_start_s": int(start), "label": int(mapped.iloc[0])}
                row.update(build_epoch_features(accel.iloc[a_start:a_end], heart_rate.iloc[h_start:h_end], start, end))
                rows.append(row)
    result = pd.DataFrame(rows, columns=["subject_id", "epoch_start_s", "label", *FEATURE_COLUMNS])
    allocation = split_subjects(result.subject_id.tolist(), seed=seed)
    result["split"] = result.subject_id.map({subject: split for split, subjects in allocation.items() for subject in subjects})
    validate_epoch_frame(result)
    manifest = {
        "dataset": "BIDSleep", "release_version": "1.0.0", "feature_columns": FEATURE_COLUMNS,
        "seed": seed, "splits": allocation, "unknown_stage_count": unknown,
        "source_layout": "Bidslab*/<night>/{motion.csv,hr.csv,labels.mat}",
        "label_mapping": {"N1": 1, "N2": 1, "Wake": 0, "N3": 0, "REM": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_dir / "epochs.parquet", index=False)
    (output_dir / "splits.json").write_text(__import__("json").dumps(allocation, indent=2) + "\n", encoding="utf-8")
    write_manifest(output_dir / "dataset_manifest.json", manifest)
    return result, manifest


def _quality(frame):
    return frame["quality"] if "quality" in frame.columns else pd.Series("VALID", index=frame.index)


def _std(values):
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def _mad(values):
    return float((values - values.median()).abs().median()) if len(values) else 0.0


def _zcr(values):
    non_zero = values[values != 0].to_numpy()
    if len(non_zero) < 2:
        return 0.0
    return float(((non_zero[:-1] * non_zero[1:]) < 0).sum() / (len(non_zero) - 1))


def build_epoch_features(accel: pd.DataFrame, heart_rate: pd.DataFrame, start_s: float, end_s: float) -> dict:
    a = accel[(accel.timestamp_s >= start_s) & (accel.timestamp_s < end_s)].copy()
    h = heart_rate[(heart_rate.timestamp_s >= start_s) & (heart_rate.timestamp_s < end_s)].copy()
    if len(a):
        a["magnitude"] = (a.x * a.x + a.y * a.y + a.z * a.z).pow(0.5)
    a_quality = _quality(a)
    h_quality = _quality(h)
    a_valid = a[a_quality == "VALID"] if len(a) else a
    h_valid = h[h_quality == "VALID"] if len(h) else h
    magnitudes = a_valid.magnitude if len(a_valid) else pd.Series(dtype=float)
    hr = h_valid.heart_rate_bpm.dropna() if len(h_valid) else pd.Series(dtype=float)
    ibi = h_valid.ibi_ms.dropna() if len(h_valid) else pd.Series(dtype=float)
    differences = ibi.diff().dropna()
    return {
        "accel_magnitude_mean": float(magnitudes.mean()) if len(magnitudes) else 0.0,
        "accel_magnitude_standard_deviation": _std(magnitudes),
        "accel_magnitude_median_absolute_deviation": _mad(magnitudes),
        "activity_count": float((magnitudes.diff().abs().dropna() >= ACTIVITY_DELTA_THRESHOLD).sum()),
        "accel_zero_crossing_rate": _zcr(magnitudes - magnitudes.mean()) if len(magnitudes) else 0.0,
        "heart_rate_mean": float(hr.mean()) if len(hr) else 0.0,
        "heart_rate_standard_deviation": _std(hr),
        "ibi_mean": float(ibi.mean()) if len(ibi) else 0.0,
        "ibi_rmssd": float(math.sqrt((differences * differences).mean())) if len(differences) else 0.0,
        "accel_valid_sample_ratio": float(len(a_valid) / len(a)) if len(a) else 0.0,
        "heart_rate_valid_sample_ratio": float(len(h_valid) / len(h)) if len(h) else 0.0,
        "off_body_flag": float(((a_quality == "OFF_BODY").any() if len(a) else False) or ((h_quality == "OFF_BODY").any() if len(h) else False)),
    }


def prepare_bidsleep_epochs(accel: pd.DataFrame, heart_rate: pd.DataFrame, labels: pd.DataFrame, seed: int = 42):
    """Explicit BIDSleep adapter. Inputs use normalized adapter columns."""
    rows = []
    unknown = 0
    for _, label_row in labels.sort_values(["subject_id", "epoch_start_s"]).iterrows():
        mapped, dropped = map_stage(pd.Series([label_row["stage"]]))
        unknown += dropped
        if pd.isna(mapped.iloc[0]):
            continue
        subject = label_row["subject_id"]
        a = accel[accel.subject_id == subject]
        h = heart_rate[heart_rate.subject_id == subject]
        row = {"subject_id": subject, "epoch_start_s": int(label_row.epoch_start_s), "label": int(mapped.iloc[0])}
        row.update(build_epoch_features(a, h, label_row.epoch_start_s, label_row.epoch_start_s + 30))
        rows.append(row)
    result = pd.DataFrame(rows, columns=["subject_id", "epoch_start_s", "label", *FEATURE_COLUMNS])
    allocation = split_subjects(result.subject_id.tolist(), seed=seed) if len(result) else {"train": [], "validation": [], "test": []}
    reverse = {subject: split for split, values in allocation.items() for subject in values}
    result["split"] = result.subject_id.map(reverse)
    validate_epoch_frame(result)
    manifest = {
        "dataset": "BIDSleep", "release_version": "not specified by source export",
        "access_date": date.today().isoformat(),
        "licence_terms": "BIDSleep open-access terms; verify the downloaded release terms before redistribution",
        "participants": 47, "nights": 253,
        "selected_signal_files": ["accelerometer.csv", "heart_rate.csv", "labels.csv"],
        "label_mapping": {"LIGHT": 1, "AWAKE": 0, "DEEP": 0, "REM": 0},
        "unknown_stage_count": unknown,
        "exclusions": {"unknown_stage": unknown, "duplicate_subject_epoch": 0, "invalid_quality": "retained in denominators and excluded from valid feature numerators"},
        "seed": seed, "splits": allocation, "feature_columns": FEATURE_COLUMNS,
    }
    return result, manifest


def prepare_from_directory(data_dir: Path, output_dir: Path, seed: int = 42):
    """Read normalized BIDSleep CSV exports; raw files remain outside the repository."""
    accel = pd.read_csv(data_dir / "accelerometer.csv")
    heart_rate = pd.read_csv(data_dir / "heart_rate.csv")
    labels = pd.read_csv(data_dir / "labels.csv")
    epochs, manifest = prepare_bidsleep_epochs(accel, heart_rate, labels, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs.to_parquet(output_dir / "epochs.parquet", index=False)
    (output_dir / "splits.json").write_text(__import__("json").dumps(manifest["splits"], indent=2) + "\n", encoding="utf-8")
    (output_dir / "dataset_manifest.json").write_text(__import__("json").dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return epochs, manifest
