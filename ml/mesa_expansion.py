"""Cohort selection and selective download support for MESA expansion."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import pandas as pd
import requests
from sklearn.model_selection import StratifiedShuffleSplit


PERMITTED_PATHS = {
    "actigraphy": re.compile(r"actigraphy/mesa-sleep-(\d{4})\.csv"),
    "events": re.compile(r"polysomnography/annotations-events-nsrr/mesa-sleep-(\d{4})-nsrr\.xml"),
    "rpoints": re.compile(r"polysomnography/annotations-rpoints/mesa-sleep-(\d{4})-rpoint\.csv"),
}


def build_file_catalog(entries: list[dict]) -> dict[str, dict[str, dict]]:
    """Index only the three explicitly permitted non-EDF MESA modalities."""
    catalog: dict[str, dict[str, dict]] = {}
    for entry in entries:
        if not entry.get("is_file", True):
            continue
        path = str(entry.get("full_path", ""))
        for kind, pattern in PERMITTED_PATHS.items():
            match = pattern.fullmatch(path)
            if not match:
                continue
            subject_id = match.group(1)
            if kind in catalog.setdefault(subject_id, {}):
                raise ValueError(f"duplicate {kind} file for MESA participant {subject_id}")
            catalog[subject_id][kind] = {
                "path": path,
                "size": int(entry["file_size"]),
                "md5": str(entry["file_checksum_md5"]).lower(),
            }
    return catalog


def _age_band(age: float) -> str:
    if age < 65:
        return "under65"
    if age < 75:
        return "65to74"
    return "75plus"


def _stratified_indices(frame: pd.DataFrame, train_size: int, seed: int) -> tuple[list[int], list[int]]:
    strata = frame[["race1c", "gender1", "age_band"]].astype(str).agg("|".join, axis=1)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    selected, remainder = next(splitter.split(frame, strata))
    return selected.tolist(), remainder.tolist()


def select_expansion_cohort(
    phenotype: pd.DataFrame,
    catalogs: dict[str, dict[str, dict]],
    excluded_ids: set[str],
    count: int = 500,
    seed: int = 20260916,
    eligible_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Select and split a demographic-stratified cohort without reading labels."""
    required = {"mesaid", "match5", "havepsg5", "haveact5", "race1c", "gender1", "sleepage5c"}
    missing = sorted(required - set(phenotype.columns))
    if missing:
        raise ValueError(f"phenotype dataset missing expansion columns: {missing}")
    frame = phenotype.loc[:, sorted(required)].copy()
    frame["subject_id"] = frame["mesaid"].map(lambda value: f"{int(value):04d}")
    valid_flags = (frame[["match5", "havepsg5", "haveact5"]].fillna(0).astype(int) == 1).all(axis=1)
    complete_files = frame["subject_id"].map(lambda subject: set(catalogs.get(subject, {})) == set(PERMITTED_PATHS))
    complete_demographics = frame[["race1c", "gender1", "sleepage5c"]].notna().all(axis=1)
    eligible = True if eligible_ids is None else frame["subject_id"].isin(eligible_ids)
    frame = frame[
        valid_flags & complete_files & complete_demographics & eligible & ~frame["subject_id"].isin(excluded_ids)
    ].copy()
    if len(frame) < count:
        raise ValueError(f"only {len(frame)} eligible independent MESA participants are available")
    frame["age_band"] = frame["sleepage5c"].astype(float).map(_age_band)
    selected_indices, _ = _stratified_indices(frame, count, seed)
    selected = frame.iloc[selected_indices].reset_index(drop=True)
    train_indices, held_indices = _stratified_indices(selected, 350, seed + 1)
    held = selected.iloc[held_indices].reset_index(drop=True)
    validation_indices, test_indices = _stratified_indices(held, 75, seed + 2)
    split_by_index = {index: "train" for index in train_indices}
    split_by_index.update({held_indices[index]: "validation" for index in validation_indices})
    split_by_index.update({held_indices[index]: "test" for index in test_indices})
    selected["split"] = selected.index.map(split_by_index)
    if selected["split"].isna().any():
        raise ValueError("expansion split allocation is incomplete")

    for kind in PERMITTED_PATHS:
        selected[f"{kind}_path"] = selected["subject_id"].map(lambda subject: catalogs[subject][kind]["path"])
        selected[f"{kind}_bytes"] = selected["subject_id"].map(lambda subject: catalogs[subject][kind]["size"])
        selected[f"{kind}_md5"] = selected["subject_id"].map(lambda subject: catalogs[subject][kind]["md5"])
    columns = [
        "subject_id", "mesaid", "sleepage5c", "gender1", "race1c", "age_band", "split",
        *[f"{kind}_{suffix}" for kind in PERMITTED_PATHS for suffix in ("path", "bytes", "md5")],
    ]
    return selected.loc[:, columns].sort_values("subject_id", ignore_index=True)


def write_expansion_manifest(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _file_specs(manifest: pd.DataFrame, root: Path) -> list[dict]:
    root = Path(root).resolve()
    specs = []
    for row in manifest.itertuples(index=False):
        for kind, pattern in PERMITTED_PATHS.items():
            relative = str(getattr(row, f"{kind}_path"))
            if not pattern.fullmatch(relative):
                raise ValueError(f"download path must be a permitted MESA file within the data root: {relative}")
            target = (root / relative).resolve()
            if root not in target.parents:
                raise ValueError("download path must stay within the data root")
            specs.append(
                {
                    "path": relative,
                    "target": target,
                    "size": int(getattr(row, f"{kind}_bytes")),
                    "md5": str(getattr(row, f"{kind}_md5")).lower(),
                }
            )
    paths = [spec["path"] for spec in specs]
    if len(paths) != len(set(paths)):
        raise ValueError("expansion manifest contains duplicate download paths")
    return specs


def _matches(spec: dict) -> bool:
    target = spec["target"]
    if not target.is_file() or target.stat().st_size != spec["size"]:
        return False
    digest = hashlib.md5()
    with target.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == spec["md5"]


def required_download_bytes(manifest: pd.DataFrame, root: Path) -> int:
    return sum(spec["size"] for spec in _file_specs(manifest, root) if not _matches(spec))


def download_expansion(
    manifest: pd.DataFrame,
    root: Path,
    token_path: Path,
    workers: int = 8,
    base_url: str = "https://sleepdata.org/datasets/mesa/files",
) -> dict:
    """Download only manifest files, with checksum resume and token-safe errors."""
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    specs = _file_specs(manifest, root)
    missing_bytes = sum(spec["size"] for spec in specs if not _matches(spec))
    required_free = int(missing_bytes * 1.10)
    if shutil.disk_usage(root).free < required_free:
        raise OSError(f"insufficient free space: need {required_free} bytes")
    token = Path(token_path).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("NSRR token file is empty")

    def fetch(spec: dict) -> dict:
        if _matches(spec):
            return {"downloaded": 0, "skipped": 1, "bytes": 0}
        target = spec["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")
        url = (
            f"{base_url.rstrip('/')}/a/{quote(token, safe='')}/m/nsrr-gem-v8-0-0/"
            f"{quote(spec['path'], safe='/')}"
        )
        for _ in range(2):
            digest = hashlib.md5()
            size = 0
            try:
                with requests.get(url, stream=True, timeout=(60, 60)) as response:
                    if response.status_code != 200:
                        continue
                    with part.open("wb") as output:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                output.write(chunk)
                                digest.update(chunk)
                                size += len(chunk)
                if size == spec["size"] and digest.hexdigest() == spec["md5"]:
                    part.replace(target)
                    return {"downloaded": 1, "skipped": 0, "bytes": size}
            except Exception:
                pass
            finally:
                if part.exists() and (size != spec["size"] or digest.hexdigest() != spec["md5"]):
                    part.unlink()
        raise RuntimeError(f"download failed checksum or HTTP validation for {spec['path']}")

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, spec): spec["path"] for spec in specs}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append(f"{futures[future]} ({type(exc).__name__})")
    if failures:
        raise RuntimeError(f"{len(failures)} MESA downloads failed: {', '.join(sorted(failures)[:5])}")
    downloaded = sum(result["downloaded"] for result in results)
    skipped = sum(result["skipped"] for result in results)
    return {
        "verified": len(results),
        "downloaded": downloaded,
        "skipped": skipped,
        "bytes_downloaded": sum(result["bytes"] for result in results),
    }
