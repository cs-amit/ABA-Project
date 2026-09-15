import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from urllib.parse import urlparse

import pandas as pd
import pytest

from ml.mesa_expansion import (
    build_file_catalog,
    download_expansion,
    required_download_bytes,
    select_expansion_cohort,
)


def _metadata(subjects):
    entries = []
    for sid in subjects:
        entries.extend(
            [
                {"full_path": f"actigraphy/mesa-sleep-{sid}.csv", "file_size": 10, "file_checksum_md5": "a" * 32, "is_file": True},
                {"full_path": f"polysomnography/annotations-events-nsrr/mesa-sleep-{sid}-nsrr.xml", "file_size": 20, "file_checksum_md5": "b" * 32, "is_file": True},
                {"full_path": f"polysomnography/annotations-rpoints/mesa-sleep-{sid}-rpoint.csv", "file_size": 30, "file_checksum_md5": "c" * 32, "is_file": True},
                {"full_path": f"polysomnography/edfs/mesa-sleep-{sid}.edf", "file_size": 999, "file_checksum_md5": "d" * 32, "is_file": True},
            ]
        )
    return entries


def _phenotype(count=600):
    rows = []
    for index in range(count):
        rows.append(
            {
                "mesaid": index + 1,
                "match5": 1,
                "havepsg5": 1,
                "haveact5": 1,
                "race1c": index % 4 + 1,
                "gender1": index % 2,
                "sleepage5c": [60, 70, 80][index % 3],
            }
        )
    return pd.DataFrame(rows)


def test_build_file_catalog_accepts_only_three_permitted_modalities():
    catalog = build_file_catalog(_metadata(["0001"]))

    assert set(catalog) == {"0001"}
    assert set(catalog["0001"]) == {"actigraphy", "events", "rpoints"}
    assert all("edf" not in item["path"].lower() for item in catalog["0001"].values())


def test_select_expansion_cohort_is_deterministic_excludes_pilot_and_has_exact_splits():
    phenotype = _phenotype()
    ids = [f"{value:04d}" for value in phenotype.mesaid]
    catalog = build_file_catalog(_metadata(ids))
    excluded = set(ids[:24])

    first = select_expansion_cohort(phenotype, catalog, excluded)
    second = select_expansion_cohort(phenotype, catalog, excluded)

    assert first.equals(second)
    assert len(first) == first.subject_id.nunique() == 500
    assert not set(first.subject_id) & excluded
    assert first.split.value_counts().to_dict() == {"train": 350, "validation": 75, "test": 75}
    assert first.groupby(["race1c", "gender1", "age_band"]).size().size == 12
    assert not first[["actigraphy_path", "events_path", "rpoints_path"]].apply(
        lambda column: column.str.contains("edf", case=False)
    ).any().any()


def test_select_expansion_cohort_requires_flags_and_all_three_files():
    phenotype = _phenotype()
    ids = [f"{value:04d}" for value in phenotype.mesaid]
    phenotype.loc[24, "match5"] = 0
    catalog = build_file_catalog(_metadata(ids))
    del catalog["0026"]["rpoints"]

    selected = select_expansion_cohort(phenotype, catalog, set(ids[:24]))

    assert "0025" not in set(selected.subject_id)
    assert "0026" not in set(selected.subject_id)


def _download_manifest(contents):
    row = {}
    for kind, (path, body) in contents.items():
        row[f"{kind}_path"] = path
        row[f"{kind}_bytes"] = len(body)
        row[f"{kind}_md5"] = hashlib.md5(body).hexdigest()
    return pd.DataFrame([row])


class _Handler(BaseHTTPRequestHandler):
    contents = {}
    requests = 0

    def do_GET(self):
        type(self).requests += 1
        name = urlparse(self.path).path.rsplit("/", 1)[-1]
        body = type(self).contents.get(name)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


class _Server:
    def __init__(self, contents):
        _Handler.contents = contents
        _Handler.requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def __exit__(self, *_):
        self.server.shutdown()
        self.thread.join()


def test_download_expansion_skips_verified_files_and_atomically_downloads_missing(tmp_path):
    contents = {
        "actigraphy": ("actigraphy/mesa-sleep-0001.csv", b"activity"),
        "events": ("polysomnography/annotations-events-nsrr/mesa-sleep-0001-nsrr.xml", b"stages"),
        "rpoints": ("polysomnography/annotations-rpoints/mesa-sleep-0001-rpoint.csv", b"beats"),
    }
    manifest = _download_manifest(contents)
    existing = tmp_path / contents["actigraphy"][0]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"activity")
    token_path = tmp_path / "token"
    token_path.write_text("secret-token", encoding="utf-8")

    assert required_download_bytes(manifest, tmp_path) == len(b"stages") + len(b"beats")
    with _Server({Path(path).name: body for path, body in contents.values()}) as base_url:
        result = download_expansion(manifest, tmp_path, token_path, workers=2, base_url=base_url)

    assert result == {"verified": 3, "downloaded": 2, "skipped": 1, "bytes_downloaded": 11}
    assert _Handler.requests == 2
    assert not list(tmp_path.rglob("*.part"))


def test_download_expansion_removes_bad_partial_without_exposing_token(tmp_path):
    contents = {
        "actigraphy": ("actigraphy/mesa-sleep-0001.csv", b"expected"),
        "events": ("polysomnography/annotations-events-nsrr/mesa-sleep-0001-nsrr.xml", b"xml"),
        "rpoints": ("polysomnography/annotations-rpoints/mesa-sleep-0001-rpoint.csv", b"beats"),
    }
    manifest = _download_manifest(contents)
    token_path = tmp_path / "token"
    token_path.write_text("secret-token", encoding="utf-8")
    served = {Path(path).name: (b"wrong" if kind == "actigraphy" else body) for kind, (path, body) in contents.items()}

    with _Server(served) as base_url:
        with pytest.raises(RuntimeError) as error:
            download_expansion(manifest, tmp_path, token_path, workers=1, base_url=base_url)

    assert "secret-token" not in str(error.value)
    assert not list(tmp_path.rglob("*.part"))


def test_download_expansion_rejects_manifest_path_traversal(tmp_path):
    contents = {
        "actigraphy": ("../outside.csv", b"bad"),
        "events": ("polysomnography/annotations-events-nsrr/mesa-sleep-0001-nsrr.xml", b"xml"),
        "rpoints": ("polysomnography/annotations-rpoints/mesa-sleep-0001-rpoint.csv", b"beats"),
    }
    manifest = _download_manifest(contents)
    token_path = tmp_path / "token"
    token_path.write_text("secret-token", encoding="utf-8")

    with pytest.raises(ValueError, match="within"):
        download_expansion(manifest, tmp_path, token_path)
