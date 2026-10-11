"""OpenML benchmark acquisition and time-ordered data handling."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff

OPENML_DATASET_ID = 1597
SOURCE_PAGE = "https://www.openml.org/d/1597"
ZENODO_RECORD_URL = "https://zenodo.org/records/7395559"
ZENODO_DOWNLOAD_URL = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
EXPECTED_BYTES = 150_828_752
EXPECTED_MD5 = "e90efcb83d69faf99fcab8b0255024de"
EXPECTED_ROWS = 284_807
EXPECTED_FRAUD_ROWS = 492
CHUNK_BYTES = 4 * 1024 * 1024
RAW_NAME = "creditcard.csv"
LABEL_COLUMN = "Class"
RAW_FEATURES = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(destination: Path, overwrite: bool = False) -> dict[str, Any]:
    """Download the complete CC-BY Zenodo mirror in verified byte ranges."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = destination.parent / "manifest.json"
    if destination.exists() and not overwrite:
        md5 = hashlib.md5()
        with destination.open("rb") as existing:
            for block in iter(lambda: existing.read(1024 * 1024), b""):
                md5.update(block)
        if destination.stat().st_size != EXPECTED_BYTES or md5.hexdigest() != EXPECTED_MD5:
            raise RuntimeError(
                f"Existing file {destination} is incomplete or has the wrong checksum. "
                "Remove it or rerun with --overwrite."
            )
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))

    chunk_dir = destination.parent / f".{destination.name}.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_count = (EXPECTED_BYTES + CHUNK_BYTES - 1) // CHUNK_BYTES

    def fetch_chunk(index: int) -> Path:
        start = index * CHUNK_BYTES
        end = min(start + CHUNK_BYTES, EXPECTED_BYTES) - 1
        chunk_path = chunk_dir / f"{start:012d}.part"
        expected_length = end - start + 1
        if chunk_path.exists() and chunk_path.stat().st_size == expected_length:
            return chunk_path
        chunk_path.unlink(missing_ok=True)
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                request = urllib.request.Request(
                    ZENODO_DOWNLOAD_URL,
                    headers={
                        "User-Agent": "Topic3-FraudAutoencoder/1.0 (academic benchmark)",
                        "Accept-Encoding": "identity",
                        "Range": f"bytes={start}-{end}",
                    },
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or content_range != f"bytes {start}-{end}/{EXPECTED_BYTES}":
                        raise RuntimeError(
                            f"Unexpected range response for bytes {start}-{end}: "
                            f"status={response.status}, Content-Range={content_range!r}"
                        )
                    payload = response.read()
                if len(payload) != expected_length:
                    raise RuntimeError(
                        f"Incomplete byte range {start}-{end}: got {len(payload)} bytes, "
                        f"expected {expected_length}."
                    )
                temporary = chunk_path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, chunk_path)
                return chunk_path
            except Exception as exc:
                last_error = exc
                time.sleep(min(attempt, 4))
        raise RuntimeError(f"Failed to download byte range {start}-{end}: {last_error}")

    temp_path = destination.parent / f".{destination.name}.download.part"
    try:
        completed = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(fetch_chunk, index) for index in range(chunk_count)]
            for future in as_completed(futures):
                future.result()
                completed += 1
                print(f"Downloaded verified range {completed}/{chunk_count}", flush=True)
        with temp_path.open("wb") as out:
            for index in range(chunk_count):
                start = index * CHUNK_BYTES
                with (chunk_dir / f"{start:012d}.part").open("rb") as chunk:
                    shutil.copyfileobj(chunk, out)
            out.flush()
            os.fsync(out.fileno())
        total = temp_path.stat().st_size
        digest = hashlib.md5()
        with temp_path.open("rb") as out:
            for block in iter(lambda: out.read(1024 * 1024), b""):
                digest.update(block)
        if total != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_MD5:
            raise RuntimeError(
                f"Downloaded dataset integrity check failed: bytes={total}, md5={digest.hexdigest()}; "
                f"expected bytes={EXPECTED_BYTES}, md5={EXPECTED_MD5}."
            )
        os.replace(temp_path, destination)
        manifest = {
            "dataset_id": OPENML_DATASET_ID,
            "source_page": SOURCE_PAGE,
            "source_url": ZENODO_DOWNLOAD_URL,
            "mirror_record": ZENODO_RECORD_URL,
            "doi": "10.5281/zenodo.7395559",
            "license": "CC-BY-4.0 (as stated on the Zenodo mirror record)",
            "file": destination.name,
            "bytes": total,
            "sha256": sha256_file(destination),
            "md5_verified_against_zenodo_record": digest.hexdigest(),
            "expected_bytes": EXPECTED_BYTES,
            "expected_md5": EXPECTED_MD5,
            "dataset_description": "Public, real European credit-card transaction benchmark (OpenML 1597).",
            "downloaded_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(chunk_dir, ignore_errors=True)
        return manifest
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _decode_arff_frame(path: Path) -> pd.DataFrame:
    records, _ = arff.loadarff(str(path))
    frame = pd.DataFrame(records)
    for column in frame.select_dtypes(include=["object"]).columns:
        frame[column] = frame[column].map(
            lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
        )
    return frame


def load_dataset(raw_path: Path, processed_csv: Path | None = None) -> pd.DataFrame:
    """Load the benchmark, verify its schema, normalize types, and preserve time order."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {raw_path}. Run `python scripts/download_dataset.py` first."
        )
    if processed_csv is not None and Path(processed_csv).exists():
        frame = pd.read_csv(processed_csv)
    else:
        if raw_path.suffix.lower() == ".csv":
            frame = pd.read_csv(raw_path)
        else:
            frame = _decode_arff_frame(raw_path)
        if processed_csv is not None:
            Path(processed_csv).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(processed_csv, index=False)

    frame.columns = [str(column).strip() for column in frame.columns]
    lower_names = {name.lower(): name for name in frame.columns}
    if LABEL_COLUMN not in frame.columns and "class" in lower_names:
        frame = frame.rename(columns={lower_names["class"]: LABEL_COLUMN})
    missing = [name for name in [*RAW_FEATURES, LABEL_COLUMN] if name not in frame.columns]
    if missing:
        raise ValueError(f"OpenML dataset is missing required columns: {missing}")
    frame = frame[[*RAW_FEATURES, LABEL_COLUMN]].copy()
    for column in [*RAW_FEATURES, LABEL_COLUMN]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Dataset contains missing or non-finite numeric values.")
    labels = frame[LABEL_COLUMN].astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError("Expected binary Class values 0 (genuine) and 1 (fraud).")
    if len(frame) != EXPECTED_ROWS or int(labels.sum()) != EXPECTED_FRAUD_ROWS:
        raise ValueError(
            f"Dataset count mismatch: rows={len(frame)}, fraud={int(labels.sum())}; "
            f"expected rows={EXPECTED_ROWS}, fraud={EXPECTED_FRAUD_ROWS}."
        )
    frame[LABEL_COLUMN] = labels
    frame = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    return frame


def chronological_splits(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return contiguous, non-overlapping train/validation/test row-index arrays."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must both be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must be less than 1.")
    n_rows = len(frame)
    train_end = int(n_rows * train_fraction)
    validation_end = train_end + int(n_rows * validation_fraction)
    if train_end < 1 or validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("Dataset is too small for the requested chronological split.")
    return (
        np.arange(0, train_end),
        np.arange(train_end, validation_end),
        np.arange(validation_end, n_rows),
    )
