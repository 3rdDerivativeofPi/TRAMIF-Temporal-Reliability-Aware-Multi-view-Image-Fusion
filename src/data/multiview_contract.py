"""Shared, deterministic dataset contract for distributed TRAMIF image generation.

The original metadata SHA is a sample ID. It is not expected to equal
SHA-256 of the disarmed executable. Never derive labels from future data.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

MODALITIES = ("raw_byte", "entropy", "sbsmi")
SPLITS = ("train", "validation", "test")
SPLIT_RANGE = {
    "train": ("2019-08-01", "2020-02-01"),
    "validation": ("2020-02-01", "2020-04-01"),
    "test": ("2020-04-01", "2020-10-01"),
}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


def file_sha256(path: Path, chunk_size: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True,
                                   allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV or missing header: {path}")
        return list(reader)


def write_csv(path: Path, records: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_mapping(path: Path) -> dict[str, int]:
    """Repo's authoritative training-derived CSV: family,class_id,..."""
    rows = read_csv(path)
    mapping = {}
    indices = set()
    for r in rows:
        family = str(r.get("family", "")).strip().lower()
        if not family or family in mapping:
            raise ValueError(f"Missing/duplicate family in mapping: {family!r}")
        index = int(r["class_id"])
        if index < 0 or index in indices or str(index) != str(r["class_id"]).strip():
            raise ValueError(f"Invalid/duplicate class_id in mapping: {r['class_id']!r}")
        mapping[family] = index
        indices.add(index)
    if len(mapping) != 51 or indices != set(range(51)):
        raise ValueError("Expected repository's locked 51 classes, IDs 0..50")
    return mapping


def normalize_split(value: str) -> str:
    result = str(value).strip().lower()
    result = "test" if result == "future_test" else result
    if result not in SPLITS:
        raise ValueError(f"Unknown split: {value!r}")
    return result


def parse_timestamp(text: str) -> tuple[str, str]:
    """Return UTC YYYY-MM-DD and YYYY-MM. Builder resolves BODMAS mixed formats."""
    value = str(text).strip()
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {text!r}") from exc
    if date.tzinfo is not None:
        date = date.astimezone(timezone.utc)
    return date.date().isoformat(), date.strftime("%Y-%m")


def load_manifest(path: Path, mapping: dict[str, int]) -> list[dict]:
    """Fail closed on sample IDs, date/split mismatches and class mapping.

    CSV requires sha, family, split, timestamp, month, class_id.
    Optional file_size and zip_crc32 are verified when present.
    """
    rows = read_csv(path)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    required = {"sha", "family", "split", "timestamp", "month", "class_id"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(rows[0]))}")
    seen = set()
    result = []
    for r in rows:
        sha = str(r["sha"]).strip().lower()
        if SHA_RE.fullmatch(sha) is None or sha in seen:
            raise ValueError(f"Invalid or duplicate original SHA: {sha!r}")
        seen.add(sha)
        family = str(r["family"]).strip().lower()
        if family not in mapping or str(r["class_id"]).strip() != str(mapping[family]):
            raise ValueError(f"Frozen family mapping mismatch: {sha}")
        split = normalize_split(r["split"])
        day, month = parse_timestamp(r["timestamp"])
        start, end = SPLIT_RANGE[split]
        if not start <= day < end:
            raise ValueError(f"Date outside {split}: {sha}: {day}")
        if str(r["month"]).strip() != month:
            raise ValueError(f"Month mismatch: {sha}")
        size = str(r.get("file_size") or "").strip()
        crc = str(r.get("zip_crc32") or "").strip().lower()
        if size and (not size.isdecimal() or int(size) <= 0):
            raise ValueError(f"Invalid file size for {sha}")
        if crc and re.fullmatch(r"[0-9a-f]{8}", crc) is None:
            raise ValueError(f"Invalid indexed CRC32 for {sha}")
        result.append({"sha": sha, "family": family, "class_id": mapping[family],
                       "split": split, "timestamp": str(r["timestamp"]).strip(),
                       "month": month, "file_size": int(size) if size else None,
                       "zip_crc32": crc or None})
    return sorted(result, key=lambda x: (SPLITS.index(x["split"]), x["month"], x["sha"]))


def select_rows(rows: list[dict], splits: list[str], months: list[str] | None,
                limit: int | None, shard_id: int, num_shards: int) -> list[dict]:
    if num_shards < 1 or shard_id < 0 or shard_id >= num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards and num-shards >= 1")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    chosen = []
    count_by_split = {}
    for row in rows:
        if row["split"] not in splits or (months and row["month"] not in months):
            continue
        # SHA-independent of manifest row ordering and other machines' files.
        if int(row["sha"], 16) % num_shards != shard_id:
            continue
        count = count_by_split.get(row["split"], 0)
        if limit is not None and count >= limit:
            continue
        count_by_split[row["split"]] = count + 1
        chosen.append(row)
    return chosen


def output_paths(root: Path, row: dict, modality: str) -> tuple[Path, Path]:
    if modality not in MODALITIES:
        raise ValueError(modality)
    ext = ".png" if modality == "sbsmi" else ".npy"
    folder = Path(root) / row["split"] / row["month"] / modality
    image = folder / (row["sha"] + ext)
    report = Path(root) / "reports" / row["split"] / row["month"] / modality / (row["sha"] + ".json")
    return image, report


def load_image(path: Path, modality: str) -> np.ndarray:
    if modality == "sbsmi":
        with Image.open(path) as img:
            if img.format != "PNG" or img.mode != "L":
                raise ValueError(f"Expected grayscale PNG: {path}")
            array = np.asarray(img).copy()
    else:
        array = np.load(path, allow_pickle=False)
    validate_image(array, modality)
    return array


def validate_image(array: np.ndarray, modality: str) -> None:
    if not isinstance(array, np.ndarray) or array.shape != (64, 64):
        raise ValueError(f"{modality} shape is not 64x64")
    if modality == "sbsmi":
        if array.dtype != np.uint8:
            raise ValueError("SBSMI must have uint8 storage")
    else:
        if array.dtype != np.dtype("<f4"):
            raise ValueError(f"{modality} must have float32 storage")
        if not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError(f"{modality} values must be finite and in [0,1]")


def expected_report(row: dict, modality: str, config_id: str) -> dict:
    return {"original_sha": row["sha"], "family": row["family"],
            "class_id": row["class_id"], "split": row["split"],
            "month": row["month"], "timestamp": row["timestamp"],
            "modality": modality, "config_id": config_id}


def valid_report(root: Path, row: dict, modality: str, config_id: str) -> dict:
    image, report_path = output_paths(root, row, modality)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError(f"Report status not PASS: {report_path}")
    for k, v in expected_report(row, modality, config_id).items():
        if report.get(k) != v:
            raise ValueError(f"Report {k} mismatch: {report_path}")
    if SHA_RE.fullmatch(str(report.get("disarmed_content_sha256", ""))) is None:
        raise ValueError(f"Disarmed content SHA missing: {report_path}")
    if report.get("output_sha256") != file_sha256(image):
        raise ValueError(f"Output digest mismatch: {image}")
    actual = load_image(image, modality)
    if report.get("shape") != [64, 64] or report.get("dtype") != str(actual.dtype):
        raise ValueError(f"Output metadata mismatch: {report_path}")
    return report
