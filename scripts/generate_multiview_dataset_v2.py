"""Distributed, streamed, split/month-aware TRAMIF V2 image generator.

Run THE SAME script on each laptop, setting --modalities to its assigned view.
Each job has a local output directory. Merge later by original sample SHA.
The future-test split is explicitly guarded by a frozen config ID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import platform
import zlib
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZipFile, is_zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.multiview_contract import (
    MODALITIES, SHA_RE, canonical_hash, expected_report, file_sha256, load_image,
    load_manifest, load_mapping, output_paths, select_rows, validate_image,
    valid_report, write_json_atomic,
)
from src.preprocessing.entropy_image_stream_fast import (
    stream_to_entropy_image_fast as stream_to_entropy_image,
)
from src.preprocessing.raw_byte_stream import stream_to_raw_byte
from src.preprocessing.sbsmi_vectorized import (
    stream_to_sbsmi_fast as stream_to_sbsmi,
)

LOG_FIELDS = ("sha", "split", "month", "modality", "status", "seconds", "error")
SOURCE_MODULES = (
    "scripts/generate_multiview_dataset_v2.py",
    "src/data/multiview_contract.py",
    "src/preprocessing/raw_byte_stream.py",
    "src/preprocessing/entropy_core.py",
    "src/preprocessing/entropy_stream.py",
    "src/preprocessing/entropy_image_stream.py",
    "src/preprocessing/implementation_sbsmi.py",
    "src/preprocessing/entropy_stream_fast.py",
    "src/preprocessing/entropy_image_stream_fast.py",
    "src/preprocessing/sbsmi_vectorized.py",
)


def normalized_source_digest(path: Path) -> str:
    """Normalize only line endings; ensure equal code fingerprint across Windows/Unix."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def build_config(manifest: Path, mapping: Path) -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    modules = {rel: normalized_source_digest(repo_root / rel) for rel in SOURCE_MODULES}
    return {
        "schema": 2,
        "version": "tramif-v2-fractional-area-2026-10",
        "manifest_sha256": file_sha256(manifest),
        "mapping_sha256": file_sha256(mapping),
        "source_code_sha256": modules,
        "fixed_preprocessing": {
            "raw_byte": "width=256; valid-mask; fractional-area resize 64x64; float32 [0,1]",
            "entropy": "256-byte windows stride128; mean overlapping normalized H/8; fractional-area resize; float32 [0,1]",
            "sbsmi": "MSB-first; non-overlapping 6-bit blocks; high-order-zero tail; row probability; floor(255*P); uint8 PNG",
        },
        "extensions": {"raw_byte": ".npy", "entropy": ".npy", "sbsmi": ".png"},
    }


def ensure_output_config(out: Path, config: dict, splits: list[str], frozen_id: str | None,
                         dry_run: bool) -> str:
    config_id = canonical_hash(config)
    existing_file = out / "config.json"
    if "test" in splits:
        if splits != ["test"]:
            raise ValueError("Process future test separately: --splits test")
        if frozen_id != config_id:
            raise ValueError("Future test needs --frozen-config-id with the exact historical config ID")
        if not existing_file.is_file():
            raise ValueError("Future test requires previous historical config.json in this output directory")
    if existing_file.exists():
        saved = json.loads(existing_file.read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError("Configuration mismatch! Use a NEW output directory/version")
    elif not dry_run:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Refusing nonempty output directory without config.json")
        write_json_atomic(existing_file, config)
    return config_id


def build_source_index(path: Path) -> tuple[str, dict[str, list[dict]]]:
    """Select by ORIGINAL SHA filename, never by disarmed-content hash."""
    index: dict[str, list[dict]] = {}
    def add(relative: str, size: int, crc: str | None = None) -> None:
        basename = PurePosixPath(relative.replace("\\", "/")).name.lower()
        sample_sha = basename[:-4] if basename.endswith(".exe") else basename
        if SHA_RE.fullmatch(sample_sha):
            index.setdefault(sample_sha, []).append({"relative": relative, "size": size, "zip_crc32": crc})

    if path.is_dir():
        mode = "folder"
        for file in path.rglob("*"):
            if file.is_file():
                add(file.relative_to(path).as_posix(), file.stat().st_size)
    elif path.is_file() and is_zipfile(path):
        mode = "zip"
        with ZipFile(path) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    add(info.filename, info.file_size, f"{info.CRC:08x}")
    else:
        raise ValueError(f"Source is not a directory or readable ZIP: {path}")
    return mode, index


def select_sources(rows: list[dict], index: dict[str, list[dict]]) -> list[dict]:
    selected = []
    for row in rows:
        sources = index.get(row["sha"], [])
        if len(sources) != 1:
            raise ValueError(f"Missing/ambiguous binary for {row['sha']}: {len(sources)} matches")
        entry = sources[0]
        if not 0 < entry["size"]:
            raise ValueError(f"Empty source binary: {row['sha']}")
        if row["file_size"] is not None and entry["size"] != row["file_size"]:
            raise ValueError(f"Source size differs from indexed size: {row['sha']}")
        if row["zip_crc32"] and entry["zip_crc32"] and entry["zip_crc32"] != row["zip_crc32"]:
            raise ValueError(f"ZIP CRC differs from indexed CRC: {row['sha']}")
        selected.append({**row, "source": entry})
    return selected


class DigestReader:
    """Capture the disarmed bytes ACTUALLY seen by the preprocessor."""
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()
        self.crc = 0
        self.count = 0

    def read(self, n=-1):
        data = self.stream.read(n)
        self.sha.update(data)
        self.crc = zlib.crc32(data, self.crc)
        self.count += len(data)
        return data

    def fingerprint(self) -> dict:
        return {"disarmed_content_sha256": self.sha.hexdigest(),
                "zip_crc32": f"{self.crc & 0xffffffff:08x}", "file_size": self.count}


def open_source(source: Path, mode: str, archive: ZipFile | None, member: str):
    if mode == "zip":
        assert archive is not None
        return archive.open(member, "r")
    relative = PurePosixPath(member.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source path: {member}")
    return (source / Path(*relative.parts)).open("rb")


def verified_source(reader: DigestReader, row: dict) -> dict:
    fingerprint = reader.fingerprint()
    if fingerprint["file_size"] != row["source"]["size"]:
        raise ValueError(f"Binary changed or reading was incomplete: {row['sha']}")
    crc = row["zip_crc32"] or row["source"]["zip_crc32"]
    if crc is not None and fingerprint["zip_crc32"] != crc:
        raise ValueError(f"Input CRC mismatch: {row['sha']}")
    return fingerprint


def make_image(reader: DigestReader, modality: str, size: int, chunk_size: int) -> np.ndarray:
    if modality == "raw_byte":
        array = stream_to_raw_byte(reader, file_size=size, chunk_size=chunk_size)
    elif modality == "entropy":
        array = stream_to_entropy_image(reader, file_size=size, chunk_size=chunk_size)
    elif modality == "sbsmi":
        array = stream_to_sbsmi(reader, bit_num=6, chunk_size=chunk_size)
    else:
        raise ValueError(modality)
    if modality != "sbsmi":
        array = array.astype(np.float32)
    validate_image(array, modality)
    return array


def save_image_atomic(path: Path, array: np.ndarray, modality: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        if modality == "sbsmi":
            Image.fromarray(array, mode="L").save(temporary, format="PNG")
        else:
            with temporary.open("wb") as f:
                np.save(f, array, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def cached_or_none(out: Path, row: dict, modality: str, config_id: str) -> dict | None:
    try:
        return valid_report(out, row, modality, config_id)
    except (ValueError, OSError, KeyError, TypeError, EOFError):
        return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="ZIP or extracted folder")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/multiview_v2_manifest.csv"))
    parser.add_argument("--mapping", type=Path, default=Path("data/manifests/bodmas_eligible_families_v0.csv"))
    parser.add_argument("--output-dir", type=Path, required=True, help="LOCAL laptop output directory")
    parser.add_argument("--modalities", choices=("all", *MODALITIES), nargs="+", default=["all"])
    parser.add_argument("--splits", choices=("train", "validation", "test"), nargs="+", default=["train", "validation"])
    parser.add_argument("--months", nargs="+", help="Optional YYYY-MM month selection")
    parser.add_argument("--worker-id", default="unnamed", help="Laptop/member label in run metadata")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N samples PER selected split on this shard")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-source-on-skip", action="store_true", help="Rehash source bytes on cache hits")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-previews", action="store_true", help="Extra PNG previews, not for model input")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--frozen-config-id", help="Required for future test; obtained from historical run")
    return parser.parse_args(argv)


def run(args) -> dict:
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive")
    splits = list(dict.fromkeys(args.splits))
    modalities = list(MODALITIES) if "all" in args.modalities else list(dict.fromkeys(args.modalities))
    if "test" in splits and splits != ["test"]:
        raise ValueError("Process future test separately")
    if args.frozen_config_id and splits != ["test"]:
        raise ValueError("--frozen-config-id is only for future test")
    mapping = load_mapping(args.mapping)
    rows = load_manifest(args.manifest, mapping)
    selected = select_rows(rows, splits, args.months, args.limit, args.shard_id, args.num_shards)
    if not selected:
        raise ValueError("No samples selected: check manifest, splits, months, and shard")

    source = args.source.expanduser().resolve()
    mode, index = build_source_index(source)
    selected = select_sources(selected, index)  # strict preflight BEFORE any writes
    config = build_config(args.manifest, args.mapping)
    out = args.output_dir.expanduser().resolve()
    config_id = ensure_output_config(out, config, splits, args.frozen_config_id, args.dry_run)
    print("MODE:", mode, "MODALITIES:", modalities, "SPLITS:", splits)
    print("SELECTED:", len(selected), "SHARD:", f"{args.shard_id}/{args.num_shards}")
    print("CONFIG ID:", config_id)
    if args.dry_run:
        return {"selected": len(selected), "config_id": config_id, "counts": {}}

    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    log_path = logs / f"{now}_{'_'.join(modalities)}.csv"
    write_json_atomic(log_path.with_suffix(".json"), {
        "worker_id": args.worker_id, "source_mode": mode, "config_id": config_id,
        "modalities": modalities, "splits": splits, "selected": len(selected),
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "chunk_size": args.chunk_size, "python": platform.python_version(),
        "numpy": np.__version__, "pillow": Image.__version__,
        "utc_started": datetime.now(timezone.utc).isoformat(),
    })
    counts = {"CREATED": 0, "VERIFIED": 0, "ERROR": 0}
    archive = ZipFile(source, "r") if mode == "zip" else None
    try:
        with log_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            writer.writeheader()
            f.flush()
            for row in tqdm(selected, desc="Generating", unit="file"):
                for modality in modalities:
                    start = perf_counter()
                    status, error = "CREATED", ""
                    try:
                        cached = None if args.overwrite else cached_or_none(out, row, modality, config_id)
                        if cached is not None:
                            if args.verify_source_on_skip:
                                with open_source(source, mode, archive, row["source"]["relative"]) as stream:
                                    reader = DigestReader(stream)
                                    while reader.read(args.chunk_size):
                                        pass
                                    fingerprint = verified_source(reader, row)
                                if cached["disarmed_content_sha256"] != fingerprint["disarmed_content_sha256"]:
                                    raise ValueError("Cached source hash differs from source bytes")
                            status = "VERIFIED"
                        else:
                            with open_source(source, mode, archive, row["source"]["relative"]) as stream:
                                reader = DigestReader(stream)
                                image = make_image(reader, modality, row["source"]["size"], args.chunk_size)
                                fingerprint = verified_source(reader, row)
                            path, report_path = output_paths(out, row, modality)
                            save_image_atomic(path, image, modality)
                            restored = load_image(path, modality)
                            if not np.array_equal(image, restored):
                                raise ValueError("Saved pixels differ from computed pixels")
                            report = {**expected_report(row, modality, config_id),
                                      **fingerprint,
                                      "status": "PASS", "source_member": row["source"]["relative"],
                                      "shape": [64, 64],
                                      "dtype": str(image.dtype), "output_sha256": file_sha256(path)}
                            write_json_atomic(report_path, report)
                        if args.save_previews and modality != "sbsmi":
                            path, _ = output_paths(out, row, modality)
                            dest = out / "previews" / row["split"] / row["month"] / modality / f"{row['sha']}.png"
                            pixels = np.floor(np.clip(load_image(path, modality) * 255, 0, 255)).astype(np.uint8)
                            save_image_atomic(dest, pixels, "sbsmi")
                    except Exception as exc:
                        status, error = "ERROR", f"{type(exc).__name__}: {exc}"
                    counts[status] += 1
                    writer.writerow({"sha": row["sha"], "split": row["split"], "month": row["month"],
                                     "modality": modality, "status": status,
                                     "seconds": f"{perf_counter() - start:.6f}", "error": error})
                    f.flush()
                    if status == "ERROR":
                        print("ERROR:", row["sha"], modality, error, file=sys.stderr)
                        if not args.continue_on_error:
                            raise RuntimeError(error)
    finally:
        if archive is not None:
            archive.close()
    print("Completed:", counts, "Log:", log_path)
    if counts["ERROR"]:
        raise RuntimeError(f"{counts['ERROR']} failed outputs; see {log_path}")
    return {"selected": len(selected), "config_id": config_id, "counts": counts, "log": str(log_path)}


def main():
    try:
        run(parse_args())
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("FAILED:", exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
