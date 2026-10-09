"""Visual and numerical comparison of reference V2 vs optimized V2 on train-only bytes.

Usage (Windows PowerShell, from TRAMIF repository root):
  python -m scripts.generate_compare_10 --source 'E:\\BODMAS_GW\\BODMAS_disarmed_malware_binaries.zip' --metadata 'E:\\BODMAS_GW\\bodmas_metadata.csv' --limit 10 --output-dir 'data\\derived\\compare_10'

This is an *engineering trial* only, not a scientific performance evaluation.
The original filename SHA is the sample ID; the content SHA is from disarmed bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from time import perf_counter
from zipfile import ZipFile, is_zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.preprocessing.implementation_sbsmi import stream_to_sbsmi
from src.preprocessing.sbsmi_vectorized import stream_to_sbsmi_fast
from src.preprocessing.entropy_image_stream import stream_to_entropy_image
from src.preprocessing.entropy_image_stream_fast import stream_to_entropy_image_fast

SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
TRAIN_START = pd.Timestamp("2019-08-01T00:00:00Z")
TRAIN_STOP = pd.Timestamp("2020-02-01T00:00:00Z")


def select_training_samples(csv_path: Path, mapping_path: Path, is_manifest: bool) -> list[dict]:
    mapping = pd.read_csv(mapping_path, dtype="string")
    if not {"family", "class_id"}.issubset(mapping.columns):
        raise ValueError("Mapping needs family and class_id columns")
    mapping["family"] = mapping["family"].str.strip().str.lower()
    if mapping["family"].isna().any() or mapping["family"].duplicated().any():
        raise ValueError("Invalid family mapping")
    families = set(mapping["family"].tolist())
    if len(families) != 51 or set(pd.to_numeric(mapping["class_id"])) != set(range(51)):
        raise ValueError("Expected the existing train-derived 51-family mapping")

    data = pd.read_csv(csv_path, dtype="string")
    id_col = next((c for c in ("sha", "sha256", "sample_id") if c in data), None)
    fam_col = next((c for c in ("family", "family_norm") if c in data), None)
    if id_col is None or fam_col is None or "timestamp" not in data:
        raise ValueError("Manifest/metadata needs SHA, family, and timestamp")
    data["original_sha"] = data[id_col].str.strip().str.lower()
    data["family_clean"] = data[fam_col].str.strip().str.lower()
    data = data[data["family_clean"].isin(families)].copy()
    if data.empty:
        raise ValueError("No samples from the frozen train-derived families")
    if data["original_sha"].isna().any() or not data["original_sha"].str.fullmatch(SHA_RE.pattern).fillna(False).all():
        raise ValueError("Invalid original SHA in eligible-family metadata")
    if data["original_sha"].duplicated().any():
        raise ValueError("Duplicate original SHA in eligible-family metadata")
    times = pd.to_datetime(data["timestamp"], format="mixed", utc=True, errors="coerce")
    if times.isna().any():
        raise ValueError("Invalid timestamp for eligible-family sample")
    data["time_utc"] = times
    selected = data[(times >= TRAIN_START) & (times < TRAIN_STOP)].copy()
    if is_manifest:
        if "split" not in data:
            raise ValueError("--manifest requires split column")
        normalized = data["split"].str.strip().str.lower().replace({"future_test": "test"})
        if not normalized.loc[selected.index].eq("train").all():
            raise ValueError("Manifest's 'train' labels disagree with timestamps")
        if normalized.eq("train").sum() != len(selected):
            raise ValueError("Manifest has train-labeled records outside the training period")
    if selected.empty:
        raise ValueError("No Aug 2019 - Jan 2020 samples selected")
    selected = selected.sort_values("original_sha")
    return [{"sha": row.original_sha, "month": row.time_utc.strftime("%Y-%m")}
            for row in selected.itertuples(index=False)]


def source_index(source_path: Path) -> tuple[str, dict[str, list[tuple[str, int]]]]:
    index: dict[str, list[tuple[str, int]]] = {}
    def record(rel: str, size: int) -> None:
        filename = PurePosixPath(rel.replace("\\", "/")).name.lower()
        sha = filename[:-4] if filename.endswith(".exe") else filename
        if SHA_RE.fullmatch(sha):
            index.setdefault(sha, []).append((rel, size))

    if source_path.is_file() and is_zipfile(source_path):
        with ZipFile(source_path) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    record(member.filename, member.file_size)
        return "zip", index
    if source_path.is_dir():
        for file in source_path.rglob("*"):
            if file.is_file():
                record(file.relative_to(source_path).as_posix(), file.stat().st_size)
        return "folder", index
    raise ValueError(f"Source must be a ZIP or folder: {source_path}")


@contextmanager
def open_source(source_path: Path, mode: str, member: str):
    if mode == "zip":
        with ZipFile(source_path) as archive:
            with archive.open(member, "r") as binary:
                yield binary
    else:
        relative = PurePosixPath(member.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid relative input path")
        with (source_path / Path(*relative.parts)).open("rb") as binary:
            yield binary


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.sha = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.sha.update(data)
        self.count += len(data)
        return data


def run_transform(source_path, mode, member, size, implementation, modality, chunk_size):
    with open_source(source_path, mode, member) as stream:
        reader = HashingReader(stream)
        start = perf_counter()
        if modality == "sbsmi":
            fn = stream_to_sbsmi if implementation == "reference_v2" else stream_to_sbsmi_fast
            image = fn(reader, bit_num=6, chunk_size=chunk_size)
        else:
            fn = stream_to_entropy_image if implementation == "reference_v2" else stream_to_entropy_image_fast
            image = fn(reader, file_size=size, chunk_size=chunk_size)
        seconds = perf_counter() - start
        if reader.count != size:
            raise ValueError(f"Incomplete source read for {member}: {reader.count} != {size}")
    if image.shape != (64, 64):
        raise ValueError("Wrong image shape")
    if modality == "sbsmi" and image.dtype != np.uint8:
        raise ValueError("SBSMI must be uint8")
    if modality == "entropy" and (not np.isfinite(image).all() or image.min() < 0 or image.max() > 1):
        raise ValueError("Entropy must be finite in [0,1]")
    return image, reader.sha.hexdigest(), seconds


def write_view(output_root: Path, sha: str, implementation: str, modality: str, array):
    directory = output_root / implementation / modality
    directory.mkdir(parents=True, exist_ok=True)
    if modality == "sbsmi":
        path = directory / f"{sha}.png"
        Image.fromarray(array).save(path, format="PNG")
        with Image.open(path) as readback:
            if not np.array_equal(np.array(readback), array):
                raise RuntimeError("SBSMI save/load mismatch")
        return np.asarray(array)
    data = np.asarray(array, dtype=np.float32)
    path = directory / f"{sha}.npy"
    np.save(path, data, allow_pickle=False)
    if not np.array_equal(np.load(path, allow_pickle=False), data):
        raise RuntimeError("Entropy save/load mismatch")
    preview = np.floor(np.clip(data * 255.0, 0, 255)).astype(np.uint8)
    png = output_root / implementation / "entropy_preview" / f"{sha}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview).save(png, format="PNG")
    return preview


def make_contact_sheet(output_root: Path, samples: list[dict], modality: str):
    n = len(samples)
    scale = 3
    tile_size = 64 * scale
    left = 160
    row_h = tile_size + 13
    sheet = Image.new("RGB", (left + 2 * (tile_size + 18), 38 + n * row_h), "white")
    drawer = ImageDraw.Draw(sheet)
    drawer.text((left + 4, 10), "Reference V2", fill="black")
    drawer.text((left + tile_size + 22, 10), "Optimized V2", fill="black")
    for index, sample in enumerate(samples):
        y = 34 + index * row_h
        drawer.text((8, y + 6), sample["sha"][:14], fill="black")
        drawer.text((8, y + 24), sample["month"], fill="black")
        view_dir = "sbsmi" if modality == "sbsmi" else "entropy_preview"
        for col, impl in enumerate(("reference_v2", "fast_v2")):
            image_file = output_root / impl / view_dir / (sample["sha"] + ".png")
            with Image.open(image_file) as im:
                resized = im.convert("RGB").resize((tile_size, tile_size), Image.Resampling.NEAREST)
            sheet.paste(resized, (left + col * (tile_size + 18), y))
    destination = output_root / f"{modality}_comparison_sheet.png"
    sheet.save(destination)
    return destination


def compare_10(source: Path, csv_file: Path, mapping: Path, is_manifest: bool,
               output: Path, limit: int = 10, max_size_mib: float | None = None,
               chunk_size: int = 65536, atol: float = 1e-6) -> list[dict]:
    if limit < 1 or chunk_size < 1 or atol < 0 or (max_size_mib is not None and max_size_mib <= 0):
        raise ValueError("Invalid limit, chunk size, tolerance or max-size setting")
    candidates = select_training_samples(csv_file, mapping, is_manifest)
    mode, index = source_index(source)
    choices = []
    missing = 0
    for sample in candidates:
        matches = index.get(sample["sha"], [])
        if not matches:
            missing += 1
            continue  # This is a smoke test, not the full-dataset audit.
        if len(matches) != 1:
            raise ValueError(f"Ambiguous archive entries for {sample['sha']}")
        member, size = matches[0]
        if size <= 0 or (max_size_mib and size > max_size_mib * (1024**2)):
            continue
        choices.append({**sample, "member": member, "size": size})
        if len(choices) == limit:
            break
    if len(choices) < limit:
        raise ValueError(f"Found only {len(choices)} accessible train samples (requested {limit})")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Selected {len(choices)} TRAIN binaries | source mode: {mode} | missing skipped: {missing}")
    results = []
    for number, sample in enumerate(choices, 1):
        images = {}
        checksums = set()
        timings = {}
        previews = {}
        for modality in ("sbsmi", "entropy"):
            for implementation in ("reference_v2", "fast_v2"):
                image, digest, elapsed = run_transform(source, mode, sample["member"],
                                                       sample["size"], implementation, modality, chunk_size)
                checksums.add(digest)
                images[(modality, implementation)] = image
                timings[(modality, implementation)] = elapsed
                previews[(modality, implementation)] = write_view(output, sample["sha"], implementation, modality, image)
        if len(checksums) != 1:
            raise RuntimeError(f"Different source bytes processed for {sample['sha']}")
        sbs_reference = images[("sbsmi", "reference_v2")]
        sbs_fast = images[("sbsmi", "fast_v2")]
        entropy_reference = images[("entropy", "reference_v2")]
        entropy_fast = images[("entropy", "fast_v2")]
        sbs_difference = int(np.count_nonzero(sbs_reference != sbs_fast))
        entropy_max_difference = float(np.max(np.abs(entropy_reference.astype(np.float64) - entropy_fast.astype(np.float64))))
        ent_f32_difference = int(np.count_nonzero(entropy_reference.astype(np.float32) != entropy_fast.astype(np.float32)))
        ent_preview_difference = int(np.count_nonzero(previews[("entropy", "reference_v2")] != previews[("entropy", "fast_v2")]))
        passed = sbs_difference == 0 and entropy_max_difference <= atol
        result = {"sha": sample["sha"], "month": sample["month"], "split": "train",
                  "bytes": sample["size"], "disarmed_content_sha256": next(iter(checksums)),
                  "sbsmi_differing_pixels": sbs_difference, "entropy_max_difference": f"{entropy_max_difference:.14g}",
                  "entropy_differing_float32_values": ent_f32_difference,
                  "entropy_preview_differing_pixels": ent_preview_difference,
                  "sbsmi_reference_seconds": f"{timings[('sbsmi','reference_v2')]:.6f}",
                  "sbsmi_fast_seconds": f"{timings[('sbsmi','fast_v2')]:.6f}",
                  "entropy_reference_seconds": f"{timings[('entropy','reference_v2')]:.6f}",
                  "entropy_fast_seconds": f"{timings[('entropy','fast_v2')]:.6f}",
                  "status": "PASS" if passed else "FAIL"}
        results.append(result)
        print(f"[{number}/{len(choices)}] {sample['sha'][:12]} | SBSMI {sbs_difference} differing pixels | "
              f"Entropy maxdiff={entropy_max_difference:.3g} | {result['status']}", flush=True)
    summary_path = output / "comparison.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for modality in ("sbsmi", "entropy"):
        print("Contact sheet:", make_contact_sheet(output, choices, modality))
    print("Comparison report:", summary_path)
    if any(r["status"] != "PASS" for r in results):
        raise RuntimeError("Some outputs differ beyond tolerance; review comparison.csv")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="BODMAS disarmed ZIP or altered/ folder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metadata", type=Path, help="BODMAS bodmas_metadata.csv")
    group.add_argument("--manifest", type=Path, help="Canonical multiview_v2_manifest.csv")
    parser.add_argument("--mapping", type=Path, default=Path("data/manifests/bodmas_eligible_families_v0.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/derived/compare_old_fast_10"))
    parser.add_argument("--limit", type=int, default=10, help="Number of TRAIN binary samples, not number of image files")
    parser.add_argument("--max-size-mib", type=float, help="Optional cap for quicker engineering smoke test")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--entropy-atol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    compare_10(args.source, args.manifest or args.metadata, args.mapping,
               args.manifest is not None, args.output_dir, args.limit,
               args.max_size_mib, args.chunk_size, args.entropy_atol)


if __name__ == "__main__":
    main()
