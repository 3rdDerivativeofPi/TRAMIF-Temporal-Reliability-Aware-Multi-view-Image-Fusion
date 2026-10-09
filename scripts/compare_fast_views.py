"""Compare original V2 and experimental fast Entropy/SBSMI on one binary.

Operates on inert binary bytes; never executes the input. Do NOT compare the
V1 PyTorch-area entropy output with V2 geometric-area entropy: those are
intentionally different preprocessing specifications.

Usage (PowerShell):
  python -m scripts.compare_fast_views --file 'E:\\BODMAS_GW\\altered\\SHA.exe'
  python -m scripts.compare_fast_views --zip 'E:\\BODMAS_GW\\BODMAS_disarmed_malware_binaries.zip' --member 'altered/SHA.exe'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from src.preprocessing.implementation_sbsmi import stream_to_sbsmi
from src.preprocessing.sbsmi_vectorized import stream_to_sbsmi_fast
from src.preprocessing.entropy_image_stream import stream_to_entropy_image
from src.preprocessing.entropy_image_stream_fast import stream_to_entropy_image_fast


@contextmanager
def open_binary(file_path: Path | None, zip_path: Path | None, member: str | None):
    if file_path is not None:
        with file_path.open("rb") as binary:
            yield binary, file_path.stat().st_size
    else:
        with ZipFile(zip_path, "r") as archive:
            info = archive.getinfo(member)
            if info.is_dir():
                raise ValueError("ZIP member is a directory")
            with archive.open(info, "r") as binary:
                yield binary, info.file_size


def compare(
    file_path: Path | None = None,
    zip_path: Path | None = None,
    member: str | None = None,
    chunk_size: int = 65536,
    entropy_atol: float = 1e-6,
) -> dict:
    """Compare on the same specified source; no image files are required."""
    if (file_path is None) == (zip_path is None):
        raise ValueError("Provide exactly one of file_path or zip_path")
    if zip_path is not None and not member:
        raise ValueError("ZIP input needs member")
    if file_path is not None and member:
        raise ValueError("member is only valid with zip_path")
    if chunk_size <= 0 or entropy_atol < 0:
        raise ValueError("Invalid chunk size or tolerance")

    with open_binary(file_path, zip_path, member) as (binary, file_size):
        if file_size == 0:
            raise ValueError("Empty files are not valid inputs")
        sha = hashlib.sha256()
        total = 0
        while block := binary.read(65536):
            sha.update(block)
            total += len(block)
        if total != file_size:
            raise RuntimeError("Input size changed during hashing")
    print(f"Source size: {file_size:,} bytes")
    print(f"Disarmed-content SHA-256: {sha.hexdigest()}")

    class FingerprintReader:
        """Verify each compared function actually consumes the same source bytes."""
        def __init__(self, binary):
            self.binary = binary
            self.digest = hashlib.sha256()
            self.bytes_read = 0

        def read(self, count=-1):
            data = self.binary.read(count)
            self.digest.update(data)
            self.bytes_read += len(data)
            return data

    def run(fn, needs_size):
        with open_binary(file_path, zip_path, member) as (binary, current_size):
            if current_size != file_size:
                raise RuntimeError("Input size changed between implementations")
            reader = FingerprintReader(binary)
            start = perf_counter()
            if needs_size:
                out = fn(reader, file_size=file_size, chunk_size=chunk_size)
            else:
                out = fn(reader, bit_num=6, chunk_size=chunk_size)
            elapsed = perf_counter() - start
            if reader.bytes_read != file_size or reader.digest.hexdigest() != sha.hexdigest():
                raise RuntimeError("Preprocessors read different source bytes; comparison invalid")
        return out, elapsed

    sbs_ref, st_ref = run(stream_to_sbsmi, needs_size=False)
    sbs_fast, st_fast = run(stream_to_sbsmi_fast, needs_size=False)
    sbs_different = int(np.count_nonzero(sbs_ref != sbs_fast))
    sbs_ok = (sbs_ref.dtype == sbs_fast.dtype == np.uint8 and
              sbs_ref.shape == sbs_fast.shape == (64, 64) and sbs_different == 0)

    ent_ref, et_ref = run(stream_to_entropy_image, needs_size=True)
    ent_fast, et_fast = run(stream_to_entropy_image_fast, needs_size=True)
    difference = np.abs(ent_ref.astype(np.float64) - ent_fast.astype(np.float64))
    # In V2 training, both entropy image paths are stored as float32 NPY.
    f32_ref = ent_ref.astype(np.float32)
    f32_fast = ent_fast.astype(np.float32)
    # Diagnostic only: V2 stores entropy in float32 NPY, not this 8-bit preview.
    preview_ref = np.clip(ent_ref * 255.0, 0, 255).astype(np.uint8)
    preview_fast = np.clip(ent_fast * 255.0, 0, 255).astype(np.uint8)
    preview_different = int(np.count_nonzero(preview_ref != preview_fast))
    ent_ok = (ent_ref.shape == ent_fast.shape == (64, 64)
              and np.isfinite(ent_ref).all() and np.isfinite(ent_fast).all()
              and np.allclose(ent_ref, ent_fast, rtol=0.0, atol=entropy_atol))

    result = {
        "source": str(file_path if file_path is not None else zip_path),
        "zip_member": member,
        "source_size": file_size,
        "disarmed_content_sha256": sha.hexdigest(),
        "chunk_size": chunk_size,
        "sbsmi": {
            "pass": bool(sbs_ok), "different_pixels": sbs_different,
            "old_seconds": st_ref, "fast_seconds": st_fast,
        },
        "entropy": {
            "pass": bool(ent_ok),
            "max_abs_difference": float(difference.max()),
            "mean_abs_difference": float(difference.mean()),
            "different_float32_values": int(np.count_nonzero(f32_ref != f32_fast)),
            "different_8bit_preview_pixels": preview_different,
            "entropy_atol": entropy_atol,
            "old_seconds": et_ref, "fast_seconds": et_fast,
        },
        "overall_pass": bool(sbs_ok and ent_ok),
    }
    print(f"SBSMI: {'PASS' if sbs_ok else 'FAIL'} | differing uint8 pixels = {sbs_different}")
    print(f"Entropy: {'PASS' if ent_ok else 'FAIL'} | max difference = {difference.max():.3g}; "
          f"mean difference = {difference.mean():.3g}; float32 differing values = "
          f"{result['entropy']['different_float32_values']}; "
          f"8-bit preview differences = {preview_different}")
    print(f"SBSMI time: {st_ref:.4f}s original / {st_fast:.4f}s fast")
    print(f"Entropy time: {et_ref:.4f}s original / {et_fast:.4f}s fast")
    print(f"OVERALL: {'PASS' if result['overall_pass'] else 'FAIL'}")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path)
    source.add_argument("--zip", type=Path)
    parser.add_argument("--member", help="ZIP member path, e.g., altered/<sha>.exe")
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--entropy-atol", type=float, default=1e-6)
    parser.add_argument("--report", type=Path, help="Optional JSON comparison report")
    args = parser.parse_args(argv)
    output = compare(args.file, args.zip, args.member, args.chunk_size, args.entropy_atol)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
