"""Streaming SBSMI generation from ordinary files and ZIP members."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def _check_bit_num(bit_num: int) -> None:
    if isinstance(bit_num, bool) or not isinstance(bit_num, int) or not 1 <= bit_num <= 8:
        raise ValueError("bit_num must be an integer from 1 to 8.")


def _check_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")


def _relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Sample paths must be relative and must not contain '..'.")
    return path


def iterate_lbit_states(b_input_strm, bit_num: int = 6, chunk_size: int = 65536):
    """Yield non-overlapping states, MSB first, across byte/chunk boundaries."""
    _check_bit_num(bit_num)
    _check_chunk_size(chunk_size)

    bit_buffer = 0
    bit_count = 0
    state_mask = (1 << bit_num) - 1

    while chunk := b_input_strm.read(chunk_size):
        for byte in chunk:
            bit_buffer = (bit_buffer << 8) | byte
            bit_count += 8

            while bit_count >= bit_num:
                remaining_bits = bit_count - bit_num
                yield (bit_buffer >> remaining_bits) & state_mask
                bit_buffer &= (1 << remaining_bits) - 1
                bit_count = remaining_bits

    # Pad missing HIGH-order bits: tail 101 becomes 000101, not 101000.
    if bit_count > 0:
        yield bit_buffer


def stream_to_sbsmi(stream, bit_num: int = 6, chunk_size: int = 65536) -> np.ndarray:
    """Count state transitions, normalize each row, and quantize to uint8."""
    _check_bit_num(bit_num)
    _check_chunk_size(chunk_size)

    n_states = 1 << bit_num
    counts = np.zeros((n_states, n_states), dtype=np.uint64)
    previous_state = None

    for state in iterate_lbit_states(stream, bit_num, chunk_size):
        if previous_state is not None:
            counts[previous_state, state] += 1
        previous_state = state

    if previous_state is None:
        raise ValueError("Cannot construct SBSMI from empty input.")

    probabilities = np.zeros((n_states, n_states), dtype=np.float64)
    for row in range(n_states):
        row_total = counts[row].sum()
        if row_total > 0:
            probabilities[row] = counts[row] / row_total

    return np.floor(probabilities * 255).astype(np.uint8)


def file_to_sbsmi(path: str | Path, bit_num: int = 6, chunk_size: int = 65536) -> np.ndarray:
    """Read one ordinary binary file."""
    with Path(path).open("rb") as stream:
        return stream_to_sbsmi(stream, bit_num, chunk_size)


def zip_entry_to_sbsmi(
    zip_file: ZipFile, member_path: str, bit_num: int = 6, chunk_size: int = 65536
) -> np.ndarray:
    """Read one ZIP member without extracting the archive."""
    with zip_file.open(member_path, "r") as stream:
        return stream_to_sbsmi(stream, bit_num, chunk_size)


def validate_sbsmi(image: np.ndarray, bit_num: int = 6) -> None:
    """Require a square uint8 image; uint8 guarantees values in [0, 255]."""
    _check_bit_num(bit_num)
    if not isinstance(image, np.ndarray):
        raise TypeError("SBSMI must be a NumPy array.")

    size = 1 << bit_num
    if image.shape != (size, size):
        raise ValueError(f"Expected SBSMI shape {(size, size)}, got {image.shape}.")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 SBSMI, got {image.dtype}.")


def save_sbsmi(image: np.ndarray, output_path: str | Path, bit_num: int = 6) -> None:
    """Validate and save one grayscale image."""
    validate_sbsmi(image, bit_num)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)


def _validate_cached_image(path: Path, bit_num: int) -> None:
    size = 1 << bit_num
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (size, size):
            raise ValueError(f"Expected {(size, size)} grayscale SBSMI: {path}")
        image.load()  # Decode pixels to detect damaged image data.


def _output_path_for_sample(output_root: Path, sample: dict) -> Path:
    relative_input = _relative_path(sample["path"])
    sha = str(sample.get("sha") or "").strip()
    family = str(sample.get("family") or "").strip()
    filename = f"{sha or relative_input.stem}.png"

    if family:
        return output_root / family / filename
    if sha:
        return output_root / filename
    return output_root / relative_input.with_suffix(".png")


def generate_sbsmi_dataset(
    source_path: str | Path,
    output_root: str | Path,
    source_mode: str,
    samples: list[dict] | None = None,
    bit_num: int = 6,
    chunk_size: int = 65536,
    overwrite: bool = False,
    max_files: int | None = None,
) -> list[dict]:
    """Generate SBSMIs with one success/skipped/failed record per sample.

    Folder mode: samples=None scans recursively; a list selects specific files.
    ZIP mode: an explicit sample list is required; nothing is extracted.
    Each sample needs a relative 'path'; optional 'sha'/'family' set PNG naming.
    max_files=None allows any count; a positive limit rejects oversized jobs.
    """
    _check_bit_num(bit_num)
    _check_chunk_size(chunk_size)
    source_path = Path(source_path)
    output_root = Path(output_root)
    source_mode = source_mode.lower().strip()

    if source_mode not in {"folder", "zip"}:
        raise ValueError("source_mode must be 'folder' or 'zip'.")
    if max_files is not None and (
        isinstance(max_files, bool) or not isinstance(max_files, int) or max_files <= 0
    ):
        raise ValueError("max_files must be a positive integer or None.")

    if source_mode == "folder":
        if not source_path.is_dir():
            raise ValueError("source_path must be a folder in folder mode.")
        if samples is None:
            samples = [
                {"path": path.relative_to(source_path).as_posix()}
                for path in sorted(source_path.rglob("*")) if path.is_file()
            ]
    else:
        if not source_path.is_file():
            raise ValueError("source_path must be a ZIP file in zip mode.")
        if samples is None:
            raise ValueError("samples is required in zip mode to select the intended cohort.")

    if not samples:
        raise RuntimeError("No input files were selected or found.")
    if max_files is not None and len(samples) > max_files:
        raise RuntimeError(f"Selected {len(samples)} samples, which exceeds max_files={max_files}.")

    print(f"{source_mode.upper()} mode: {len(samples)} selected sample(s).", flush=True)
    records = []
    archive = ZipFile(source_path, "r") if source_mode == "zip" else None

    try:
        for index, sample in enumerate(samples, start=1):
            print(
                f"START {index}/{len(samples)}: {sample['path']}",
                flush=True,
            )
            relative_input = _relative_path(sample["path"])
            output_path = _output_path_for_sample(output_root, sample)
            record = {
                "path": relative_input.as_posix(),
                "sha": str(sample.get("sha") or ""),
                "family": str(sample.get("family") or ""),
                "output_path": str(output_path),
                "error_type": "",
                "error_message": "",
            }

            try:
                if output_path.exists() and not overwrite:
                    _validate_cached_image(output_path, bit_num)
                    record["status"] = "skipped"
                else:
                    if source_mode == "folder":
                        image = file_to_sbsmi(source_path / relative_input, bit_num, chunk_size)
                    else:
                        assert archive is not None
                        image = zip_entry_to_sbsmi(
                            archive, relative_input.as_posix(), bit_num, chunk_size
                        )
                    save_sbsmi(image, output_path, bit_num)
                    record["status"] = "success"
            except Exception as exc:
                record.update(
                    status="failed", error_type=type(exc).__name__, error_message=str(exc)
                )

            records.append(record)
            print(f"DONE {index}: {record['status']}", flush=True)
    finally:
        if archive is not None:
            archive.close()

    return records