from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def _check_bit_num(bit_num: int) -> None:
    if (
        isinstance(bit_num, bool)
        or not isinstance(bit_num, int)
        or not 1 <= bit_num <= 8
    ):
        raise ValueError("bit_num must be an integer from 1 to 8.")


def _check_chunk_size(chunk_size: int) -> None:
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")


def _relative_path(value: str | Path) -> Path:
    path = Path(value)

    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Sample paths must be relative and must not contain '..'.")

    return path


def iterate_lbit_states(
    b_input_strm,
    bit_num: int = 6,
    chunk_size: int = 65536,
):
    """Yield non-overlapping bit_num-bit states from a binary stream."""
    _check_bit_num(bit_num)
    _check_chunk_size(chunk_size)

    bit_buffer = 0
    bit_count = 0
    state_mask = (1 << bit_num) - 1

    while True:
        chunk = b_input_strm.read(chunk_size)

        if not chunk:
            break

        for byte in chunk:
            bit_buffer = (bit_buffer << 8) | byte
            bit_count += 8

            while bit_count >= bit_num:
                remaining_bits = bit_count - bit_num
                state = (bit_buffer >> remaining_bits) & state_mask

                yield state

                if remaining_bits == 0:
                    bit_buffer = 0
                else:
                    remaining_mask = (1 << remaining_bits) - 1
                    bit_buffer &= remaining_mask

                bit_count = remaining_bits

    if bit_count > 0:
        yield bit_buffer


def stream_to_sbsmi(
    stream,
    bit_num: int = 6,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Convert a binary stream into an SBSMI uint8 matrix."""
    _check_bit_num(bit_num)
    _check_chunk_size(chunk_size)

    n_states = 1 << bit_num

    counts = np.zeros(
        (n_states, n_states),
        dtype=np.uint64,
    )

    previous_state = None
    number_of_states = 0

    for current_state in iterate_lbit_states(
        stream,
        bit_num=bit_num,
        chunk_size=chunk_size,
    ):
        number_of_states += 1

        if previous_state is not None:
            counts[previous_state, current_state] += 1

        previous_state = current_state

    if number_of_states == 0:
        raise ValueError("Cannot construct SBSMI from empty input.")

    probabilities = np.zeros(
        (n_states, n_states),
        dtype=np.float64,
    )

    for row in range(n_states):
        row_total = counts[row].sum()

        if row_total > 0:
            probabilities[row] = counts[row] / row_total

    return np.floor(
        probabilities * 255
    ).astype(np.uint8)


def file_to_sbsmi(
    path: str | Path,
    bit_num: int = 6,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Generate SBSMI from one normal file on disk."""
    with Path(path).open("rb") as stream:
        return stream_to_sbsmi(
            stream,
            bit_num=bit_num,
            chunk_size=chunk_size,
        )


def zip_entry_to_sbsmi(
    zip_file: ZipFile,
    member_path: str,
    bit_num: int = 6,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Generate SBSMI from one file stored inside an already-open ZIP."""
    with zip_file.open(member_path, "r") as stream:
        return stream_to_sbsmi(
            stream,
            bit_num=bit_num,
            chunk_size=chunk_size,
        )


def validate_sbsmi(
    image: np.ndarray,
    bit_num: int = 6,
) -> None:
    """Check the expected SBSMI shape, dtype, and value range."""
    _check_bit_num(bit_num)

    if not isinstance(image, np.ndarray):
        raise TypeError("SBSMI must be a NumPy array.")

    expected_size = 1 << bit_num
    expected_shape = (expected_size, expected_size)

    if image.shape != expected_shape:
        raise ValueError(
            f"Invalid SBSMI shape: expected {expected_shape}, got {image.shape}."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"Invalid SBSMI dtype: expected uint8, got {image.dtype}."
        )

    if image.min() < 0 or image.max() > 255:
        raise ValueError("SBSMI pixel values must be in [0, 255].")


def save_sbsmi(
    image: np.ndarray,
    output_path: str | Path,
    bit_num: int = 6,
) -> None:
    """Validate and save one SBSMI as a grayscale PNG."""
    validate_sbsmi(image, bit_num=bit_num)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(image, mode="L").save(output_path)


def _output_path_for_sample(
    output_root: Path,
    sample: dict,
) -> Path:
    relative_input = _relative_path(sample["path"])

    sha = str(sample.get("sha") or "").strip()
    family = str(sample.get("family") or "").strip()

    if sha:
        filename = f"{sha}.png"
    else:
        filename = f"{relative_input.stem}.png"

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
) -> list[dict]:
    """
    Generate many SBSMIs from either a folder or a ZIP archive.

    Folder mode:
        If samples is None, process every file under source_path recursively.

    ZIP mode:
        samples is required.
        Each sample needs a relative member path in sample["path"].
    """
    _check_bit_num(bit_num)
    _check_chunk_size(chunk_size)

    source_path = Path(source_path)
    output_root = Path(output_root)
    source_mode = source_mode.lower().strip()

    if source_mode not in {"folder", "zip"}:
        raise ValueError("source_mode must be 'folder' or 'zip'.")

    if source_mode == "folder":
        if not source_path.is_dir():
            raise ValueError("source_path must be a folder in folder mode.")

        if samples is None:
            samples = [
                {"path": str(path.relative_to(source_path))}
                for path in sorted(source_path.rglob("*"))
                if path.is_file()
            ]

    else:
        if not source_path.is_file():
            raise ValueError("source_path must be a ZIP file in zip mode.")

        if samples is None:
            raise ValueError(
                "samples is required in zip mode so the intended cohort "
                "is selected explicitly."
            )

    records: list[dict] = []

    zip_file = (
        ZipFile(source_path, "r")
        if source_mode == "zip"
        else None
    )

    try:
        for sample in samples:
            relative_input = _relative_path(sample["path"])
            output_path = _output_path_for_sample(
                output_root,
                sample,
            )

            base_record = {
                "path": str(relative_input),
                "sha": str(sample.get("sha") or ""),
                "family": str(sample.get("family") or ""),
                "output_path": str(output_path),
            }

            if output_path.exists() and not overwrite:
                records.append({
                    **base_record,
                    "status": "skipped",
                    "error_type": "",
                    "error_message": "",
                })
                continue

            try:
                if source_mode == "folder":
                    image = file_to_sbsmi(
                        source_path / relative_input,
                        bit_num=bit_num,
                        chunk_size=chunk_size,
                    )
                else:
                    assert zip_file is not None
                    
                    image = zip_entry_to_sbsmi(
                        zip_file,
                        relative_input.as_posix(),
                        bit_num=bit_num,
                        chunk_size=chunk_size,
                    )

                validate_sbsmi(
                    image,
                    bit_num=bit_num,
                )

                save_sbsmi(
                    image,
                    output_path,
                    bit_num=bit_num,
                )

                records.append({
                    **base_record,
                    "status": "success",
                    "error_type": "",
                    "error_message": "",
                })

            except Exception as exc:
                records.append({
                    **base_record,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

    finally:
        if zip_file is not None:
            zip_file.close()

    return records
