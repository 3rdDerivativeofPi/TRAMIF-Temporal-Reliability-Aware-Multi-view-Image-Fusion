import numpy as np

from .common import mask_aware_area_resize


WINDOW_SIZE = 256
STRIDE = 128
RAW_WIDTH = 256
OUTPUT_SIZE = (64, 64)


# entropy window generation

def normalized_shannon_entropy(
    values: np.ndarray,
) -> float:
    """
    Compute Shannon entropy of byte values and normalize by 8 bits.
    """

    values = np.asarray(
        values,
        dtype=np.uint8,
    )

    if len(values) == 0:
        raise ValueError(
            "Entropy requires at least one byte"
        )

    counts = np.bincount(
        values,
        minlength=256,
    )

    counts = counts[counts > 0]

    probabilities = (
        counts.astype(np.float64)
        / counts.sum()
    )

    entropy_bits = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy_bits / 8.0)


# generate entropy-window start positions

def entropy_window_starts(
    n_bytes: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[int]:
    """
    Generate full windows followed, where needed, by one trailing
    partial window.

    Files shorter than one complete window use a single window
    beginning at offset 0.
    """

    if n_bytes <= 0:
        raise ValueError(
            "n_bytes must be positive"
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive"
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive"
        )

    if n_bytes <= window_size:
        return [0]

    starts = list(
        range(
            0,
            n_bytes - window_size + 1,
            stride,
        )
    )

    last_start = starts[-1]
    last_end = last_start + window_size

    if last_end < n_bytes:
        next_start = last_start + stride

        if next_start < n_bytes:
            starts.append(next_start)

    return starts

# map entropy back to individual offsets

def entropy_per_offset(
    data: bytes,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> np.ndarray:
    """
    Assign window entropy values to the byte offsets covered by each
    window. Offsets covered by multiple windows receive the mean of
    those entropy estimates.
    """

    values = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    n_bytes = len(values)

    if n_bytes == 0:
        raise ValueError(
            "Cannot construct entropy from an empty file"
        )

    entropy_sum = np.zeros(
        n_bytes,
        dtype=np.float64,
    )

    entropy_count = np.zeros(
        n_bytes,
        dtype=np.int32,
    )

    starts = entropy_window_starts(
        n_bytes=n_bytes,
        window_size=window_size,
        stride=stride,
    )

    for start in starts:
        end = min(
            start + window_size,
            n_bytes,
        )

        window = values[start:end]

        entropy = normalized_shannon_entropy(
            window
        )

        entropy_sum[start:end] += entropy
        entropy_count[start:end] += 1

    if np.any(entropy_count == 0):
        uncovered = np.flatnonzero(
            entropy_count == 0
        )

        raise RuntimeError(
            "Entropy windowing left byte offsets uncovered: "
            f"{uncovered[:10]}"
        )

    result = (
        entropy_sum
        / entropy_count
    )

    return result.astype(np.float32)


# arrange entropy values using the same byte layout as the raw-byte representation

def entropy_layout(
    data: bytes,
    width: int = RAW_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map per-byte entropy estimates onto the same width-256 layout
    used by the raw-byte representation.
    """

    if width <= 0:
        raise ValueError("width must be positive")

    entropy = entropy_per_offset(data)

    n_values = len(entropy)

    n_rows = (
        n_values + width - 1
    ) // width

    image = np.zeros(
        (n_rows, width),
        dtype=np.float32,
    )

    mask = np.zeros(
        (n_rows, width),
        dtype=bool,
    )

    image.flat[:n_values] = entropy
    mask.flat[:n_values] = True

    return image, mask

def entropy_image(
    data: bytes,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> np.ndarray:
    """
    Construct the final fixed-size entropy image.
    """

    image, mask = entropy_layout(data)

    return mask_aware_area_resize(
        image=image,
        mask=mask,
        output_size=output_size,
    )