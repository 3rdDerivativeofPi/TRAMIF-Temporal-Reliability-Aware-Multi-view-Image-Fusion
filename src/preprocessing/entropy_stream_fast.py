"""Batched, constant-memory entropy-offset generator for TRAMIF V2.

Matches the historical V2 entropy definition: 256-byte windows spaced by
128 bytes, one final (possibly partial) window, H/8 normalization, and the
mean across windows covering each *real* byte offset.

Optimization: each input block of 128 bytes is histogrammed once. A 256-byte
window histogram is the sum of two adjacent block histograms. We evaluate
many windows together with NumPy, then emit all finalized byte offsets in
order. No binary bytes are executed or altered.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing.entropy_stream import iter_local_entropy

_BLOCK = 128
_BINS = 256
_INDEX = np.arange(_BINS + 1, dtype=np.float64)
_C_LOG_C = np.zeros(_BINS + 1, dtype=np.float64)
_C_LOG_C[1:] = _INDEX[1:] * np.log2(_INDEX[1:])


def _histograms_for_blocks(data: bytes) -> np.ndarray:
    """Produce 256-bin counts per 128-byte block, in one NumPy operation."""
    if len(data) % _BLOCK:
        raise ValueError("Internal error: expected complete 128-byte blocks")
    n = len(data) // _BLOCK
    if n == 0:
        return np.empty((0, _BINS), dtype=np.uint16)
    source = np.frombuffer(data, dtype=np.uint8).reshape(n, _BLOCK)
    keys = source.astype(np.int32) + (_BINS * np.arange(n, dtype=np.int32)[:, None])
    counts = np.bincount(keys.ravel(), minlength=n * _BINS).reshape(n, _BINS)
    return counts.astype(np.uint16)


def _normalized_entropy(counts: np.ndarray) -> np.ndarray:
    """Vectorized H/8 from one or more integer count histograms."""
    lengths = counts.sum(axis=1, dtype=np.int64).astype(np.float64)
    if np.any(lengths <= 0):
        raise ValueError("Entropy cannot be computed from an empty block")
    bits = np.log2(lengths) - _C_LOG_C[counts].sum(axis=1) / lengths
    return np.clip(bits / 8.0, 0.0, 1.0)


def _process_histograms(
    histograms: np.ndarray,
    previous_histogram: np.ndarray | None,
    previous_entropy: float | None,
) -> tuple[np.ndarray, float | None, np.ndarray]:
    """Return (last histogram, last entropy, finalized 128-offset values)."""
    if previous_histogram is None:
        left = histograms[:-1]
        right = histograms[1:]
    else:
        left = np.concatenate((previous_histogram[None, :], histograms[:-1]))
        right = histograms

    if len(right) == 0:
        return histograms[-1].copy(), previous_entropy, np.empty(0, dtype=np.float64)

    windows = left + right  # max count = 256, safe in uint16
    entropies = _normalized_entropy(windows)

    if previous_entropy is None:
        finalized = np.empty(len(entropies), dtype=np.float64)
        finalized[0] = entropies[0]
        if len(entropies) > 1:
            finalized[1:] = (entropies[:-1] + entropies[1:]) / 2.0
    else:
        finalized = np.empty(len(entropies), dtype=np.float64)
        finalized[0] = (previous_entropy + entropies[0]) / 2.0
        if len(entropies) > 1:
            finalized[1:] = (entropies[:-1] + entropies[1:]) / 2.0

    return histograms[-1].copy(), float(entropies[-1]), finalized


def iter_local_entropy_fast(
    stream,
    file_size: int,
    window_size: int = 256,
    stride: int = 128,
    chunk_size: int = 65536,
):
    """Yield float64 arrays, with exactly one normalized entropy per input byte.

    Memory use is O(chunk_size + window_size) for the standard settings.
    For nonstandard window/stride settings, defer to the unoptimized V2
    reference implementation instead of silently changing semantics.
    """
    for name, value in (("file_size", file_size), ("window_size", window_size),
                        ("stride", stride), ("chunk_size", chunk_size)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if stride > window_size:
        raise ValueError("stride must not exceed window_size")

    if (window_size, stride) != (256, 128):
        yield from iter_local_entropy(stream, file_size=file_size,
                                      window_size=window_size, stride=stride,
                                      chunk_size=chunk_size)
        return

    previous_histogram = None
    previous_entropy = None
    remaining = file_size
    carry = b""
    previous_block_length = 0
    values_emitted = 0

    while remaining > 0:
        requested = min(chunk_size, remaining)
        chunk = stream.read(requested)
        if not chunk:
            raise ValueError("Input ended before the declared file size")
        if len(chunk) > requested:
            raise ValueError("Input returned more bytes than requested")
        remaining -= len(chunk)

        data = carry + chunk
        full_size = len(data) // _BLOCK * _BLOCK
        carry = data[full_size:]
        if not full_size:
            continue
        histograms = _histograms_for_blocks(data[:full_size])
        previous_histogram, previous_entropy, finalized = _process_histograms(
            histograms, previous_histogram, previous_entropy,
        )
        previous_block_length = _BLOCK
        if len(finalized):
            batch = np.repeat(finalized, _BLOCK)
            values_emitted += len(batch)
            yield batch

    # Strict input-size check, also important for source fingerprint reports.
    if stream.read(1):
        raise ValueError("Input exceeds the declared file size")

    if carry:
        tail = np.bincount(np.frombuffer(carry, dtype=np.uint8), minlength=_BINS)
        tail = tail.astype(np.uint16)[None, :]
        previous_histogram, previous_entropy, finalized = _process_histograms(
            tail, previous_histogram, previous_entropy,
        )
        if len(finalized):
            batch = np.repeat(finalized, _BLOCK)
            values_emitted += len(batch)
            yield batch
        previous_block_length = len(carry)

    if previous_histogram is None:
        # file_size > 0 ensures there is a block by now.
        raise RuntimeError("No input blocks were processed")

    if previous_entropy is None:
        # A single block, shorter than or equal to 128 bytes.
        previous_entropy = float(_normalized_entropy(previous_histogram[None, :])[0])

    final_values = np.full(previous_block_length, previous_entropy, dtype=np.float64)
    values_emitted += len(final_values)
    yield final_values
    if values_emitted != file_size:
        raise RuntimeError(f"Entropy offset count mismatch: {values_emitted} != {file_size}")
