"""Experimental vectorized 6-bit SBSMI generator (defensive preprocessing).

Exact integer-state semantics match implementation_sbsmi.stream_to_sbsmi for
6-bit MSB-first nonoverlapping states, high-order-zero tail, and floor(255 P).
Only use after checking identity on your locked BODMAS smoke-test cohort.
"""
from __future__ import annotations

import numpy as np

from src.preprocessing.implementation_sbsmi import stream_to_sbsmi as original_stream_to_sbsmi


def stream_to_sbsmi_fast(stream, bit_num: int = 6, chunk_size: int = 65536) -> np.ndarray:
    """Read binary input in bounded chunks and return 64x64 uint8 SBSMI.

    Uses the periodic 24-bit/3-byte alignment of nonoverlapping 6-bit states.
    Other state widths fall back to the existing implementation.
    """
    if isinstance(bit_num, bool) or not isinstance(bit_num, int) or not 1 <= bit_num <= 8:
        raise ValueError("bit_num must be an integer from 1 to 8.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if bit_num != 6:
        return original_stream_to_sbsmi(stream, bit_num=bit_num, chunk_size=chunk_size)

    counts = np.zeros((64, 64), dtype=np.uint64)
    carry = b""
    previous_state: int | None = None
    encountered_bytes = False

    while chunk := stream.read(chunk_size):
        encountered_bytes = True
        data = carry + chunk
        complete_size = (len(data) // 3) * 3
        carry = data[complete_size:]
        if not complete_size:
            continue

        # Exactly 3 input bytes (24 bits) become exactly four 6-bit states.
        triples = np.frombuffer(data[:complete_size], dtype=np.uint8).reshape(-1, 3)
        b0 = triples[:, 0].astype(np.uint16)
        b1 = triples[:, 1].astype(np.uint16)
        b2 = triples[:, 2].astype(np.uint16)
        states = np.empty(triples.shape[0] * 4, dtype=np.uint8)
        states[0::4] = b0 >> 2
        states[1::4] = ((b0 & 3) << 4) | (b1 >> 4)
        states[2::4] = ((b1 & 15) << 2) | (b2 >> 6)
        states[3::4] = b2 & 63

        # Connect the first state with the previous chunk's final state.
        if previous_state is not None:
            counts[previous_state, int(states[0])] += 1
        transitions = states[:-1].astype(np.int32) * 64 + states[1:].astype(np.int32)
        counts += np.bincount(transitions, minlength=4096).reshape(64, 64).astype(np.uint64)
        previous_state = int(states[-1])

    if not encountered_bytes:
        raise ValueError("Cannot construct SBSMI from empty input.")

    # Original code retains a partial 6-bit state without shifting its bits
    # to the high end: missing high-order bits are zero.
    if carry:
        b0 = carry[0]
        if len(carry) == 1:
            tail_states = [b0 >> 2, b0 & 3]
        else:
            b1 = carry[1]
            tail_states = [b0 >> 2, ((b0 & 3) << 4) | (b1 >> 4), b1 & 15]
        for state in tail_states:
            if previous_state is not None:
                counts[previous_state, state] += 1
            previous_state = state

    probabilities = np.zeros((64, 64), dtype=np.float64)
    for row in range(64):
        total = counts[row].sum()
        if total > 0:
            probabilities[row] = counts[row] / total
    return np.floor(probabilities * 255).astype(np.uint8)
