import numpy as np


STATE_BITS = 6
NUM_STATES = 2 ** STATE_BITS


def bytes_to_bits_msb_first(data: bytes) -> np.ndarray:
    """
    Convert bytes to a flat bitstream.

    IMPORTANT:
    MSB-first is currently an implementation assumption unless verified
    from the paper's bit-reader definition.
    """

    values = np.frombuffer(data, dtype=np.uint8)

    if len(values) == 0:
        raise ValueError("Cannot construct SBSMI from an empty file")

    return np.unpackbits(
        values,
        bitorder="big",
    )


def bits_to_states(
    bits: np.ndarray,
    state_bits: int = STATE_BITS,
) -> np.ndarray:
    """
    Split a bitstream into non-overlapping l-bit states.

    Current tail policy:
    discard an incomplete final state.

    This matches the likely interpretation of ReadBits(..., l) returning
    EOF when a complete state cannot be read, but must remain documented
    until the source semantics are verified.
    """

    n_complete = len(bits) // state_bits

    if n_complete == 0:
        return np.empty(0, dtype=np.uint8)

    usable = bits[: n_complete * state_bits]

    chunks = usable.reshape(
        n_complete,
        state_bits,
    )

    powers = 2 ** np.arange(
        state_bits - 1,
        -1,
        -1,
        dtype=np.uint64,
    )

    states = chunks @ powers

    return states.astype(np.uint8)

def transition_counts(
    states: np.ndarray,
    num_states: int = NUM_STATES,
) -> np.ndarray:

    counts = np.zeros(
        (num_states, num_states),
        dtype=np.float64,
    )

    if len(states) < 2:
        return counts

    previous = states[:-1]
    current = states[1:]

    np.add.at(
        counts,
        (previous, current),
        1,
    )

    return counts

def row_normalize(
    counts: np.ndarray,
) -> np.ndarray:

    row_sums = counts.sum(
        axis=1,
        keepdims=True,
    )

    probabilities = np.zeros_like(
        counts,
        dtype=np.float64,
    )

    np.divide(
        counts,
        row_sums,
        out=probabilities,
        where=row_sums > 0,
    )

    return probabilities

def sbsmi_float(
    data: bytes,
) -> np.ndarray:

    bits = bytes_to_bits_msb_first(data)

    states = bits_to_states(bits)

    counts = transition_counts(states)

    probabilities = row_normalize(counts)

    return (
        probabilities * 255.0
    ).astype(np.float32)

def quantize_sbsmi(
    image: np.ndarray,
) -> np.ndarray:
    """
    Pending exact source-compatible MatrixConvImage semantics.
    Not sure if the source doesnt tell us if it should be quantized to 8-bit or rounding first and then quantized to 8-bit. The pseudocode is not clear on this.
    """
    raise NotImplementedError(
        "Source quantization convention not yet verified"
    )