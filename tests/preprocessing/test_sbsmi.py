import numpy as np

from src.preprocessing.sbsmi import (
    bits_to_states,
    quantize_sbsmi,
    row_normalize,
    transition_counts,
)


def test_known_state_conversion():

    bits = np.array(
        [
            0, 0, 0, 0, 0, 0,  # 0
            0, 0, 0, 0, 0, 1,  # 1
            1, 1, 1, 1, 1, 1,  # 63
        ],
        dtype=np.uint8,
    )

    states = bits_to_states(bits)

    np.testing.assert_array_equal(
        states,
        np.array([0, 1, 63]),
    )

def test_known_transition_counts():

    states = np.array(
        [0, 1, 0, 1],
        dtype=np.uint8,
    )

    counts = transition_counts(states)

    assert counts[0, 1] == 2
    assert counts[1, 0] == 1

    assert counts.sum() == 3

def test_row_normalization():

    states = np.array(
        [0, 1, 0, 1],
        dtype=np.uint8,
    )

    counts = transition_counts(states)

    probabilities = row_normalize(counts)

    assert probabilities[0, 1] == 1.0
    assert probabilities[1, 0] == 1.0

    assert probabilities[2].sum() == 0.0

def test_row_normalization_with_multiple_targets():

    states = np.array(
        [
            0, 1,
            0, 1,
            0, 2,
        ],
        dtype=np.uint8,
    )

    counts = transition_counts(states)

    probabilities = row_normalize(counts)

    np.testing.assert_allclose(
        probabilities[0, 1],
        2 / 3,
    )

    np.testing.assert_allclose(
        probabilities[0, 2],
        1 / 3,
    )

def test_incomplete_state_zero_fills_missing_high_order_bits():

    bits = np.array(
        [
            0, 0, 0, 0, 0, 1,  # complete state = 1
            1, 1, 1,             # incomplete tail -> 000111 = 7
        ],
        dtype=np.uint8,
    )

    states = bits_to_states(bits)

    np.testing.assert_array_equal(
        states,
        np.array([1, 7]),
    )

# Current source interpretation:
# incomplete final l-bit block is discarded.
#



def test_quantize_sbsmi_uses_floor():

    image = np.array(
        [
            [0.0, 127.5, 170.0, 255.0],
        ],
        dtype=np.float32,
    )

    quantized = quantize_sbsmi(image)

    np.testing.assert_array_equal(
        quantized,
        np.array(
            [
                [0, 127, 170, 255],
            ],
            dtype=np.uint8,
        ),
    )

    assert quantized.dtype == np.uint8
    
