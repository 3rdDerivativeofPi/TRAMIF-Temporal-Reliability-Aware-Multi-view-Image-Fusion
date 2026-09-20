import numpy as np
import pytest

from src.preprocessing.entropy import (
    entropy_image,
    entropy_per_offset,
    entropy_window_starts,
    normalized_shannon_entropy,
)


def test_constant_bytes_have_zero_entropy():
    values = np.zeros(
        256,
        dtype=np.uint8,
    )

    entropy = normalized_shannon_entropy(
        values
    )

    np.testing.assert_allclose(
        entropy,
        0.0,
        atol=1e-12,
    )


def test_all_256_byte_values_have_maximum_entropy():
    values = np.arange(
        256,
        dtype=np.uint8,
    )

    entropy = normalized_shannon_entropy(
        values
    )

    np.testing.assert_allclose(
        entropy,
        1.0,
        atol=1e-12,
    )


def test_short_file_has_single_partial_window():
    starts = entropy_window_starts(
        n_bytes=100,
        window_size=256,
        stride=128,
    )

    assert starts == [0]


def test_trailing_partial_window():
    starts = entropy_window_starts(
        n_bytes=300,
        window_size=256,
        stride=128,
    )

    assert starts == [0, 128]


def test_exact_window_has_no_extra_partial_window():
    starts = entropy_window_starts(
        n_bytes=256,
        window_size=256,
        stride=128,
    )

    assert starts == [0]


def test_overlap_is_averaged():
    # Small synthetic example so we can reason about every position.
    #
    # Window size = 4
    # Stride      = 2
    #
    # Window 0: [0, 0, 0, 0] -> entropy = 0
    # Window 1: [0, 0, 0, 1] -> entropy = h
    #
    data = bytes([
        0, 0, 0, 0,
        0, 1,
    ])

    result = entropy_per_offset(
        data,
        window_size=4,
        stride=2,
    )

    h = normalized_shannon_entropy(
        np.array(
            [0, 0, 0, 1],
            dtype=np.uint8,
        )
    )

    expected = np.array(
        [
            0.0,
            0.0,
            h / 2,
            h / 2,
            h,
            h,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=1e-7,
    )


def test_entropy_image_shape():
    data = bytes(range(256)) * 20

    image = entropy_image(data)

    assert image.shape == (64, 64)
    assert image.dtype == np.float32


def test_entropy_image_is_in_unit_range():
    data = bytes(range(256)) * 20

    image = entropy_image(data)

    assert image.min() >= -1e-7
    assert image.max() <= 1.0 + 1e-7


def test_entropy_processing_is_deterministic():
    data = bytes(range(256)) * 10

    first = entropy_image(data)
    second = entropy_image(data)

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_empty_file_fails_explicitly():
    with pytest.raises(ValueError):
        entropy_image(b"")