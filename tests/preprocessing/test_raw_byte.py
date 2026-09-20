import numpy as np
import pytest

from src.preprocessing.raw_byte import (
    raw_byte_image,
    raw_byte_layout,
)


def test_exactly_256_bytes_produces_one_row():
    data = bytes(range(256))

    image, mask = raw_byte_layout(data)
    print(image)
    print(mask)

    assert image.shape == (1, 256)
    assert mask.shape == (1, 256)

    assert mask.all()

    assert image[0, 0] == 0.0
    assert image[0, 255] == 1.0

# test exactly 256 bytes produces one row correctly


def test_byte_256_starts_second_row():
    data = bytes(range(256)) + bytes([128])

    image, mask = raw_byte_layout(data)
    print(image)
    print(mask)

    assert image.shape == (2, 256)

    assert mask[0].sum() == 256
    assert mask[1].sum() == 1

    np.testing.assert_allclose(
        image[1, 0],
        128 / 255,
        atol=1e-7,
    )

# test byte 256 starts second row correctly


def test_padding_positions_are_invalid():
    data = bytes([255]) * 257

    image, mask = raw_byte_layout(data)
    print(image)
    print(mask)

    assert mask[1, 0]
    assert not mask[1, 1]
    assert not mask[1, 255]

# test padding positions are invalid


def test_final_image_shape():
    data = bytes(range(256)) * 20

    image = raw_byte_image(data)
    print(image)

    assert image.shape == (64, 64)
    assert image.dtype == np.float32

# test final image shape


def test_final_image_is_in_unit_range():
    data = bytes(range(256)) * 20

    image = raw_byte_image(data)
    print(image)

    assert image.min() >= -1e-7
    assert image.max() <= 1.0 + 1e-7

# test final image is in [0.0 , 1.0]


def test_raw_processing_is_deterministic():
    data = bytes(range(256)) * 10

    first = raw_byte_image(data)
    second = raw_byte_image(data)

    print(first)
    print(second)

    np.testing.assert_array_equal(
        first,
        second,
    )

# test raw processing is deterministic (same output everytime)


def test_empty_file_fails_explicitly():
    with pytest.raises(ValueError):
        raw_byte_image(b"")

# test empty file fails explicitly
