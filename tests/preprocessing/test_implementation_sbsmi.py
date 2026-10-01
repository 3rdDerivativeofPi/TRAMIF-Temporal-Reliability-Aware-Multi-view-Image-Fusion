from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from preprocessing.implementation_sbsmi import (
    file_to_sbsmi,
    generate_sbsmi_dataset,
    iterate_lbit_states,
    save_sbsmi,
    stream_to_sbsmi,
    validate_sbsmi,
    zip_entry_to_sbsmi,
)

# command for testing:
# pytest -v --tb=short --disable-warnings tests/preprocessing/test_implementation_sbsmi.py

# ---------------------------------------------------------------------------
# TESTS FOR iterate_lbit_states()
# ---------------------------------------------------------------------------


def test_iterate_lbit_states_known_example():
    """
    Hand-checkable example:

        11010110 01101100
        -> 110101 | 100110 | 1100
        -> 53, 38, 12
    """
    data = bytes([
        0b11010110,
        0b01101100,
    ])

    states = list(
        iterate_lbit_states(
            BytesIO(data),
            bit_num=6,
            chunk_size=1,
        )
    )

    assert states == [
        53,
        38,
        12,
    ]



def test_iterate_lbit_states_tail_is_not_left_shifted():
    """
    Source-compatible tail rule:

        10101101 with bit_num=5
        -> 10101 | 101
        -> 21, 5

    Tail 101 means 00101 = 5, not 10100 = 20.
    """
    data = bytes([
        0b10101101,
    ])

    states = list(
        iterate_lbit_states(
            BytesIO(data),
            bit_num=5,
            chunk_size=1,
        )
    )

    assert states == [
        21,
        5,
    ]



def test_iterate_lbit_states_zero_tail_is_kept():
    """
    A tail can exist even when its integer value is zero.

        00000000 with bit_num=6
        -> 000000 | 00
        -> 0, 0
    """
    data = bytes([
        0b00000000,
    ])

    states = list(
        iterate_lbit_states(
            BytesIO(data),
            bit_num=6,
        )
    )

    assert states == [
        0,
        0,
    ]



def test_iterate_lbit_states_is_chunk_independent():
    """
    Changing chunk_size must NOT change the state sequence.

    This catches bugs where leftover bits are lost at chunk boundaries.
    """
    data = bytes(range(100))

    expected = list(
        iterate_lbit_states(
            BytesIO(data),
            bit_num=6,
            chunk_size=1,
        )
    )

    for chunk_size in [
        2,
        7,
        64,
        65536,
    ]:
        actual = list(
            iterate_lbit_states(
                BytesIO(data),
                bit_num=6,
                chunk_size=chunk_size,
            )
        )

        assert actual == expected


# ---------------------------------------------------------------------------
# TESTS FOR stream_to_sbsmi()
# ---------------------------------------------------------------------------


def test_stream_to_sbsmi_known_transitions_and_floor():
    """
    These bytes produce four 6-bit states:

        5 -> 12 -> 5 -> 20

    Therefore:
        5 -> 12 occurs once
        5 -> 20 occurs once
        12 -> 5 occurs once

    Row 5:
        P(12 | 5) = 1/2 -> floor(127.5) = 127
        P(20 | 5) = 1/2 -> floor(127.5) = 127

    Row 12:
        P(5 | 12) = 1 -> 255
    """
    data = bytes([
        0x14,
        0xC1,
        0x54,
    ])

    image = stream_to_sbsmi(
        BytesIO(data),
        bit_num=6,
    )

    assert image.shape == (
        64,
        64,
    )
    assert image.dtype == np.uint8

    assert image[5, 12] == 127
    assert image[5, 20] == 127
    assert image[12, 5] == 255

    # Only these three pixels should be non-zero.
    assert np.count_nonzero(image) == 3



def test_stream_to_sbsmi_rejects_empty_input():
    """Empty input has no states, so SBSMI generation must fail."""
    with pytest.raises(ValueError):
        stream_to_sbsmi(
            BytesIO(b""),
            bit_num=6,
        )



def test_stream_to_sbsmi_is_chunk_independent():
    """Different chunk sizes must produce exactly the same SBSMI."""
    data = bytes(range(128))

    expected = stream_to_sbsmi(
        BytesIO(data),
        bit_num=6,
        chunk_size=1,
    )

    for chunk_size in [
        2,
        7,
        64,
        65536,
    ]:
        actual = stream_to_sbsmi(
            BytesIO(data),
            bit_num=6,
            chunk_size=chunk_size,
        )

        assert np.array_equal(
            actual,
            expected,
        )


# ---------------------------------------------------------------------------
# TESTS FOR file_to_sbsmi()
# ---------------------------------------------------------------------------


def test_file_to_sbsmi_matches_stream(tmp_path):
    """
    The same bytes read from a normal file and an in-memory stream
    must produce the same SBSMI.
    """
    data = bytes([
        0x14,
        0xC1,
        0x54,
    ])

    file_path = (
        tmp_path
        / "sample.bin"
    )
    file_path.write_bytes(data)

    from_file = file_to_sbsmi(
        file_path
    )

    from_stream = stream_to_sbsmi(
        BytesIO(data)
    )

    assert np.array_equal(
        from_file,
        from_stream,
    )


# ---------------------------------------------------------------------------
# TESTS FOR zip_entry_to_sbsmi()
# ---------------------------------------------------------------------------


def test_zip_entry_to_sbsmi_matches_stream(tmp_path):
    """
    The same bytes read from a ZIP member and an in-memory stream
    must produce the same SBSMI.
    """
    data = bytes([
        0x14,
        0xC1,
        0x54,
    ])

    zip_path = (
        tmp_path
        / "sample.zip"
    )

    with ZipFile(
        zip_path,
        "w",
    ) as zip_file:
        zip_file.writestr(
            "inside/sample.bin",
            data,
        )

    with ZipFile(
        zip_path,
        "r",
    ) as zip_file:
        from_zip = zip_entry_to_sbsmi(
            zip_file,
            "inside/sample.bin",
        )

    from_stream = stream_to_sbsmi(
        BytesIO(data)
    )

    assert np.array_equal(
        from_zip,
        from_stream,
    )


# ---------------------------------------------------------------------------
# TESTS FOR validate_sbsmi()
# ---------------------------------------------------------------------------


def test_validate_sbsmi_accepts_valid_image():
    """A 64 x 64 uint8 matrix is valid for bit_num=6."""
    image = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    # Test passes if no exception is raised.
    validate_sbsmi(image)


@pytest.mark.parametrize(
    ("image", "error_type"),
    [
        (
            # Wrong shape.
            np.zeros(
                (32, 32),
                dtype=np.uint8,
            ),
            ValueError,
        ),
        (
            # Wrong dtype.
            np.zeros(
                (64, 64),
                dtype=np.float32,
            ),
            ValueError,
        ),
        (
            # Not a NumPy array.
            [
                [0] * 64
                for _ in range(64)
            ],
            TypeError,
        ),
    ],
)
def test_validate_sbsmi_rejects_invalid_images(
    image,
    error_type,
):
    """Invalid SBSMI formats must fail clearly."""
    with pytest.raises(error_type):
        validate_sbsmi(image)


# ---------------------------------------------------------------------------
# TESTS FOR save_sbsmi()
# ---------------------------------------------------------------------------


def test_save_sbsmi_round_trip(tmp_path):
    """
    Save an SBSMI as PNG, load it again, and verify that all pixel
    values remain unchanged.
    """
    image = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    image[5, 12] = 127
    image[12, 5] = 255

    output_path = (
        tmp_path
        / "images"
        / "sample.png"
    )

    save_sbsmi(
        image,
        output_path,
    )

    loaded = np.array(
        Image.open(output_path)
    )

    assert output_path.exists()
    assert np.array_equal(
        loaded,
        image,
    )


# ---------------------------------------------------------------------------
# TESTS FOR generate_sbsmi_dataset() - FOLDER MODE
# ---------------------------------------------------------------------------


def test_generate_sbsmi_dataset_folder_mode(tmp_path):
    """
    Folder mode should:
    - recursively find normal files
    - generate one PNG per file
    - preserve relative structure when no SHA/family is supplied
    """
    input_root = (
        tmp_path
        / "raw"
    )

    output_root = (
        tmp_path
        / "processed"
    )

    (
        input_root
        / "nested"
    ).mkdir(
        parents=True
    )

    (
        input_root
        / "a.bin"
    ).write_bytes(
        bytes([
            0x14,
            0xC1,
            0x54,
        ])
    )

    (
        input_root
        / "nested"
        / "b.bin"
    ).write_bytes(
        bytes([
            0x12,
            0x34,
            0x56,
        ])
    )

    records = generate_sbsmi_dataset(
        source_path=input_root,
        output_root=output_root,
        source_mode="folder",
    )

    assert (
        output_root
        / "a.png"
    ).exists()

    assert (
        output_root
        / "nested"
        / "b.png"
    ).exists()

    assert len(records) == 2

    assert all(
        record["status"] == "success"
        for record in records
    )


# ---------------------------------------------------------------------------
# TESTS FOR generate_sbsmi_dataset() - ZIP MODE
# ---------------------------------------------------------------------------


def test_generate_sbsmi_dataset_zip_mode(tmp_path):
    """
    ZIP mode should:
    - process only entries listed in samples
    - not extract/process every ZIP member
    - save successful outputs
    - record failures instead of silently skipping them
    """
    zip_path = (
        tmp_path
        / "dataset.zip"
    )

    output_root = (
        tmp_path
        / "processed"
    )

    with ZipFile(
        zip_path,
        "w",
    ) as zip_file:
        # Selected file: should be processed.
        zip_file.writestr(
            "samples/a.bin",
            bytes([
                0x14,
                0xC1,
                0x54,
            ]),
        )

        # Exists in ZIP but is not selected: should stay untouched.
        zip_file.writestr(
            "samples/not_selected.bin",
            b"not selected",
        )

    samples = [
        {
            "path": "samples/a.bin",
            "sha": "aaa",
            "family": "family_a",
        },
        {
            # Deliberately missing so failure logging is tested.
            "path": "samples/missing.bin",
            "sha": "bbb",
            "family": "family_b",
        },
    ]

    records = generate_sbsmi_dataset(
        source_path=zip_path,
        output_root=output_root,
        source_mode="zip",
        samples=samples,
    )

    # Successful selected file.
    assert (
        output_root
        / "family_a"
        / "aaa.png"
    ).exists()

    # Missing member must not create an image.
    assert not (
        output_root
        / "family_b"
        / "bbb.png"
    ).exists()

    # Unselected ZIP member must not be processed.
    assert not (
        output_root
        / "samples"
        / "not_selected.png"
    ).exists()

    assert [
        record["status"]
        for record in records
    ] == [
        "success",
        "failed",
    ]

    # zipfile reports a missing member as KeyError.
    assert (
        records[1]["error_type"]
        == "KeyError"
    )
