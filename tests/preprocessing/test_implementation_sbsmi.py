from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image # install Pillow btw

from src.preprocessing.implementation_sbsmi import (
    file_to_sbsmi,
    generate_sbsmi_dataset,
    iterate_lbit_states,
    save_sbsmi,
    stream_to_sbsmi,
    validate_sbsmi,
    zip_entry_to_sbsmi,
)

# python -m pytest tests/preprocessing/test_implementation_sbsmi.py -v


def test_iterate_lbit_states_known_example():
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

    assert states == [53, 38, 12]


def test_iterate_lbit_states_tail_is_not_left_shifted():
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

    assert states == [21, 5]


def test_iterate_lbit_states_zero_tail_is_kept():
    data = bytes([
        0b00000000,
    ])

    states = list(
        iterate_lbit_states(
            BytesIO(data),
            bit_num=6,
        )
    )

    assert states == [0, 0]


def test_iterate_lbit_states_is_chunk_independent():
    data = bytes(range(100))

    expected = list(
        iterate_lbit_states(
            BytesIO(data),
            bit_num=6,
            chunk_size=1,
        )
    )

    for chunk_size in [2, 7, 64, 65536]:
        actual = list(
            iterate_lbit_states(
                BytesIO(data),
                bit_num=6,
                chunk_size=chunk_size,
            )
        )

        assert actual == expected


def test_stream_to_sbsmi_known_transitions_and_floor():
    data = bytes([
        0x14,
        0xC1,
        0x54,
    ])

    image = stream_to_sbsmi(
        BytesIO(data),
        bit_num=6,
    )

    assert image.shape == (64, 64)
    assert image.dtype == np.uint8

    assert image[5, 12] == 127
    assert image[5, 20] == 127
    assert image[12, 5] == 255

    assert np.count_nonzero(image) == 3


def test_stream_to_sbsmi_rejects_empty_input():
    with pytest.raises(ValueError):
        stream_to_sbsmi(
            BytesIO(b""),
            bit_num=6,
        )


def test_stream_to_sbsmi_is_chunk_independent():
    data = bytes(range(128))

    expected = stream_to_sbsmi(
        BytesIO(data),
        bit_num=6,
        chunk_size=1,
    )

    for chunk_size in [2, 7, 64, 65536]:
        actual = stream_to_sbsmi(
            BytesIO(data),
            bit_num=6,
            chunk_size=chunk_size,
        )

        assert np.array_equal(
            actual,
            expected,
        )


def test_file_to_sbsmi_matches_stream(tmp_path):
    data = bytes([
        0x14,
        0xC1,
        0x54,
    ])

    file_path = tmp_path / "sample.bin"
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


def test_zip_entry_to_sbsmi_matches_stream(tmp_path):
    data = bytes([
        0x14,
        0xC1,
        0x54,
    ])

    zip_path = tmp_path / "sample.zip"

    with ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr(
            "inside/sample.bin",
            data,
        )

    with ZipFile(zip_path, "r") as zip_file:
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


def test_validate_sbsmi_accepts_valid_image():
    image = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    validate_sbsmi(image)


@pytest.mark.parametrize(
    ("image", "error_type"),
    [
        (
            np.zeros(
                (32, 32),
                dtype=np.uint8,
            ),
            ValueError,
        ),
        (
            np.zeros(
                (64, 64),
                dtype=np.float32,
            ),
            ValueError,
        ),
        (
            [[0] * 64 for _ in range(64)],
            TypeError,
        ),
    ],
)
def test_validate_sbsmi_rejects_invalid_images(
    image,
    error_type,
):
    with pytest.raises(error_type):
        validate_sbsmi(image)


def test_save_sbsmi_round_trip(tmp_path):
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


def test_generate_sbsmi_dataset_folder_mode(tmp_path):
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"

    (input_root / "nested").mkdir(
        parents=True
    )

    (input_root / "a.bin").write_bytes(
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


def test_generate_sbsmi_dataset_zip_mode(tmp_path):
    zip_path = tmp_path / "dataset.zip"
    output_root = tmp_path / "processed"

    with ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr(
            "samples/a.bin",
            bytes([
                0x14,
                0xC1,
                0x54,
            ]),
        )

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

    assert (
        output_root
        / "family_a"
        / "aaa.png"
    ).exists()

    assert not (
        output_root
        / "family_b"
        / "bbb.png"
    ).exists()

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

    assert (
        records[1]["error_type"]
        == "KeyError"
    )
