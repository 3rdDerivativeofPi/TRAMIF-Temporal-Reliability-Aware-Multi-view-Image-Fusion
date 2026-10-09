"""Synthetic regression tests: V2 reference vs fast transformations.

Run in the TRAMIF repo after installing the distributed V2 modules:
  python -m pytest -q tests/test_fast_views.py
"""
import io
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.implementation_sbsmi import stream_to_sbsmi
from src.preprocessing.sbsmi_vectorized import stream_to_sbsmi_fast
from src.preprocessing.entropy_stream import iter_local_entropy
from src.preprocessing.entropy_stream_fast import iter_local_entropy_fast
from src.preprocessing.entropy_image_stream import stream_to_entropy_image
from src.preprocessing.entropy_image_stream_fast import stream_to_entropy_image_fast
from scripts.compare_fast_views import compare


LENGTHS = [
    1, 2, 3, 7, 8, 31, 63, 64, 65, 127, 128, 129, 130,
    255, 256, 257, 258, 383, 384, 385, 386, 511, 512, 513,
    1023, 1024, 1025, 4095, 4096, 4097, 65535, 65536, 65537,
]


def binary_data(length, kind="random"):
    if kind == "zero":
        return b"\0" * length
    if kind == "repeated":
        return (bytes(range(256)) * ((length + 255) // 256))[:length]
    rng = np.random.default_rng(length)
    return rng.integers(0, 256, length, dtype=np.uint8).tobytes()


@pytest.mark.parametrize("length", LENGTHS)
def test_entropy_image_matches_v2_reference(length):
    data = binary_data(length)
    baseline = stream_to_entropy_image(io.BytesIO(data), file_size=length)
    optimized = stream_to_entropy_image_fast(io.BytesIO(data), file_size=length)
    assert baseline.shape == optimized.shape == (64, 64)
    assert np.isfinite(optimized).all()
    assert (optimized >= 0).all() and (optimized <= 1).all()
    np.testing.assert_allclose(optimized, baseline, atol=1e-12, rtol=0)


@pytest.mark.parametrize("length", LENGTHS)
def test_sbsmi_uint8_pixels_identical(length):
    data = binary_data(length)
    baseline = stream_to_sbsmi(io.BytesIO(data), bit_num=6)
    optimized = stream_to_sbsmi_fast(io.BytesIO(data), bit_num=6)
    assert optimized.dtype == np.uint8
    assert optimized.shape == (64, 64)
    np.testing.assert_array_equal(optimized, baseline)


@pytest.mark.parametrize("length", [1, 128, 129, 256, 257, 385, 512, 4097, 65537])
@pytest.mark.parametrize("chunk_size", [1, 3, 31, 127, 128, 129, 65536])
def test_entropy_chunk_independence(length, chunk_size):
    data = binary_data(length)
    ref = stream_to_entropy_image(io.BytesIO(data), file_size=length)
    actual = stream_to_entropy_image_fast(io.BytesIO(data), file_size=length, chunk_size=chunk_size)
    np.testing.assert_allclose(actual, ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("length", [1, 128, 129, 256, 257, 385, 512, 4097, 65537])
@pytest.mark.parametrize("chunk_size", [1, 3, 31, 127, 128, 129, 65536])
def test_sbsmi_chunk_independence(length, chunk_size):
    data = binary_data(length)
    ref = stream_to_sbsmi(io.BytesIO(data), bit_num=6)
    actual = stream_to_sbsmi_fast(io.BytesIO(data), bit_num=6, chunk_size=chunk_size)
    np.testing.assert_array_equal(actual, ref)


@pytest.mark.parametrize("kind", ["zero", "repeated", "random"])
@pytest.mark.parametrize("length", [16, 257, 1025, 8193])
def test_entropy_patterns_and_per_offset_values(kind, length):
    data = binary_data(length, kind)
    old = np.concatenate(list(iter_local_entropy(io.BytesIO(data), length)))
    new = np.concatenate(list(iter_local_entropy_fast(io.BytesIO(data), length)))
    assert len(new) == len(data)
    np.testing.assert_allclose(new, old, atol=1e-12, rtol=0)


@pytest.mark.parametrize("kind", ["zero", "repeated", "random"])
@pytest.mark.parametrize("length", [16, 257, 1025, 8193])
def test_sbsmi_patterns(kind, length):
    data = binary_data(length, kind)
    np.testing.assert_array_equal(
        stream_to_sbsmi_fast(io.BytesIO(data)),
        stream_to_sbsmi(io.BytesIO(data)),
    )


def test_nondefault_entropy_window_delegates_to_reference():
    data = binary_data(21)
    ref = np.concatenate(list(iter_local_entropy(io.BytesIO(data), 21, window_size=5, stride=2)))
    fast = np.concatenate(list(iter_local_entropy_fast(io.BytesIO(data), 21, window_size=5, stride=2)))
    np.testing.assert_array_equal(ref, fast)


def test_other_sbsmi_widths_delegate_to_reference():
    data = binary_data(257)
    for bits in [4, 5, 7, 8]:
        np.testing.assert_array_equal(
            stream_to_sbsmi_fast(io.BytesIO(data), bit_num=bits),
            stream_to_sbsmi(io.BytesIO(data), bit_num=bits),
        )


@pytest.mark.parametrize("method", ["entropy", "sbsmi"])
def test_empty_input_fails(method):
    if method == "entropy":
        with pytest.raises(ValueError):
            stream_to_entropy_image_fast(io.BytesIO(b""), file_size=0)
    else:
        with pytest.raises(ValueError):
            stream_to_sbsmi_fast(io.BytesIO(b""))


def test_entropy_rejects_short_and_long_streams():
    with pytest.raises(ValueError):
        stream_to_entropy_image_fast(io.BytesIO(b"\x05" * 255), file_size=256)
    with pytest.raises(ValueError):
        stream_to_entropy_image_fast(io.BytesIO(b"\x05" * 257), file_size=256)


def test_entropy_invalid_args():
    for chunk_size in [0, -1]:
        with pytest.raises(ValueError):
            stream_to_entropy_image_fast(io.BytesIO(b"1"), file_size=1, chunk_size=chunk_size)


def test_png_npy_roundtrip_does_not_change_pixels(tmp_path):
    data = binary_data(1025)
    sbs = stream_to_sbsmi_fast(io.BytesIO(data))
    path_png = tmp_path / "sbsmi.png"
    Image.fromarray(sbs, mode="L").save(path_png)
    with Image.open(path_png) as img:
        restored_png = np.asarray(img)
    np.testing.assert_array_equal(sbs, restored_png)

    ent = stream_to_entropy_image_fast(io.BytesIO(data), file_size=len(data)).astype(np.float32)
    path_npy = tmp_path / "entropy.npy"
    np.save(path_npy, ent, allow_pickle=False)
    restored_npy = np.load(path_npy, allow_pickle=False)
    np.testing.assert_array_equal(ent, restored_npy)


def test_compare_command_on_file_and_zip(tmp_path):
    data = binary_data(2049)
    path = tmp_path / "test.bin"
    path.write_bytes(data)
    direct = compare(file_path=path)
    assert direct["overall_pass"]
    assert direct["sbsmi"]["different_pixels"] == 0
    archive = tmp_path / "binary.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("altered/test.bin", data)
    zipped = compare(zip_path=archive, member="altered/test.bin")
    assert zipped["overall_pass"]
    assert zipped["disarmed_content_sha256"] == direct["disarmed_content_sha256"]
