import torch
import pytest

from src.models.malsbslcnet_blocks import (
    BaseBlock,
    channel_shuffle,
)


def test_channel_shuffle_preserves_shape():
    x = torch.randn(
        2,
        32,
        16,
        16,
    )

    y = channel_shuffle(x)

    assert y.shape == x.shape


def test_stride1_preserves_shape():
    block = BaseBlock(
        in_channels=32,
        out_channels=32,
        stride=1,
    )

    x = torch.randn(
        2,
        32,
        32,
        32,
    )

    y = block(x)

    assert y.shape == (
        2,
        32,
        32,
        32,
    )


def test_stride2_halves_spatial_size():
    block = BaseBlock(
        in_channels=32,
        out_channels=64,
        stride=2,
    )

    x = torch.randn(
        2,
        32,
        32,
        32,
    )

    y = block(x)

    assert y.shape == (
        2,
        64,
        16,
        16,
    )


def test_stride1_requires_matching_channels():
    with pytest.raises(ValueError):
        BaseBlock(
            in_channels=32,
            out_channels=64,
            stride=1,
        )


def test_invalid_stride_fails():
    with pytest.raises(ValueError):
        BaseBlock(
            in_channels=32,
            out_channels=32,
            stride=3,
        )