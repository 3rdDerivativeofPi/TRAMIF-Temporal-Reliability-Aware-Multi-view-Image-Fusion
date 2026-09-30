"""
Tests for src.data.datasets.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import io

from src.data.datasets import MalwareImageDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_image_dir():
    """
    Create a temporary directory containing 5 synthetic uint8 PNG images
    (64 × 64) and return its path plus a manifest referencing them.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        manifest_rows = []

        for i in range(5):
            sample_id = f"synthetic_{i:03d}"
            arr = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            img_path = root / f"{sample_id}.png"

            # Write PNG using torchvision.io (CHW uint8 tensor).
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 64, 64)
            io.write_png(tensor, str(img_path))

            manifest_rows.append({
                "sample_id": sample_id,
                "class_id": i % 3,  # 3 classes: 0, 1, 2
            })

        manifest = pd.DataFrame(manifest_rows)
        yield root, manifest


# ---------------------------------------------------------------------------
# Basic construction and indexing
# ---------------------------------------------------------------------------

def test_len(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    assert len(dataset) == 5


def test_getitem_returns_tensor_and_int(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)


def test_getitem_shape_and_dtype(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    x, y = dataset[0]
    assert x.shape == (1, 64, 64)
    assert x.dtype == torch.uint8
    assert 0 <= y < 3


def test_labels_match_manifest(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    sorted_manifest = manifest.sort_values("sample_id").reset_index(drop=True)
    for idx in range(len(dataset)):
        _, y = dataset[idx]
        expected = int(sorted_manifest.loc[idx, "class_id"])
        assert y == expected


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------

def test_dataLoader_batch_shapes(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    x_batch, y_batch = next(iter(loader))
    assert x_batch.shape == (4, 1, 64, 64)
    assert x_batch.dtype == torch.uint8
    assert y_batch.shape == (4,)
    assert y_batch.dtype == torch.long


def test_dataLoader_last_batch(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)  # 5 samples -> last batch = 1

    all_batches = list(loader)
    last_x, last_y = all_batches[-1]
    assert last_x.shape == (1, 1, 64, 64)


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------

def test_image_values_in_byte_range(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")
    for idx in range(len(dataset)):
        x, _ = dataset[idx]
        assert x.min() >= 0
        assert x.max() <= 255


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_deterministic_order(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(manifest, root, suffix=".png")

    items_first = [dataset[idx] for idx in range(len(dataset))]
    items_second = [dataset[idx] for idx in range(len(dataset))]

    for (x1, y1), (x2, y2) in zip(items_first, items_second):
        assert torch.equal(x1, x2)
        assert y1 == y2


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def test_transform_applied(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    dataset = MalwareImageDataset(
        manifest,
        root,
        suffix=".png",
        transform=lambda t: t.float() / 255.0,  # uint8 -> float [0, 1]
    )

    x, _ = dataset[0]
    assert x.dtype == torch.float32
    assert x.min() >= 0.0
    assert x.max() <= 1.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_required_columns():
    # Pass only "class_id" — "sample_id" is missing.
    manifest = pd.DataFrame({"class_id": [0, 1]})
    with pytest.raises(ValueError, match="missing required columns"):
        MalwareImageDataset(manifest, Path("."), suffix=".png")


def test_duplicate_sample_ids():
    manifest = pd.DataFrame({
        "sample_id": ["a", "a"],
        "class_id": [0, 1],
    })
    with pytest.raises(ValueError, match="sample_id values must be unique"):
        MalwareImageDataset(manifest, Path("."), suffix=".png")


def test_file_not_found_error(synthetic_image_dir):
    root, manifest = synthetic_image_dir
    # Replace manifest with a non-existent sample ID.
    # torchvision.io.read_image raises RuntimeError (not FileNotFoundError)
    # when the file is missing.
    bad_manifest = pd.DataFrame({
        "sample_id": ["nonexistent_file_xyz"],
        "class_id": [0],
    })
    dataset = MalwareImageDataset(bad_manifest, root, suffix=".png")
    with pytest.raises(RuntimeError):
        dataset[0]
