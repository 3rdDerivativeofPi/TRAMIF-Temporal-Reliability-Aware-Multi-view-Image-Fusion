"""Synthetic-only tests for the optional 10-sample reference/fast preview command."""
import csv
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from scripts.generate_compare_10 import compare_10, select_training_samples


def fixtures(tmp_path):
    mapping = tmp_path / "eligible.csv"
    pd.DataFrame([{"family": f"family{i}", "class_id": i} for i in range(51)]).to_csv(mapping, index=False)
    rng = np.random.default_rng(42)
    rows = []
    contents = {}
    for number, (split, dt, length) in enumerate([
        ("train", "2019-08-15", 1),
        ("train", "2019-09-15", 257),
        ("train", "2019-11-15", 512),
        ("validation", "2020-02-15", 16385),
        ("future_test", "2020-04-15", 8192),
        ("train", "2019-10-12", 12800),
    ], start=1):
        sha = f"{number:064x}"
        contents[sha] = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
        rows.append({"sha": sha, "family": "family0", "timestamp": dt, "split": split, "month": dt[:7], "class_id": 0})
    meta = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(meta, index=False)
    folder = tmp_path / "altered"; folder.mkdir()
    archive = tmp_path / "binary.zip"
    with ZipFile(archive, "w") as zf:
        for sha, data in contents.items():
            (folder / f"{sha}.exe").write_bytes(data)
            zf.writestr(f"altered/{sha}.exe", data)
    return mapping, meta, archive, folder


def test_compare_zip_images_and_contact_sheets(tmp_path):
    mapping, meta, archive, folder = fixtures(tmp_path)
    out = tmp_path / "comparison"
    result = compare_10(archive, meta, mapping, False, out, limit=3, chunk_size=7)
    assert len(result) == 3
    assert all(x["status"] == "PASS" and x["split"] == "train" for x in result)
    assert all(int(x["sbsmi_differing_pixels"]) == 0 for x in result)
    assert all(float(x["entropy_max_difference"]) <= 1e-6 for x in result)
    assert (out / "comparison.csv").is_file()
    for modality in ("entropy", "sbsmi"):
        with Image.open(out / f"{modality}_comparison_sheet.png") as im:
            assert im.size[0] > 400 and im.size[1] > 500
    for r in result:
        sha = r["sha"]
        for variant in ("reference_v2", "fast_v2"):
            assert (out / variant / "sbsmi" / f"{sha}.png").is_file()
            image = np.load(out / variant / "entropy" / f"{sha}.npy", allow_pickle=False)
            assert image.shape == (64, 64) and image.dtype == np.float32
            assert (out / variant / "entropy_preview" / f"{sha}.png").is_file()


def test_compare_extracted_folder(tmp_path):
    mapping, meta, archive, folder = fixtures(tmp_path)
    result = compare_10(folder, meta, mapping, False, tmp_path / "out", limit=1)
    assert len(result) == 1 and result[0]["status"] == "PASS"


def test_manifest_timestamp_guard(tmp_path):
    mapping, meta, archive, folder = fixtures(tmp_path)
    manifest = tmp_path / "manifest.csv"
    data = pd.read_csv(meta, dtype="string")
    data.loc[0, "split"] = "validation"
    data.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="disagree"):
        select_training_samples(manifest, mapping, True)


def test_requires_enough_train_samples(tmp_path):
    mapping, meta, archive, folder = fixtures(tmp_path)
    with pytest.raises(ValueError, match="Found only"):
        compare_10(archive, meta, mapping, False, tmp_path / "out", limit=10)
