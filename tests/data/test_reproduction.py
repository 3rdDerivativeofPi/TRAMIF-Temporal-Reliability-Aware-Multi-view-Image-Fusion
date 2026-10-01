import json
import subprocess
import sys

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.cross_validation import build_stratified_fold_manifest
from src.data.reproduction import ReproductionDataset, audit_reproduction_folds
from src.data.reproduction_datamodule import ReproductionDataModule


@pytest.fixture
def samples(tmp_path):
    rows = []
    # Uneven totals exercise allowable +/-1 stratification differences.
    for class_id, count in [(0, 7), (1, 8)]:
        for index in range(count):
            name = f"{class_id}_{index}.png"
            array = np.full((64, 64), index + class_id * 20, dtype=np.uint8)
            array[0, 0], array[-1, -1] = 0, 255
            Image.fromarray(array).save(tmp_path / name)
            rows.append(dict(sha=f"c{class_id}s{index}", class_id=class_id,
                             family=f"family{class_id}", image_path=name))
    frame = build_stratified_fold_manifest(pd.DataFrame(rows), n_splits=3)
    return frame, tmp_path


def test_tensor_contract_and_exact_pixel_scaling(samples):
    frame, root = samples
    dataset = ReproductionDataset(frame, image_root=root)
    x, y = dataset[0]
    assert x.shape == (1, 64, 64) and x.dtype == torch.float32
    assert isinstance(y, int) and y == int(frame.iloc[0].class_id)
    assert x[0, 0, 0] == 0 and x[0, -1, -1] == 1
    raw, _ = ReproductionDataset(frame, image_root=root, scale_to_unit=False)[0]
    torch.testing.assert_close(x, raw / 255)
    # A subset lacking class 0 must keep class 1, not remap it to 0.
    assert ReproductionDataset(frame[frame.class_id == 1], image_root=root)[0][1] == 1


@pytest.mark.parametrize("mode,size", [("RGB", (64, 64)), ("L", (32, 64))])
def test_wrong_preprocessing_is_rejected(samples, mode, size):
    frame, root = samples
    Image.new(mode, size).save(root / frame.iloc[0].image_path)
    with pytest.raises(ValueError, match="Expected 64x64"):
        ReproductionDataset(frame, image_root=root)[0]


def test_every_fold_has_exact_complement_and_all_samples_once(samples):
    frame, root = samples
    report = audit_reproduction_folds(frame, n_splits=3, image_root=root)
    seen = []
    for fold_id in range(3):
        dm = ReproductionDataModule(frame, fold_id, n_splits=3, image_root=root, batch_size=4)
        dm.setup("test")
        test_ids = set(frame.loc[frame.test_fold == fold_id, "sha"])
        assert set(dm.test_manifest.sha) == test_ids
        assert set(dm.train_manifest.sha) == set(frame.sha) - test_ids
        assert dm.val_dataloader() == []
        test_labels = torch.cat([y for _, y in dm.test_dataloader()])
        assert test_labels.dtype == torch.int64
        assert test_labels.tolist() == dm.test_manifest.class_id.tolist()
        assert sum(len(y) for _, y in dm.train_dataloader()) == len(dm.train_manifest)
        assert report["folds"][fold_id]["overlap_count"] == 0
        seen.extend(dm.test_manifest.sha)
    assert sorted(seen) == sorted(frame.sha)


@pytest.mark.parametrize("column,value", [("test_fold", 0.5), ("test_fold", -1),
    ("test_fold", None), ("class_id", 0.5), ("class_id", -1), ("class_id", None)])
def test_invalid_numeric_metadata_fails_before_training(samples, column, value):
    frame, root = samples
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        ReproductionDataModule(frame, 0, n_splits=3, image_root=root).setup("fit")


def test_cross_fold_duplicate_id_and_alias_path_are_rejected(samples):
    frame, root = samples
    other = frame.index[frame.test_fold != frame.iloc[0].test_fold][0]
    duplicate = frame.copy()
    duplicate.loc[other, "sha"] = frame.iloc[0].sha.upper()
    with pytest.raises(ValueError, match="unique"):
        audit_reproduction_folds(duplicate, n_splits=3)
    duplicate = frame.copy()
    duplicate.loc[other, "image_path"] = str(root / frame.iloc[0].image_path)
    with pytest.raises(ValueError, match="Duplicate resolved"):
        audit_reproduction_folds(duplicate, n_splits=3, image_root=root)


def test_missing_file_and_corrupted_image_fail(samples):
    frame, root = samples
    path = root / frame.iloc[0].image_path
    path.unlink()
    with pytest.raises(FileNotFoundError):
        audit_reproduction_folds(frame, n_splits=3, image_root=root)
    path.write_bytes(b"invalid PNG")
    with pytest.raises(OSError):
        ReproductionDataset(frame, image_root=root)[0]


def test_unbalanced_folds_and_inconsistent_labels_rejected(samples):
    frame, _ = samples
    wrong = frame.copy()
    wrong.loc[wrong.class_id == 0, "test_fold"] = 0
    with pytest.raises(ValueError, match="Every class"):
        audit_reproduction_folds(wrong, n_splits=3)
    wrong = frame.copy()
    wrong.loc[0, "family"] = "family1"
    with pytest.raises(ValueError, match="one-to-one"):
        audit_reproduction_folds(wrong, n_splits=3)


def test_seeded_train_shuffle_is_repeatable(samples):
    frame, root = samples
    def order():
        dm = ReproductionDataModule(frame, 0, n_splits=3, image_root=root, seed=123, batch_size=2)
        return torch.cat([x[:, 0, 1, 1] for x, _ in dm.train_dataloader()])
    torch.testing.assert_close(order(), order())


class TinyClassifier(pl.LightningModule):
    """Integration fixture only; NOT the MalSBSLCNet reproduction model."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.layer = torch.nn.Linear(4096, num_classes)
        self.train_seen = self.test_seen = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.train_seen += len(y)
        return torch.nn.functional.cross_entropy(self.layer(x.flatten(1)), y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        self.test_seen += len(y)
        self.log("test_loss", torch.nn.functional.cross_entropy(self.layer(x.flatten(1)), y))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001)


def test_real_lightning_fit_test_for_every_fold(samples):
    frame, root = samples
    for fold_id in range(3):
        dm = ReproductionDataModule(frame, fold_id, n_splits=3, image_root=root, batch_size=4)
        model = TinyClassifier(dm.num_classes)
        trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            limit_val_batches=0, num_sanity_val_steps=0)
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, verbose=False)
        assert model.train_seen == len(dm.train_manifest)
        assert model.test_seen == len(dm.test_manifest)


def test_audit_cli_decodes_all_images_and_writes_report(samples):
    frame, root = samples
    manifest, output = root / "folds.csv", root / "audit.json"
    frame.to_csv(manifest, index=False)
    subprocess.run([sys.executable, "-m", "scripts.audit_reproduction_data",
        "--manifest", str(manifest), "--image-root", str(root), "--n-splits", "3",
        "--check-all-images", "--output", str(output)], check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["status"] == "passed"
    assert report["decoded_image_count"] == len(frame)
    assert len(report["folds"]) == 3
