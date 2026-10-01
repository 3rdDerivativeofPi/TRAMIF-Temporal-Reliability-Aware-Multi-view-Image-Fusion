"""One independent Lightning data module per outer reproduction CV fold."""
from __future__ import annotations

import random
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.cross_validation import split_fold_manifest
from src.data.reproduction import (make_reproduction_dataset, audit_reproduction_folds,
                                   integer_column, read_manifest)


def seed_worker(worker_id: int) -> None:
    """Top-level function is picklable by Windows/spawn workers."""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class ReproductionDataModule(pl.LightningDataModule):
    def __init__(self, manifest: str | Path | pd.DataFrame, fold_id: int, *,
                 image_root: str | Path = ".", image_column: str = "image_path",
                 n_splits: int = 10, batch_size: int = 64, num_workers: int = 0,
                 seed: int = 42, pin_memory: bool = False, scale_to_unit: bool = True):
        super().__init__()
        for name, value, minimum in [("fold_id", fold_id, 0), ("batch_size", batch_size, 1),
                                      ("num_workers", num_workers, 0)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        self.manifest = read_manifest(manifest)
        self.fold_id, self.n_splits = fold_id, n_splits
        self.image_root, self.image_column = image_root, image_column
        self.batch_size, self.num_workers = batch_size, num_workers
        self.seed, self.pin_memory = seed, pin_memory
        self.scale_to_unit = scale_to_unit
        self.train_dataset = self.test_dataset = None
        self.audit_report = None

    def setup(self, stage: str | None = None) -> None:
        # Validate even when invoked directly with stage='test'. Cache immutable
        # split datasets so Lightning's repeated stage hooks do not reshuffle.
        if self.train_dataset is not None:
            return
        self.audit_report = audit_reproduction_folds(
            self.manifest, n_splits=self.n_splits,
            image_root=self.image_root, image_column=self.image_column)
        frame = self.manifest.copy()
        frame["test_fold"] = integer_column(frame, "test_fold")
        self.train_manifest, self.test_manifest = split_fold_manifest(frame, self.fold_id)
        kwargs = dict(image_root=self.image_root, image_column=self.image_column,
                      scale_to_unit=self.scale_to_unit)
        self.train_dataset = make_reproduction_dataset(self.train_manifest, **kwargs)
        self.test_dataset = make_reproduction_dataset(self.test_manifest, **kwargs)

    @property
    def num_classes(self) -> int:
        self.setup()
        return self.audit_report["num_classes"]

    def _loader(self, training: bool) -> DataLoader:
        self.setup()
        generator = torch.Generator().manual_seed(self.seed + self.fold_id)
        return DataLoader(
            self.train_dataset if training else self.test_dataset,
            batch_size=self.batch_size, shuffle=training, drop_last=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            worker_init_fn=seed_worker, generator=generator)

    def train_dataloader(self) -> DataLoader:
        return self._loader(True)

    def val_dataloader(self) -> list:
        # Outer test data must NEVER select checkpoints or stop training.
        # Use predeclared epochs, or implement an inner split from train only.
        return []

    def test_dataloader(self) -> DataLoader:
        return self._loader(False)
