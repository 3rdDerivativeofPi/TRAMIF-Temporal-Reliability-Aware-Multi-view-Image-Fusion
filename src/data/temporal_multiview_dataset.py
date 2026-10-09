"""PyTorch loader for merged chronological TRAMIF V2 images.

Independent of src/data/reproduction_datamodule.py and its 10-fold protocol.
Returns ({modality: [1,64,64] float32 tensor}, class_id); optionally SHA.
"""
from __future__ import annotations

from pathlib import Path
import json

import torch

from src.data.multiview_contract import (
    MODALITIES, file_sha256, load_image, load_manifest, load_mapping, output_paths,
)


class TemporalMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, root: str | Path, manifest: str | Path, mapping: str | Path,
                 *, split: str, month: str | None = None,
                 modalities: tuple[str, ...] = MODALITIES,
                 allow_test: bool = False, return_sha: bool = False):
        self.root = Path(root)
        config = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        if config.get("manifest_sha256") != file_sha256(Path(manifest)):
            raise ValueError("Dataset manifest differs from stored preprocessing config")
        if config.get("mapping_sha256") != file_sha256(Path(mapping)):
            raise ValueError("Family mapping differs from stored preprocessing config")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test")
        if split == "test" and not allow_test:
            raise ValueError("Future test must be explicitly enabled for frozen evaluation")
        if not modalities or len(set(modalities)) != len(modalities) or any(m not in MODALITIES for m in modalities):
            raise ValueError("Invalid modalities")
        self.modalities = modalities
        self.return_sha = return_sha
        mapping_data = load_mapping(Path(mapping))
        rows = load_manifest(Path(manifest), mapping_data)
        self.rows = [r for r in rows if r["split"] == split and (month is None or r["month"] == month)]
        if not self.rows:
            raise ValueError("Empty split/month selection")
        # Eager existence check; training cannot silently skip missing modalities.
        for r in self.rows:
            for modality in self.modalities:
                path, _ = output_paths(self.root, r, modality)
                if not path.is_file():
                    raise FileNotFoundError(path)
        self.num_classes = len(mapping_data)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        r = self.rows[index]
        views = {}
        for modality in self.modalities:
            path, _ = output_paths(self.root, r, modality)
            array = load_image(path, modality)
            if modality == "sbsmi":
                tensor = torch.from_numpy(array).float().div_(255.0)
            else:
                tensor = torch.from_numpy(array).float()
            views[modality] = tensor.unsqueeze(0)
        label = int(r["class_id"])
        if self.return_sha:
            return views, label, r["sha"]
        return views, label
