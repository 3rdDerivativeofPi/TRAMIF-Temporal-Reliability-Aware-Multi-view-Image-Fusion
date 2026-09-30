"""
Lazy-loading dataset for malware binary-image classification.

Each sample is identified by a sample_id (typically the SHA-256 hash of
the original file).  Images are loaded on-demand from disk so that large
corpora do not need to fit in memory.

The Dataset is agnostic to the image representation (raw-byte, entropy,
SBSMI, etc.).  Callers specify the image root and a filename suffix so
that the same interface works for all three views.

Example
-------
>>> import pandas as pd
>>> manifest = pd.DataFrame({
...     "sample_id": ["abc123", "def456"],
...     "class_id":  [0, 1],
... })
>>> dataset = MalwareImageDataset(
...     manifest=manifest,
...     image_root=Path("data/images/sbsmi"),
...     suffix=".png",          # files are {sample_id}.png
...     transform=None,
... )
>>> x, y = dataset[0]  # x: Tensor (1, H, W), y: int
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torchvision import io


class MalwareImageDataset(torch.utils.data.Dataset):
    """
    Lazy-loading dataset for disk-stored malware binary images.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must contain at least the columns:

        - ``sample_id`` — unique identifier for each sample (e.g. SHA-256).
        - ``class_id`` — integer class label in range ``[0, num_classes)``.

        Extra columns are ignored.
    image_root : Path | str
        Root directory containing one image file per sample.
    suffix : str
        Filename suffix appended to each ``sample_id`` to locate the image
        on disk, e.g. ``".npy"`` or ``".png"``.
    transform : callable | None
        Optional transform applied to the image tensor after loading.
        Must return a ``torch.Tensor``.  If ``None`` the raw loaded tensor
        is returned.

    Raises
    ------
    FileNotFoundError
        If an image file does not exist at the expected path.
    ValueError
        If the manifest is missing required columns or ``sample_id`` values
        are not unique.
    """

    _REQUIRED_COLUMNS = frozenset({"sample_id", "class_id"})

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: Path | str,
        suffix: str = "",
        transform: callable | None = None,
    ) -> None:
        super().__init__()

        self.image_root = Path(image_root)
        self.suffix = suffix
        self.transform = transform

        # Validate columns
        missing = self._REQUIRED_COLUMNS - set(manifest.columns)
        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        # Validate unique sample IDs
        if manifest["sample_id"].duplicated().any():
            raise ValueError("sample_id values must be unique")

        # Store a sorted copy so that __getitem__ order is deterministic
        self._manifest = (
            manifest[["sample_id", "class_id"]]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample_id = self._manifest.loc[index, "sample_id"]
        class_id: int = int(self._manifest.loc[index, "class_id"])

        image_path = self.image_root / f"{sample_id}{self.suffix}"

        # Read image as grayscale (1 channel).  torchvision.io.read_image
        # returns uint8 Tensor with shape (C, H, W).  We force C=1 by
        # converting from RGB if needed.
        image = io.read_image(
            str(image_path),
            mode=io.ImageReadMode.GRAY,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id
