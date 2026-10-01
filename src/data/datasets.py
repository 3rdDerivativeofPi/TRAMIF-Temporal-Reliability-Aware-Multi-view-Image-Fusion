from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torchvision import io


def validate_image_file(
    path: str | Path,
    expected_size: tuple[int, int] | None = None,
    strict_grayscale: bool = False,
) -> None:
    """Decode an image and optionally check its original mode and size."""

    with Image.open(path) as image:
        wrong_size = (
            expected_size is not None
            and image.size != expected_size
        )
        wrong_mode = strict_grayscale and image.mode != "L"

        if wrong_size or wrong_mode:
            raise ValueError(
                (
                    f"Expected {expected_size[0]}x{expected_size[1]} "
                    if expected_size is not None
                    else "Expected "
                )
                + ("mode L image" if strict_grayscale else "image")
                + f", got {image.size} {image.mode}: {path}"
            )

        image.load()


@dataclass(frozen=True)
class ImageToFloat:
    """Picklable transform for Windows DataLoader workers."""

    scale_to_unit: bool = True

    def __call__(self, image: Tensor) -> Tensor:
        image = image.to(torch.float32)

        if self.scale_to_unit:
            return image / 255.0

        return image


class MalwareImageDataset(torch.utils.data.Dataset):
    """Shared image loader for SBSMI, raw-byte, and entropy images."""

    _REQUIRED_COLUMNS = frozenset({
        "sample_id",
        "class_id",
    })

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: Path | str,
        suffix: str = "",
        transform: Callable[[Tensor], Tensor] | None = None,
        *,
        image_column: str | None = None,
        expected_size: tuple[int, int] | None = None,
        strict_grayscale: bool = False,
        sort_by_sample_id: bool = True,
    ) -> None:
        super().__init__()

        self.image_root = Path(image_root)
        self.suffix = suffix
        self.transform = transform
        self.image_column = image_column
        self.expected_size = expected_size
        self.strict_grayscale = strict_grayscale

        missing = self._REQUIRED_COLUMNS - set(manifest.columns)

        if missing:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing)}"
            )

        if manifest["sample_id"].duplicated().any():
            raise ValueError("sample_id values must be unique")

        if (
            manifest["sample_id"].isna().any()
            or manifest["sample_id"]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError("sample_id values must be non-empty")

        if image_column is not None:
            if image_column not in manifest:
                raise ValueError(
                    f"Missing image column: {image_column}"
                )

            if (
                manifest[image_column].isna().any()
                or manifest[image_column]
                .astype(str)
                .str.strip()
                .eq("")
                .any()
            ):
                raise ValueError(
                    f"{image_column} contains an empty path"
                )

        self._manifest = manifest.copy()

        if sort_by_sample_id:
            self._manifest = self._manifest.sort_values(
                "sample_id",
                kind="mergesort",
            )

        self._manifest = self._manifest.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._manifest)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self._manifest.iloc[index]
        class_id = int(row["class_id"])

        if self.image_column is None:
            image_path = (
                self.image_root
                / f"{row['sample_id']}{self.suffix}"
            )
        else:
            image_path = Path(str(row[self.image_column]))

            if not image_path.is_absolute():
                image_path = self.image_root / image_path

        if self.expected_size is not None or self.strict_grayscale:
            validate_image_file(
                image_path,
                self.expected_size,
                self.strict_grayscale,
            )

        image = io.read_image(
            str(image_path),
            mode=io.ImageReadMode.GRAY,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id