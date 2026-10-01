"""SBSMI reproduction data contract; independent of TRAMIF temporal splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets import (
    ImageToFloat,
    MalwareImageDataset,
)
from src.data.labels import (
    family_label_space_from_frame,
)

from src.data.cross_validation import split_fold_manifest, validate_fold_manifest


def integer_column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise ValueError(f"Missing required column: {name}")
    numeric = pd.to_numeric(frame[name], errors="coerce")
    if (numeric.isna().any() or not np.isfinite(numeric).all()
            or (numeric < 0).any() or (numeric % 1 != 0).any()
            or (numeric >= 2**63).any() or frame[name].map(lambda x: isinstance(x, bool)).any()):
        raise ValueError(f"{name} must contain non-negative integer values")
    return numeric.astype("int64")


def read_manifest(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    # Keep IDs as text, including numeric-looking IDs with leading zeroes.
    return (source.copy() if isinstance(source, pd.DataFrame)
            else pd.read_csv(source, dtype={"sha": "string"}))


def resolve_image_paths(frame: pd.DataFrame, image_root: str | Path,
                        image_column: str = "image_path") -> list[Path]:
    if image_column not in frame:
        raise ValueError(f"Missing required image column: {image_column}")
    root = Path(image_root).expanduser().resolve()
    paths = []
    for value in frame[image_column]:
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"{image_column} contains an empty path")
        path = Path(str(value)).expanduser()
        paths.append((path if path.is_absolute() else root / path).resolve())
    return paths


def audit_reproduction_folds(source: str | Path | pd.DataFrame, *,
                             n_splits: int = 10,
                             image_root: str | Path | None = None,
                             image_column: str = "image_path") -> dict:
    """Fail closed on invalid IDs/labels/folds or unbalanced stratification.

    Pass image_root to additionally verify file existence and reject reuse of
    the same resolved image path. This is an ID/path audit, not near-duplicate
    binary detection. No fold assignments or class labels are changed.
    """
    frame = read_manifest(source)
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if frame.empty:
        raise ValueError("Manifest must not be empty")
    if "sha" not in frame or frame["sha"].isna().any():
        raise ValueError("sha is missing or contains missing values")
    ids = frame["sha"].astype(str).str.strip().str.lower()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("sha must contain unique non-empty IDs (case insensitive)")
    frame["sha"] = ids
    frame["class_id"] = integer_column(frame, "class_id")
    frame["test_fold"] = integer_column(frame, "test_fold")
    validate_fold_manifest(frame, n_splits=n_splits)
    classes = sorted(frame["class_id"].unique().tolist())
    if classes != list(range(len(classes))):
        raise ValueError("class_id must be contiguous 0..C-1; preserve one global class mapping")
    if "family" in frame:
        if frame["family"].isna().any() or frame["family"].astype(str).str.strip().eq("").any():
            raise ValueError("family contains missing/empty values")
        if (frame.groupby("family")["class_id"].nunique().max() != 1
                or frame.groupby("class_id")["family"].nunique().max() != 1):
            raise ValueError("family and class_id must have a one-to-one mapping")
        family_label_space_from_frame(
            frame[["family", "class_id"]].drop_duplicates()
        )
    paths = None
    if image_root is not None:
        paths = resolve_image_paths(frame, image_root, image_column)
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate resolved image paths: different IDs reuse one image")
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing image files ({len(missing)}): {missing[:5]}")
    table = pd.crosstab(frame["test_fold"], frame["class_id"]).reindex(
        index=range(n_splits), columns=classes, fill_value=0)
    if (table == 0).any().any():
        raise ValueError("Every class must appear in every test fold")
    if ((table.max() - table.min()) > 1).any():
        raise ValueError("Class distributions are not stratified: per-class fold counts differ by > 1")
    folds = []
    for fold_id in range(n_splits):
        train, test = split_fold_manifest(frame, fold_id)
        overlap = set(train.sha) & set(test.sha)
        if overlap or len(train) + len(test) != len(frame):
            raise ValueError(f"Invalid train/test partition for fold {fold_id}")
        folds.append({
            "fold_id": fold_id, "train_samples": len(train), "test_samples": len(test),
            "overlap_count": len(overlap),
            "train_class_counts": {str(k): int(v) for k, v in train.class_id.value_counts().sort_index().items()},
            "test_class_counts": {str(k): int(v) for k, v in test.class_id.value_counts().sort_index().items()},
        })
    return {"samples": len(frame), "num_classes": len(classes), "n_splits": n_splits,
            "each_sample_tested_once": True, "image_paths_checked": paths is not None,
            "scope": "Exact sample IDs and optional resolved image paths; not near-duplicate detection.",
            "folds": folds}


def make_reproduction_dataset(
    manifest: str | Path | pd.DataFrame,
    *,
    image_root: str | Path = ".",
    image_column: str = "image_path",
    scale_to_unit: bool = True,
) -> MalwareImageDataset:
    """Adapt reproduction metadata to the shared image loader."""

    frame = read_manifest(manifest)

    if frame.empty:
        raise ValueError("Dataset manifest must not be empty")

    if "sha" not in frame:
        raise ValueError("Missing required column: sha")

    if (
        "sample_id" in frame
        and not frame["sample_id"]
        .astype(str)
        .equals(frame["sha"].astype(str))
    ):
        raise ValueError(
            "sample_id and sha must identify the same samples"
        )

    frame["sample_id"] = frame["sha"]
    frame["class_id"] = integer_column(frame, "class_id")

    frame[image_column] = [
        str(path)
        for path in resolve_image_paths(
            frame,
            image_root,
            image_column,
        )
    ]

    return MalwareImageDataset(
        frame,
        image_root=image_root,
        image_column=image_column,
        transform=ImageToFloat(scale_to_unit),
        expected_size=(64, 64),
        strict_grayscale=True,
        sort_by_sample_id=False,
    )
