"""Deterministic stratified cross-validation utilities.

This module is intended for the MalCSBSV/BODMAS reproduction track.
It is deliberately separate from the chronological TRAMIF split logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


TEST_FOLD_COLUMN = "test_fold"


def _validate_source_manifest(
    manifest: pd.DataFrame,
    *,
    sample_id_column: str,
    label_column: str,
    n_splits: int,
) -> None:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    required = {sample_id_column, label_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if manifest.empty:
        raise ValueError("Manifest must not be empty")

    if manifest[sample_id_column].isna().any():
        raise ValueError(f"{sample_id_column} contains missing values")

    if manifest[label_column].isna().any():
        raise ValueError(f"{label_column} contains missing values")

    if manifest[sample_id_column].duplicated().any():
        duplicates = manifest.loc[
            manifest[sample_id_column].duplicated(keep=False),
            sample_id_column,
        ].astype(str).tolist()
        preview = duplicates[:5]
        raise ValueError(
            f"{sample_id_column} must be unique; duplicate examples: {preview}"
        )

    class_counts = manifest[label_column].value_counts()
    too_small = class_counts[class_counts < n_splits]
    if not too_small.empty:
        details = ", ".join(
            f"{label}={count}" for label, count in too_small.items()
        )
        raise ValueError(
            "Every class must contain at least n_splits samples for "
            f"{n_splits}-fold stratification; too-small classes: {details}"
        )


def build_stratified_fold_manifest(
    manifest: pd.DataFrame,
    *,
    sample_id_column: str = "sha",
    label_column: str = "class_id",
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Assign each sample to exactly one deterministic test fold.

    The returned table contains every input row exactly once plus a
    ``test_fold`` column in the range ``0 .. n_splits - 1``.

    Before splitting, rows are sorted by sample ID so the assignment is
    reproducible even if the input CSV is read in a different row order.
    """

    _validate_source_manifest(
        manifest,
        sample_id_column=sample_id_column,
        label_column=label_column,
        n_splits=n_splits,
    )

    ordered = manifest.sort_values(
        sample_id_column,
        kind="mergesort",
    ).reset_index(drop=True)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    assignments = np.full(
        len(ordered),
        fill_value=-1,
        dtype=np.int64,
    )

    labels = ordered[label_column].to_numpy()
    dummy_x = np.zeros(len(ordered), dtype=np.uint8)

    for fold_id, (_, test_indices) in enumerate(
        splitter.split(dummy_x, labels)
    ):
        assignments[test_indices] = fold_id

    if np.any(assignments < 0):
        raise RuntimeError("At least one sample was not assigned to a test fold")

    result = ordered.copy()
    result[TEST_FOLD_COLUMN] = assignments

    validate_fold_manifest(
        result,
        sample_id_column=sample_id_column,
        n_splits=n_splits,
    )

    return result


def validate_fold_manifest(
    fold_manifest: pd.DataFrame,
    *,
    sample_id_column: str = "sha",
    n_splits: int = 10,
) -> None:
    """Validate basic invariants of an already-generated fold manifest."""

    required = {sample_id_column, TEST_FOLD_COLUMN}
    missing = required - set(fold_manifest.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if fold_manifest[sample_id_column].duplicated().any():
        raise ValueError(
            "Fold manifest must contain each sample exactly once"
        )

    if fold_manifest[TEST_FOLD_COLUMN].isna().any():
        raise ValueError("test_fold contains missing values")

    expected_folds = set(range(n_splits))
    observed_folds = set(
        fold_manifest[TEST_FOLD_COLUMN].astype(int).unique().tolist()
    )

    if observed_folds != expected_folds:
        raise ValueError(
            "Fold IDs do not match the expected range: "
            f"expected {sorted(expected_folds)}, observed {sorted(observed_folds)}"
        )


def split_fold_manifest(
    fold_manifest: pd.DataFrame,
    fold_id: int,
    *,
    sample_id_column: str = "sha",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train/test rows for one fold.

    Samples assigned to ``fold_id`` form the test set. All other samples form
    the training set.
    """

    if TEST_FOLD_COLUMN not in fold_manifest.columns:
        raise ValueError(f"Missing required column: {TEST_FOLD_COLUMN}")

    available_folds = set(
        fold_manifest[TEST_FOLD_COLUMN].astype(int).unique().tolist()
    )
    if fold_id not in available_folds:
        raise ValueError(
            f"Unknown fold_id {fold_id}; available folds: {sorted(available_folds)}"
        )

    train = fold_manifest.loc[
        fold_manifest[TEST_FOLD_COLUMN] != fold_id
    ].copy()
    test = fold_manifest.loc[
        fold_manifest[TEST_FOLD_COLUMN] == fold_id
    ].copy()

    train_ids = set(train[sample_id_column].astype(str))
    test_ids = set(test[sample_id_column].astype(str))

    overlap = train_ids & test_ids
    if overlap:
        preview = sorted(overlap)[:5]
        raise RuntimeError(
            f"Train/test leakage detected in fold {fold_id}: {preview}"
        )

    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_stratified_fold_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    sample_id_column: str = "sha",
    label_column: str = "class_id",
    n_splits: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Read a CSV manifest, assign folds, save the result, and return it."""

    source_path = Path(source_path)
    output_path = Path(output_path)

    manifest = pd.read_csv(source_path)
    fold_manifest = build_stratified_fold_manifest(
        manifest,
        sample_id_column=sample_id_column,
        label_column=label_column,
        n_splits=n_splits,
        random_state=random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fold_manifest.to_csv(output_path, index=False)

    return fold_manifest
