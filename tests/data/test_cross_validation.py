from collections import Counter

import pandas as pd
import pytest

from src.data.cross_validation import (
    build_stratified_fold_manifest,
    split_fold_manifest,
)


def _synthetic_manifest(
    n_classes: int = 10,
    samples_per_class: int = 20,
) -> pd.DataFrame:
    rows = []

    for class_id in range(n_classes):
        for sample_index in range(samples_per_class):
            rows.append(
                {
                    "sha": f"class{class_id:02d}-sample{sample_index:03d}",
                    "family": f"family_{class_id}",
                    "class_id": class_id,
                }
            )

    return pd.DataFrame(rows)


def test_fold_generation_is_deterministic():
    manifest = _synthetic_manifest()

    first = build_stratified_fold_manifest(
        manifest,
        random_state=123,
    )
    second = build_stratified_fold_manifest(
        manifest.sample(frac=1.0, random_state=999),
        random_state=123,
    )

    first_map = dict(zip(first["sha"], first["test_fold"]))
    second_map = dict(zip(second["sha"], second["test_fold"]))

    assert first_map == second_map


def test_train_and_test_are_disjoint_for_every_fold():
    fold_manifest = build_stratified_fold_manifest(
        _synthetic_manifest(),
        random_state=123,
    )

    for fold_id in range(10):
        train, test = split_fold_manifest(
            fold_manifest,
            fold_id,
        )

        train_ids = set(train["sha"])
        test_ids = set(test["sha"])

        assert train_ids.isdisjoint(test_ids)
        assert len(train) + len(test) == len(fold_manifest)


def test_every_sample_appears_in_exactly_one_test_fold():
    fold_manifest = build_stratified_fold_manifest(
        _synthetic_manifest(),
        random_state=123,
    )

    test_appearances = Counter()

    for fold_id in range(10):
        _, test = split_fold_manifest(
            fold_manifest,
            fold_id,
        )

        test_appearances.update(test["sha"].tolist())

    assert set(test_appearances) == set(fold_manifest["sha"])
    assert all(count == 1 for count in test_appearances.values())


def test_stratification_balances_each_class_across_folds():
    fold_manifest = build_stratified_fold_manifest(
        _synthetic_manifest(
            n_classes=10,
            samples_per_class=20,
        ),
        random_state=123,
    )

    counts = (
        fold_manifest
        .groupby(["test_fold", "class_id"])
        .size()
    )

    # 20 examples per class / 10 folds = exactly 2 per class per fold.
    assert (counts == 2).all()


def test_class_with_too_few_samples_is_rejected():
    manifest = _synthetic_manifest(
        n_classes=2,
        samples_per_class=9,
    )

    with pytest.raises(ValueError, match="at least n_splits"):
        build_stratified_fold_manifest(
            manifest,
            n_splits=10,
            random_state=123,
        )
