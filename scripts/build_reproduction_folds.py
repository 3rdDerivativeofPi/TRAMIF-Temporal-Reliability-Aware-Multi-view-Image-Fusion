"""Build the deterministic 10-fold manifest for MalCSBSV reproduction."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.cross_validation import save_stratified_fold_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/manifests/malcsbsv_bodmas_reproduction_v1.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/manifests/malcsbsv_bodmas_reproduction_folds_v1.csv"
        ),
    )
    parser.add_argument("--sample-id-column", default="sha")
    parser.add_argument("--label-column", default="class_id")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=(
            "Project-selected deterministic CV seed. Change this only through "
            "the reproduction configuration; do not tune it from results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fold_manifest = save_stratified_fold_manifest(
        args.input,
        args.output,
        sample_id_column=args.sample_id_column,
        label_column=args.label_column,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )

    print(f"Saved {len(fold_manifest)} samples to {args.output}")
    print()
    print("Test samples per fold:")
    print(
        fold_manifest["test_fold"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print()
    print("Class counts per test fold:")
    print(
        fold_manifest.groupby(
            ["test_fold", args.label_column]
        ).size().unstack(fill_value=0).to_string()
    )


if __name__ == "__main__":
    main()
