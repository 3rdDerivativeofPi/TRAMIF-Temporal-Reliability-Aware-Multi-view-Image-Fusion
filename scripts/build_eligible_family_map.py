"""
Build the known-family class mapping using training-period
support only.

This script must not use validation or future-test family
performance to select or reorder classes.
"""

from pathlib import Path

import pandas as pd


SUPPORT_PATH = Path(
    "data/manifests/bodmas_family_support_train_v0.csv"
)

OUTPUT_PATH = Path(
    "data/manifests/bodmas_eligible_families_v0.csv"
)


def main() -> None:
    support = pd.read_csv(SUPPORT_PATH)

    required_columns = {
        "family",
        "n_files",
        "n_months",
        "eligible_provisional",
    }

    missing = required_columns - set(support.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    eligible = support[
        support["eligible_provisional"]
    ].copy()

    # Deterministic class ordering.
    eligible = eligible.sort_values(
        "family"
    ).reset_index(drop=True)

    eligible["class_id"] = range(
        len(eligible)
    )

    assert len(eligible) == 51

    assert eligible["family"].is_unique
    assert eligible["class_id"].is_unique

    assert eligible["class_id"].tolist() == list(
        range(51)
    )

    output = eligible[
        [
            "family",
            "class_id",
            "n_files",
            "n_months",
        ]
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved {len(output)} eligible families "
        f"to {OUTPUT_PATH}"
    )

    print()
    print(output)


if __name__ == "__main__":
    main()