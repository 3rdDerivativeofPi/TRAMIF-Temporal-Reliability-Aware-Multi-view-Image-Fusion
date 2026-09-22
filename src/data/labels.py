from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FamilyLabelSpace:
    families: tuple[str, ...]
    family_to_id: dict[str, int]
    id_to_family: dict[int, str]

    @property
    def num_classes(self) -> int:
        return len(self.families)


def load_family_label_space(
    path: str | Path,
) -> FamilyLabelSpace:
    path = Path(path)

    df = pd.read_csv(path)

    required_columns = {
        "family",
        "class_id",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["family"].duplicated().any():
        raise ValueError(
            "Family names must be unique"
        )

    if df["class_id"].duplicated().any():
        raise ValueError(
            "Class IDs must be unique"
        )

    df = df.sort_values(
        "class_id"
    ).reset_index(drop=True)

    expected_ids = list(
        range(len(df))
    )

    actual_ids = df[
        "class_id"
    ].tolist()

    if actual_ids != expected_ids:
        raise ValueError(
            "Class IDs must be contiguous "
            "starting from zero"
        )

    families = tuple(
        df["family"].tolist()
    )

    family_to_id = {
        family: class_id
        for class_id, family
        in enumerate(families)
    }

    id_to_family = {
        class_id: family
        for family, class_id
        in family_to_id.items()
    }

    return FamilyLabelSpace(
        families=families,
        family_to_id=family_to_id,
        id_to_family=id_to_family,
    )