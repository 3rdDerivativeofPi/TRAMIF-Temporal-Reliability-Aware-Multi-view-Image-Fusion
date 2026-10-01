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
    return family_label_space_from_frame(
        pd.read_csv(Path(path))
    )


def family_label_space_from_frame(
    df: pd.DataFrame,
) -> FamilyLabelSpace:
    """Validate an existing mapping without changing class IDs."""

    df = df.copy()

    required_columns = {"family", "class_id"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Family label space must not be empty")

    if df["family"].map(
        lambda value: (
            not isinstance(value, str)
            or not value.strip()
        )
    ).any():
        raise ValueError(
            "Family names must be non-empty strings"
        )

    numeric = pd.to_numeric(
        df["class_id"],
        errors="coerce",
    )

    if (
        numeric.isna().any()
        or (numeric < 0).any()
        or (numeric % 1 != 0).any()
        or df["class_id"].map(
            lambda value: isinstance(value, bool)
        ).any()
    ):
        raise ValueError(
            "Class IDs must be non-negative integers"
        )

    df["class_id"] = numeric.astype("int64")

    if df["family"].duplicated().any():
        raise ValueError("Family names must be unique")

    if df["class_id"].duplicated().any():
        raise ValueError("Class IDs must be unique")

    df = df.sort_values("class_id").reset_index(drop=True)

    if df["class_id"].tolist() != list(range(len(df))):
        raise ValueError(
            "Class IDs must be contiguous starting from zero"
        )

    families = tuple(df["family"].tolist())

    family_to_id = {
        family: class_id
        for class_id, family in enumerate(families)
    }

    id_to_family = {
        class_id: family
        for family, class_id in family_to_id.items()
    }

    return FamilyLabelSpace(
        families=families,
        family_to_id=family_to_id,
        id_to_family=id_to_family,
    )