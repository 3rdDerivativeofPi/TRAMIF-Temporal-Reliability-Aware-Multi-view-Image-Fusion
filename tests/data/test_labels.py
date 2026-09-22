from pathlib import Path

from src.data.labels import (
    load_family_label_space,
)


LABEL_PATH = Path(
    "data/manifests/bodmas_eligible_families_v0.csv"
)


def test_label_space_has_51_classes():
    labels = load_family_label_space(
        LABEL_PATH
    )

    assert labels.num_classes == 51


def test_class_ids_are_contiguous():
    labels = load_family_label_space(
        LABEL_PATH
    )

    ids = sorted(
        labels.id_to_family.keys()
    )

    assert ids == list(
        range(51)
    )


def test_family_mapping_is_bidirectional():
    labels = load_family_label_space(
        LABEL_PATH
    )

    for family, class_id in (
        labels.family_to_id.items()
    ):
        assert (
            labels.id_to_family[class_id]
            == family
        )


def test_family_order_is_deterministic():
    labels = load_family_label_space(
        LABEL_PATH
    )

    assert list(
        labels.families
    ) == sorted(
        labels.families
    )