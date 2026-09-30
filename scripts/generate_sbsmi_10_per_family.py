import csv
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from src.preprocessing.implementation_sbsmi import (
    generate_sbsmi_dataset,
)

# python -m scripts.generate_sbsmi_10_per_family

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

# Change these two paths if your BODMAS files are somewhere else.
ZIP_PATH = Path(
    r"E:\BODMAS_GW\BODMAS_disarmed_malware_binaries.zip"
)

METADATA_PATH = Path(
    r"E:\BODMAS_GW\bodmas_metadata.csv"
)


# Keep this pilot separate from the future full reproduction cache.
OUTPUT_ROOT = Path(
    "data/manifests/test"
)


# Save exactly which samples were selected.
SELECTION_MANIFEST_PATH = (
    OUTPUT_ROOT.parent
    / "test" / "sbsmi_pilot_10_per_family_manifest.csv"
)


# Save success / skipped / failed status for every selected sample.
GENERATION_LOG_PATH = (
    OUTPUT_ROOT.parent
    / "test" / "sbsmi_pilot_10_per_family_log.csv"
)


FAMILIES = [
    "ceeinject",
    "drolnux",
    "gandcrab",
    "mira",
    "musecador",
    "sfone",
    "sillyp2p",
    "small",
    "upatre",
    "wabot",
]


SAMPLES_PER_FAMILY = 10

# Fixed seed means everyone on the team should select
# the same samples from the same metadata + ZIP.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# FIND REAL BINARIES THAT EXIST INSIDE THE ZIP
# ---------------------------------------------------------------------------

def find_available_candidates():
    """
    Find samples belonging to the 10 reproduction families
    whose binary actually exists inside the BODMAS ZIP.

    Returns:

        {
            "ceeinject": [
                {
                    "path": "altered/<sha>.exe",
                    "sha": "<sha>",
                    "family": "ceeinject",
                },
                ...
            ],

            ...
        }
    """

    candidates = {
        family: []
        for family in FAMILIES
    }

    # Read the ZIP directory only.
    #
    # This does NOT extract the archive and does NOT load
    # every malware binary into memory.
    with ZipFile(
        ZIP_PATH,
        "r",
    ) as zip_file:

        zip_members = set(
            zip_file.namelist()
        )

    # Now read the metadata CSV.
    with METADATA_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            family = (
                row["family"]
                .strip()
                .lower()
            )

            # Ignore families outside the 10-family reproduction set.
            if family not in candidates:
                continue

            sha = (
                row["sha"]
                .strip()
                .lower()
            )

            if not sha:
                continue

            # In this BODMAS ZIP, disarmed binaries are stored as:
            #
            # altered/<original SHA>.exe
            member_path = (
                f"altered/{sha}.exe"
            )

            # Only keep metadata rows whose binary
            # really exists in the ZIP.
            if member_path not in zip_members:
                continue

            candidates[family].append({
                "path": member_path,
                "sha": sha,
                "family": family,
            })

    return candidates


# ---------------------------------------------------------------------------
# CHOOSE EXACTLY 10 PER FAMILY
# ---------------------------------------------------------------------------

def choose_samples(
    candidates,
):
    """
    Select 10 available binaries from each family.

    Selection is random but reproducible because RANDOM_SEED
    is fixed.
    """

    rng = random.Random(
        RANDOM_SEED
    )

    selected = []

    for family in FAMILIES:
        family_candidates = candidates[
            family
        ]

        # Sort first so selection does not depend on
        # arbitrary CSV/dictionary ordering.
        family_candidates = sorted(
            family_candidates,
            key=lambda sample: sample["sha"],
        )

        available_count = len(
            family_candidates
        )

        print(
            f"{family:12s}: "
            f"{available_count} available"
        )

        if available_count < SAMPLES_PER_FAMILY:
            raise RuntimeError(
                f"{family} only has "
                f"{available_count} usable binaries; "
                f"need {SAMPLES_PER_FAMILY}."
            )

        chosen = rng.sample(
            family_candidates,
            SAMPLES_PER_FAMILY,
        )

        # Sorting the chosen samples makes the manifest easier to read.
        chosen = sorted(
            chosen,
            key=lambda sample: sample["sha"],
        )

        selected.extend(
            chosen
        )

    return selected


# ---------------------------------------------------------------------------
# SAVE THE SELECTION MANIFEST
# ---------------------------------------------------------------------------

def save_selection_manifest(
    selected,
):
    """
    Record exactly which 100 binaries were selected.
    """

    SELECTION_MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SELECTION_MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "family",
                "sha",
                "path",
            ],
        )

        writer.writeheader()

        writer.writerows(
            selected
        )


# ---------------------------------------------------------------------------
# SAVE THE GENERATION LOG
# ---------------------------------------------------------------------------

def save_generation_log(
    records,
):
    """
    Save success / skipped / failed information for each sample.
    """

    GENERATION_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "family",
        "sha",
        "path",
        "output_path",
        "status",
        "error_type",
        "error_message",
    ]

    with GENERATION_LOG_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                field: record.get(
                    field,
                    "",
                )
                for field in fields
            })


# ---------------------------------------------------------------------------
# PRINT A SIMPLE RESULT SUMMARY
# ---------------------------------------------------------------------------

def print_summary(
    records,
):
    print()
    print("=" * 60)
    print("SBSMI PILOT SUMMARY")
    print("=" * 60)

    for family in FAMILIES:
        family_records = [
            record
            for record in records
            if record["family"] == family
        ]

        status_counts = Counter(
            record["status"]
            for record in family_records
        )

        # "skipped" normally means the PNG already existed,
        # so it is still ready for use.
        ready = (
            status_counts["success"]
            + status_counts["skipped"]
        )

        print(
            f"{family:12s} "
            f"ready={ready:2d}/10  "
            f"success={status_counts['success']:2d}  "
            f"skipped={status_counts['skipped']:2d}  "
            f"failed={status_counts['failed']:2d}"
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print(
        "Finding real binaries "
        "for the 10 reproduction families..."
    )

    candidates = (
        find_available_candidates()
    )

    print()
    print(
        "Selecting 10 samples "
        "from each family..."
    )

    selected = choose_samples(
        candidates
    )

    # We expect:
    #
    # 10 families x 10 files = 100 files
    expected_total = (
        len(FAMILIES)
        * SAMPLES_PER_FAMILY
    )

    if len(selected) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} samples, "
            f"but selected {len(selected)}."
        )

    print()
    print(
        f"Selected {len(selected)} "
        "real binaries."
    )

    save_selection_manifest(
        selected
    )

    print(
        "Selection manifest:"
    )

    print(
        SELECTION_MANIFEST_PATH
    )

    print()
    print(
        "Generating SBSMI images "
        "directly from the ZIP..."
    )

    # IMPORTANT:
    #
    # We reuse the existing tested SBSMI implementation.
    #
    # This script does NOT implement SBSMI again.
    records = generate_sbsmi_dataset(
        source_path=ZIP_PATH,
        output_root=OUTPUT_ROOT,
        source_mode="zip",
        samples=selected,
        bit_num=6,
        chunk_size=65536,
        overwrite=False,
    )

    save_generation_log(
        records
    )

    print_summary(
        records
    )

    print()
    print(
        "Generation log:"
    )

    print(
        GENERATION_LOG_PATH
    )

    print()
    print(
        "SBSMI output directory:"
    )

    print(
        OUTPUT_ROOT
    )

    # Final safety check.
    #
    # A failed sample remains in the log,
    # but the script finishes with an error so we cannot
    # accidentally call an incomplete pilot successful.
    incomplete_families = []

    for family in FAMILIES:
        ready = sum(
            1
            for record in records
            if (
                record["family"] == family
                and record["status"]
                in {"success", "skipped"}
            )
        )

        if ready != SAMPLES_PER_FAMILY:
            incomplete_families.append(
                family
            )

    if incomplete_families:
        raise RuntimeError(
            "Pilot generation is incomplete for: "
            + ", ".join(
                incomplete_families
            )
            + ". Check the generation log."
        )

    print()
    print(
        "Pilot complete: "
        "10 usable SBSMI images "
        "for each of the 10 families."
    )


if __name__ == "__main__":
    main()