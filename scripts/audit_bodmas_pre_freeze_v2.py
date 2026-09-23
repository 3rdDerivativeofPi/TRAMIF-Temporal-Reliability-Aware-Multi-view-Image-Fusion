"""
BODMAS pre-freeze audit v2.

Read-only audit:
- does not extract malware
- does not execute malware
- does not run PE_modifier.py
- does not re-arm samples
- does not inspect future per-family performance

Key correction from v1:
BODMAS metadata contains mixed timestamp formats, so timestamps are parsed with
format="mixed".
"""

from pathlib import Path
import json
import re
import zipfile

import pandas as pd


ROOT = Path(r"E:\BODMAS_GW")

METADATA = ROOT / "bodmas_metadata.csv"
META_DISARM = ROOT / "meta_disarm.csv"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_id(value):
    return str(value).strip().lower()


def is_sha256(value):
    return bool(SHA256_RE.fullmatch(normalize_id(value)))


def print_section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------
# 1. Metadata
# ---------------------------------------------------------------------

meta = pd.read_csv(METADATA)

required = {"sha", "timestamp", "family"}
if not required.issubset(meta.columns):
    raise RuntimeError(
        f"Expected metadata columns {sorted(required)}, "
        f"got {meta.columns.tolist()}"
    )

meta["sample_id"] = meta["sha"].map(normalize_id)
meta["family_clean"] = meta["family"].fillna("").astype(str).str.strip()
meta["is_malware"] = meta["family_clean"].ne("")

# IMPORTANT: BODMAS uses mixed timestamp formats.
meta["timestamp_parsed"] = pd.to_datetime(
    meta["timestamp"],
    format="mixed",
    errors="coerce",
    utc=True,
)

meta["valid_sha256"] = meta["sample_id"].map(is_sha256)

malware = meta.loc[meta["is_malware"]].copy()
benign = meta.loc[~meta["is_malware"]].copy()

print_section("BODMAS METADATA")

print("rows:", len(meta))
print("unique sample identifiers:", meta["sample_id"].nunique())
print("duplicate identifier rows:", int(meta["sample_id"].duplicated().sum()))
print("malware rows:", len(malware))
print("blank-family / benign rows:", len(benign))

print("non-standard IDs, all records:", int((~meta["valid_sha256"]).sum()))
print("non-standard IDs, malware:", int((~malware["valid_sha256"]).sum()))
print("non-standard IDs, benign:", int((~benign["valid_sha256"]).sum()))

bad_id_examples = (
    meta.loc[~meta["valid_sha256"], "sha"]
    .astype(str)
    .head(10)
    .tolist()
)
print("non-standard ID examples:", bad_id_examples)

print("invalid timestamps, all records:", int(meta["timestamp_parsed"].isna().sum()))
print("invalid timestamps, malware:", int(malware["timestamp_parsed"].isna().sum()))
print("invalid timestamps, benign:", int(benign["timestamp_parsed"].isna().sum()))

print("earliest metadata timestamp:", meta["timestamp_parsed"].min())
print("latest metadata timestamp:", meta["timestamp_parsed"].max())
print("earliest malware timestamp:", malware["timestamp_parsed"].min())
print("latest malware timestamp:", malware["timestamp_parsed"].max())


# ---------------------------------------------------------------------
# 2. Chronology
# ---------------------------------------------------------------------

malware["month"] = (
    malware["timestamp_parsed"]
    .dt.tz_convert(None)
    .dt.to_period("M")
)

monthly_counts = malware.groupby("month").size().rename("malware_files")

train_mask = (
    malware["timestamp_parsed"].ge("2019-08-01")
    & malware["timestamp_parsed"].lt("2020-02-01")
)
validation_mask = (
    malware["timestamp_parsed"].ge("2020-02-01")
    & malware["timestamp_parsed"].lt("2020-04-01")
)
future_mask = (
    malware["timestamp_parsed"].ge("2020-04-01")
    & malware["timestamp_parsed"].lt("2020-10-01")
)

train = malware.loc[train_mask].copy()
validation = malware.loc[validation_mask].copy()
future = malware.loc[future_mask].copy()

assigned = train_mask | validation_mask | future_mask

print_section("MALWARE CHRONOLOGY")
print(monthly_counts)
print()
print("training files:", len(train))
print("validation files:", len(validation))
print("future-test files:", len(future))
print("malware outside protocol periods:", int((~assigned).sum()))


# ---------------------------------------------------------------------
# 3. Training-only family support
# ---------------------------------------------------------------------

support = (
    train.groupby("family_clean")
    .agg(
        n_files=("sample_id", "nunique"),
        n_months=("month", "nunique"),
    )
    .reset_index()
    .rename(columns={"family_clean": "family"})
)

support["eligible_provisional"] = (
    support["n_files"].ge(50)
    & support["n_months"].ge(3)
)

support = support.sort_values(
    ["eligible_provisional", "n_files", "n_months", "family"],
    ascending=[False, False, False, True],
).reset_index(drop=True)

eligible = (
    support.loc[
        support["eligible_provisional"],
        ["family", "n_files", "n_months"],
    ]
    .sort_values("family")
    .reset_index(drop=True)
)

eligible.insert(1, "class_id", range(len(eligible)))

eligible_files = int(eligible["n_files"].sum())
training_files = int(len(train))
coverage = 100.0 * eligible_files / training_files if training_files else float("nan")

print_section("TRAINING-ONLY FAMILY SUPPORT")
print("training families:", support["family"].nunique())
print("provisionally eligible families:", len(eligible))
print("eligible training files:", eligible_files)
print("total training files:", training_files)
print(f"eligible-family training coverage: {coverage:.2f}%")
print("excluded families:", int((~support["eligible_provisional"]).sum()))
print(
    "excluded-family training files:",
    int(
        support.loc[
            ~support["eligible_provisional"],
            "n_files",
        ].sum()
    ),
)

print("\neligible-family month-support distribution:")
print(eligible["n_months"].value_counts().sort_index())

print("\nlowest-support eligible families:")
print(
    eligible.sort_values(
        ["n_files", "n_months", "family"]
    ).head(15).to_string(index=False)
)

print("\nhighest-support excluded families:")
print(
    support.loc[~support["eligible_provisional"]]
    .sort_values(["n_files", "n_months", "family"], ascending=[False, False, True])
    .head(15)
    .to_string(index=False)
)


# ---------------------------------------------------------------------
# 4. meta_disarm
# ---------------------------------------------------------------------

disarm = pd.read_csv(META_DISARM)

if "sha256" not in disarm.columns:
    raise RuntimeError("meta_disarm.csv is missing expected sha256 column")

disarm["sample_id"] = disarm["sha256"].map(normalize_id)

print_section("META_DISARM")
print("rows:", len(disarm))
print("unique SHA identifiers:", disarm["sample_id"].nunique())
print("invalid SHA strings:", int((~disarm["sample_id"].map(is_sha256)).sum()))
print("duplicate SHA rows:", int(disarm["sample_id"].duplicated().sum()))


# ---------------------------------------------------------------------
# 5. Disarmed malware archive
# ---------------------------------------------------------------------

archives = [
    p
    for p in ROOT.iterdir()
    if p.is_file()
    and "disarmed_malware_binaries" in p.name.lower()
    and zipfile.is_zipfile(p)
]

if len(archives) != 1:
    raise RuntimeError(
        "Expected exactly one disarmed malware ZIP, found "
        + repr([p.name for p in archives])
    )

archive = archives[0]

with zipfile.ZipFile(archive, "r") as zf:
    members = [info for info in zf.infolist() if not info.is_dir()]
    member_paths = [info.filename for info in members]

    archive_ids = []
    invalid_member_names = []

    for name in member_paths:
        stem = Path(name).stem.lower()
        if is_sha256(stem):
            archive_ids.append(stem)
        else:
            invalid_member_names.append(name)

    archive_uncompressed = sum(info.file_size for info in members)
    archive_compressed_members = sum(info.compress_size for info in members)

print_section("DISARMED BINARY ARCHIVE")
print("archive:", archive.name)
print("archive file size:", archive.stat().st_size)
print("file members:", len(members))
print("total member compressed bytes:", archive_compressed_members)
print("total uncompressed bytes:", archive_uncompressed)
print("duplicate archive paths:", len(member_paths) - len(set(member_paths)))
print("valid SHA-like filenames:", len(archive_ids))
print("non-SHA-like filenames:", len(invalid_member_names))


# ---------------------------------------------------------------------
# 6. Identifier cross-check
# ---------------------------------------------------------------------

metadata_malware_ids = set(malware["sample_id"])
archive_ids = set(archive_ids)
disarm_ids = set(disarm["sample_id"])

print_section("IDENTIFIER CROSS-CHECK")
print("metadata malware IDs:", len(metadata_malware_ids))
print("archive filename IDs:", len(archive_ids))
print("meta_disarm IDs:", len(disarm_ids))
print("metadata IDs missing from archive:", len(metadata_malware_ids - archive_ids))
print("archive IDs missing from metadata:", len(archive_ids - metadata_malware_ids))
print("archive IDs missing from meta_disarm:", len(archive_ids - disarm_ids))
print("meta_disarm IDs missing from archive:", len(disarm_ids - archive_ids))


# ---------------------------------------------------------------------
# 7. Feature-vector package
# ---------------------------------------------------------------------

feature_packages = [
    p
    for p in ROOT.iterdir()
    if p.is_file()
    and "feature_vector" in p.name.lower()
    and zipfile.is_zipfile(p)
]

feature_members = []

print_section("FEATURE-VECTOR PACKAGE")
for package in feature_packages:
    print("package:", package.name, package.stat().st_size, "bytes")
    with zipfile.ZipFile(package, "r") as zf:
        for info in zf.infolist():
            if not info.is_dir():
                feature_members.append(
                    {
                        "package": package.name,
                        "member": info.filename,
                        "uncompressed_bytes": int(info.file_size),
                        "compressed_bytes": int(info.compress_size),
                    }
                )
                print(
                    " ",
                    info.filename,
                    "uncompressed=",
                    info.file_size,
                    "compressed=",
                    info.compress_size,
                )


# ---------------------------------------------------------------------
# 8. Save derived audit artifacts
# ---------------------------------------------------------------------

support_out = ROOT / "audit_training_family_support_v2.csv"
eligible_out = ROOT / "audit_eligible_families_v2.csv"
summary_out = ROOT / "audit_summary_v2.json"

support.to_csv(support_out, index=False)
eligible.to_csv(eligible_out, index=False)

summary = {
    "metadata_rows": int(len(meta)),
    "unique_sample_ids": int(meta["sample_id"].nunique()),
    "duplicate_id_rows": int(meta["sample_id"].duplicated().sum()),
    "malware_rows": int(len(malware)),
    "benign_or_blank_family_rows": int(len(benign)),
    "nonstandard_ids_all": int((~meta["valid_sha256"]).sum()),
    "nonstandard_ids_malware": int((~malware["valid_sha256"]).sum()),
    "nonstandard_ids_benign": int((~benign["valid_sha256"]).sum()),
    "invalid_timestamps_all": int(meta["timestamp_parsed"].isna().sum()),
    "invalid_timestamps_malware": int(malware["timestamp_parsed"].isna().sum()),
    "invalid_timestamps_benign": int(benign["timestamp_parsed"].isna().sum()),
    "earliest_metadata_timestamp": str(meta["timestamp_parsed"].min()),
    "latest_metadata_timestamp": str(meta["timestamp_parsed"].max()),
    "earliest_malware_timestamp": str(malware["timestamp_parsed"].min()),
    "latest_malware_timestamp": str(malware["timestamp_parsed"].max()),
    "monthly_malware_counts": {
        str(month): int(count)
        for month, count in monthly_counts.items()
    },
    "training_files": int(len(train)),
    "validation_files": int(len(validation)),
    "future_test_files": int(len(future)),
    "malware_outside_protocol": int((~assigned).sum()),
    "training_families": int(support["family"].nunique()),
    "eligible_families_provisional": int(len(eligible)),
    "eligible_training_files": eligible_files,
    "eligible_training_coverage_percent": round(coverage, 4),
    "excluded_families": int((~support["eligible_provisional"]).sum()),
    "excluded_training_files": int(
        support.loc[~support["eligible_provisional"], "n_files"].sum()
    ),
    "archive_file": archive.name,
    "archive_file_bytes": int(archive.stat().st_size),
    "archive_members": int(len(members)),
    "archive_uncompressed_bytes": int(archive_uncompressed),
    "archive_sha_filenames": int(len(archive_ids)),
    "meta_disarm_rows": int(len(disarm)),
    "metadata_missing_from_archive": int(len(metadata_malware_ids - archive_ids)),
    "archive_missing_from_metadata": int(len(archive_ids - metadata_malware_ids)),
    "archive_missing_from_meta_disarm": int(len(archive_ids - disarm_ids)),
    "meta_disarm_missing_from_archive": int(len(disarm_ids - archive_ids)),
    "feature_package_members": feature_members,
}

with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print_section("DONE")
print("Created:", support_out)
print("Created:", eligible_out)
print("Created:", summary_out)
