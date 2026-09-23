from pathlib import Path
import json
import re
import zipfile

import pandas as pd


ROOT = Path(r"E:\BODMAS_GW")

METADATA = ROOT / "bodmas_metadata.csv"
META_DISARM = ROOT / "meta_disarm.csv"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize_sha(value):
    return str(value).strip().lower()


def is_valid_sha(value):
    return bool(SHA256_RE.fullmatch(normalize_sha(value)))


print("=" * 70)
print("BODMAS PRE-FREEZE DATA AUDIT")
print("=" * 70)


# =========================================================
# 1. TOP-LEVEL SOURCE FILES
# =========================================================

print("\n=== TOP-LEVEL FILES ===")

for path in sorted(ROOT.iterdir()):
    if path.is_file():
        print(f"{path.name:45s} {path.stat().st_size:>15,} bytes")


# =========================================================
# 2. BODMAS METADATA
# =========================================================

print("\n=== BODMAS METADATA ===")

meta = pd.read_csv(METADATA)

print("shape:", meta.shape)
print("columns:", meta.columns.tolist())

required_columns = {"sha", "timestamp", "family"}

if not required_columns.issubset(meta.columns):
    raise RuntimeError(
        f"Expected metadata columns {required_columns}, "
        f"got {set(meta.columns)}"
    )

meta["sha_norm"] = meta["sha"].map(normalize_sha)

meta["family_clean"] = (
    meta["family"]
    .fillna("")
    .astype(str)
    .str.strip()
)

meta["is_malware"] = meta["family_clean"].ne("")

meta["timestamp_parsed"] = pd.to_datetime(
    meta["timestamp"],
    errors="coerce",
    utc=True,
)

print("total rows:", len(meta))
print("malware-labelled rows:", int(meta["is_malware"].sum()))
print("blank-family rows:", int((~meta["is_malware"]).sum()))

print(
    "invalid SHA strings:",
    int((~meta["sha_norm"].map(is_valid_sha)).sum()),
)

print(
    "duplicate SHA rows:",
    int(meta["sha_norm"].duplicated().sum()),
)

print(
    "unparseable timestamps:",
    int(meta["timestamp_parsed"].isna().sum()),
)


# =========================================================
# 3. MALWARE-ONLY TEMPORAL AUDIT
# =========================================================

malware = meta.loc[meta["is_malware"]].copy()

print("\n=== MALWARE-ONLY TIMESTAMP AUDIT ===")

print(
    "earliest malware timestamp:",
    malware["timestamp_parsed"].min(),
)

print(
    "latest malware timestamp:",
    malware["timestamp_parsed"].max(),
)

malware["month"] = (
    malware["timestamp_parsed"]
    .dt.tz_convert(None)
    .dt.to_period("M")
)


# We may inspect aggregate month totals for data availability.
# Do NOT use future per-family statistics for tuning.

monthly_counts = (
    malware
    .groupby("month")
    .size()
    .rename("malware_files")
)

print("\nmalware files by month:")
print(monthly_counts)


# =========================================================
# 4. TRAINING-ONLY FAMILY SUPPORT
#
# Framework §11.2:
# Train = Aug 2019-Jan 2020
# >=50 unique files
# >=3 training months
# =========================================================

print("\n=== TRAINING-ONLY FAMILY SUPPORT ===")

train_mask = (
    malware["timestamp_parsed"].ge("2019-08-01")
    & malware["timestamp_parsed"].lt("2020-02-01")
)

train = malware.loc[train_mask].copy()

support = (
    train
    .groupby("family_clean")
    .agg(
        unique_files=("sha_norm", "nunique"),
        training_months=("month", "nunique"),
    )
)

support["eligible"] = (
    support["unique_files"].ge(50)
    & support["training_months"].ge(3)
)

support = support.sort_values(
    ["eligible", "unique_files", "training_months"],
    ascending=[False, False, False],
)

print("training malware rows:", len(train))
print("training families:", train["family_clean"].nunique())

print(
    "eligible families:",
    int(support["eligible"].sum()),
)

support.to_csv(
    ROOT / "audit_training_family_support.csv"
)


# =========================================================
# 5. DISARM METADATA
# READ ONLY
# =========================================================

print("\n=== META_DISARM ===")

disarm = pd.read_csv(META_DISARM)

print("shape:", disarm.shape)
print("columns:", disarm.columns.tolist())

if "sha256" not in disarm.columns:
    raise RuntimeError(
        "meta_disarm.csv does not contain expected sha256 column"
    )

disarm["sha_norm"] = disarm["sha256"].map(normalize_sha)

print(
    "invalid SHA strings:",
    int((~disarm["sha_norm"].map(is_valid_sha)).sum()),
)

print(
    "duplicate SHA rows:",
    int(disarm["sha_norm"].duplicated().sum()),
)


# =========================================================
# 6. FIND MALWARE ZIP
# =========================================================

binary_archives = [
    p
    for p in ROOT.iterdir()
    if p.is_file()
    and "disarmed_malware_binaries" in p.name.lower()
    and zipfile.is_zipfile(p)
]

if len(binary_archives) != 1:
    raise RuntimeError(
        "Expected exactly one disarmed malware ZIP, found: "
        + repr([p.name for p in binary_archives])
    )

binary_archive = binary_archives[0]

print("\n=== DISARMED BINARY ARCHIVE ===")
print("archive:", binary_archive.name)

with zipfile.ZipFile(binary_archive, "r") as zf:

    members = [
        info
        for info in zf.infolist()
        if not info.is_dir()
    ]

    member_names = [
        info.filename
        for info in members
    ]

    member_sha = []

    invalid_archive_names = []

    for name in member_names:

        stem = Path(name).stem.lower()

        if is_valid_sha(stem):
            member_sha.append(stem)
        else:
            invalid_archive_names.append(name)

    print("file members:", len(members))

    print(
        "total uncompressed bytes:",
        f"{sum(x.file_size for x in members):,}",
    )

    print(
        "duplicate archive paths:",
        len(member_names) - len(set(member_names)),
    )

    print(
        "valid SHA-like filenames:",
        len(member_sha),
    )

    print(
        "non-SHA-like filenames:",
        len(invalid_archive_names),
    )

    print("\nfirst 5 archive members:")

    for info in members[:5]:
        print(
            info.filename,
            f"{info.file_size:,} bytes",
        )


# =========================================================
# 7. CROSS-CHECK IDENTIFIERS
# =========================================================

print("\n=== IDENTIFIER CROSS-CHECK ===")

metadata_malware_sha = set(
    malware["sha_norm"]
)

archive_sha = set(member_sha)

disarm_sha = set(
    disarm["sha_norm"]
)

print(
    "unique malware metadata SHA:",
    len(metadata_malware_sha),
)

print(
    "unique archive filename SHA:",
    len(archive_sha),
)

print(
    "unique meta_disarm SHA:",
    len(disarm_sha),
)

print(
    "metadata malware SHA missing from archive:",
    len(metadata_malware_sha - archive_sha),
)

print(
    "archive SHA missing from malware metadata:",
    len(archive_sha - metadata_malware_sha),
)

print(
    "archive SHA missing from meta_disarm:",
    len(archive_sha - disarm_sha),
)

print(
    "meta_disarm SHA missing from archive:",
    len(disarm_sha - archive_sha),
)


# =========================================================
# 8. FEATURE-VECTOR OUTER ARCHIVE
#
# Only inspect wrapper.
# Do not load 724 MB bodmas.npz yet.
# =========================================================

print("\n=== FEATURE-VECTOR PACKAGE ===")

feature_candidates = [
    p
    for p in ROOT.iterdir()
    if p.is_file()
    and "feature_vector" in p.name.lower()
]

for path in feature_candidates:

    print(
        "candidate:",
        path.name,
        f"({path.stat().st_size:,} bytes)",
    )

    if zipfile.is_zipfile(path):

        with zipfile.ZipFile(path, "r") as zf:

            print("ZIP members:")

            for info in zf.infolist():
                if not info.is_dir():
                    print(
                        " ",
                        info.filename,
                        f"{info.file_size:,} bytes",
                    )


# =========================================================
# 9. MACHINE-READABLE SUMMARY
# =========================================================

summary = {
    "metadata_rows": int(len(meta)),
    "malware_rows": int(meta["is_malware"].sum()),
    "blank_family_rows": int((~meta["is_malware"]).sum()),
    "training_malware_rows": int(len(train)),
    "training_families": int(
        train["family_clean"].nunique()
    ),
    "eligible_training_families": int(
        support["eligible"].sum()
    ),
    "archive_members": int(len(members)),
    "archive_sha_filenames": int(len(archive_sha)),
    "meta_disarm_unique_sha": int(len(disarm_sha)),
    "metadata_malware_unique_sha": int(
        len(metadata_malware_sha)
    ),
    "metadata_missing_from_archive": int(
        len(metadata_malware_sha - archive_sha)
    ),
    "archive_missing_from_metadata": int(
        len(archive_sha - metadata_malware_sha)
    ),
    "archive_missing_from_meta_disarm": int(
        len(archive_sha - disarm_sha)
    ),
}

with open(
    ROOT / "audit_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n=== DONE ===")
print(
    "Created:",
    ROOT / "audit_training_family_support.csv",
)

print(
    "Created:",
    ROOT / "audit_summary.json",
)