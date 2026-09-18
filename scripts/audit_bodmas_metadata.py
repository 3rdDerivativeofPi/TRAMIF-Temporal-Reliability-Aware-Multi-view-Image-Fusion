from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data/raw/bodmas/bodmas_metadata.csv")


df = pd.read_csv(METADATA_PATH)

# --------------------------------------------------
# 1. Basic schema
# --------------------------------------------------

required_columns = {"sha", "timestamp", "family"}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 2. Normalize SHA-256
# --------------------------------------------------

df["sha256"] = (
    df["sha"]
    .astype("string")
    .str.strip()
    .str.lower()
)

valid_sha256 = df["sha256"].str.fullmatch(r"[0-9a-f]{64}")

print("Invalid SHA-256:", (~valid_sha256.fillna(False)).sum())
print("Unique SHA-256:", df["sha256"].nunique())
print("Duplicate SHA-256 rows:", df["sha256"].duplicated().sum())


# --------------------------------------------------
# 3. Normalize family
# --------------------------------------------------

df["family"] = (
    df["family"]
    .astype("string")
    .str.strip()
)

df.loc[df["family"] == "", "family"] = pd.NA

df["is_malware"] = df["family"].notna()

print("Rows with family:", df["is_malware"].sum())
print("Rows without family:", (~df["is_malware"]).sum())


# --------------------------------------------------
# 4. Parse timestamps
# --------------------------------------------------

df["timestamp_raw"] = df["timestamp"]

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    errors="coerce",
    utc=True,
)

print("Invalid timestamps:", df["timestamp"].isna().sum())
print("Earliest timestamp:", df["timestamp"].min())
print("Latest timestamp:", df["timestamp"].max())

# Month is used for grouping/support counts.
# A YYYY-MM string avoids dropping timezone information.
df["month"] = df["timestamp"].dt.strftime("%Y-%m")


# --------------------------------------------------
# 5. Chronological split
# --------------------------------------------------

TRAIN_START = pd.Timestamp("2019-08-01", tz="UTC")
VALID_START = pd.Timestamp("2020-02-01", tz="UTC")
TEST_START = pd.Timestamp("2020-04-01", tz="UTC")
TEST_END = pd.Timestamp("2020-10-01", tz="UTC")

df["split"] = "outside_protocol"

df.loc[df["timestamp"].isna(), "split"] = "invalid"

df.loc[
    (df["timestamp"] >= TRAIN_START)
    & (df["timestamp"] < VALID_START),
    "split",
] = "train"

df.loc[
    (df["timestamp"] >= VALID_START)
    & (df["timestamp"] < TEST_START),
    "split",
] = "validation"

df.loc[
    (df["timestamp"] >= TEST_START)
    & (df["timestamp"] < TEST_END),
    "split",
] = "future_test"

print("\nSplit counts:")
print(df["split"].value_counts(dropna=False))

# --------------------------------------------------
# 6. Audit invalid timestamps
# --------------------------------------------------

bad_ts = df[df["timestamp"].isna()].copy()

print("\n=== Invalid timestamp audit ===")
print("Total:", len(bad_ts))
print("Malware:", bad_ts["is_malware"].sum())
print("Benign:", (~bad_ts["is_malware"]).sum())

print("\nExample raw timestamp values:")
print(
    bad_ts[
        ["sha256", "timestamp_raw", "family"]
    ].head(20).to_string(index=False)
)

# --------------------------------------------------
# 7. Audit invalid SHA-256
# --------------------------------------------------

bad_sha = df[~valid_sha256.fillna(False)].copy()

print("\n=== Invalid SHA-256 audit ===")
print(
    bad_sha[
        ["sha", "timestamp_raw", "family"]
    ].to_string(index=False)
)

# --------------------------------------------------
# 8. Malware-only chronology
# --------------------------------------------------

malware_df = df[df["is_malware"]].copy()

print("\n=== Malware-only chronology ===")
print("Malware rows:", len(malware_df))
print("Earliest malware timestamp:", malware_df["timestamp"].min())
print("Latest malware timestamp:", malware_df["timestamp"].max())
print("Invalid malware timestamps:", malware_df["timestamp"].isna().sum())

print(
    "Malware rows without valid SHA-256:",
    malware_df["sha256"].isna().sum()
)
                                                                                                                                                                                                               
print("\nMalware split counts:")
print(malware_df["split"].value_counts(dropna=False))

# --------------------------------------------------
# Build standardized manifest fields
# --------------------------------------------------

MANIFEST_DIR = Path("data/manifests")
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

df["sample_id"] = df["sha"].astype("string").str.strip()

df["is_valid_sha256"] = (
    df["sample_id"]
    .str.fullmatch(r"[0-9a-fA-F]{64}")
    .fillna(False)
)

df["sha256"] = (
    df["sample_id"]
    .str.lower()
    .where(df["is_valid_sha256"], pd.NA)
)

df["binary_available"] = False


manifest_cols = [
    "sample_id",
    "sha256",
    "is_valid_sha256",
    "timestamp",
    "month",
    "family",
    "is_malware",
    "split",
    "binary_available",
]


# --------------------------------------------------
# Save complete BODMAS manifest
# --------------------------------------------------

df[manifest_cols].to_csv(
    MANIFEST_DIR / "bodmas_all_v0.csv",
    index=False,
)


# --------------------------------------------------
# Save malware-family cohort manifest
# --------------------------------------------------

malware_df = df[df["is_malware"]].copy()

assert len(malware_df) == 57293
assert malware_df["timestamp"].notna().all()
assert malware_df["sha256"].notna().all()

assert set(malware_df["split"]) == {
    "train",
    "validation",
    "future_test",
}

malware_df[manifest_cols].to_csv(
    MANIFEST_DIR / "bodmas_family_v0.csv",
    index=False,
)

print("\nSaved manifests:")
print(MANIFEST_DIR / "bodmas_all_v0.csv")
print(MANIFEST_DIR / "bodmas_family_v0.csv")

# --------------------------------------------------
# Build family-level training support statistics
# --------------------------------------------------

train = malware_df[
    malware_df["split"] == "train"
].copy()

support = (
    train
    .groupby("family")
    .agg(
        n_files=("sha256", "nunique"),
        n_months=("month", "nunique"),
    )
    .sort_values("n_files", ascending=False)
)

support["eligible_provisional"] = (
    (support["n_files"] >= 50)
    & (support["n_months"] >= 3)
)

support.to_csv(
    MANIFEST_DIR / "bodmas_family_support_train_v0.csv"
)

print("\nTraining families:", len(support))
print(
    "Provisionally eligible families:",
    support["eligible_provisional"].sum()
)