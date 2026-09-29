from pathlib import Path

import pandas as pd


# ==================================================
# Configuration
# ==================================================

METADATA_PATH = Path(
    "E:/BODMAS_GW/bodmas_metadata.csv"
)

MANIFEST_DIR = Path(
    "data/manifests"
)


# 10 BODMAS families used by Zhang et al.
MALCSBSV_FAMILIES = {
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
}


# Sample counts reported in Zhang et al.
PAPER_COUNTS = {
    "ceeinject": 1163,
    "drolnux": 920,
    "gandcrab": 939,
    "mira": 1829,
    "musecador": 1054,
    "sfone": 1377,
    "sillyp2p": 1526,
    "small": 3217,
    "upatre": 2988,
    "wabot": 3485,
}


# ==================================================
# 1. Load metadata
# ==================================================

df = pd.read_csv(METADATA_PATH)


required_columns = {
    "sha",
    "timestamp",
    "family",
}

missing_columns = (
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing expected columns: "
        f"{missing_columns}"
    )


# ==================================================
# 2. Normalize SHA-256
# ==================================================

df["sha256"] = (
    df["sha"]
    .astype("string")
    .str.strip()
    .str.lower()
)


valid_sha = (
    df["sha256"]
    .str.fullmatch(r"[0-9a-f]{64}")
    .fillna(False)
)


# ==================================================
# 3. Normalize family labels
# ==================================================

family_clean = (
    df["family"]
    .astype("string")
    .str.strip()
)


df["family_norm"] = (
    family_clean
    .mask(
        family_clean.eq(""),
        pd.NA,
    )
    .str.lower()
)


is_malware = (
    df["family_norm"].notna()
)


# ==================================================
# 4. Parse timestamps
# ==================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    errors="coerce",
    utc=True,
)


# Convert timezone-aware timestamps into
# simple monthly labels such as "2020-03".
#
# tz_convert(None) avoids the warning produced
# by directly converting timezone-aware values
# into Period objects.
df["month"] = (
    df["timestamp"]
    .dt.tz_convert(None)
    .dt.to_period("M")
    .astype("string")
)


df["year"] = (
    df["timestamp"]
    .dt.year
)


# ==================================================
# 5. Basic BODMAS overview
# ==================================================

print("\n========================================")
print("BODMAS OVERVIEW")
print("========================================")

print(
    "Total metadata rows:",
    len(df),
)

print(
    "Malware rows:",
    is_malware.sum(),
)

print(
    "Rows without family:",
    (~is_malware).sum(),
)

print(
    "Invalid SHA-256:",
    (~valid_sha).sum(),
)

print(
    "Invalid timestamps:",
    df["timestamp"].isna().sum(),
)

print(
    "Duplicate valid SHA-256:",
    df.loc[
        valid_sha,
        "sha256",
    ].duplicated().sum(),
)


# ==================================================
# 6. Malware timestamp range
# ==================================================

malware_timestamps = df.loc[
    is_malware & valid_sha,
    "timestamp",
]


print("\n========================================")
print("MALWARE TIMESTAMP RANGE")
print("========================================")

print(
    "All malware:",
    malware_timestamps.min(),
    "->",
    malware_timestamps.max(),
)


# ==================================================
# 7. Select the 10 MalCSBSV families
# ==================================================

candidate_df = df[
    is_malware
    & valid_sha
    & df["family_norm"].isin(
        MALCSBSV_FAMILIES
    )
].copy()


print("\n========================================")
print("MALCSBSV 10-FAMILY CANDIDATE POOL")
print("========================================")

print(
    "Rows:",
    len(candidate_df),
)

print(
    "Unique SHA-256:",
    candidate_df["sha256"].nunique(),
)

print(
    "Timestamp range:",
    candidate_df["timestamp"].min(),
    "->",
    candidate_df["timestamp"].max(),
)

print(
    "Earliest month:",
    candidate_df["month"].min(),
)

print(
    "Latest month:",
    candidate_df["month"].max(),
)


# ==================================================
# 8. Yearly family distribution
# ==================================================

yearly_counts = (
    candidate_df
    .groupby(
        [
            "year",
            "family_norm",
        ]
    )["sha256"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(
        columns=sorted(
            MALCSBSV_FAMILIES
        ),
        fill_value=0,
    )
)


yearly_counts["TOTAL"] = (
    yearly_counts.sum(axis=1)
)


print("\n========================================")
print("SAMPLES PER FAMILY BY YEAR")
print("========================================")

print(
    yearly_counts.to_string()
)


# ==================================================
# 9. Monthly family distribution
# ==================================================

monthly_counts = (
    candidate_df
    .groupby(
        [
            "month",
            "family_norm",
        ]
    )["sha256"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(
        columns=sorted(
            MALCSBSV_FAMILIES
        ),
        fill_value=0,
    )
    .sort_index()
)


monthly_counts["TOTAL"] = (
    monthly_counts.sum(axis=1)
)


print("\n========================================")
print("SAMPLES PER FAMILY BY MONTH")
print("========================================")

print(
    monthly_counts.to_string()
)

# ==================================================
# 10. Peak month for each family
# ==================================================

family_monthly = monthly_counts.drop(columns="TOTAL")

print("\n========================================")
print("PEAK MONTH FOR EACH FAMILY")
print("========================================")

for family in map(str, family_monthly.columns):

    family_counts = family_monthly[family]

    peak_month = str(
        family_counts.idxmax()
    )

    peak_count = int(
        family_counts.max()
    )

    print(
        f"{family:12s} -> "
        f"{peak_month}: "
        f"{peak_count}"
    )

# ==================================================
# 12. Monthly total peaks
# ==================================================

print("\n========================================")
print("TOTAL MALCSBSV SAMPLES PER MONTH")
print("========================================")


monthly_totals = (
    monthly_counts["TOTAL"]
    .sort_values(
        ascending=False
    )
)


print(
    monthly_totals.to_string()
)


print(
    "\nHighest-volume month:",
    monthly_totals.index[0],
)

print(
    "Samples:",
    monthly_totals.iloc[0],
)


# ==================================================
# 13. Compare current BODMAS with Zhang et al.
# ==================================================

paper_counts = pd.Series(
    PAPER_COUNTS,
    name="paper",
)


current_counts = (
    candidate_df
    .groupby(
        "family_norm"
    )["sha256"]
    .nunique()
    .reindex(
        list(
            PAPER_COUNTS.keys()
        ),
        fill_value=0,
    )
)


comparison = pd.DataFrame(
    {
        "paper": paper_counts,
        "current": current_counts,
    }
)


comparison["difference"] = (
    comparison["current"]
    - comparison["paper"]
)


comparison["percent_difference"] = (
    comparison["difference"]
    / comparison["paper"]
    * 100
).round(2)


print("\n========================================")
print("ZHANG ET AL. VS CURRENT BODMAS")
print("========================================")

print(
    comparison.to_string()
)


paper_total = (
    comparison["paper"].sum()
)

current_total = (
    comparison["current"].sum()
)


print("\nTotals:")

print(
    "Paper:",
    paper_total,
)

print(
    "Current:",
    current_total,
)

print(
    "Difference:",
    current_total - paper_total,
)


# ==================================================
# 14. Deterministic class IDs
# ==================================================

family_to_id = {
    family: class_id
    for class_id, family
    in enumerate(
        sorted(
            MALCSBSV_FAMILIES
        )
    )
}


candidate_df["class_id"] = (
    candidate_df[
        "family_norm"
    ].map(
        family_to_id
    )
)



# ==================================================
# 15. Save essential audit outputs only
# ==================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

manifest_columns = [
    "sha256",
    "timestamp",
    "month",
    "year",
    "family",
    "family_norm",
    "class_id",
]


# 1. Main candidate manifest
candidate_df[
    manifest_columns
].to_csv(
    MANIFEST_DIR
    / "malcsbsv_candidate_pool.csv",
    index=False,
)




# 2. Monthly family distribution
monthly_counts.to_csv(
    MANIFEST_DIR
    / "malcsbsv_family_counts_by_month.csv",
)


# 3. Paper vs current BODMAS comparison
comparison.to_csv(
    MANIFEST_DIR
    / "malcsbsv_paper_count_comparison.csv",
)


print("\n========================================")
print("AUDIT COMPLETE")
print("========================================")

print("Saved:")
print("- malcsbsv_candidate_pool_v1.csv")
print("- malcsbsv_family_counts_by_month.csv")
print("- malcsbsv_paper_count_comparison.csv")