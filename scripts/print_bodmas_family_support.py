from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


DATA_PATH = Path("E:\\BODMAS_GW\\bodmas_metadata.csv")
OUTPUT_DIR = Path("data/manifests")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 1. Load BODMAS metadata
# =========================================================

df = pd.read_csv(DATA_PATH)

df["family"] = (
    df["family"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# Blank family = benign, so remove benign rows.
malware = df[df["family"] != ""].copy()


# =========================================================
# 2. Count support for every malware family
# =========================================================

family_support = (
    malware
    .groupby("family")["sha"]
    .nunique()
    .reset_index(name="support")
    .sort_values("family")
    .reset_index(drop=True)
)


# Save first CSV
support_path = OUTPUT_DIR / "bodmas_family_support.csv"

family_support.to_csv(
    support_path,
    index=False,
)


# =========================================================
# 3. Find suspiciously similar family names
# =========================================================

families = family_support["family"].tolist()

support_lookup = dict(
    zip(
        family_support["family"],
        family_support["support"],
    )
)

suspicious_pairs = []


for i in range(len(families)):
    for j in range(i + 1, len(families)):

        family_a = families[i]
        family_b = families[j]

        similarity = SequenceMatcher(
            None,
            family_a.lower(),
            family_b.lower(),
        ).ratio()

        # 0.80 = 80% similar
        if similarity >= 0.80:
            suspicious_pairs.append(
                {
                    "family_a": family_a,
                    "support_a": support_lookup[family_a],
                    "family_b": family_b,
                    "support_b": support_lookup[family_b],
                    "similarity": round(
                        similarity * 100,
                        2,
                    ),
                }
            )


suspicious_df = pd.DataFrame(suspicious_pairs)

if not suspicious_df.empty:
    suspicious_df = suspicious_df.sort_values(
        "similarity",
        ascending=False,
    )


# Save second CSV
suspicious_path = (
    OUTPUT_DIR
    / "bodmas_suspicious_family_names.csv"
)

suspicious_df.to_csv(
    suspicious_path,
    index=False,
)


# =========================================================
# 4. Print summary
# =========================================================

print("Done!")

print(
    f"Total malware families: "
    f"{len(family_support)}"
)

print(
    f"Suspicious name pairs: "
    f"{len(suspicious_df)}"
)

print()
print("Created:")
print(support_path)
print(suspicious_path)