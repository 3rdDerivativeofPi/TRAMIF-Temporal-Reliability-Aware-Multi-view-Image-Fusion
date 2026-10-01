from pathlib import Path
import hashlib

import pandas as pd


METADATA_PATH = Path(
    "E:/BODMAS_GW/bodmas_metadata.csv"
)


# File identity
sha256 = hashlib.sha256(
    METADATA_PATH.read_bytes()
).hexdigest()


df = pd.read_csv(METADATA_PATH)

family = (
    df["family"]
    .astype("string")
    .str.strip()
    .str.lower()
)


print("File:", METADATA_PATH.resolve())
print("File size:", METADATA_PATH.stat().st_size)
print("CSV SHA-256:", sha256)
print("Rows:", len(df))

print("\n=== Family counts ===")

for name in [
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
]:
    print(
        f"{name:12s}",
        int((family == name).sum()),
    )