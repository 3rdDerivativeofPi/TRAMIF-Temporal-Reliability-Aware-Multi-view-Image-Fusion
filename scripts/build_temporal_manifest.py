"""Make one canonical, training-mapped BODMAS manifest to copy to all laptops.

Reads existing BODMAS metadata (raw CSV or bodmas_family_v0.csv). The family
set/class indices come ONLY from the existing training-derived eligible CSV.
Future rows may be listed for the eventual frozen evaluation; they must not
be used to select models, preprocessing, reliability rules or hyperparameters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.multiview_contract import SPLIT_RANGE, load_mapping, write_csv

COLUMNS = ("sha", "family", "class_id", "split", "month", "timestamp",
           "file_size", "zip_crc32")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="BODMAS original metadata CSV or derived bodmas_family_v0.csv")
    parser.add_argument("--mapping", type=Path, default=Path("data/manifests/bodmas_eligible_families_v0.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/multiview_v2_manifest.csv"))
    parser.add_argument("--index", type=Path, help="Optional indexed file_size/zip_crc32 CSV, if available")
    args = parser.parse_args()

    mapping = load_mapping(args.mapping)
    data = pd.read_csv(args.metadata, dtype="string", keep_default_na=True)
    id_column = next((k for k in ("sha256", "sha", "sample_id") if k in data), None)
    family_column = next((k for k in ("family_norm", "family") if k in data), None)
    if not id_column or not family_column or "timestamp" not in data:
        raise ValueError("Metadata requires SHA identifier, family and timestamp columns")

    data["sha"] = data[id_column].str.strip().str.lower()
    data["family_clean"] = data[family_column].str.strip().str.lower()
    data = data[data["family_clean"].isin(mapping)].copy()
    if data.empty:
        raise ValueError("No eligible samples in metadata")
    if not data["sha"].str.fullmatch(r"[0-9a-f]{64}").fillna(False).all():
        raise ValueError("Invalid SHA among selected samples")
    if data["sha"].duplicated().any():
        raise ValueError("Duplicate original SHA among selected samples")

    dates = pd.to_datetime(data["timestamp"], format="mixed", utc=True, errors="coerce")
    if dates.isna().any():
        raise ValueError("Unparseable timestamps among selected samples")
    data["time_utc"] = dates
    data["month"] = dates.dt.strftime("%Y-%m")
    data["timestamp_utc"] = dates.dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    data["day"] = dates.dt.strftime("%Y-%m-%d")
    data["split"] = "outside_protocol"
    for split, (start, stop) in SPLIT_RANGE.items():
        data.loc[(data["day"] >= start) & (data["day"] < stop), "split"] = split
    data = data[data["split"] != "outside_protocol"].copy()
    if data.empty:
        raise ValueError("No eligible samples in the protocol date intervals")
    # Normalize date string so input CSV format does not leak into per-laptop hashes.
    data["class_id"] = data["family_clean"].map(mapping)

    index = {}
    if args.index:
        from src.data.multiview_contract import read_csv
        for row in read_csv(args.index):
            key = str(row.get("sha") or row.get("sha256") or "").strip().lower()
            if key in index:
                raise ValueError(f"Duplicate SHA in optional index: {key}")
            index[key] = row

    rows = []
    for r in data.to_dict("records"):
        src = index.get(r["sha"], {})
        rows.append({
            "sha": r["sha"], "family": r["family_clean"],
            "class_id": mapping[r["family_clean"]], "split": r["split"],
            "month": r["month"], "timestamp": r["timestamp_utc"],
            "file_size": src.get("file_size") or "",
            "zip_crc32": src.get("zip_crc32") or "",
        })
    rows.sort(key=lambda r: (list(SPLIT_RANGE).index(r["split"]), r["month"], r["sha"]))
    write_csv(args.output, rows, COLUMNS)
    print("Manifest:", args.output)
    print("Selected eligible samples:", len(rows))
    for split in SPLIT_RANGE:
        print(split, sum(r["split"] == split for r in rows))
    print("Future rows are evaluation-only. Do not use their labels/counts for tuning.")


if __name__ == "__main__":
    main()
