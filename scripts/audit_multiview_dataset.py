"""Verify split/month/modality coverage and content of a generated or merged dataset."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.data.multiview_contract import (
    MODALITIES, canonical_hash, file_sha256, load_manifest, load_mapping,
    output_paths, select_rows, valid_report,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/multiview_v2_manifest.csv"))
    parser.add_argument("--mapping", type=Path, default=Path("data/manifests/bodmas_eligible_families_v0.csv"))
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"), default=["train", "validation"])
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--limit", type=int, help="Same smoke-test limit as generation")
    parser.add_argument("--check-extras", action="store_true", help="Reject unexpected images in selected folders")
    return parser


def audit(args) -> dict:
    mapping = load_mapping(args.mapping)
    config = json.loads((args.root / "config.json").read_text(encoding="utf-8"))
    if config.get("manifest_sha256") != file_sha256(args.manifest):
        raise ValueError("Manifest hash mismatch with dataset config")
    if config.get("mapping_sha256") != file_sha256(args.mapping):
        raise ValueError("Family mapping hash mismatch with dataset config")
    config_id = canonical_hash(config)
    selected = select_rows(load_manifest(args.manifest, mapping), args.splits, args.months,
                           args.limit, 0, 1)
    if not selected:
        raise ValueError("No selected manifest rows")

    totals = Counter()
    errors = []
    folder_expected = {}
    content_by_sha = {}
    for row in selected:
        source_hashes = set()
        for modality in args.modalities:
            image, _ = output_paths(args.root, row, modality)
            folder_expected.setdefault(image.parent, set()).add(image.name)
            try:
                report = valid_report(args.root, row, modality, config_id)
                source_hashes.add(report["disarmed_content_sha256"])
                totals[(row["split"], row["month"], modality)] += 1
            except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
                errors.append(f"{row['sha']} {row['split']} {row['month']} {modality}: {exc}")
        if len(source_hashes) > 1:
            errors.append(f"Cross-view content mismatch: {row['sha']}")
        elif len(source_hashes) == 1:
            content_by_sha[row["sha"]] = (next(iter(source_hashes)), row["split"])
    if args.check_extras and args.limit is None:
        for folder, expected in folder_expected.items():
            ext = ".png" if folder.name == "sbsmi" else ".npy"
            actual = {path.name for path in folder.glob("*" + ext)}
            for filename in sorted(actual - expected):
                errors.append(f"Unexpected output: {folder / filename}")
    # Descriptive exact-duplicate audit. Do not silently remove any rows;
    # near-duplicate policy remains a separate prespecified research decision.
    by_content = {}
    cross_split_duplicates = []
    for sha, (content_sha, split) in content_by_sha.items():
        previous = by_content.get(content_sha)
        if previous is not None and previous[1] != split and previous[0] != sha:
            cross_split_duplicates.append((previous[0], sha, previous[1], split))
        else:
            by_content.setdefault(content_sha, (sha, split))
    for item in cross_split_duplicates[:10]:
        print("WARNING: Identical disarmed contents across splits:", item)
    if cross_split_duplicates:
        print("REVIEW REQUIRED: Fix/explain exact cross-split duplicates under the locked duplicate policy.")
    for (split, month, modality), count in sorted(totals.items()):
        print(f"{split:10} {month:7} {modality:9} verified={count}")
    print(f"Samples={len(selected)} requested images={len(selected)*len(args.modalities)} "
          f"verified={sum(totals.values())} errors={len(errors)}")
    for msg in errors[:20]:
        print("ERROR:", msg)
    print("AUDIT:", "PASS" if not errors else "FAIL")
    return {"pass": not errors, "errors": errors, "verified": sum(totals.values()),
            "cross_split_exact_duplicates": cross_split_duplicates}


def main():
    try:
        result = audit(build_parser().parse_args())
    except (OSError, ValueError, KeyError) as exc:
        print("AUDIT FAILED:", exc, file=sys.stderr)
        raise SystemExit(1) from exc
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
