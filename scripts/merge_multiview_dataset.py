"""Safely combine independent laptop modality outputs by original SHA.

Fail closed on missing files, source-content mismatches, configuration drift,
uncertain provenance, corrupt outputs, or duplicate candidates. Never match
images by processing order or family folder names.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from src.data.multiview_contract import (
    MODALITIES, canonical_hash, file_sha256, load_manifest, load_mapping,
    output_paths, read_csv, select_rows, valid_report, write_csv,
    write_json_atomic,
)

MERGED_COLUMNS = ("sha", "family", "class_id", "split", "month", "timestamp",
                  "raw_byte_path", "entropy_path", "sbsmi_path",
                  "disarmed_content_sha256", "config_id")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Laptop output roots; one or more modalities in each")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Merged dataset root (separate from laptop roots)")
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/multiview_v2_manifest.csv"))
    parser.add_argument("--mapping", type=Path, default=Path("data/manifests/bodmas_eligible_families_v0.csv"))
    parser.add_argument("--splits", nargs="+", choices=("train", "validation", "test"),
                        default=["train", "validation"])
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--limit", type=int, help="For smoke test: first N samples per split")
    parser.add_argument("--frozen-config-id", help="Required when merging test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Full verification without copying")
    return parser


def load_inputs_config(inputs: list[Path], manifest: Path, mapping: Path) -> tuple[dict, str]:
    configs = []
    for root in inputs:
        path = root / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        configs.append(config)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Laptops have different preprocessing configurations")
    config = configs[0]
    if config.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Laptop manifests differ from the provided authoritative manifest")
    if config.get("mapping_sha256") != file_sha256(mapping):
        raise ValueError("Laptop family mapping differs from the provided mapping")
    return config, canonical_hash(config)


def candidate_from_root(inputs: list[Path], row: dict, modality: str,
                        config_id: str) -> tuple[Path, Path, dict]:
    found = []
    for root in inputs:
        image, report = output_paths(root, row, modality)
        if image.exists() or report.exists():
            if not image.is_file() or not report.is_file():
                raise ValueError(f"Image/report incomplete: {image}")
            details = valid_report(root, row, modality, config_id)
            found.append((image, report, details))
    if len(found) != 1:
        raise ValueError(f"Expected ONE {modality} output for {row['sha']}; found {len(found)}")
    return found[0]


def atomic_copy(source: Path, destination: Path, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if file_sha256(source) != file_sha256(destination):
            raise ValueError(f"Output differs and --overwrite was not provided: {destination}")
        return
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != file_sha256(source):
            raise ValueError(f"Copy integrity failed: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def merge(args) -> dict:
    inputs = [p.expanduser().resolve() for p in args.inputs]
    output = args.output_dir.expanduser().resolve()
    if len(set(inputs)) != len(inputs):
        raise ValueError("Input roots must be unique")
    if any(output == p or p in output.parents or output in p.parents for p in inputs):
        raise ValueError("Merged output cannot be an input folder or its ancestor/descendant")
    splits = list(dict.fromkeys(args.splits))
    modalities = list(dict.fromkeys(args.modalities))
    if "test" in splits and splits != ["test"]:
        raise ValueError("Future test merging must be run separately")

    mapping = load_mapping(args.mapping)
    rows = load_manifest(args.manifest, mapping)
    selected = select_rows(rows, splits, args.months, args.limit, 0, 1)
    if not selected:
        raise ValueError("No matching rows")
    config, config_id = load_inputs_config(inputs, args.manifest, args.mapping)
    if "test" in splits and args.frozen_config_id != config_id:
        raise ValueError("Future test requires --frozen-config-id from historical run")
    existing_config = output / "config.json"
    if existing_config.exists():
        if json.loads(existing_config.read_text(encoding="utf-8")) != config:
            raise ValueError("Merged destination has a different preprocessing configuration")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Destination nonempty without config.json")

    # Full preflight first. This will identify even ONE incomplete SHA before copying.
    validated = []
    counts = Counter()
    for row in selected:
        matched = {}
        content_shas = set()
        sizes = set()
        crcs = set()
        for modality in modalities:
            image, report_path, report = candidate_from_root(inputs, row, modality, config_id)
            matched[modality] = (image, report_path, report)
            content_shas.add(report["disarmed_content_sha256"])
            sizes.add(report["file_size"])
            crcs.add(report["zip_crc32"])
            counts[(row["split"], row["month"], modality)] += 1
        if len(content_shas) != 1 or len(sizes) != 1 or len(crcs) != 1:
            raise ValueError(f"Views were generated from DIFFERENT bytes: {row['sha']}")
        validated.append((row, matched, next(iter(content_shas))))

    print("PRECHECK PASS:", len(selected), "samples,", len(selected) * len(modalities), "images")
    print("Config ID:", config_id)
    for (split, month, modality), n in sorted(counts.items()):
        print(f"  {split:10s} {month} {modality:9s} {n}")
    if args.dry_run:
        return {"selected": len(selected), "images": len(selected) * len(modalities), "config_id": config_id}

    if not existing_config.exists():
        write_json_atomic(existing_config, config)
    existing_manifest = output / "merged_manifest.csv"
    entries = {}
    if existing_manifest.is_file():
        for row in read_csv(existing_manifest):
            if row["sha"] in entries:
                raise ValueError("Duplicate SHA in old merged manifest")
            entries[row["sha"]] = row
    for row, matched, source_sha in validated:
        for modality, (source, report_source, _) in matched.items():
            target, report_dest = output_paths(output, row, modality)
            atomic_copy(source, target, overwrite=args.overwrite)
            atomic_copy(report_source, report_dest, overwrite=args.overwrite)
            valid_report(output, row, modality, config_id)
        entry = {"sha": row["sha"], "family": row["family"],
                 "class_id": row["class_id"], "split": row["split"],
                 "month": row["month"], "timestamp": row["timestamp"],
                 "disarmed_content_sha256": source_sha, "config_id": config_id}
        for modality in MODALITIES:
            if modality in matched:
                image, _ = output_paths(output, row, modality)
                entry[modality + "_path"] = image.relative_to(output).as_posix()
            else:
                entry[modality + "_path"] = entries.get(row["sha"], {}).get(modality + "_path", "")
        if row["sha"] in entries:
            previous = entries[row["sha"]]
            for k in ("family", "class_id", "split", "month", "timestamp", "disarmed_content_sha256", "config_id"):
                if str(previous.get(k)) != str(entry.get(k)):
                    raise ValueError(f"Existing merge record conflicts for {row['sha']} ({k})")
        entries[row["sha"]] = entry
    ordered = sorted(entries.values(), key=lambda x: (x["split"], x["month"], x["sha"]))
    write_csv(existing_manifest, ordered, MERGED_COLUMNS)
    print("MERGE COMPLETE:", len(validated), "new/verified samples")
    print("Merged manifest:", existing_manifest)
    return {"selected": len(selected), "images": len(selected) * len(modalities), "config_id": config_id}


def main():
    try:
        merge(build_parser().parse_args())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print("MERGE FAILED:", exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
