"""Prepare the complete 10-family SBSMI cohort from a ZIP or folder."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, is_zipfile

import pandas as pd
from PIL import Image

from src.data.cross_validation import build_stratified_fold_manifest
from src.data.labels import load_family_label_space
from src.data.reproduction import audit_reproduction_folds, integer_column
from src.preprocessing.implementation_sbsmi import generate_sbsmi_dataset


FAMILIES = {
    "ceeinject", "drolnux", "gandcrab", "mira", "musecador",
    "sfone", "sillyp2p", "small", "upatre", "wabot",
}

# Project-selected seeds; the paper does not supply its exact seed values.
REPEAT_SEEDS = tuple(range(42, 52))


def index_source(source_path, wanted_ids):
    """Locate <sha> or <sha>.exe files without reading binary contents."""
    matches = defaultdict(list)

    def record(relative_path):
        name = PurePosixPath(relative_path).name.lower()
        sha = name[:-4] if name.endswith(".exe") else name
        if sha in wanted_ids:
            matches[sha].append(relative_path)

    if source_path.is_dir():
        source_mode = "folder"
        for path in source_path.rglob("*"):
            if path.is_file():
                record(path.relative_to(source_path).as_posix())

    elif source_path.is_file() and is_zipfile(source_path):
        source_mode = "zip"
        with ZipFile(source_path) as archive:
            for entry in archive.infolist():
                if not entry.is_dir():
                    record(entry.filename)

    else:
        raise ValueError(f"Expected an existing folder or valid ZIP: {source_path}")

    return source_mode, matches


def prepare(source_path, cohort_path, image_root, manifest_dir, *, overwrite=False):
    source_path = Path(source_path).expanduser()
    cohort_path = Path(cohort_path)
    image_root = Path(image_root)
    manifest_dir = Path(manifest_dir)

    # Load the complete cohort and preserve its existing class IDs.
    frame = pd.read_csv(cohort_path, dtype={"sha256": "string"})
    required = {"sha256", "family_norm", "class_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing cohort columns: {sorted(required - set(frame.columns))}")

    frame = frame.rename(columns={"sha256": "sha"})
    frame["sha"] = frame.sha.str.strip().str.lower()
    frame["family"] = frame.family_norm.astype("string").str.strip().str.lower()

    if (
        frame.empty
        or not frame.sha.str.fullmatch(r"[0-9a-f]{64}").fillna(False).all()
        or frame.sha.duplicated().any()
    ):
        raise ValueError("Cohort must contain unique, valid SHA-256 identifiers")

    if frame.family.isna().any() or set(frame.family) != FAMILIES:
        raise ValueError("Cohort must contain exactly the paper's 10 families")

    frame["class_id"] = integer_column(frame, "class_id")
    if frame.groupby("family").size().min() < 10:
        raise ValueError("Each family needs at least 10 samples for 10-fold CV")

    frame = frame.sort_values("sha", kind="stable").reset_index(drop=True)
    frame["sample_id"] = frame.sha
    frame["image_path"] = frame.family + "/" + frame.sha + ".png"
    frame["experiment_scope"] = "full_10_family_reproduction"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = manifest_dir / "family_map.csv"
    mapping = frame[["family", "class_id"]].drop_duplicates().sort_values("class_id")
    mapping.to_csv(mapping_path, index=False)
    labels = load_family_label_space(mapping_path)
    if labels.num_classes != 10:
        raise ValueError("Expected exactly 10 globally consistent class IDs")

    # Resolve every sample before generation; missing or duplicate matches stop the run.
    source_mode, matches = index_source(source_path, set(frame.sha))
    availability = frame[["sha", "family"]].copy()
    availability["source_match_count"] = frame.sha.map(lambda sha: len(matches[sha]))
    availability["candidate_paths"] = frame.sha.map(lambda sha: json.dumps(matches[sha]))
    availability.to_csv(manifest_dir / "binary_availability.csv", index=False)

    unavailable = availability.loc[availability.source_match_count != 1]
    if not unavailable.empty:
        raise RuntimeError(
            f"{len(unavailable)} cohort binaries are missing or ambiguous in the source. "
            "See binary_availability.csv; no smaller cohort was substituted."
        )

    frame["path"] = frame.sha.map(lambda sha: matches[sha][0])
    frame.to_csv(manifest_dir / "selection.csv", index=False)
    print(f"Selected ALL {len(frame):,} samples:", flush=True)
    print(frame.groupby("family").size().to_string(), flush=True)

    # Generate every sample, saving progress after each batch.
    selected = frame[["sha", "family", "path"]].to_dict("records")
    records = []

    for start in range(0, len(selected), 1000):
        batch = selected[start:start + 1000]
        result = generate_sbsmi_dataset(
            source_path=source_path,
            output_root=image_root,
            source_mode=source_mode,
            samples=batch,
            bit_num=6,
            chunk_size=65536,
            overwrite=overwrite,
        )

        for record in result:
            if record["status"] in {"success", "skipped"}:
                try:
                    expected = image_root / record["family"] / f"{record['sha']}.png"
                    if Path(record["output_path"]).resolve() != expected.resolve():
                        raise ValueError("Unexpected output path")

                    with Image.open(expected) as image:
                        if image.mode != "L" or image.size != (64, 64):
                            raise ValueError("Expected 64x64 grayscale SBSMI")
                        image.load()

                except Exception as exc:
                    record.update(
                        status="failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

        records.extend(result)
        pd.DataFrame(records).to_csv(manifest_dir / "generation_log.csv", index=False)
        print(f"Checked {len(records):,}/{len(selected):,} images", flush=True)

    # Check that generation covered the complete cohort exactly once.
    log = pd.DataFrame(records)
    if log.sha.duplicated().any() or set(log.sha) != set(frame.sha):
        raise RuntimeError("Generation log does not cover the complete cohort exactly once")

    logged = log.set_index("sha").loc[frame.sha]
    logged_paths = logged.path.astype(str).str.replace("\\", "/", regex=False)
    if (
        logged.family.tolist() != frame.family.tolist()
        or logged_paths.tolist() != frame.path.tolist()
    ):
        raise RuntimeError("Generation log disagrees with the selected cohort")

    failed = log.loc[~log.status.isin(["success", "skipped"])]
    if not failed.empty:
        raise RuntimeError(
            f"{len(failed)} images failed; inspect generation_log.csv and rerun"
        )

    frame.to_csv(manifest_dir / "full_cohort.csv", index=False)

    # Each repetition uses every sample; only fold assignments change.
    audits = []
    for repeat, seed in enumerate(REPEAT_SEEDS, start=1):
        folds = build_stratified_fold_manifest(frame, n_splits=10, random_state=seed)
        audit = audit_reproduction_folds(folds, n_splits=10, image_root=image_root)
        name = f"folds_repeat_{repeat:02d}_seed_{seed}.csv"
        folds.to_csv(manifest_dir / name, index=False)
        audits.append({
            "repeat": repeat,
            "split_seed": seed,
            "manifest": name,
            **audit,
        })

    report = {
        "samples": len(frame),
        "num_classes": labels.num_classes,
        "family_counts": frame.groupby("family").size().to_dict(),
        "cohort_path": str(cohort_path),
        "cohort_csv_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "source_path": str(source_path),
        "source_mode": source_mode,
        "image_root": str(image_root),
        "bit_num": 6,
        "repeat_seeds": REPEAT_SEEDS,
        "seed_origin": "Project-selected; author seeds unavailable",
        "paper_samples": 18498,
        "repetitions": audits,
    }
    (manifest_dir / "preparation_audit.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Ready: {len(frame):,} images; 10 repetitions of 10-fold CV prepared.")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", "--zip", dest="source", type=Path, required=True,
        help="ZIP archive or folder containing the cohort binaries",
    )
    parser.add_argument(
        "--cohort", type=Path,
        default=Path("data/manifests/malcsbsv_candidate_pool.csv"),
    )
    parser.add_argument(
        "--image-root", type=Path,
        default=Path("data/processed/sbsmi_10_families"),
    )
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=Path("data/manifests/sbsmi_full"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prepare(
        args.source, args.cohort, args.image_root, args.manifest_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()