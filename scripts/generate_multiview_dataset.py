"""
Distributed, Resumable & Trial-Ready Multi-View Image Dataset Generator.

Features:
  1. Terminal Progress Bar: Displays live progress, ETA, and speed using tqdm.
  2. Direct ZIP/Folder Processing: Read binaries directly from disk or compressed ZIPs.
  3. Flexible Metadata Columns: Accepts 'sha' or 'sha256', and 'family' or 'family_norm'.
  4. Modality Selection: Generate 'all', 'raw_byte', 'entropy', or 'sbsmi' per machine.
  5. Crash Recovery: Logs progress to disk and skips already processed files on restart.
  6. Trial Run Mode: Use `--limit N` to test on N real samples before full execution.
"""

import argparse
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, is_zipfile
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.preprocessing.raw_byte import raw_byte_image
from src.preprocessing.entropy import entropy_image
from src.preprocessing.implementation_sbsmi import stream_to_sbsmi

SUPPORTED_MODALITIES = {"raw_byte", "entropy", "sbsmi"}


def load_eligible_families(eligible_csv_path: Path) -> tuple[set[str], dict[str, int]]:
    """Load eligible family names and class ID mapping."""
    df = pd.read_csv(eligible_csv_path)
    df["family"] = df["family"].astype(str).str.strip().str.lower()
    df["class_id"] = df["class_id"].astype(int)
    return set(df["family"]), dict(zip(df["family"], df["class_id"]))


def index_source(source_path: Path) -> tuple[str, dict[str, list[str]]]:
    """Index source binaries in a directory or ZIP file mapping SHA256 -> relative path."""
    matches = {}

    def record(relative_path: str):
        name = PurePosixPath(relative_path).name.lower()
        sha = name[:-4] if name.endswith(".exe") else name
        if len(sha) == 64:
            matches.setdefault(sha, []).append(relative_path)

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
        raise ValueError(f"Invalid source path (must be directory or ZIP): {source_path}")

    return source_mode, matches


def save_image(array: np.ndarray, output_path: Path) -> None:
    """Save a 2D array as a grayscale PNG image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if array.dtype in (np.float32, np.float64):
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    Image.fromarray(array, mode="L").save(output_path)


def process_sample_stream(
    stream,
    sha: str,
    family: str,
    output_root: Path,
    modalities_to_generate: set[str],
    output_size: tuple[int, int] = (64, 64),
    sbsmi_bit_num: int = 6,
) -> dict:
    """Read a binary stream and generate active modality images."""
    stem = f"{family}/{sha}.png"
    targets = {
        "raw_byte": output_root / "raw_byte" / stem,
        "entropy": output_root / "entropy" / stem,
        "sbsmi": output_root / "sbsmi" / stem,
    }

    record = {
        "sha": sha,
        "family": family,
        "status": "success",
        "error": "",
    }

    try:
        # 1. SBSMI
        if "sbsmi" in modalities_to_generate:
            sbs_img = stream_to_sbsmi(stream, bit_num=sbsmi_bit_num)
            save_image(sbs_img, targets["sbsmi"])
            if len(modalities_to_generate) > 1 and hasattr(stream, "seek"):
                stream.seek(0)

        # 2. Raw-Byte & Entropy
        remaining_modalities = modalities_to_generate - {"sbsmi"}
        if remaining_modalities:
            data = stream.read()
            if not data:
                raise ValueError("Empty binary file")

            if "raw_byte" in modalities_to_generate:
                raw_img = raw_byte_image(data, output_size=output_size)
                save_image(raw_img, targets["raw_byte"])

            if "entropy" in modalities_to_generate:
                ent_img = entropy_image(data, output_size=output_size)
                save_image(ent_img, targets["entropy"])

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {str(exc)}"

    return record


def prepare_cohort(
    source_path: Path,
    eligible_csv: Path,
    cohort_csv: Path,
    output_dir: Path,
    modalities: list[str],
    output_size: tuple[int, int] = (64, 64),
    sbsmi_bit_num: int = 6,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    source_path = Path(source_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Determine requested modalities
    if "all" in modalities:
        active_modalities = SUPPORTED_MODALITIES.copy()
    else:
        active_modalities = set(modalities) & SUPPORTED_MODALITIES

    print(f"Active Modalities on this Machine: {sorted(active_modalities)}")

    # Load Eligible Families
    eligible_families, family_to_class = load_eligible_families(eligible_csv)
    
    # Save family map
    family_map_df = pd.DataFrame([
        {"family": fam, "class_id": cid} 
        for fam, cid in sorted(family_to_class.items(), key=lambda x: x[1])
    ])
    family_map_df.to_csv(manifest_dir / "family_map.csv", index=False)

    # Filter Cohort with Flexible Column Names
    cohort_df = pd.read_csv(cohort_csv)
    sha_col = "sha256" if "sha256" in cohort_df.columns else "sha"
    fam_col = "family_norm" if "family_norm" in cohort_df.columns else "family"

    if sha_col not in cohort_df.columns or fam_col not in cohort_df.columns:
        raise ValueError(
            f"Cohort CSV must contain hash column ('sha256' or 'sha') and family column ('family_norm' or 'family'). "
            f"Found: {list(cohort_df.columns)}"
        )

    cohort_df["sha"] = cohort_df[sha_col].astype(str).str.strip().str.lower()
    cohort_df["family"] = cohort_df[fam_col].astype(str).str.strip().str.lower()

    selected_df = cohort_df[cohort_df["family"].isin(eligible_families)].copy()
    selected_df["class_id"] = selected_df["family"].map(family_to_class)

    # Index Source Files
    source_mode, matches = index_source(source_path)
    selected_df["source_match_count"] = selected_df["sha"].map(lambda s: len(matches.get(s, [])))
    
    valid_samples = selected_df[selected_df["source_match_count"] == 1].copy()
    valid_samples["path"] = valid_samples["sha"].map(lambda s: matches[s][0])

    # --- Trial Run / Limit Filter ---
    if limit is not None and limit > 0:
        valid_samples = valid_samples.iloc[:limit].copy()
        print(f"[TRIAL MODE] Limited cohort execution to first {len(valid_samples)} samples.")

    valid_samples.to_csv(manifest_dir / "selected_cohort.csv", index=False)

    # --- Persistent Log & Resumption Engine ---
    modality_prefix = "_".join(sorted(active_modalities))
    log_file = manifest_dir / f"generation_log_{modality_prefix}.csv"

    completed_shas = set()
    records = []

    if log_file.exists() and not overwrite:
        existing_log = pd.read_csv(log_file)
        finished = existing_log[existing_log["status"].isin(["success", "skipped"])]
        completed_shas = set(finished["sha"].astype(str))
        records = existing_log.to_dict("records")
        print(f"Loaded existing log ({log_file.name}): {len(completed_shas):,} samples already finished.")

    archive = ZipFile(source_path, "r") if source_mode == "zip" else None

    try:
        sample_dicts = valid_samples[["sha", "family", "path"]].to_dict("records")
        total = len(sample_dicts)

        # Interactive Progress Bar Setup
        pbar = tqdm(sample_dicts, total=total, desc="Generating Images", unit="sample")

        for idx, sample in enumerate(pbar, start=1):
            sha, family, rel_path = sample["sha"], sample["family"], sample["path"]

            # Crash Recovery: Skip already completed files
            if sha in completed_shas and not overwrite:
                pbar.set_postfix_str(f"Skipped: {family}/{sha[:8]}")
                continue

            pbar.set_postfix_str(f"Processing: {family}/{sha[:8]}")

            if source_mode == "folder":
                with (source_path / rel_path).open("rb") as stream:
                    rec = process_sample_stream(
                        stream, sha, family, output_dir, active_modalities, output_size, sbsmi_bit_num
                    )
            else:
                assert archive is not None
                with archive.open(rel_path, "r") as stream:
                    rec = process_sample_stream(
                        stream, sha, family, output_dir, active_modalities, output_size, sbsmi_bit_num
                    )

            records.append(rec)
            completed_shas.add(sha)

            if rec["status"] != "success":
                tqdm.write(f"⚠️ Failed [{family}/{sha[:12]}]: {rec['error']}")

            # Periodic log save every 10 samples
            if idx % 10 == 0 or idx == total:
                pd.DataFrame(records).to_csv(log_file, index=False)

    finally:
        if archive is not None:
            archive.close()

    pd.DataFrame(records).to_csv(log_file, index=False)
    print(f"\nProcessing completed for modalities: {sorted(active_modalities)}")


def main():
    parser = argparse.ArgumentParser(description="Distributed, Resumable & Trial-Ready Multi-View Generator.")
    parser.add_argument("--source", type=Path, required=True, help="Path to binary directory or ZIP archive")
    parser.add_argument("--cohort", type=Path, required=True, help="Metadata CSV file (sha/sha256, family/family_norm)")
    parser.add_argument("--eligible-csv", type=Path, default=Path("audit_eligible_families.csv"), help="Eligible families CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/multiview_eligible"), help="Output directory")
    
    parser.add_argument(
        "--modalities", nargs="+", choices=["all", "raw_byte", "entropy", "sbsmi"], default=["all"],
        help="Modality images to generate on this worker machine",
    )
    
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit execution to first N samples for trial runs",
    )
    
    parser.add_argument("--size", type=int, default=64, help="Target image dimension for raw_byte/entropy (default: 64)")
    parser.add_argument("--bit-num", type=int, default=6, help="Bit length for SBSMI states (default: 6 -> 64x64 output image)")
    parser.add_argument("--overwrite", action="store_true", help="Ignore log and overwrite existing files")

    args = parser.parse_args()
    prepare_cohort(
        source_path=args.source,
        eligible_csv=args.eligible_csv,
        cohort_csv=args.cohort,
        output_dir=args.output_dir,
        modalities=args.modalities,
        output_size=(args.size, args.size),
        sbsmi_bit_num=args.bit_num,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()