"""Sequential GPU launcher for the existing single-fold reproduction runner.

Run from the repository root. Default: at most ten unfinished jobs.
Uses only Python's standard library; starts a fresh Python process per fold.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def manifest_info(path):
    """Compare CSV content despite harmless changes in quoting/line endings."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"sha", "class_id", "test_fold"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing {sorted(required)} columns: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    rows.sort(key=lambda row: row["sha"])
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    counts = {fold: 0 for fold in range(10)}
    for row in rows:
        fold = int(row["test_fold"])
        if fold not in counts:
            raise ValueError(f"Expected test_fold IDs 0..9: {path}")
        counts[fold] += 1
    if not all(counts.values()):
        raise ValueError(f"Every test fold must contain samples: {path}")
    return hashlib.sha256(encoded).hexdigest(), counts


def find_saved_run(output_dir, fold, seed, digest, sample_count, args):
    """Reuse only runs matching the current data and requested configuration."""
    pending = None
    for folder in sorted(output_dir.glob(f"fold_{fold}_seed_{seed}_*")):
        config = read_json(folder / "config.json")
        if not config:
            continue
        expected = {
            "fold_id": fold, "seed": seed, "n_splits": 10,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "scale_to_unit": True,
        }
        if any(config.get(key) != value for key, value in expected.items()):
            continue
        if config.get("accelerator") not in {"gpu", "auto"}:
            continue
        if "+cu" not in config.get("versions", {}).get("torch", ""):
            continue
        if Path(config.get("image_root", "")).resolve() != args.image_root:
            continue
        snapshot = folder / "fold_manifest.csv"
        checkpoint = folder / "final.ckpt"
        if not snapshot.is_file() or not checkpoint.is_file():
            continue
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != config.get("manifest_snapshot_sha256"):
            continue
        try:
            if manifest_info(snapshot)[0] != digest:
                continue
        except (OSError, ValueError):
            continue
        report = read_json(folder / "test_metrics.json")
        if (report and report.get("fold_id") == fold and report.get("seed") == seed
                and report.get("metrics", {}).get("samples") == sample_count
                and all(key in report["metrics"] for key in
                        ["accuracy", "macro_f1", "balanced_accuracy", "uncalibrated_nll"])):
            return "complete", folder
        if pending is None:
            pending = folder
    return ("evaluate", pending) if pending else ("train", None)


@contextmanager
def keep_windows_awake():
    """Temporarily block idle sleep, while still allowing the screen to turn off."""
    if os.name != "nt":
        yield
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    set_state = kernel.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_uint
    continuous = 0x80000000
    if not set_state(continuous | 0x00000001):
        raise OSError("Could not request wakefulness. Set plugged-in sleep to Never and retry.")
    try:
        yield
    finally:
        set_state(continuous)


def run_queue(args):
    launched = 0
    skipped = 0
    for repeat in range(args.first_repeat, args.last_repeat + 1):
        seed = 41 + repeat
        manifest = args.manifest_dir / f"folds_repeat_{repeat:02d}_seed_{seed}.csv"
        digest, counts = manifest_info(manifest)
        for fold in range(10):
            status, folder = find_saved_run(
                args.output_dir, fold, seed, digest, counts[fold], args)
            label = f"repeat {repeat:02d}, seed {seed}, fold {fold}"
            if status == "complete":
                skipped += 1
                print(f"SKIP completed {label}: {folder}", flush=True)
                continue
            if args.max_runs and launched >= args.max_runs:
                print(f"Reached limit: {launched} jobs; {skipped} already completed.", flush=True)
                return 0
            command = [sys.executable, "-u", "-m", "src.training.reproduction"]
            if status == "evaluate":
                command += ["--evaluate-only", str(folder), "--accelerator", "gpu"]
            else:
                command += [
                    "--manifest", str(manifest), "--image-root", str(args.image_root),
                    "--fold-id", str(fold), "--seed", str(seed),
                    "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers),
                    "--output-dir", str(args.output_dir), "--accelerator", "gpu",
                ]
            launched += 1
            print(f"START {launched}: {status.upper()} {label}", flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
                continue
            logs = args.output_dir / "overnight_logs"
            logs.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = logs / f"repeat_{repeat:02d}_fold_{fold}_{stamp}_{uuid4().hex[:8]}.log"
            print(f"Log: {log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                log.write(subprocess.list2cmdline(command) + "\n")
                log.flush()
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f"STOP: exit code {result.returncode}. Read {log_path}", flush=True)
                return result.returncode
            status_after, _ = find_saved_run(
                args.output_dir, fold, seed, digest, counts[fold], args)
            if status_after != "complete":
                print(f"STOP: process finished but matching results are missing. Read {log_path}", flush=True)
                return 1
            print(f"DONE {label}", flush=True)
    print(f"Queue finished: {launched} jobs; {skipped} already completed.", flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--first-repeat", type=int, default=1)
    parser.add_argument("--last-repeat", type=int, default=10)
    parser.add_argument("--max-runs", type=int, default=10, help="Maximum unfinished jobs; 0 means all.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print queued commands without launching them.")
    args = parser.parse_args()
    if not 1 <= args.first_repeat <= args.last_repeat <= 10:
        parser.error("Repeat range must lie within 1..10.")
    if args.max_runs < 0 or args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid run limit, epoch count, batch size or worker count.")
    for name in ["manifest_dir", "image_root", "output_dir"]:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if not Path("src/training/reproduction.py").is_file():
        parser.error("Run this launcher from the repository root.")
    if not args.image_root.is_dir():
        parser.error("--image-root must be an existing directory.")
    for repeat in range(args.first_repeat, args.last_repeat + 1):
        manifest_info(args.manifest_dir / f"folds_repeat_{repeat:02d}_seed_{41 + repeat}.csv")
    if args.dry_run:
        return run_queue(args)
    check = subprocess.run([
        sys.executable, "-c",
        "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in this Python environment'",
    ])
    if check.returncode:
        return check.returncode
    with keep_windows_awake():
        return run_queue(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped. Restart the same command to skip completed jobs.", flush=True)
        raise SystemExit(130)
