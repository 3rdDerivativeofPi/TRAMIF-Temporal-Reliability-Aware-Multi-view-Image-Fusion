"""Export historical logits and embeddings from existing TRAMIF final.pt files.

Run from the repository root with python -m scripts.export_temporal_predictions.
Defaults: all three views, five seeds, February and March 2020. No training.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.multiview_contract import MODALITIES, file_sha256, load_mapping, write_json_atomic
from src.data.temporal_multiview_dataset import TemporalMultiViewDataset
from src.training.temporal import _sha_obj, create_branch

HISTORICAL_MONTHS = {
    "train": ("2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01"),
    "validation": ("2020-02", "2020-03"),
}


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(v.strip()) for v in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer seeds") from exc
    if not seeds or min(seeds) < 0 or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be unique non-negative integers")
    return seeds


def select_months(split: str, requested: list[str] | None) -> tuple[str, ...]:
    if split not in HISTORICAL_MONTHS:
        raise ValueError("This exporter accepts only historical train/validation data")
    months = tuple(requested) if requested else HISTORICAL_MONTHS[split]
    if len(months) != len(set(months)) or any(m not in HISTORICAL_MONTHS[split] for m in months):
        raise ValueError(f"Invalid or repeated month for {split}: {months}")
    return months


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "src/models").rglob("*.py")) + [
        Path(__file__).resolve(),
        root / "src/data/multiview_contract.py",
        root / "src/data/temporal_multiview_dataset.py",
        root / "src/training/temporal.py",
    ]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(set(paths))}


def load_final(checkpoint_path: Path, view: str, seed: int, data_identity: dict,
               device: torch.device, plan: dict | None = None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    identity = metadata["identity"]
    if metadata["view"] != view or identity["view"] != view or metadata["seed"] != seed:
        raise ValueError(f"Checkpoint view/seed mismatch: {checkpoint_path}")
    if metadata["signature"] != _sha_obj(identity):
        raise ValueError(f"Invalid checkpoint identity signature: {checkpoint_path}")
    if metadata["num_classes"] != data_identity["num_classes"] or metadata["epochs"] < 1:
        raise ValueError(f"Invalid checkpoint class count/epochs: {checkpoint_path}")
    for key, expected in data_identity.items():
        if identity.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from current dataset: {checkpoint_path}")
    if plan is not None:
        selected = plan["views"][view]
        if (selected["signature"] != metadata["signature"] or
                selected["selected_epoch"] != metadata["epochs"] or seed not in selected["seeds"]):
            raise ValueError(f"Checkpoint does not match the epoch plan: {checkpoint_path}")
    model = create_branch(identity["model"], metadata["num_classes"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).requires_grad_(False).eval()
    return model, metadata


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def export_month(model, dataset, *, view: str, seed: int, month: str, split: str,
                 class_names: list[str], context: dict, output_root: Path,
                 device: torch.device, batch_size: int, num_workers: int,
                 skip_existing: bool = False) -> dict:
    """Rows retain the manifest order; logits are saved before any softmax/scaling."""
    model.requires_grad_(False).eval()
    path = output_root / view / f"seed_{seed}" / f"{month}.npz"
    report_path = path.with_suffix(".json")
    identity = {**context, "view": view, "seed": seed, "month": month, "split": split,
                "batch_size": batch_size, "num_workers": num_workers}
    fingerprint = _sha_obj(identity)
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("fingerprint") != fingerprint:
            raise ValueError(f"Existing export has different inputs/settings; use a new --output-root: {path}")
        if not path.is_file() or report.get("output_sha256") != file_sha256(path):
            raise ValueError(f"Existing export is missing or corrupted: {path}")
        if not skip_existing:
            raise FileExistsError(f"Export already exists; use --skip-existing: {path}")
        print(f"SKIP {view}/seed_{seed}/{month}: {report['n_samples']} samples", flush=True)
        return report
    if path.exists() and not skip_existing:
        raise FileExistsError(f"Incomplete export; use --skip-existing to regenerate: {path}")

    expected_sha = np.asarray([r["sha"] for r in dataset.rows], dtype="U64")
    expected_labels = np.asarray([r["class_id"] for r in dataset.rows], dtype=np.int64)
    if any(r["month"] != month or r["split"] != split for r in dataset.rows):
        raise ValueError("Dataset contains rows outside the requested month/split")
    n = len(dataset)
    classes = len(class_names)
    logits_array = np.empty((n, classes), dtype=np.float32)
    embeddings_array = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    cursor = 0
    for views, labels, shas in loader:
        count = len(labels)
        stop = cursor + count
        if (list(shas) != expected_sha[cursor:stop].tolist() or
                not np.array_equal(labels.numpy(), expected_labels[cursor:stop])):
            raise ValueError("Sample IDs/labels do not match manifest row ordering")
        images = views[view].to(device, dtype=torch.float32, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(images)
        logits, embeddings = output.logits, output.embedding
        if logits.shape != (count, classes) or embeddings.ndim != 2 or embeddings.shape[0] != count:
            raise ValueError("Model returned an invalid logits/embedding shape")
        if embeddings.shape[1] < 1 or not torch.isfinite(logits).all() or not torch.isfinite(embeddings).all():
            raise ValueError("Model returned empty or non-finite output")
        if embeddings_array is None:
            embeddings_array = np.empty((n, embeddings.shape[1]), dtype=np.float32)
        if embeddings.shape[1] != embeddings_array.shape[1]:
            raise ValueError("Embedding dimension changed between batches")
        logits_array[cursor:stop] = logits.float().cpu().numpy()
        embeddings_array[cursor:stop] = embeddings.float().cpu().numpy()
        cursor = stop
    if cursor != n or embeddings_array is None:
        raise ValueError("Export did not cover every selected sample")
    arrays = {"sha": expected_sha, "labels": expected_labels,
              "month": np.full(n, month, dtype="U7"), "logits": logits_array,
              "embeddings": embeddings_array, "class_names": np.asarray(class_names, dtype=str)}
    write_npz_atomic(path, arrays)
    report = {"fingerprint": fingerprint, "identity": identity, "n_samples": n,
              "num_classes": classes, "embedding_dim": int(embeddings_array.shape[1]),
              "class_counts": np.bincount(expected_labels, minlength=classes).tolist(),
              "output_sha256": file_sha256(path),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json_atomic(report_path, report)
    print(f"DONE {view}/seed_{seed}/{month}: {n} samples; "
          f"logits={logits_array.shape}, embeddings={embeddings_array.shape}", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_temporal.yaml"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs/temporal"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/exports"))
    parser.add_argument("--views", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("11,22,33,44,55"))
    parser.add_argument("--split", choices=tuple(HISTORICAL_MONTHS), default="validation")
    parser.add_argument("--months", nargs="+")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    months = select_months(args.split, args.months)
    if args.batch_size < 1 or args.num_workers < 0 or len(args.views) != len(set(args.views)):
        parser.error("Require positive batch size, non-negative workers, and unique views")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this Python environment")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root, manifest, mapping_path = (Path(cfg["dataset"][k]).resolve()
                                    for k in ("root", "manifest", "mapping"))
    data_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    mapping = load_mapping(mapping_path)
    names = [name for name, index in sorted(mapping.items(), key=lambda item: item[1])]
    data_identity = {"manifest_sha256": file_sha256(manifest),
                     "mapping_sha256": file_sha256(mapping_path),
                     "preprocessing_config_id": _sha_obj(data_config), "num_classes": len(names)}
    plan_path = args.runs_root / "epoch_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    base_context = {"export_schema": 1, "data": data_identity, "source_sha256": source_hashes(),
                    "torch_version": str(torch.__version__), "device": str(device),
                    "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                    "precision": "float32 tensors; autocast disabled", "temperature": 1.0,
                    "model_mode": "eval", "gradients": False,
                    "epoch_plan_sha256": file_sha256(plan_path) if plan is not None else None}
    total = 0
    for view in args.views:
        datasets = {month: TemporalMultiViewDataset(root, manifest, mapping_path, split=args.split,
                    month=month, modalities=(view,), return_sha=True) for month in months}
        reference = None
        for seed in args.seeds:
            checkpoint_path = args.runs_root / view / "final" / f"seed_{seed}" / "final.pt"
            checkpoint_hash = file_sha256(checkpoint_path)
            model, metadata = load_final(checkpoint_path, view, seed, data_identity, device, plan)
            current = (metadata["signature"], metadata["epochs"])
            if reference is not None and current != reference:
                raise ValueError(f"Final seeds for {view} use different configurations/epoch counts")
            reference = current
            context = {**base_context, "checkpoint_sha256": checkpoint_hash,
                       "training_signature": metadata["signature"], "epochs": metadata["epochs"],
                       "model_spec": metadata["identity"]["model"]}
            for month, dataset in datasets.items():
                export_month(model, dataset, view=view, seed=seed, month=month, split=args.split,
                             class_names=names, context=context, output_root=args.output_root,
                             device=device, batch_size=args.batch_size, num_workers=args.num_workers,
                             skip_existing=args.skip_existing)
                total += 1
            del model
    print(f"Finished: {total} monthly exports ready under {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
