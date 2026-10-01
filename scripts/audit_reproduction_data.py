"""Run via python -m scripts.audit_reproduction_data from the project root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.reproduction import make_reproduction_dataset, audit_reproduction_folds
from src.data.reproduction_datamodule import ReproductionDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--check-all-images", action="store_true")
    parser.add_argument("--raw-pixels", action="store_true", help="Use float32 [0,255] instead of [0,1]")
    parser.add_argument("--output", type=Path, default=Path("reports/reproduction_data_audit.json"))
    args = parser.parse_args()
    report = audit_reproduction_folds(args.manifest, n_splits=args.n_splits,
                                     image_root=args.image_root, image_column=args.image_column)
    kwargs = dict(image_root=args.image_root, image_column=args.image_column,
                  scale_to_unit=not args.raw_pixels)
    checked = 0
    if args.check_all_images:
        dataset = make_reproduction_dataset(args.manifest, **kwargs)
        for index in range(len(dataset)):
            dataset[index]
            checked += 1
    report["all_images_decoded"] = args.check_all_images
    report["decoded_image_count"] = checked
    for fold in report["folds"]:
        dm = ReproductionDataModule(args.manifest, fold["fold_id"], n_splits=args.n_splits,
                                    batch_size=args.batch_size, num_workers=args.num_workers, **kwargs)
        shapes = {}
        for split, loader in [("train", dm.train_dataloader()), ("test", dm.test_dataloader())]:
            images, labels = next(iter(loader))
            if images.shape[1:] != (1, 64, 64) or images.dtype != torch.float32 or labels.dtype != torch.int64:
                raise RuntimeError(f"Invalid {split} batch contract in fold {fold['fold_id']}")
            shapes[split] = list(images.shape)
        fold["first_batch_shapes"] = shapes
        print(f"Fold {fold['fold_id']}: train={fold['train_samples']}, test={fold['test_samples']}, overlap=0", flush=True)
    report["status"] = "passed"
    report["scaling"] = "float32 [0,255]" if args.raw_pixels else "float32 [0,1]"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Audit saved: {args.output}")


if __name__ == "__main__":
    main()
