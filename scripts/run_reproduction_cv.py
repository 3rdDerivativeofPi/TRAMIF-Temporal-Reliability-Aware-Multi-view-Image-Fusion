"""Fixed-epoch CV runner; factory must return a NEW LightningModule each call."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import lightning.pytorch as pl

from src.data.reproduction_datamodule import ReproductionDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--model-factory", required=True, help="module:function accepting num_classes=...")
    parser.add_argument("--epochs", type=int, required=True, help="Predeclared fixed epoch budget; never tune on outer test")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--raw-pixels", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/reproduction_cv"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    module, separator, name = args.model_factory.partition(":")
    if not separator or not module or not name:
        parser.error("--model-factory must have module:function format")
    factory = getattr(importlib.import_module(module), name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for fold_id in range(args.n_splits):
        pl.seed_everything(args.seed + fold_id, workers=True)
        dm = ReproductionDataModule(
            args.manifest, fold_id, image_root=args.image_root, image_column=args.image_column,
            n_splits=args.n_splits, batch_size=args.batch_size, num_workers=args.num_workers,
            seed=args.seed, scale_to_unit=not args.raw_pixels)
        dm.setup("fit")
        model = factory(num_classes=dm.num_classes)
        if not isinstance(model, pl.LightningModule):
            raise TypeError("Factory must return lightning.pytorch.LightningModule")
        # New model, optimizer (via configure_optimizers), and Trainer per fold.
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                             logger=False, enable_checkpointing=False, limit_val_batches=0,
                             num_sanity_val_steps=0, default_root_dir=str(args.output_dir / f"fold_{fold_id}"))
        trainer.fit(model, datamodule=dm)
        metrics = trainer.test(model, datamodule=dm, ckpt_path=None)
        result = {"fold_id": fold_id, "epochs": args.epochs, "seed": args.seed + fold_id,
                  "split_audit": dm.audit_report["folds"][fold_id], "metrics": metrics}
        (args.output_dir / f"fold_{fold_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        results.append(result)
        del trainer, model, dm
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
