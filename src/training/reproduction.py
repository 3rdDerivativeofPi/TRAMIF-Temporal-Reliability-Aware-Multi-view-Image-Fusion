"""Single-fold MalCSBSV reproduction using ReproductionDataModule.

Paper Section 5.1 / Table 3: Adam, lr=1e-4, weight decay=1e-6,
batch size 32, maximum 100 epochs. This runner explicitly uses the final
epoch, without validation-based checkpoint selection. Run all outer folds
with fresh models for a complete CV experiment; this is not TRAMIF's
chronological experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.nn import functional as F

from src.data.reproduction_datamodule import ReproductionDataModule
from src.models.malsbslcnet import MalSBSLCNet
from src.training.lightning_classifier import LightningMalwareClassifier


def _wrap_network(network: MalSBSLCNet) -> LightningMalwareClassifier:
    return LightningMalwareClassifier(
        model=network,
        learning_rate=1e-4,
        weight_decay=1e-6,
    )


def make_malsbslcnet(*, num_classes: int) -> LightningMalwareClassifier:
    """Preserve the original model-factory interface."""
    return _wrap_network(MalSBSLCNet(num_classes=num_classes))


def _save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _summarize_metrics(targets, predictions, nll_sum, num_classes) -> dict:
    if not targets:
        raise ValueError("The test fold is empty.")
    return {
        "samples": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets, predictions, labels=list(range(num_classes)),
            average="macro", zero_division=0,
        )),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "uncalibrated_nll": float(nll_sum / len(targets)),
    }


def _extract_logits(output):
    """Accept a logits tensor or the project's BranchOutput.logits."""
    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        fields = sorted(vars(output)) if hasattr(output, "__dict__") else []
        raise TypeError(
            f"Expected a tensor or an object with a tensor .logits field; "
            f"got {type(output).__name__} with fields {fields}. "
            "Check the BranchOutput definition for the classification-logits field."
        )
    return logits


def evaluate_test_fold(network, datamodule) -> dict:
    """Evaluate the trained network without requiring a wrapper test_step.

    Accepts unnormalized logits directly or in a BranchOutput.logits field.
    Metrics are sample-weighted across all batches, including the final batch.
    """
    network.eval()
    device = next(network.parameters()).device
    num_classes = datamodule.num_classes
    targets, predictions = [], []
    nll_sum = 0.0

    with torch.inference_mode():
        for images, labels in datamodule.test_dataloader():
            images = images.to(device)
            labels = labels.to(device)
            logits = _extract_logits(network(images))
            if logits.shape != (labels.shape[0], num_classes):
                raise ValueError("Expected logits with shape [batch_size, num_classes].")
            if not torch.isfinite(logits).all():
                raise ValueError("Test logits contain NaN or infinity.")
            nll_sum += F.cross_entropy(logits, labels, reduction="sum").item()
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return _summarize_metrics(targets, predictions, nll_sum, num_classes)


def run_reproduction_fold(
    manifest: str | Path,
    fold_id: int,
    *,
    image_root: str | Path = ".",
    image_column: str = "image_path",
    output_dir: str | Path = "runs/reproduction",
    n_splits: int = 10,
    epochs: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    accelerator: str = "auto",
    scale_to_unit: bool = True,
) -> dict:
    """Train a fresh model on the other folds, then test fold_id once.

    The existing LightningMalwareClassifier must implement training_step and
    configure_optimizers for the supplied network. Its optimizer/loss code
    must still be checked against the paper; this file cannot establish that.
    """
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    pl.seed_everything(seed, workers=True)
    datamodule = ReproductionDataModule(
        manifest=manifest, fold_id=fold_id, image_root=image_root,
        image_column=image_column, n_splits=n_splits,
        batch_size=batch_size, num_workers=num_workers, seed=seed,
        pin_memory=accelerator != "cpu" and torch.cuda.is_available(),
        scale_to_unit=scale_to_unit,
    )
    datamodule.setup("fit")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir).resolve() / f"fold_{fold_id}_seed_{seed}_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = run_dir / "fold_manifest.csv"
    datamodule.manifest.to_csv(snapshot, index=False)
    _save_json(run_dir / "fold_audit.json", datamodule.audit_report)
    config = {
        "track": "MalCSBSV cross-validation reproduction",
        "manifest": str(Path(manifest).resolve()),
        "manifest_snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "fold_id": fold_id, "n_splits": n_splits,
        "num_classes": datamodule.num_classes, "seed": seed,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": 1e-4, "weight_decay": 1e-6,
        "num_workers": num_workers, "accelerator": accelerator,
        "deterministic": "warn", "cudnn_benchmark": False,
        "image_root": str(Path(image_root).expanduser().resolve()),
        "image_column": image_column, "scale_to_unit": scale_to_unit,
        "checkpoint_policy": "final epoch; no held-out-fold selection",
        "versions": {"torch": torch.__version__, "lightning": pl.__version__,
                     "numpy": np.__version__},
    }
    _save_json(run_dir / "config.json", config)

    network = MalSBSLCNet(num_classes=datamodule.num_classes)
    classifier = _wrap_network(network)
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, min_epochs=epochs, max_epochs=epochs,
        # Keep deterministic algorithms where available; warn for CUDA ops
        # without a deterministic implementation (e.g. adaptive pool backward).
        deterministic="warn", benchmark=False, precision="32-true",
        default_root_dir=str(run_dir),
        logger=CSVLogger(save_dir=str(run_dir), name="training", version=0),
        enable_checkpointing=False, num_sanity_val_steps=0, limit_val_batches=0,
    )
    # Apply after Trainer construction, which can set global determinism.
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("Runner file:", Path(__file__).resolve())
    print("Deterministic:", torch.are_deterministic_algorithms_enabled())
    print("Warn only:", torch.is_deterministic_algorithms_warn_only_enabled())
    started = perf_counter()
    trainer.fit(classifier, datamodule=datamodule)
    training_seconds = perf_counter() - started
    if trainer.interrupted:
        raise RuntimeError("Training was interrupted; no final-fold score was produced.")
    checkpoint = run_dir / "final.ckpt"
    trainer.save_checkpoint(str(checkpoint))

    started = perf_counter()
    metrics = evaluate_test_fold(network, datamodule)
    result = {
        "fold_id": fold_id, "seed": seed, "metrics": metrics,
        "training_seconds": training_seconds,
        "evaluation_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint), "run_dir": str(run_dir),
        "complete_cv_experiment": False,
    }
    _save_json(run_dir / "test_metrics.json", result)
    return result


def evaluate_saved_fold(run_dir: str | Path, *, accelerator: str = "auto") -> dict:
    """Recover evaluation after training, using this runner's own checkpoint.

    Read the saved configuration and exact manifest snapshot. Never call fit
    or change the fold, class mapping, scaling, or trained parameters.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    snapshot = run_dir / "fold_manifest.csv"
    checkpoint_path = run_dir / "final.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No saved final checkpoint: {checkpoint_path}")
    snapshot_sha = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    if snapshot_sha != config["manifest_snapshot_sha256"]:
        raise ValueError("The saved manifest snapshot has changed; evaluation stopped.")
    if accelerator not in {"auto", "cpu", "gpu"}:
        raise ValueError("accelerator must be auto, cpu or gpu.")
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --accelerator cpu.")
    use_cuda = accelerator != "cpu" and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    pl.seed_everything(config["seed"], workers=True)
    datamodule = ReproductionDataModule(
        manifest=snapshot, fold_id=config["fold_id"],
        image_root=config["image_root"], image_column=config["image_column"],
        n_splits=config["n_splits"], batch_size=config["batch_size"],
        num_workers=config["num_workers"], seed=config["seed"],
        pin_memory=use_cuda, scale_to_unit=config["scale_to_unit"],
    )
    datamodule.setup("test")
    if datamodule.num_classes != config["num_classes"]:
        raise ValueError("Class count differs from the saved training configuration.")
    network = MalSBSLCNet(num_classes=datamodule.num_classes)
    classifier = _wrap_network(network)
    # Full Lightning checkpoint from this runner's own completed training run.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classifier.load_state_dict(checkpoint["state_dict"], strict=True)
    classifier.to(device)

    started = perf_counter()
    metrics = evaluate_test_fold(network, datamodule)
    result = {
        "fold_id": config["fold_id"], "seed": config["seed"],
        "metrics": metrics, "evaluation_only": True,
        "training_seconds": None,
        "evaluation_seconds": perf_counter() - started,
        "checkpoint": str(checkpoint_path), "run_dir": str(run_dir),
        "complete_cv_experiment": False,
    }
    report_path = run_dir / "test_metrics.json"
    if report_path.exists():
        report_path = run_dir / f"test_metrics_recheck_{uuid4().hex[:8]}.json"
    result["report_path"] = str(report_path)
    _save_json(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fold-id", type=int)
    parser.add_argument("--evaluate-only", type=Path, metavar="RUN_DIR",
                        help="Evaluate RUN_DIR/final.ckpt using its saved configuration; skip training.")
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reproduction"))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--no-scale-to-unit", action="store_true")
    args = vars(parser.parse_args())
    saved_run = args.pop("evaluate_only")
    if saved_run is not None:
        if args["manifest"] is not None or args["fold_id"] is not None:
            parser.error("Evaluation-only reads the saved manifest/fold; omit --manifest and --fold-id.")
        print(json.dumps(evaluate_saved_fold(saved_run, accelerator=args["accelerator"]), indent=2))
        return
    if args["manifest"] is None or args["fold_id"] is None:
        parser.error("Training requires --manifest and --fold-id.")
    args["scale_to_unit"] = not args.pop("no_scale_to_unit")
    print(json.dumps(run_reproduction_fold(**args), indent=2))


if __name__ == "__main__":
    main()
