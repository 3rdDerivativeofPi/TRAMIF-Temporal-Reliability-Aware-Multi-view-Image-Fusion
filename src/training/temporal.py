"""Single-view chronological training for the frozen TRAMIF V2 image contract.

The MalCSBSV 10-fold reproduction in reproduction.py is a separate experiment.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset

from src.data.multiview_contract import MODALITIES, file_sha256
from src.data.temporal_multiview_dataset import TemporalMultiViewDataset
from src.models.branch import MalwareImageBranch

PILOT_FIT_END = "2019-11"
PILOT_VAL_START = "2019-12"
PILOT_MONTHS = ("2019-12", "2020-01")


def _json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".tmp-", suffix=".json", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def _torch_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", suffix=".pt",
                                     delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".tmp-", suffix=".jsonl", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    os.replace(tmp, path)


def _sha_obj(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_settings(settings: dict) -> dict:
    required = ("batch_size", "epochs", "learning_rate", "weight_decay",
                "num_workers", "grad_accumulation", "grad_clip", "amp", "deterministic")
    missing = set(required) - set(settings)
    if missing:
        raise ValueError(f"Missing training settings: {sorted(missing)}")
    for key in ("batch_size", "epochs", "grad_accumulation"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(settings["num_workers"]) is not int or settings["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    for key in ("learning_rate", "weight_decay", "grad_clip"):
        v = settings[key]
        if type(v) not in (int, float) or not np.isfinite(v) or v < 0:
            raise ValueError(f"{key} must be a finite non-negative number")
    if settings["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    for key in ("amp", "deterministic"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    return settings


def experiment_identity(config: dict, view: str, data_config: dict, manifest: Path,
                        mapping: Path) -> dict:
    model_config = config["models"][view]
    return {
        "view": view,
        "model": model_config,
        "training": validate_settings(config["training"]),
        "manifest_sha256": file_sha256(manifest),
        "mapping_sha256": file_sha256(mapping),
        "preprocessing_config_id": _sha_obj(data_config),
        "num_classes": 51,
        "pilot_fit_through": PILOT_FIT_END,
        "pilot_validation_months": list(PILOT_MONTHS),
    }


class SafeShuffleBatchSampler(Sampler[list[int]]):
    """Fixed per-epoch permutation, with no singleton batch for BatchNorm1d."""

    def __init__(self, size: int, batch_size: int, seed: int):
        if size < 2 or batch_size < 2:
            raise ValueError("Training requires at least two samples and batch_size >= 2")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch = 0

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch * 100003)
        indices = torch.randperm(self.size, generator=gen).tolist()
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self):
        n, r = divmod(self.size, self.batch_size)
        return n + (1 if r and r != 1 else 0)


def seed_all(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def create_branch(spec: dict, classes: int) -> MalwareImageBranch:
    name = spec["class_path"]
    if ":" not in name:
        raise ValueError("class_path must look like src.models.malsbslcnet:MalSBSLCNet")
    module_name, class_name = name.split(":", 1)
    klass = getattr(importlib.import_module(module_name), class_name)
    branch = klass(num_classes=classes, **spec.get("kwargs", {}))
    if not isinstance(branch, MalwareImageBranch):
        raise TypeError("Model must inherit MalwareImageBranch")
    return branch


def _branch_logits(output, labels, num_classes: int) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    embedding = getattr(output, "embedding", None)
    if not isinstance(logits, torch.Tensor) or logits.shape != (len(labels), num_classes):
        raise ValueError("Branch must return logits of shape [batch, num_classes]")
    if not isinstance(embedding, torch.Tensor) or embedding.shape[0] != len(labels):
        raise ValueError("Branch must return an embedding for each sample")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits")
    return logits


def _amp_options(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return False, torch.float32
    return True, torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _unpack(batch, view: str, device: torch.device):
    views, labels = batch
    return views[view].to(device, non_blocking=True), labels.to(device, non_blocking=True)


def _train_epoch(model, loader, sampler, view, optimizer, scaler, settings, device, epoch):
    model.train()
    sampler.epoch = epoch
    optimizer.zero_grad(set_to_none=True)
    total_loss, correct, n, pending = 0.0, 0, 0, 0
    amp, dtype = _amp_options(device, settings["amp"])
    for batch_idx, batch in enumerate(loader):
        images, labels = _unpack(batch, view, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            logits = _branch_logits(model(images), labels, model.num_classes)
            loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        batch_n = len(labels)
        scaler.scale(loss * batch_n).backward()
        total_loss += float(loss.detach()) * batch_n
        correct += int((logits.detach().argmax(1) == labels).sum())
        n += batch_n
        pending += batch_n
        if (batch_idx + 1) % settings["grad_accumulation"] == 0 or batch_idx + 1 == len(loader):
            scaler.unscale_(optimizer)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(pending)
            if settings["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), settings["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
    return {"train_loss": total_loss / n, "train_accuracy": correct / n}


@torch.inference_mode()
def _validate(model, loader, view, device):
    model.eval()
    labels_by_month, preds_by_month = defaultdict(list), defaultdict(list)
    loss_sum, sample_count = 0.0, 0
    amp, dtype = _amp_options(device, True)
    for views, labels, months in loader:
        images = views[view].to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            logits = _branch_logits(model(images), labels, model.num_classes)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss)
        sample_count += len(labels)
        for month, y, pred in zip(months, labels.cpu().tolist(), logits.argmax(1).cpu().tolist()):
            labels_by_month[month].append(y)
            preds_by_month[month].append(pred)
    if not sample_count or tuple(sorted(labels_by_month)) != PILOT_MONTHS:
        raise ValueError("Both pilot validation months must contain images")
    scores = {month: float(f1_score(labels_by_month[month], preds_by_month[month],
                                   labels=list(range(model.num_classes)),
                                   average="macro", zero_division=0))
              for month in PILOT_MONTHS}
    return {"val_loss": loss_sum / sample_count,
            "mean_month_macro_f1": float(np.mean(list(scores.values()))),
            "worst_month_macro_f1": float(min(scores.values())),
            "month_macro_f1": scores}


class MonthDataset(torch.utils.data.Dataset):
    def __init__(self, source, indices):
        self.source = source
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        views, label = self.source[true_idx]
        return views, label, self.source.rows[true_idx]["month"]


def make_datasets(root: Path, manifest: Path, mapping: Path, view: str, phase: str):
    source = TemporalMultiViewDataset(root, manifest, mapping, split="train", modalities=(view,))
    if phase == "final":
        return source, None
    fit_idx = [i for i, r in enumerate(source.rows) if r["month"] <= PILOT_FIT_END]
    val_idx = [i for i, r in enumerate(source.rows) if r["month"] >= PILOT_VAL_START]
    if not fit_idx or not val_idx:
        raise ValueError("Empty pilot fit/holdout")
    months = {source.rows[i]["month"] for i in val_idx}
    if months != set(PILOT_MONTHS):
        raise ValueError(f"Pilot requires validation months {PILOT_MONTHS}, got {sorted(months)}")
    return Subset(source, fit_idx), MonthDataset(source, val_idx)


def run_seed(*, cfg: dict, view: str, seed: int, phase: str, epochs: int,
             output_root: Path, device: torch.device, skip_completed: bool = False) -> dict:
    if phase not in {"pilot", "final"} or view not in MODALITIES:
        raise ValueError("Invalid phase or view")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be positive")
    dataset_cfg = cfg["dataset"]
    root, manifest, mapping = (Path(dataset_cfg[k]).resolve()
                              for k in ("root", "manifest", "mapping"))
    settings = validate_settings(cfg["training"])
    if view not in cfg["models"]:
        raise ValueError(f"Missing model configuration for {view}")
    data_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    identity = experiment_identity(cfg, view, data_config, manifest, mapping)
    signature = _sha_obj(identity)
    run_dir = output_root / view / phase / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path, last_path = run_dir / "summary.json", run_dir / "last.pt"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("signature") != signature or summary.get("epochs") != epochs:
            raise ValueError(f"Existing run incompatible with config: {run_dir}")
        if skip_completed:
            files = [run_dir / "metadata.json", run_dir / "history.jsonl", last_path]
            if phase == "final":
                files.append(run_dir / "final.pt")
            missing = [str(path) for path in files if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Completed run is missing artifacts: {missing}")
            return summary
        raise FileExistsError(f"Run already completed: {run_dir}; use --skip-completed")
    fit, holdout = make_datasets(root, manifest, mapping, view, phase)
    if len(fit) < 2:
        raise ValueError("Need at least 2 training samples")
    seed_all(seed, settings["deterministic"])
    branch = create_branch(cfg["models"][view], 51).to(device)
    sampler = SafeShuffleBatchSampler(len(fit), settings["batch_size"], seed)
    opts = {"num_workers": settings["num_workers"], "pin_memory": device.type == "cuda",
            "persistent_workers": False}
    if settings["num_workers"]:
        opts["prefetch_factor"] = 2
    loader = DataLoader(fit, batch_sampler=sampler, **opts)
    val_loader = None
    if holdout is not None:
        val_loader = DataLoader(holdout, batch_size=settings["batch_size"], shuffle=False, **opts)
    optimizer = torch.optim.AdamW(branch.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp, dtype = _amp_options(device, settings["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    history, start = [], 0
    if last_path.exists():
        # Trusted local run artifact only; PyTorch checkpoints use pickle.
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["epochs"] != epochs or checkpoint["seed"] != seed:
            raise ValueError(f"Resume checkpoint differs from requested run: {last_path}")
        branch.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_rng(checkpoint["rng"])
        history = checkpoint["history"]
        start = checkpoint["epoch"] + 1
        _write_history(run_dir / "history.jsonl", history)
    _json_atomic(run_dir / "metadata.json", {"signature": signature, "identity": identity,
                                             "seed": seed, "phase": phase, "epochs": epochs,
                                             "torch": torch.__version__})
    for epoch in range(start, epochs):
        row = {"epoch": epoch + 1}
        row.update(_train_epoch(branch, loader, sampler, view, optimizer, scaler, settings, device, epoch))
        if val_loader is not None:
            row.update(_validate(branch, val_loader, view, device))
        scheduler.step()
        row["learning_rate_after_step"] = scheduler.get_last_lr()[0]
        history.append(row)
        _torch_atomic(last_path, {"model": branch.state_dict(), "optimizer": optimizer.state_dict(),
                                  "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                                  "rng": _rng_state(), "history": history, "epoch": epoch,
                                  "signature": signature, "epochs": epochs, "seed": seed})
        _write_history(run_dir / "history.jsonl", history)
        print(f"[{view}/{phase}/seed={seed}] epoch {epoch + 1}/{epochs}: "
              f"loss={row['train_loss']:.4f}" +
              (f" val_month_F1={row['mean_month_macro_f1']:.4f}" if val_loader else ""), flush=True)
    if phase == "final":
        _torch_atomic(run_dir / "final.pt", {"model": branch.state_dict(),
                    "metadata": {"identity": identity, "signature": signature, "seed": seed,
                                 "epochs": epochs, "view": view, "num_classes": 51}})
    summary = {"signature": signature, "epochs": epochs, "seed": seed,
               "phase": phase, "view": view, "completed": True,
               "samples_fit": len(fit), "samples_validation": len(holdout) if holdout else 0}
    _json_atomic(summary_path, summary)
    return summary
