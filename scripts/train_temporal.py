"""Train the chronological TRAMIF classifiers without touching validation/test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.multiview_contract import MODALITIES, file_sha256, load_manifest, load_mapping
from src.training.temporal import experiment_identity, run_seed, validate_settings


def _seed_list(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds expects integers separated by commas") from exc
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique non-negative integers")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_temporal.yaml"))
    parser.add_argument("--phase", choices=("pilot", "final"), required=True)
    parser.add_argument("--view", choices=MODALITIES, required=True)
    parser.add_argument("--seeds", type=_seed_list, default=_seed_list("11,22,33,44,55"))
    parser.add_argument("--epoch-plan", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("runs/temporal"))
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping")
    settings = validate_settings(cfg["training"])
    root, manifest, mapping = (Path(cfg["dataset"][key]).resolve()
                              for key in ("root", "manifest", "mapping"))
    data_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if data_config.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Manifest does not match the generated image dataset")
    if data_config.get("mapping_sha256") != file_sha256(mapping):
        raise ValueError("Mapping does not match the generated image dataset")
    rows = load_manifest(manifest, load_mapping(mapping))
    train_rows = [r for r in rows if r["split"] == "train"]
    fit_rows = [r for r in train_rows if r["month"] <= "2019-11"]
    held_rows = [r for r in train_rows if r["month"] >= "2019-12"]
    if not train_rows or not fit_rows or not held_rows:
        raise ValueError("Missing chronological training/pilot data")
    if {r["month"] for r in held_rows} != {"2019-12", "2020-01"}:
        raise ValueError("Pilot requires both 2019-12 and 2020-01")
    identity = experiment_identity(cfg, args.view, data_config, manifest, mapping)

    if args.phase == "pilot":
        epochs = settings["epochs"]
        if args.epoch_plan:
            parser.error("--epoch-plan is only used for final training")
    else:
        if args.epoch_plan is None:
            parser.error("Final training requires --epoch-plan from completed pilot runs")
        plan = json.loads(args.epoch_plan.read_text(encoding="utf-8"))
        view_plan = plan["views"][args.view]
        from src.training.temporal import _sha_obj
        if view_plan["signature"] != _sha_obj(identity):
            raise ValueError("Epoch plan does not match the current dataset/model/settings")
        if sorted(args.seeds) != sorted(view_plan["seeds"]):
            raise ValueError("Final seeds must match the seeds used to select the epoch")
        epochs = int(view_plan["selected_epoch"])

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    print(f"phase={args.phase} view={args.view} device={device} epochs={epochs} "
          f"seeds={args.seeds} train={len(train_rows)} "
          f"pilot_fit={len(fit_rows)} pilot_holdout={len(held_rows)}", flush=True)
    if args.dry_run:
        print("Validated manifest, class mapping, chronology and preprocessing hashes. "
              "Images have not been opened.")
        return
    for seed in args.seeds:
        run_seed(cfg=cfg, view=args.view, seed=seed, phase=args.phase, epochs=epochs,
                 output_root=args.output_root, device=device, skip_completed=args.skip_completed)


if __name__ == "__main__":
    main()
