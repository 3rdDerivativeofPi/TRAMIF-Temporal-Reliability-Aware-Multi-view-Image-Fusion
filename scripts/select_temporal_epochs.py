"""Choose one epoch count per view from training-period temporal pilot runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data.multiview_contract import MODALITIES, write_json_atomic


def choose_epochs(root: Path, views: tuple[str, ...], seeds: list[int], metric: str) -> dict:
    plan: dict = {"metric": metric, "views": {}}
    for view in views:
        runs = []
        signature = None
        epochs = None
        for seed in seeds:
            folder = root / view / "pilot" / f"seed_{seed}"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            history = [json.loads(line) for line in (folder / "history.jsonl").read_text(
                encoding="utf-8").splitlines()]
            if not summary["completed"] or summary["phase"] != "pilot" or summary["seed"] != seed:
                raise ValueError(f"Incomplete/wrong pilot run in {folder}")
            if metadata["phase"] != "pilot" or metadata["seed"] != seed:
                raise ValueError(f"Invalid pilot metadata in {folder}")
            if signature is None:
                signature, epochs = summary["signature"], summary["epochs"]
            if (summary["signature"] != signature or metadata["signature"] != signature or
                    summary["epochs"] != epochs or len(history) != epochs or
                    [row["epoch"] for row in history] != list(range(1, epochs + 1))):
                raise ValueError(f"Inconsistent pilot run in {folder}")
            if any(metric not in row for row in history):
                raise ValueError(f"Missing {metric} in {folder}")
            values = [float(row[metric]) for row in history]
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"Non-finite pilot metric in {folder}")
            runs.append(values)
        means = [sum(values[epoch] for values in runs) / len(runs) for epoch in range(epochs)]
        selected = max(range(epochs), key=lambda i: (means[i], -i))
        plan["views"][view] = {
            "selected_epoch": selected + 1,
            "pilot_epochs": epochs,
            "signature": signature,
            "seeds": sorted(seeds),
            "mean_pilot_metric_at_epoch": means[selected],
        }
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/temporal"))
    parser.add_argument("--views", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--metric", choices=("mean_month_macro_f1", "worst_month_macro_f1"),
                        default="mean_month_macro_f1")
    parser.add_argument("--save", type=Path, default=Path("runs/temporal/epoch_plan.json"))
    args = parser.parse_args()
    seeds = [int(v) for v in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be unique")
    result = choose_epochs(args.output_root, tuple(args.views), seeds, args.metric)
    write_json_atomic(args.save, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
