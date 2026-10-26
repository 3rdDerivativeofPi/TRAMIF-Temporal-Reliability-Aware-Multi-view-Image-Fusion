"""Evaluate equal-weight fusion of the three calibrated TRAMIF views.

Run from the repository root:
    python -m scripts.evaluate_uniform_fusion

Requires scripts/report_temporal_predictions.py and existing exports and
temperature.json files. Uses NumPy on CPU. No training or temperature fitting.
For each seed separately: mean_v softmax(logits_v / T_v), with weights 1/3.
Evaluates February and March only and reports all 51 families.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts import report_temporal_predictions as reporting

METHOD = "uniform_calibrated"
VIEWS = reporting.VIEWS
MONTHS = ("2020-02", "2020-03")


def calibrated_log_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Invalid logits")
    scores = scores / temperature
    shifted = scores - scores.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def uniform_log_probabilities(log_probabilities: list[np.ndarray]) -> np.ndarray:
    """Arithmetic mean of probabilities, calculated stably in log space."""
    if len(log_probabilities) != 3:
        raise ValueError("Uniform baseline requires exactly three views")
    shape = log_probabilities[0].shape
    if (len(shape) != 2 or any(p.shape != shape or not np.isfinite(p).all()
                              for p in log_probabilities)):
        raise ValueError("View probabilities must have matching finite shapes")
    for p in log_probabilities:
        if not np.allclose(np.exp(p).sum(axis=1), 1, rtol=0, atol=1e-10):
            raise ValueError("Input log probabilities are not normalized")
    return np.logaddexp.reduce(np.stack(log_probabilities), axis=0) - np.log(3.0)


def checked_temperature(calibration: dict, export: dict, *, path: Path,
                        view: str, seed: int, month: str, names: list[str]) -> float:
    if (calibration.get("view") != view or calibration.get("seed") != seed or
            calibration.get("fit_month") != "2020-02" or calibration.get("class_names") != names):
        raise ValueError(f"Calibration view/seed/fit month/class order mismatch: {path}")
    identity = export["identity"]
    if (calibration.get("checkpoint_sha256") != identity.get("checkpoint_sha256") or
            calibration.get("data") != identity.get("data")):
        raise ValueError(f"Calibration checkpoint/dataset mismatch: {path}")
    source = calibration.get("source_exports", {}).get(month, {})
    if (source.get("npz_sha256") != export["output_sha256"] or
            source.get("metadata_sha256") != reporting.file_sha256(path.with_suffix(".json"))):
        raise ValueError(f"Calibration was evaluated on different source exports: {path}")
    temperature = float(calibration["fit"]["temperature"])
    if not np.isfinite(temperature) or not 0.05 <= temperature <= 20:
        raise ValueError(f"Temperature outside the declared range: {path}")
    return temperature


def aggregate_rows(rows: list[dict]) -> list[dict]:
    summaries = []
    for view in (*VIEWS, METHOD):
        for month in MONTHS:
            group = [r for r in rows if r["view"] == view and r["month"] == month]
            result = {"view": view, "month": month, "n_seeds": len(group),
                      "seeds": ",".join(str(r["seed"]) for r in group),
                      "n_samples_per_seed": group[0]["n_samples"],
                      "n_supported_classes": group[0]["n_supported_classes"]}
            for metric in reporting.SUMMARY_METRICS:
                values = np.asarray([r[metric] for r in group])
                result[metric + "_mean"] = float(values.mean())
                result[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
            summaries.append(result)
    return summaries


def prediction_csv(arrays: dict, log_probabilities: np.ndarray) -> str:
    probabilities = np.exp(log_probabilities)
    labels = arrays["labels"]
    predictions = log_probabilities.argmax(axis=1)
    true_probability = probabilities[np.arange(len(labels)), labels]
    brier = np.maximum(0, (probabilities ** 2).sum(axis=1) - 2 * true_probability + 1)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["sha", "true_class_id", "predicted_class_id", "confidence",
                     "true_class_probability", "nll", "brier"])
    for i, sha in enumerate(arrays["sha"]):
        writer.writerow([sha, int(labels[i]), int(predictions[i]),
                         float(probabilities[i, predictions[i]]), float(true_probability[i]),
                         float(-log_probabilities[i, labels[i]]), float(brier[i])])
    return stream.getvalue()


def write_fusion_report(root: Path, report: dict, predictions: str) -> None:
    folder = root / METHOD / f"seed_{report['seed']}" / report["month"]
    # Reuse the verified 51-family formatter, replacing its uncalibrated heading.
    rendered = reporting.format_report(report)
    rendered = rendered.replace("Uncalibrated (T=1);", "Uniform fusion of calibrated probabilities;", 1)
    rendered += "\nWeights: raw_byte=1/3, entropy=1/3, sbsmi=1/3.\n"
    rendered += "Per-view temperatures: " + json.dumps(report["temperatures"], sort_keys=True) + "\n"
    reporting.write_text_atomic(folder / "classification_report.txt", rendered)
    reporting.write_text_atomic(folder / "classification_report.csv", reporting.csv_text(report["per_family"]))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["true_family / predicted_family", *report["class_names"]])
    for name, row in zip(report["class_names"], report["confusion_matrix"]):
        writer.writerow([name, *row])
    reporting.write_text_atomic(folder / "confusion_matrix.csv", stream.getvalue())
    reporting.write_text_atomic(folder / "predictions.csv", predictions)
    reporting.write_text_atomic(folder / "report.json",
                                json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", type=Path, default=Path("runs/exports"))
    parser.add_argument("--calibration-root", type=Path, default=Path("runs/calibration"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/fusion"))
    parser.add_argument("--seeds", type=reporting.parse_seeds,
                        default=reporting.parse_seeds("11,22,33,44,55"))
    args = parser.parse_args(argv)
    required = []
    for seed in args.seeds:
        for view in VIEWS:
            required.append(args.calibration_root / view / f"seed_{seed}" / "temperature.json")
            for month in MONTHS:
                path = args.exports_root / view / f"seed_{seed}" / f"{month}.npz"
                required.extend((path, path.with_suffix(".json")))
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        parser.error("Complete exports and calibration first. Missing:\n" + "\n".join(missing[:10]))
    source_code = {"fusion": reporting.file_sha256(Path(__file__).resolve()),
                   "reporting": reporting.file_sha256(Path(reporting.__file__).resolve())}
    comparisons, completed = [], []
    reference_names, reference_data = None, None
    reference_rows = {}
    for seed in args.seeds:
        calibrations = {}
        for view in VIEWS:
            path = args.calibration_root / view / f"seed_{seed}" / "temperature.json"
            calibrations[view] = (json.loads(path.read_text(encoding="utf-8")), reporting.file_sha256(path))
        for month in MONTHS:
            probabilities, sources, temperatures = [], {}, {}
            for view in VIEWS:
                path = args.exports_root / view / f"seed_{seed}" / f"{month}.npz"
                arrays, export = reporting.load_export(path, view=view, seed=seed, month=month)
                names = arrays["class_names"].tolist()
                if reference_names is not None and names != reference_names:
                    raise ValueError(f"Class ordering differs: {path}")
                reference_names = names
                data = export["identity"]["data"]
                if reference_data is not None and data != reference_data:
                    raise ValueError(f"Dataset identity differs across views/seeds: {path}")
                reference_data = data
                current_rows = (arrays["sha"], arrays["labels"])
                if month in reference_rows and any(not np.array_equal(a, b)
                                                  for a, b in zip(current_rows, reference_rows[month])):
                    raise ValueError(f"Sample IDs/labels/order differ across views/seeds: {path}")
                reference_rows[month] = current_rows
                calibration, calibration_hash = calibrations[view]
                temperature = checked_temperature(calibration, export, path=path, view=view,
                                                   seed=seed, month=month, names=names)
                logp = calibrated_log_probabilities(arrays["logits"], temperature)
                probabilities.append(logp)
                temperatures[view] = temperature
                single = reporting.classification_statistics(arrays["labels"], logp, names)
                single.update({"view": view, "seed": seed, "month": month})
                row = reporting.summary_row(single)
                row["temperature"] = temperature
                # The single-view comparison must reproduce the recorded calibration.
                for metric in reporting.SUMMARY_METRICS:
                    expected = calibration["evaluation"][month]["after"][metric]
                    if not np.isclose(row[metric], expected, rtol=0, atol=1e-10):
                        raise ValueError(f"Calibrated {metric} differs from saved evaluation: {path}")
                comparisons.append(row)
                sources[view] = {"npz_sha256": export["output_sha256"],
                                 "metadata_sha256": reporting.file_sha256(path.with_suffix(".json")),
                                 "checkpoint_sha256": export["identity"]["checkpoint_sha256"],
                                 "temperature_json_sha256": calibration_hash}
            fused = uniform_log_probabilities(probabilities)
            report = reporting.classification_statistics(arrays["labels"], fused, names)
            provenance = {"method": METHOD, "seed": seed, "month": month,
                          "weights": {view: 1 / 3 for view in VIEWS}, "temperatures": temperatures,
                          "source_exports": sources, "source_code_sha256": source_code}
            report.update({"view": METHOD, "seed": seed, "month": month, "schema": 1,
                           "split": "validation", "class_names": names,
                           "temperatures": temperatures, "provenance": provenance,
                           "created_utc": datetime.now(timezone.utc).isoformat(),
                           "definitions": {"fusion": "arithmetic mean of 3 calibrated probability vectors",
                               "seed_policy": "three views with the same seed; no averaging across seeds",
                               "temperature_fit_month": "2020-02", "weight_selection": "fixed 1/3, no fitting",
                               "macro_primary": "families with true support in this month",
                               "macro_fixed_51": "all eligible labels; undefined F1 contributes zero",
                               "confusion_matrix": "rows=true, columns=predicted in class-ID order",
                               "february": "uses temperatures fitted on February; not an independent calibration test",
                               "march": "forward evaluation with February temperatures"}})
            destination = args.output_root / METHOD / f"seed_{seed}" / month / "report.json"
            if destination.exists():
                previous = json.loads(destination.read_text(encoding="utf-8"))
                if previous.get("provenance") != provenance:
                    raise ValueError(f"Existing report uses different inputs/code; use a new --output-root: {destination}")
            row = reporting.summary_row(report)
            row["temperature"] = None  # Fusion has three separate temperatures.
            comparisons.append(row)
            completed.append((report, prediction_csv(arrays, fused)))
            print(f"CHECKED seed_{seed}/{month}: macro_F1={row['macro_f1_supported']:.4f}, "
                  f"accuracy={row['accuracy']:.4f}", flush=True)
    if set(reference_rows[MONTHS[0]][0]) & set(reference_rows[MONTHS[1]][0]):
        raise ValueError("February and March have overlapping sample IDs")
    for report, predictions in completed:
        write_fusion_report(args.output_root, report, predictions)
    folder = args.output_root / METHOD
    reporting.write_text_atomic(folder / "monthly_summary.csv", reporting.csv_text(comparisons))
    reporting.write_text_atomic(folder / "seed_summary.csv", reporting.csv_text(aggregate_rows(comparisons)))
    print(f"Finished: {len(completed)} fusion reports under {folder}. "
          f"Summary tables compare 3 calibrated single views and uniform fusion across {len(args.seeds)} seeds. "
          "Across-seed SD is not a confidence interval.", flush=True)


if __name__ == "__main__":
    main()
