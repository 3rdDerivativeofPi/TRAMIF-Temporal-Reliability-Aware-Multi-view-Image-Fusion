"""Fit 15 temperatures on February only; evaluate both February and March.

Run from the repository root:
    python -m scripts.calibrate_temporal_predictions

Requires the existing scripts/report_temporal_predictions.py and NumPy.
Reads raw logits in runs/exports. Saves temperatures and before/after reports
under runs/calibration. Does not load models or change source predictions.
The fixed search range is T in [0.05, 20]; March never enters the optimizer.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts import report_temporal_predictions as reporting

FIT_MONTH = "2020-02"
CHECK_MONTH = "2020-03"
MONTHS = (FIT_MONTH, CHECK_MONTH)
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0
PROBABILITY_METRICS = ("nll", "brier", "ece", "mean_confidence")
CLASSIFICATION_METRICS = ("accuracy", "macro_f1_supported", "macro_f1_fixed_51",
                          "weighted_f1", "balanced_accuracy_supported")


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *,
                    minimum: float = MIN_TEMPERATURE,
                    maximum: float = MAX_TEMPERATURE) -> dict:
    """Minimize ordinary sample-mean NLL, convex in inverse temperature beta.

    Use derivative bisection in float64 and keep T=1 if no meaningful NLL gain
    is found. This is not the class-balanced quality score used later in Q.
    """
    scores = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if (not np.isfinite([minimum, maximum]).all() or
            not 0 < minimum <= 1 <= maximum or minimum >= maximum):
        raise ValueError("Temperature bounds must be positive, distinct, and include 1")
    if (scores.ndim != 2 or scores.shape[1] < 2 or labels.ndim != 1 or len(labels) == 0 or
            len(labels) != len(scores) or not np.issubdtype(labels.dtype, np.integer) or
            np.any(labels < 0) or np.any(labels >= scores.shape[1]) or not np.isfinite(scores).all()):
        raise ValueError("Invalid logits or labels")
    centered = scores - scores.max(axis=1, keepdims=True)
    true_scores = centered[np.arange(len(labels)), labels]

    def objective(beta: float) -> tuple[float, float]:
        exponentials = np.exp(beta * centered)
        normalizer = exponentials.sum(axis=1)
        probabilities = exponentials / normalizer[:, None]
        nll = np.mean(np.log(normalizer) - beta * true_scores)
        derivative = np.mean((probabilities * centered).sum(axis=1) - true_scores)
        return float(nll), float(derivative)

    low, high = 1 / maximum, 1 / minimum
    baseline, _ = objective(1.0)
    iterations = 0
    if objective(low)[1] >= 0:
        beta = low
    elif objective(high)[1] <= 0:
        beta = high
    else:
        for iterations in range(1, 81):
            beta = (low + high) / 2
            if objective(beta)[1] > 0:
                high = beta
            else:
                low = beta
            if high - low <= 1e-12 * max(1.0, abs(beta)):
                break
        beta = (low + high) / 2
    fitted, derivative = objective(beta)
    if baseline - fitted <= 1e-12:
        beta, fitted = 1.0, baseline
        derivative = objective(beta)[1]
        status = "no_nll_improvement_keep_T1"
    elif beta == 1 / maximum:
        status = "maximum_temperature_bound"
    elif beta == 1 / minimum:
        status = "minimum_temperature_bound"
    else:
        status = "interior_optimum"
    if not np.isfinite([fitted, derivative, beta]).all() or fitted > baseline + 1e-10:
        raise RuntimeError("Temperature fit failed numerical checks")
    return {"temperature": float(1 / beta), "inverse_temperature": float(beta),
            "minimum_temperature": minimum, "maximum_temperature": maximum,
            "fit_nll_before": baseline, "fit_nll_after": fitted,
            "fit_nll_delta": fitted - baseline, "status": status,
            "at_search_boundary": status in {"maximum_temperature_bound", "minimum_temperature_bound"},
            "objective": "ordinary sample-mean NLL on February only",
            "solver": "bounded derivative bisection in inverse temperature; float64",
            "iterations": iterations, "derivative_at_solution": derivative}


def metric_values(stats: dict) -> dict:
    values = {key: stats[key] for key in ("accuracy", "balanced_accuracy_supported", "nll", "brier", "ece")}
    values.update({"macro_f1_supported": stats["macro_avg_supported"]["f1"],
                   "macro_f1_fixed_51": stats["macro_avg_fixed_51"]["f1"],
                   "weighted_f1": stats["weighted_avg"]["f1"],
                   "mean_confidence": sum(b["count"] * b["confidence"] for b in stats["ece_bins"]
                                          if b["count"]) / stats["n_samples"]})
    return values


def evaluate_temperature(arrays: dict, temperature: float) -> dict:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    original = np.asarray(arrays["logits"], dtype=np.float64)
    calibrated = original / temperature
    changed = int(np.count_nonzero(original.argmax(axis=1) != calibrated.argmax(axis=1)))
    if changed:
        raise RuntimeError("Scalar temperature unexpectedly changed predicted classes")
    names = arrays["class_names"].tolist()
    before = reporting.classification_statistics(arrays["labels"], original, names)
    after = reporting.classification_statistics(arrays["labels"], calibrated, names)
    before_metrics, after_metrics = metric_values(before), metric_values(after)
    if any(before_metrics[k] != after_metrics[k] for k in CLASSIFICATION_METRICS):
        raise RuntimeError("Classification metrics changed after scalar scaling")
    return {"n_samples": len(original), "n_supported_classes": before["n_supported_classes"],
            "supported_class_ids": before["supported_class_ids"], "absent_families": before["absent_families"],
            "changed_predictions": changed, "before": before_metrics, "after": after_metrics,
            "delta_after_minus_before": {k: after_metrics[k] - before_metrics[k] for k in PROBABILITY_METRICS},
            "ece_bins_before": before["ece_bins"], "ece_bins_after": after["ece_bins"]}


def summary_row(record: dict, month: str) -> dict:
    results = record["evaluation"][month]
    row = {"view": record["view"], "seed": record["seed"], "month": month,
           "fit_month": FIT_MONTH, "temperature": record["fit"]["temperature"],
           "n_samples": results["n_samples"], "n_supported_classes": results["n_supported_classes"],
           "changed_predictions": results["changed_predictions"],
           **{k: results["before"][k] for k in CLASSIFICATION_METRICS}}
    for metric in PROBABILITY_METRICS:
        row[metric + "_before"] = results["before"][metric]
        row[metric + "_after"] = results["after"][metric]
        row[metric + "_delta"] = results["delta_after_minus_before"][metric]
    return row


def seed_summary(rows: list[dict], views: list[str]) -> list[dict]:
    summary = []
    metrics = list(CLASSIFICATION_METRICS) + [m + suffix for m in PROBABILITY_METRICS
                                            for suffix in ("_before", "_after", "_delta")]
    for view in views:
        for month in MONTHS:
            group = [r for r in rows if r["view"] == view and r["month"] == month]
            result = {"view": view, "month": month, "n_seeds": len(group),
                      "seeds": ",".join(str(r["seed"]) for r in group),
                      "n_samples_per_seed": group[0]["n_samples"],
                      "n_supported_classes": group[0]["n_supported_classes"]}
            for metric in metrics:
                values = np.asarray([r[metric] for r in group])
                result[metric + "_mean"] = float(values.mean())
                result[metric + "_std"] = float(values.std(ddof=1)) if len(group) > 1 else None
            summary.append(result)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", type=Path, default=Path("runs/exports"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/calibration"))
    parser.add_argument("--views", nargs="+", choices=reporting.VIEWS, default=list(reporting.VIEWS))
    parser.add_argument("--seeds", type=reporting.parse_seeds,
                        default=reporting.parse_seeds("11,22,33,44,55"))
    args = parser.parse_args(argv)
    if len(set(args.views)) != len(args.views):
        parser.error("Views must not be repeated")
    paths = {(v, s, m): args.exports_root / v / f"seed_{s}" / f"{m}.npz"
             for v in args.views for s in args.seeds for m in MONTHS}
    missing = [str(p) for path in paths.values() for p in (path, path.with_suffix(".json"))
               if not p.is_file()]
    if missing:
        parser.error("Export both February and March first. Missing:\n" + "\n".join(missing[:10]))
    source_hashes = {"calibration_script": reporting.file_sha256(Path(__file__).resolve()),
                     "report_script": reporting.file_sha256(Path(reporting.__file__).resolve())}
    names_reference = None
    sample_references = {}
    records = []
    for view in args.views:
        for seed in args.seeds:
            arrays, exports = {}, {}
            for month in MONTHS:
                path = paths[view, seed, month]
                arrays[month], exports[month] = reporting.load_export(path, view=view, seed=seed, month=month)
                names = arrays[month]["class_names"].tolist()
                if names_reference is not None and names != names_reference:
                    raise ValueError(f"Class ordering differs: {path}")
                names_reference = names
                current = (arrays[month]["sha"], arrays[month]["labels"])
                if month in sample_references and any(not np.array_equal(a, b)
                                                      for a, b in zip(current, sample_references[month])):
                    raise ValueError(f"Sample IDs/labels/order differ: {path}")
                sample_references[month] = current
            fit_identity = exports[FIT_MONTH]["identity"]
            check_identity = exports[CHECK_MONTH]["identity"]
            for key in ("checkpoint_sha256", "data", "training_signature", "model_spec"):
                if key not in fit_identity or fit_identity[key] != check_identity.get(key):
                    raise ValueError(f"February/March {key} mismatch for {view}/seed_{seed}")
            if set(arrays[FIT_MONTH]["sha"]) & set(arrays[CHECK_MONTH]["sha"]):
                raise ValueError("February and March contain overlapping sample IDs")

            # The optimizer receives February arrays only. March is evaluation only.
            fitted = fit_temperature(arrays[FIT_MONTH]["logits"], arrays[FIT_MONTH]["labels"])
            evaluation = {month: evaluate_temperature(arrays[month], fitted["temperature"]) for month in MONTHS}
            if abs(evaluation[FIT_MONTH]["after"]["nll"] - fitted["fit_nll_after"]) > 1e-9:
                raise RuntimeError("Fitted NLL disagrees with report computation")
            source_exports = {month: {"path": str(paths[view, seed, month].resolve()),
                "npz_sha256": exports[month]["output_sha256"],
                "metadata_sha256": reporting.file_sha256(paths[view, seed, month].with_suffix(".json"))}
                for month in MONTHS}
            record = {"schema": 1, "view": view, "seed": seed, "fit_month": FIT_MONTH,
                      "evaluation_months": list(MONTHS), "fit": fitted, "evaluation": evaluation,
                      "checkpoint_sha256": fit_identity["checkpoint_sha256"], "data": fit_identity["data"],
                      "class_names": names_reference, "source_exports": source_exports,
                      "source_hashes": source_hashes, "numpy_version": np.__version__,
                      "created_utc": datetime.now(timezone.utc).isoformat(),
                      "notes": ["February post-fit metrics are in-sample calibration results.",
                                "March labels and logits are never used to fit or select temperature.",
                                "NLL/Brier/ECE improvements on March are not guaranteed.",
                                "Apply probabilities = softmax(original_logits / temperature).",
                                "Existing 51-family classification reports and confusion matrices remain valid.",
                                "Bounds are fixed in advance; report a bound hit instead of tuning on March."]}
            target = args.output_root / view / f"seed_{seed}" / "temperature.json"
            if target.exists():
                previous = json.loads(target.read_text(encoding="utf-8"))
                if (previous.get("source_exports") != source_exports or previous.get("source_hashes") != source_hashes):
                    raise ValueError(f"Existing calibration uses different inputs/code; use a new --output-root: {target}")
            records.append(record)
            print(f"FIT {view}/seed_{seed}: T={fitted['temperature']:.6f}; "
                  f"February NLL {fitted['fit_nll_before']:.4f} -> {fitted['fit_nll_after']:.4f}; "
                  f"{fitted['status']}", flush=True)

    # Complete validation/calculation before replacing any derived outputs.
    rows, temperature_rows = [], []
    for record in records:
        target = args.output_root / record["view"] / f"seed_{record['seed']}" / "temperature.json"
        reporting.write_text_atomic(target, json.dumps(record, indent=2, allow_nan=False) + "\n")
        temperature_rows.append({"view": record["view"], "seed": record["seed"], "fit_month": FIT_MONTH,
                                 "n_fit_samples": record["evaluation"][FIT_MONTH]["n_samples"], **record["fit"]})
        rows.extend(summary_row(record, month) for month in MONTHS)
    reporting.write_text_atomic(args.output_root / "temperature_summary.csv", reporting.csv_text(temperature_rows))
    reporting.write_text_atomic(args.output_root / "monthly_calibration_summary.csv", reporting.csv_text(rows))
    reporting.write_text_atomic(args.output_root / "seed_calibration_summary.csv",
                                reporting.csv_text(seed_summary(rows, args.views)))
    print(f"Finished: {len(records)} temperatures and {len(rows)} monthly comparisons under {args.output_root}. "
          "Summaries cover this command's selections. Across-seed SD is not a confidence interval.", flush=True)


if __name__ == "__main__":
    main()
