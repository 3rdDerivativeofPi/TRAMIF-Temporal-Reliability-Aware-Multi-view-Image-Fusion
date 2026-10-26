"""February quality scores, March selection, and quality-only TRAMIF fusion.

Run from the repository root:
    python -m scripts.evaluate_quality_fusion

Requires report_temporal_predictions.py and evaluate_uniform_fusion.py in
scripts, plus existing exports and temperature.json files. NumPy/CPU only.

Q_v = negative mean of per-family mean log loss on calibrated February data.
w = softmax(a * Q). A single a is selected for all seeds from the fixed grid
[0, 0.5, 1, 2, 4], by mean March true-supported macro F1; ties prefer smaller a.
March is development/selection data. No future-test files are accessed.
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
from scripts import evaluate_uniform_fusion as uniform

METHOD = "quality_only"
VIEWS = uniform.VIEWS
MONTHS = uniform.MONTHS
QUALITY_MONTH = "2020-02"
SELECTION_MONTH = "2020-03"
QUALITY_SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
TIE_TOLERANCE = 1e-12


def quality_score(log_probabilities: np.ndarray, labels: np.ndarray,
                  class_names: list[str]) -> tuple[float, list[dict]]:
    """Equal weight per true-supported family; absent families remain unavailable."""
    labels = np.asarray(labels)
    logp = np.asarray(log_probabilities, dtype=np.float64)
    classes = len(class_names)
    if (labels.ndim != 1 or len(labels) == 0 or not np.issubdtype(labels.dtype, np.integer) or
            logp.shape != (len(labels), classes) or not np.isfinite(logp).all() or
            np.any(labels < 0) or np.any(labels >= classes) or
            not np.allclose(np.exp(logp).sum(axis=1), 1.0, rtol=0, atol=1e-10)):
        raise ValueError("Invalid quality-score inputs")
    losses = -logp[np.arange(len(labels)), labels]
    counts = np.bincount(labels, minlength=classes)
    sums = np.bincount(labels, weights=losses, minlength=classes)
    family_loss = np.divide(sums, counts, out=np.zeros(classes), where=counts > 0)
    supported = counts > 0
    q = -float(family_loss[supported].mean())
    rows = [{"class_id": i, "family": name, "support": int(counts[i]),
             "mean_nll": float(family_loss[i]) if counts[i] else None,
             "status": "present" if counts[i] else "absent_from_quality_month"}
            for i, name in enumerate(class_names)]
    return q, rows


def quality_log_weights(qualities: np.ndarray, scale: float) -> np.ndarray:
    qualities = np.asarray(qualities, dtype=np.float64)
    if qualities.shape != (3,) or not np.isfinite(qualities).all() or not np.isfinite(scale) or scale < 0:
        raise ValueError("Require three finite quality scores and a non-negative scale")
    scores = scale * qualities
    shifted = scores - scores.max()
    return shifted - np.log(np.exp(shifted).sum())


def weighted_log_probabilities(logp: np.ndarray, log_weights: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    if (logp.ndim != 3 or logp.shape[0] != 3 or log_weights.shape != (3,) or
            not np.isfinite(logp).all() or not np.isfinite(log_weights).all() or
            not np.isclose(np.exp(log_weights).sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid weighted-fusion inputs")
    return np.logaddexp.reduce(logp + log_weights[:, None, None], axis=0)


def select_scale(rows: list[dict], seeds: list[int]) -> tuple[float, list[dict]]:
    """One shared development choice, never a separate best scale for each seed."""
    scores = []
    for scale in QUALITY_SCALES:
        group = [r for r in rows if r["month"] == SELECTION_MONTH and r["quality_scale"] == scale]
        if len(group) != len(seeds) or sorted(r["seed"] for r in group) != sorted(seeds):
            raise ValueError("Each scale requires the same complete set of March seeds")
        values = np.asarray([r["macro_f1_supported"] for r in group])
        if not np.isfinite(values).all():
            raise ValueError("Invalid selection metric")
        scores.append({"quality_scale": scale, "march_macro_f1_mean": float(values.mean()),
                       "march_macro_f1_std": float(values.std(ddof=1)) if len(values) > 1 else None})
    best = scores[0]
    for candidate in scores[1:]:
        if candidate["march_macro_f1_mean"] > best["march_macro_f1_mean"] + TIE_TOLERANCE:
            best = candidate
    return best["quality_scale"], scores


def aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    output = []
    for keys, group in sorted(groups.items()):
        result = {**dict(zip(group_keys, keys)), "n_seeds": len(group),
                  "seeds": ",".join(str(r["seed"]) for r in group),
                  "n_samples_per_seed": group[0]["n_samples"],
                  "n_supported_classes": group[0]["n_supported_classes"]}
        for metric in reporting.SUMMARY_METRICS:
            values = np.asarray([r[metric] for r in group])
            result[metric + "_mean"] = float(values.mean())
            result[metric + "_std"] = float(values.std(ddof=1)) if len(group) > 1 else None
        output.append(result)
    return output


def metric_row(stats: dict, view: str, seed: int, month: str, temperature=None) -> dict:
    row = reporting.summary_row({**stats, "view": view, "seed": seed, "month": month})
    row["temperature"] = temperature
    return row


def write_selected_report(folder: Path, report: dict, predictions: str) -> None:
    text = reporting.format_report(report).replace(
        "Uncalibrated (T=1);", f"Quality-only calibrated fusion (a={report['quality_scale']:g});", 1)
    text += "\nWeights: " + json.dumps(report["weights"], sort_keys=True) + "\n"
    text += "Quality scores use February. The common scale was selected on March.\n"
    text += "March scores are development-selection results, not untouched test results.\n"
    reporting.write_text_atomic(folder / "classification_report.txt", text)
    reporting.write_text_atomic(folder / "classification_report.csv", reporting.csv_text(report["per_family"]))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["true_family / predicted_family", *report["class_names"]])
    for name, row in zip(report["class_names"], report["confusion_matrix"]):
        writer.writerow([name, *row])
    reporting.write_text_atomic(folder / "confusion_matrix.csv", stream.getvalue())
    reporting.write_text_atomic(folder / "predictions.csv", predictions)
    reporting.write_text_atomic(folder / "report.json", json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", type=Path, default=Path("runs/exports"))
    parser.add_argument("--calibration-root", type=Path, default=Path("runs/calibration"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/fusion"))
    parser.add_argument("--seeds", type=reporting.parse_seeds,
                        default=reporting.parse_seeds("11,22,33,44,55"))
    args = parser.parse_args(argv)
    folder = args.output_root / METHOD
    source_hashes = {"quality_script": reporting.file_sha256(Path(__file__).resolve()),
                     "report_script": reporting.file_sha256(Path(reporting.__file__).resolve()),
                     "uniform_script": reporting.file_sha256(Path(uniform.__file__).resolve())}
    data_cache, provenance, qualities, family_rows, score_rows, comparisons = {}, {}, {}, [], [], []
    reference_names, reference_data, reference_rows = None, None, {}

    for seed in args.seeds:
        data_cache[seed], provenance[str(seed)], qualities[seed] = {}, {}, []
        for view in VIEWS:
            cal_path = args.calibration_root / view / f"seed_{seed}" / "temperature.json"
            calibration = json.loads(cal_path.read_text(encoding="utf-8"))
            provenance[str(seed)][view] = {"temperature_json_sha256": reporting.file_sha256(cal_path)}
            for month in MONTHS:
                path = args.exports_root / view / f"seed_{seed}" / f"{month}.npz"
                arrays, export = reporting.load_export(path, view=view, seed=seed, month=month)
                names = arrays["class_names"].tolist()
                if reference_names is not None and names != reference_names:
                    raise ValueError(f"Class order differs: {path}")
                reference_names = names
                data = export["identity"]["data"]
                if reference_data is not None and data != reference_data:
                    raise ValueError(f"Dataset identity differs: {path}")
                reference_data = data
                current = (arrays["sha"], arrays["labels"])
                if month in reference_rows and any(not np.array_equal(a, b)
                                                  for a, b in zip(current, reference_rows[month])):
                    raise ValueError(f"Sample IDs/labels/order differ: {path}")
                reference_rows[month] = current
                temperature = uniform.checked_temperature(calibration, export, path=path, view=view,
                                                           seed=seed, month=month, names=names)
                logp = uniform.calibrated_log_probabilities(arrays["logits"], temperature)
                cached = data_cache[seed].setdefault(month, {
                    "sha": arrays["sha"], "labels": arrays["labels"], "class_names": names,
                    "log_probabilities": [], "temperatures": {}})
                cached["log_probabilities"].append(logp)
                cached["temperatures"][view] = temperature
                provenance[str(seed)][view][month] = {
                    "npz_sha256": export["output_sha256"],
                    "metadata_sha256": reporting.file_sha256(path.with_suffix(".json")),
                    "checkpoint_sha256": export["identity"]["checkpoint_sha256"]}
                stats = reporting.classification_statistics(arrays["labels"], logp, names)
                row = metric_row(stats, view, seed, month, temperature)
                for metric in reporting.SUMMARY_METRICS:
                    if not np.isclose(row[metric], calibration["evaluation"][month]["after"][metric],
                                      rtol=0, atol=1e-10):
                        raise ValueError(f"Saved calibration metric disagrees for {view}/{seed}/{month}")
                comparisons.append(row)
                if month == QUALITY_MONTH:
                    q, by_family = quality_score(logp, arrays["labels"], names)
                    qualities[seed].append(q)
                    score_rows.append({"view": view, "seed": seed, "quality_month": QUALITY_MONTH,
                                       "n_samples": len(arrays["labels"]), "n_supported_classes": stats["n_supported_classes"],
                                       "class_balanced_nll": -q, "quality": q, "temperature": temperature})
                    family_rows.extend({"view": view, "seed": seed, "month": QUALITY_MONTH, **r} for r in by_family)
        for month in MONTHS:
            data_cache[seed][month]["log_probabilities"] = np.stack(data_cache[seed][month]["log_probabilities"])
        print(f"QUALITY seed_{seed}: " + ", ".join(f"{v}={q:.4f}" for v, q in zip(VIEWS, qualities[seed])), flush=True)
    if set(reference_rows[MONTHS[0]][0]) & set(reference_rows[MONTHS[1]][0]):
        raise ValueError("February and March sample IDs overlap")

    candidates, weight_rows = [], []
    for seed in args.seeds:
        for scale in QUALITY_SCALES:
            log_weights = quality_log_weights(qualities[seed], scale)
            weight_rows.append({"seed": seed, "quality_scale": scale,
                                **{v + "_weight": float(w) for v, w in zip(VIEWS, np.exp(log_weights))}})
            for month in MONTHS:
                data = data_cache[seed][month]
                fused = weighted_log_probabilities(data["log_probabilities"], log_weights)
                stats = reporting.classification_statistics(data["labels"], fused, data["class_names"])
                row = metric_row(stats, METHOD, seed, month)
                candidates.append({**row, "quality_scale": scale})
                if scale == 0:
                    comparisons.append({**row, "view": uniform.METHOD})
    selected, selection_scores = select_scale(candidates, args.seeds)
    print(f"SELECTED shared a={selected:g}, using mean March macro F1 across {len(args.seeds)} seeds", flush=True)
    selected_weights = {str(seed): dict(zip(VIEWS, np.exp(quality_log_weights(qualities[seed], selected)).tolist()))
                        for seed in args.seeds}
    experiment = {"quality_month": QUALITY_MONTH, "selection_month": SELECTION_MONTH,
                  "quality_scales": list(QUALITY_SCALES), "seeds": args.seeds,
                  "selected_scale": selected, "source_hashes": source_hashes,
                  "source_exports": provenance}
    selection = {"schema": 1, "method": METHOD, **experiment,
                 "selected_is_uniform": selected == 0, "selected_at_grid_maximum": selected == max(QUALITY_SCALES),
                 "criterion": "maximum mean March true-supported macro F1 across all requested seeds",
                 "tie_rule": "prefer smaller a within 1e-12 macro F1", "candidate_march_scores": selection_scores,
                 "quality_definition": "negative unweighted mean of within-family mean NLL on calibrated February data",
                 "absent_family_policy": "exclude families with zero February true support from Q; retain all 51 output classes",
                 "qualities_by_seed": {str(s): dict(zip(VIEWS, qualities[s])) for s in args.seeds},
                 "weights_by_seed": selected_weights,
                 "notes": ["February temperatures and Q use the same historical month; February metrics are development results.",
                           "March selects one scale shared across seeds; report March as development selection, not untouched test.",
                           "This is Q-only: b=0 and c=0. No MMD or bootstrap uncertainty is included.",
                           "The full reliability method must offer this same a grid at b=c=0 for a matched comparison.",
                           "If a=0 is selected, quality-only selection reduces to uniform fusion.",
                           "Keep future April-September evaluation separate; do not expand this grid from future-test results."],
                 "created_utc": datetime.now(timezone.utc).isoformat()}
    selection_path = folder / "selection.json"
    if selection_path.exists():
        previous = json.loads(selection_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in experiment.items()):
            raise ValueError(f"Previous selection uses different inputs/code/settings; use a new --output-root: {selection_path}")

    completed = []
    for seed in args.seeds:
        log_weights = quality_log_weights(qualities[seed], selected)
        for month in MONTHS:
            data = data_cache[seed][month]
            fused = weighted_log_probabilities(data["log_probabilities"], log_weights)
            report = reporting.classification_statistics(data["labels"], fused, data["class_names"])
            comparisons.append(metric_row(report, METHOD, seed, month))
            report.update({"view": METHOD, "seed": seed, "month": month, "split": "validation",
                           "schema": 1, "class_names": data["class_names"], "quality_scale": selected,
                           "quality_month": QUALITY_MONTH, "scale_selection_month": SELECTION_MONTH,
                           "qualities": dict(zip(VIEWS, qualities[seed])), "weights": selected_weights[str(seed)],
                           "temperatures": data["temperatures"], "selection": experiment,
                           "macro_primary": "true-supported families in this month",
                           "created_utc": datetime.now(timezone.utc).isoformat()})
            completed.append((folder / f"seed_{seed}" / month, report, uniform.prediction_csv(data, fused)))
    # All input checks and selection finish before any output is replaced.
    for destination, report, predictions in completed:
        write_selected_report(destination, report, predictions)
    outputs = {"quality_scores.csv": score_rows, "quality_by_family.csv": family_rows,
               "candidate_weights.csv": weight_rows, "candidate_metrics.csv": candidates,
               "candidate_summary.csv": aggregate(candidates, ("quality_scale", "month")),
               "monthly_summary.csv": comparisons, "seed_summary.csv": aggregate(comparisons, ("view", "month"))}
    for name, rows in outputs.items():
        reporting.write_text_atomic(folder / name, reporting.csv_text(rows))
    reporting.write_text_atomic(selection_path, json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print(f"Finished: {len(completed)} selected quality-only reports under {folder}. "
          "Summaries compare three calibrated single views, uniform fusion, and selected quality-only fusion. "
          "Across-seed SD is not a confidence interval.", flush=True)


if __name__ == "__main__":
    main()
