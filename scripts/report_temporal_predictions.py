"""Report all 51 families from saved February AND March TRAMIF predictions.

Run from the repository root:
    python -m scripts.report_temporal_predictions

Reads existing NPZ exports and their JSON completion records. Requires NumPy;
does not load models, run inference, fit temperatures, or modify export files.
Produces uncalibrated (T=1) reports for each view/seed/month, plus summaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VIEWS = ("raw_byte", "entropy", "sbsmi")
MONTHS = ("2020-02", "2020-03")
NUM_CLASSES = 51
ECE_BINS = 15
SUMMARY_METRICS = (
    "accuracy", "macro_f1_supported", "macro_f1_fixed_51",
    "weighted_f1", "balanced_accuracy_supported", "nll", "brier", "ece",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer seeds") from exc
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be unique non-negative integers")
    return seeds


def write_text_atomic(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_export(path: Path, *, view: str, seed: int, month: str) -> tuple[dict, dict]:
    record_path = path.with_suffix(".json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("output_sha256") != file_sha256(path):
        raise ValueError(f"Export checksum mismatch: {path}")
    identity = record.get("identity", {})
    expected = {"view": view, "seed": seed, "month": month, "split": "validation"}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Export view/seed/month/split mismatch: {path}")
    if identity.get("temperature") != 1.0:
        raise ValueError(f"Expected original uncalibrated logits (T=1): {path}")
    with np.load(path, allow_pickle=False) as saved:
        # Embeddings are not needed for a classification report.
        arrays = {key: saved[key] for key in ("sha", "labels", "month", "logits", "class_names")}
    labels, logits, names = arrays["labels"], arrays["logits"], arrays["class_names"]
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or len(labels) == 0:
        raise ValueError(f"Expected nonempty integer labels: {path}")
    n = len(labels)
    if (names.shape != (NUM_CLASSES,) or names.dtype.kind != "U" or
            len(set(names.tolist())) != NUM_CLASSES or np.any(names == "")):
        raise ValueError(f"Expected 51 unique ordered family names: {path}")
    if logits.shape != (n, NUM_CLASSES) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid logits: {path}")
    if np.any(labels < 0) or np.any(labels >= NUM_CLASSES):
        raise ValueError(f"Class IDs must be in 0..50: {path}")
    if arrays["month"].shape != (n,) or not np.all(arrays["month"] == month):
        raise ValueError(f"Wrong sample months: {path}")
    shas = arrays["sha"]
    if (shas.shape != (n,) or shas.dtype.kind != "U" or len(set(shas.tolist())) != n or
            any(len(s) != 64 or any(c not in "0123456789abcdef" for c in s) for s in shas)):
        raise ValueError(f"Invalid or duplicate sample IDs: {path}")
    if record.get("n_samples") != n or record.get("num_classes") != NUM_CLASSES:
        raise ValueError(f"Sample/class counts disagree with export metadata: {path}")
    counts = np.bincount(labels, minlength=NUM_CLASSES).tolist()
    if record.get("class_counts") != counts:
        raise ValueError(f"Family counts disagree with export metadata: {path}")
    return arrays, record


def classification_statistics(labels: np.ndarray, logits: np.ndarray,
                              class_names: list[str]) -> dict:
    """Use true-supported monthly macro metrics; also expose fixed-set averages.

    Undefined per-class precision/recall/F1 are None. Aggregate averages use
    zero for undefined ratios, explicitly matching the zero_division=0 rule.
    """
    labels = np.asarray(labels)
    scores = np.asarray(logits, dtype=np.float64)
    classes = len(class_names)
    if (labels.ndim != 1 or len(labels) == 0 or
            not np.issubdtype(labels.dtype, np.integer) or
            scores.shape != (len(labels), classes) or classes < 2 or
            np.any(labels < 0) or np.any(labels >= classes) or not np.isfinite(scores).all()):
        raise ValueError("Invalid labels/logits/class names for classification")
    predictions = scores.argmax(axis=1)
    matrix = np.bincount(labels * classes + predictions,
                         minlength=classes * classes).reshape(classes, classes)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    tp = matrix.diagonal()
    fp, fn = predicted - tp, support - tp
    precision = np.divide(tp, predicted, out=np.zeros(classes), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes), where=denominator > 0)
    supported = support > 0
    n = len(labels)
    accuracy = float(tp.sum() / n)

    def average(weights: np.ndarray) -> dict:
        return {name: float(np.average(values, weights=weights))
                for name, values in (("precision", precision), ("recall", recall), ("f1", f1))}

    rows = []
    for i, name in enumerate(class_names):
        rows.append({"class_id": i, "family": name,
                     "precision": float(precision[i]) if predicted[i] else None,
                     "recall": float(recall[i]) if support[i] else None,
                     "f1": float(f1[i]) if denominator[i] else None,
                     "support": int(support[i]), "predicted_count": int(predicted[i]),
                     "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                     "status": "present" if support[i] else "absent_this_month"})

    # Stable log-softmax: no temperature fitting or probability clipping.
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probabilities)
    true_probabilities = probabilities[np.arange(n), labels]
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    bin_ids = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    bins = []
    ece = 0.0
    for i in range(ECE_BINS):
        mask = bin_ids == i
        count = int(mask.sum())
        bin_accuracy = float(correct[mask].mean()) if count else None
        bin_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / n * abs(bin_accuracy - bin_confidence)
        bins.append({"lower": i / ECE_BINS, "upper": (i + 1) / ECE_BINS,
                     "count": count, "accuracy": bin_accuracy, "confidence": bin_confidence})

    return {"n_samples": n, "num_classes": classes, "n_supported_classes": int(supported.sum()),
            "supported_class_ids": np.flatnonzero(supported).tolist(),
            "absent_families": [class_names[i] for i in np.flatnonzero(~supported)],
            "accuracy": accuracy, "macro_avg_supported": average(supported.astype(float)),
            "macro_avg_fixed_51": average(np.ones(classes)), "weighted_avg": average(support),
            "micro_avg": {"precision": accuracy, "recall": accuracy, "f1": accuracy},
            "balanced_accuracy_supported": float(recall[supported].mean()),
            "nll": float(-log_probabilities[np.arange(n), labels].mean()),
            "brier": float(np.maximum(0, (probabilities ** 2).sum(axis=1) -
                                       2 * true_probabilities + 1).mean()),
            "ece": float(ece), "ece_bins": bins,
            "per_family": rows, "confusion_matrix": matrix.tolist()}


def summary_row(report: dict) -> dict:
    return {"view": report["view"], "seed": report["seed"], "month": report["month"],
            "n_samples": report["n_samples"], "n_supported_classes": report["n_supported_classes"],
            "temperature": 1.0, "accuracy": report["accuracy"],
            "macro_f1_supported": report["macro_avg_supported"]["f1"],
            "macro_f1_fixed_51": report["macro_avg_fixed_51"]["f1"],
            "weighted_f1": report["weighted_avg"]["f1"],
            **{k: report[k] for k in ("balanced_accuracy_supported", "nll", "brier", "ece")}}


def format_report(report: dict) -> str:
    def fmt(value):
        return "-" if value is None else f"{value:.4f}"

    width = max(25, max(len(row["family"]) for row in report["per_family"]))
    lines = [f"TRAMIF classification: {report['view']} | seed {report['seed']} | {report['month']}",
             f"Uncalibrated (T=1); samples={report['n_samples']}; "
             f"supported families={report['n_supported_classes']}/{report['num_classes']}", "",
             f"{'ID':>3}  {'Family':<{width}} {'Precision':>10} {'Recall':>10} {'F1':>10} "
             f"{'Support':>8} {'Predicted':>10}  Status"]
    for row in report["per_family"]:
        lines.append(f"{row['class_id']:>3}  {row['family']:<{width}} {fmt(row['precision']):>10} "
                     f"{fmt(row['recall']):>10} {fmt(row['f1']):>10} {row['support']:>8} "
                     f"{row['predicted_count']:>10}  {row['status']}")
    lines += ["", f"Accuracy: {report['accuracy']:.6f}",
              f"Balanced accuracy (true-supported families): {report['balanced_accuracy_supported']:.6f}"]
    for label, key in (("Macro average (true-supported)", "macro_avg_supported"),
                       ("Macro average (fixed 51)", "macro_avg_fixed_51"),
                       ("Weighted average", "weighted_avg"), ("Micro average", "micro_avg")):
        values = report[key]
        lines.append(f"{label}: precision={values['precision']:.6f}, "
                     f"recall={values['recall']:.6f}, F1={values['f1']:.6f}")
    lines += [f"NLL: {report['nll']:.6f}", f"Multiclass Brier (sum over classes): {report['brier']:.6f}",
              f"ECE ({ECE_BINS} equal-width confidence bins): {report['ece']:.6f}", "",
              "Absent families: " + (", ".join(report["absent_families"]) or "none"),
              "'-' means undefined; CSV uses blank and JSON uses null.",
              "Support=0 means this family has no true examples in this month.",
              "Aggregate ratios use zero for undefined values; supported macro excludes support=0 families.",
              "Predictions into absent families remain errors for the actual true families."]
    return "\n".join(lines) + "\n"


def write_report(output_root: Path, report: dict, class_names: list[str]) -> None:
    folder = output_root / report["view"] / f"seed_{report['seed']}" / report["month"]
    write_text_atomic(folder / "classification_report.txt", format_report(report))
    write_text_atomic(folder / "classification_report.csv", csv_text(report["per_family"]))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["true_family / predicted_family", *class_names])
    for name, row in zip(class_names, report["confusion_matrix"]):
        writer.writerow([name, *row])
    write_text_atomic(folder / "confusion_matrix.csv", stream.getvalue())
    write_text_atomic(folder / "report.json",
                      json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-root", type=Path, default=Path("runs/exports"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/reports"))
    parser.add_argument("--views", nargs="+", choices=VIEWS, default=list(VIEWS))
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("11,22,33,44,55"))
    parser.add_argument("--months", nargs="+", choices=MONTHS, default=list(MONTHS))
    args = parser.parse_args(argv)
    if len(set(args.views)) != len(args.views) or len(set(args.months)) != len(args.months):
        parser.error("Views and months must not be repeated")
    selections = [(v, s, m, args.exports_root / v / f"seed_{s}" / f"{m}.npz")
                  for v in args.views for s in args.seeds for m in args.months]
    missing = [str(p) for _, _, _, path in selections
               for p in (path, path.with_suffix(".json")) if not p.is_file()]
    if missing:
        parser.error("Run the exporter for the selected views/seeds/months first. Missing:\n" +
                     "\n".join(missing[:10]) + (f"\n... {len(missing)} missing files total" if len(missing) > 10 else ""))
    references = {}
    class_names = None
    reports = []
    script_hash = file_sha256(Path(__file__).resolve())
    for view, seed, month, path in selections:
        arrays, record = load_export(path, view=view, seed=seed, month=month)
        names = arrays["class_names"].tolist()
        if class_names is not None and names != class_names:
            raise ValueError(f"Class ordering differs across exports: {path}")
        class_names = names
        current = (arrays["sha"], arrays["labels"])
        if month in references and any(not np.array_equal(a, b)
                                       for a, b in zip(current, references[month])):
            raise ValueError(f"Samples/labels/order differ across views or seeds for {month}: {path}")
        references[month] = current
        report = classification_statistics(arrays["labels"], arrays["logits"], names)
        report.update({"report_schema": 1, "view": view, "seed": seed, "month": month,
                       "split": "validation", "temperature": 1.0,
                       "class_names": names,
                       "source_npz": str(path.resolve()), "source_npz_sha256": record["output_sha256"],
                       "source_metadata_sha256": file_sha256(path.with_suffix(".json")),
                       "source_identity": record["identity"], "report_script_sha256": script_hash,
                       "created_utc": datetime.now(timezone.utc).isoformat(),
                       "definitions": {"prediction": "argmax of uncalibrated logits",
                           "undefined_per_class": "null; blank in CSV; '-' in text",
                           "aggregate_undefined": "zero_division=0",
                           "monthly_macro_primary": "eligible families with true support > 0",
                           "fixed_51_macro_secondary": "all 51 eligible families, including support=0",
                           "brier": "mean sum of squared class probability errors; range 0..2",
                           "ece": "15 equal-width bins [lower,upper); last includes 1",
                           "confusion_matrix": "rows=true, columns=predicted; class-ID order"}})
        reports.append(report)
        print(f"CHECKED {view}/seed_{seed}/{month}: {report['n_samples']} samples", flush=True)

    # Validate every selected export before writing any derived reports.
    summaries = []
    for report in reports:
        write_report(args.output_root, report, class_names)
        summaries.append(summary_row(report))
        print(f"REPORT {report['view']}/seed_{report['seed']}/{report['month']}: "
              f"accuracy={report['accuracy']:.4f}, "
              f"macro_F1_supported={report['macro_avg_supported']['f1']:.4f}", flush=True)
    write_text_atomic(args.output_root / "monthly_summary.csv", csv_text(summaries))
    seed_summaries = []
    for view in args.views:
        for month in args.months:
            rows = [r for r in summaries if r["view"] == view and r["month"] == month]
            row = {"view": view, "month": month, "n_seeds": len(rows),
                   "seeds": ",".join(str(r["seed"]) for r in rows),
                   "n_samples_per_seed": rows[0]["n_samples"],
                   "n_supported_classes": rows[0]["n_supported_classes"]}
            for metric in SUMMARY_METRICS:
                values = np.asarray([r[metric] for r in rows])
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
            seed_summaries.append(row)
    write_text_atomic(args.output_root / "seed_summary.csv", csv_text(seed_summaries))
    print(f"Finished: {len(reports)} reports under {args.output_root}. "
          "Seed summaries are mean/sample SD, not an ensemble or confidence interval. "
          "Summary CSVs cover only this command's selections.", flush=True)


if __name__ == "__main__":
    main()
