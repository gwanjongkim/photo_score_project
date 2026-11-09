# 검증 보정으로 TOPIQ 드라이런 예측에 가드 교사 라벨을 적용하는 스크립트
from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_VALIDATION = "outputs/topiq_teacher_validation_20260531/predictions.csv"
DEFAULT_DRYRUN = "outputs/topiq_teacher_label_dryrun_20260601/predictions.csv"
DEFAULT_OUT_DIR = "outputs/topiq_teacher_label_dryrun_20260601"
SCORE_COL = "quality_score_0_100"
SCORE_COLUMNS = ["topiq_raw_x100", "calibrated_topiq_0_100", "guarded_target_0_100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply dataset-wise isotonic guarded TOPIQ labels to dry-run predictions."
    )
    parser.add_argument("--validation_predictions_csv", default=DEFAULT_VALIDATION)
    parser.add_argument("--dryrun_predictions_csv", default=DEFAULT_DRYRUN)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def clip_0_100(value: float) -> float:
    return min(100.0, max(0.0, value))


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth = safe_float(row.get(SCORE_COL))
            raw = safe_float(row.get("topiq_raw"))
            if truth is None or raw is None:
                continue
            item = dict(row)
            item[SCORE_COL] = truth
            item["topiq_raw"] = raw
            rows.append(item)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    denom_x = sum(value * value for value in dx)
    denom_y = sum(value * value for value in dy)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(denom_x * denom_y)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rankdata(x), rankdata(y))


class IsotonicStepCalibrator:
    def __init__(self, x: list[float], y: list[float]) -> None:
        if not x:
            raise ValueError("Cannot fit isotonic calibrator with no rows")
        ordered = sorted(zip(x, y), key=lambda item: item[0])
        blocks: list[dict[str, Any]] = []
        for raw, target in ordered:
            blocks.append({"x_max": raw, "sum": target, "weight": 1.0})
            while len(blocks) >= 2:
                prev = blocks[-2]
                curr = blocks[-1]
                if prev["sum"] / prev["weight"] <= curr["sum"] / curr["weight"]:
                    break
                blocks[-2:] = [{
                    "x_max": curr["x_max"],
                    "sum": prev["sum"] + curr["sum"],
                    "weight": prev["weight"] + curr["weight"],
                }]
        self.x_max = [float(block["x_max"]) for block in blocks]
        self.values = [clip_0_100(float(block["sum"]) / float(block["weight"])) for block in blocks]

    def predict_one(self, raw: float) -> float:
        index = bisect.bisect_left(self.x_max, raw)
        if index >= len(self.values):
            index = len(self.values) - 1
        return self.values[index]


def fit_datasetwise_calibrators(rows: list[dict[str, Any]]) -> dict[str, IsotonicStepCalibrator]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    return {
        dataset: IsotonicStepCalibrator(
            [float(row["topiq_raw"]) for row in dataset_rows],
            [float(row[SCORE_COL]) for row in dataset_rows],
        )
        for dataset, dataset_rows in by_dataset.items()
    }


def apply_guarded_labels(
    rows: list[dict[str, Any]], calibrators: dict[str, IsotonicStepCalibrator]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in calibrators:
            raise ValueError(f"No validation calibrator fitted for dataset: {dataset}")
        truth = float(row[SCORE_COL])
        raw = float(row["topiq_raw"])
        calibrated = calibrators[dataset].predict_one(raw)
        guarded = 0.5 * calibrated + 0.5 * truth
        item = dict(row)
        item["topiq_raw_x100"] = clip_0_100(raw * 100.0)
        item["calibrated_topiq_0_100"] = calibrated
        item["guarded_target_0_100"] = guarded
        output.append(item)
    return output


def metric_row(scope: str, dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [float(row[SCORE_COL]) for row in rows]
    y_pred = [float(row["guarded_target_0_100"]) for row in rows]
    errors = [abs(pred - truth) for truth, pred in zip(y_true, y_pred)]
    base = {
        "scope": scope,
        "dataset": dataset,
        "n": len(rows),
        "srcc": spearman(y_true, y_pred) if len(rows) >= 2 else "",
        "plcc": pearson(y_true, y_pred) if len(rows) >= 2 else "",
        "mae": statistics.fmean(errors) if errors else "",
        "false_positive_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth < 50.0 and pred > 65.0),
        "severe_false_positive_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth < 40.0 and pred > 65.0),
        "false_negative_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth > 75.0 and pred < 45.0),
    }
    for score_col in SCORE_COLUMNS:
        values = [float(row[score_col]) for row in rows]
        base[f"{score_col}_min"] = min(values) if values else ""
        base[f"{score_col}_max"] = max(values) if values else ""
        base[f"{score_col}_mean"] = statistics.fmean(values) if values else ""
    return base


def compute_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [metric_row("overall", "all", rows)]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        metrics.append(metric_row("dataset", dataset, dataset_rows))
    return metrics


def collect_cases(rows: list[dict[str, Any]], case_type: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        truth = float(row[SCORE_COL])
        pred = float(row["guarded_target_0_100"])
        if case_type == "false_positive":
            matched = truth < 50.0 and pred > 65.0
            severity = "severe" if truth < 40.0 else "standard"
            gap = pred - truth
        else:
            matched = truth > 75.0 and pred < 45.0
            severity = "standard"
            gap = truth - pred
        if matched:
            cases.append({
                "case_type": case_type,
                "severity": severity,
                "dataset": row.get("dataset", ""),
                "split": row.get("split", ""),
                "image_path": row.get("image_path", ""),
                SCORE_COL: truth,
                "topiq_raw": row.get("topiq_raw", ""),
                "topiq_raw_x100": row.get("topiq_raw_x100", ""),
                "calibrated_topiq_0_100": row.get("calibrated_topiq_0_100", ""),
                "guarded_target_0_100": pred,
                "gap": gap,
            })
    return sorted(cases, key=lambda item: float(item["gap"]), reverse=True)


def histogram_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bins = [(start, start + 10) for start in range(0, 100, 10)]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_dataset["all"] = rows
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        for score_col in SCORE_COLUMNS:
            values = [float(row[score_col]) for row in dataset_rows]
            for low, high in bins:
                if high == 100:
                    count = sum(1 for value in values if low <= value <= high)
                else:
                    count = sum(1 for value in values if low <= value < high)
                result.append({
                    "dataset": dataset,
                    "score_column": score_col,
                    "bin_low": low,
                    "bin_high": high,
                    "count": count,
                })
    return result


def choose_recommendation(metrics: list[dict[str, Any]]) -> tuple[str, str]:
    overall = next(row for row in metrics if row["scope"] == "overall")
    dataset_rows = [row for row in metrics if row["scope"] == "dataset"]
    if overall["severe_false_positive_count"] != 0:
        return "NEEDS_GUARD_REVISION", "Severe false positives are nonzero."
    if overall["false_positive_count"] > 10:
        return "NEEDS_GUARD_REVISION", "False positives are above dry-run tolerance."
    for row in dataset_rows:
        if row["srcc"] == "" or float(row["srcc"]) < 0.5 or row["plcc"] == "" or float(row["plcc"]) < 0.5:
            return "DO_NOT_DISTILL_TOPIQ_YET", f"Dataset metric collapse for {row['dataset']}."
    guarded_mean = float(overall["guarded_target_0_100_mean"])
    guarded_min = float(overall["guarded_target_0_100_min"])
    guarded_max = float(overall["guarded_target_0_100_max"])
    if guarded_min < 0.0 or guarded_max > 100.0 or not 20.0 <= guarded_mean <= 85.0:
        return "NEEDS_GUARD_REVISION", "Dry-run label distribution is not sane."
    return "READY_FOR_STUDENT_TRAINING_MANIFEST_DESIGN", "Bounded dry-run passed safety, dataset metric, and distribution checks."


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_report(
    path: Path,
    validation_rows: list[dict[str, Any]],
    dryrun_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    fp_cases: list[dict[str, Any]],
    fn_cases: list[dict[str, Any]],
    hist_csv: Path,
    recommendation: str,
    reason: str,
) -> None:
    lines = [
        "# TOPIQ Guarded Label Dry-Run Report",
        "",
        "## Executive Summary",
        "",
        f"- final recommendation: {recommendation}",
        f"- reason: {reason}",
        "- selected target: E_datasetwise_isotonic__blended_50_50",
        "- calibration fit source: validation predictions only",
        "- this target is for teacher-label design/evaluation only, not a Flutter runtime formula.",
        "",
        "## Inputs",
        "",
        f"- validation prediction rows loaded: {len(validation_rows)}",
        f"- dry-run prediction rows loaded: {len(dryrun_rows)}",
        "",
        "## Metrics",
        "",
        "| scope | dataset | n | SRCC | PLCC | MAE | FP | severe FP | FN | raw mean | calibrated mean | guarded mean | guarded min | guarded max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {scope} | {dataset} | {n} | {srcc} | {plcc} | {mae} | {fp} | {sfp} | {fn} | {raw_mean} | {cal_mean} | {guard_mean} | {guard_min} | {guard_max} |".format(
                scope=row["scope"],
                dataset=row["dataset"],
                n=row["n"],
                srcc=format_float(row["srcc"]),
                plcc=format_float(row["plcc"]),
                mae=format_float(row["mae"]),
                fp=row["false_positive_count"],
                sfp=row["severe_false_positive_count"],
                fn=row["false_negative_count"],
                raw_mean=format_float(row["topiq_raw_x100_mean"]),
                cal_mean=format_float(row["calibrated_topiq_0_100_mean"]),
                guard_mean=format_float(row["guarded_target_0_100_mean"]),
                guard_min=format_float(row["guarded_target_0_100_min"]),
                guard_max=format_float(row["guarded_target_0_100_max"]),
            )
        )
    lines.extend([
        "",
        "## False Case Summary",
        "",
        f"- false positive cases: {len(fp_cases)}",
        f"- false negative cases: {len(fn_cases)}",
        "",
        "## Histogram Output",
        "",
        f"- histogram CSV: {hist_csv}",
        "",
        "## Boundary",
        "",
        "Do not train a student from this report alone. The next step is training-manifest design if the recommendation is READY.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = parse_args()
    out_dir = Path(parser.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = read_predictions(Path(parser.validation_predictions_csv))
    dryrun_rows = read_predictions(Path(parser.dryrun_predictions_csv))
    calibrators = fit_datasetwise_calibrators(validation_rows)
    guarded_rows = apply_guarded_labels(dryrun_rows, calibrators)
    metrics = compute_metrics(guarded_rows)
    fp_cases = collect_cases(guarded_rows, "false_positive")
    fn_cases = collect_cases(guarded_rows, "false_negative")
    hist_rows = histogram_rows(guarded_rows)
    recommendation, reason = choose_recommendation(metrics)

    labels_csv = out_dir / "guarded_labels.csv"
    metrics_csv = out_dir / "guarded_label_metrics.csv"
    fp_csv = out_dir / "false_positive_cases.csv"
    fn_csv = out_dir / "false_negative_cases.csv"
    hist_csv = out_dir / "score_histograms.csv"
    report_md = out_dir / "report.md"

    label_fields = [
        "dataset",
        "split",
        "image_path",
        SCORE_COL,
        "topiq_raw",
        "topiq_raw_x100",
        "calibrated_topiq_0_100",
        "guarded_target_0_100",
    ]
    metric_fields = [
        "scope",
        "dataset",
        "n",
        "srcc",
        "plcc",
        "mae",
        "false_positive_count",
        "severe_false_positive_count",
        "false_negative_count",
    ]
    for score_col in SCORE_COLUMNS:
        metric_fields.extend([f"{score_col}_min", f"{score_col}_max", f"{score_col}_mean"])
    case_fields = [
        "case_type",
        "severity",
        "dataset",
        "split",
        "image_path",
        SCORE_COL,
        "topiq_raw",
        "topiq_raw_x100",
        "calibrated_topiq_0_100",
        "guarded_target_0_100",
        "gap",
    ]
    write_csv(labels_csv, guarded_rows, label_fields)
    write_csv(metrics_csv, metrics, metric_fields)
    write_csv(fp_csv, fp_cases, case_fields)
    write_csv(fn_csv, fn_cases, case_fields)
    write_csv(hist_csv, hist_rows, ["dataset", "score_column", "bin_low", "bin_high", "count"])
    write_report(report_md, validation_rows, guarded_rows, metrics, fp_cases, fn_cases, hist_csv, recommendation, reason)
    print(f"Wrote {labels_csv}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {fp_csv}")
    print(f"Wrote {fn_csv}")
    print(f"Wrote {hist_csv}")
    print(f"Wrote {report_md}")
    print(f"recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
