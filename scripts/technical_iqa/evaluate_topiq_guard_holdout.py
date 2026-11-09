# 검증 세트에서 맞춘 TOPIQ 가드 보정을 홀드아웃에 적용해 평가하는 스크립트
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
DEFAULT_HOLDOUT = "outputs/topiq_teacher_guard_holdout_20260601/predictions.csv"
DEFAULT_OUT_DIR = "outputs/topiq_teacher_guard_holdout_20260601"
SCORE_COL = "quality_score_0_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate guarded TOPIQ target on holdout predictions."
    )
    parser.add_argument("--validation_predictions_csv", default=DEFAULT_VALIDATION)
    parser.add_argument("--holdout_predictions_csv", default=DEFAULT_HOLDOUT)
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
            blocks.append({
                "x_min": raw,
                "x_max": raw,
                "sum": target,
                "weight": 1.0,
            })
            while len(blocks) >= 2:
                prev = blocks[-2]
                curr = blocks[-1]
                if prev["sum"] / prev["weight"] <= curr["sum"] / curr["weight"]:
                    break
                merged = {
                    "x_min": prev["x_min"],
                    "x_max": curr["x_max"],
                    "sum": prev["sum"] + curr["sum"],
                    "weight": prev["weight"] + curr["weight"],
                }
                blocks[-2:] = [merged]
        self.x_max = [float(block["x_max"]) for block in blocks]
        self.values = [clip_0_100(float(block["sum"]) / float(block["weight"])) for block in blocks]

    def predict_one(self, raw: float) -> float:
        index = bisect.bisect_left(self.x_max, raw)
        if index < 0:
            index = 0
        if index >= len(self.values):
            index = len(self.values) - 1
        return self.values[index]


def fit_datasetwise_calibrators(rows: list[dict[str, Any]]) -> dict[str, IsotonicStepCalibrator]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)
    calibrators: dict[str, IsotonicStepCalibrator] = {}
    for dataset, dataset_rows in by_dataset.items():
        calibrators[dataset] = IsotonicStepCalibrator(
            [float(row["topiq_raw"]) for row in dataset_rows],
            [float(row[SCORE_COL]) for row in dataset_rows],
        )
    return calibrators


def apply_guarded_target(
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


def metric_row(
    candidate: str, score_key: str, scope: str, dataset: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    y_true = [float(row[SCORE_COL]) for row in rows]
    y_pred = [float(row[score_key]) for row in rows]
    errors = [abs(pred - truth) for truth, pred in zip(y_true, y_pred)]
    return {
        "candidate": candidate,
        "scope": scope,
        "dataset": dataset,
        "n": len(rows),
        "srcc": spearman(y_true, y_pred) if len(rows) >= 2 else "",
        "plcc": pearson(y_true, y_pred) if len(rows) >= 2 else "",
        "mae": statistics.fmean(errors) if errors else "",
        "false_positive_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth < 50.0 and pred > 65.0),
        "severe_false_positive_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth < 40.0 and pred > 65.0),
        "false_negative_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth > 75.0 and pred < 45.0),
        "score_min": min(y_pred) if y_pred else "",
        "score_max": max(y_pred) if y_pred else "",
        "score_mean": statistics.fmean(y_pred) if y_pred else "",
    }


def compute_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row.get("dataset", ""))].append(row)

    metrics: list[dict[str, Any]] = []
    for candidate, score_key in [
        ("raw_topiq_x100", "topiq_raw_x100"),
        ("guarded_target_0_100", "guarded_target_0_100"),
    ]:
        metrics.append(metric_row(candidate, score_key, "overall", "all", rows))
        for dataset, dataset_rows in sorted(by_dataset.items()):
            metrics.append(metric_row(candidate, score_key, "dataset", dataset, dataset_rows))
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
        if not matched:
            continue
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


def choose_recommendation(metrics: list[dict[str, Any]]) -> tuple[str, str]:
    guarded = next(
        row
        for row in metrics
        if row["candidate"] == "guarded_target_0_100" and row["scope"] == "overall"
    )
    raw = next(
        row
        for row in metrics
        if row["candidate"] == "raw_topiq_x100" and row["scope"] == "overall"
    )
    dataset_rows = [
        row
        for row in metrics
        if row["candidate"] == "guarded_target_0_100" and row["scope"] == "dataset"
    ]
    if guarded["severe_false_positive_count"] != 0:
        return "NEEDS_DIFFERENT_GUARD_TARGET", "Severe false positives are nonzero."
    if guarded["false_positive_count"] > 3:
        return "NEEDS_DIFFERENT_GUARD_TARGET", "False positives are above the bounded dry-run tolerance."
    if guarded["false_negative_count"] > 0:
        return "NEEDS_DIFFERENT_GUARD_TARGET", "False negatives are nonzero."
    raw_safety = (
        raw["false_positive_count"],
        raw["severe_false_positive_count"],
        raw["false_negative_count"],
    )
    guarded_safety = (
        guarded["false_positive_count"],
        guarded["severe_false_positive_count"],
        guarded["false_negative_count"],
    )
    if guarded_safety > raw_safety:
        return "NEEDS_DIFFERENT_GUARD_TARGET", "Guarded target does not improve safety compared with raw TOPIQ."
    for row in dataset_rows:
        if row["srcc"] == "" or float(row["srcc"]) < 0.5 or row["plcc"] == "" or float(row["plcc"]) < 0.5:
            return "DO_NOT_DISTILL_TOPIQ_YET", f"Dataset metric collapse for {row['dataset']}."
    return "READY_FOR_BOUNDED_TEACHER_LABEL_DRY_RUN", "Holdout guard passed severe-FP, FP/FN, and dataset-metric gates."


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_report(
    path: Path,
    validation_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    fp_cases: list[dict[str, Any]],
    fn_cases: list[dict[str, Any]],
    recommendation: str,
    reason: str,
) -> None:
    raw_rows = [row for row in metrics if row["candidate"] == "raw_topiq_x100"]
    guarded_rows = [row for row in metrics if row["candidate"] == "guarded_target_0_100"]
    lines = [
        "# TOPIQ Guarded Holdout Evaluation Report",
        "",
        "## Executive Summary",
        "",
        f"- final recommendation: {recommendation}",
        f"- reason: {reason}",
        "- raw TOPIQ baseline: `topiq_raw * 100`.",
        "- guarded target: `0.5 * datasetwise_isotonic(topiq_raw) + 0.5 * quality_score_0_100`.",
        "- this formula is for teacher-label design/evaluation only and is not a Flutter runtime formula because app images do not have MOS labels.",
        "",
        "## Recommendation",
        "",
        f"- recommendation: {recommendation}",
        f"- reason: {reason}",
        "- selected guard target: E_datasetwise_isotonic__blended_50_50",
        "- calibration fit source: original 900-row validation predictions only",
        "",
        "## Inputs",
        "",
        f"- validation prediction rows loaded: {len(validation_rows)}",
        f"- holdout prediction rows loaded: {len(holdout_rows)}",
        "",
        "## Raw TOPIQ Holdout Metrics",
        "",
        "| scope | dataset | n | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in raw_rows:
        lines.append(
            "| {scope} | {dataset} | {n} | {srcc} | {plcc} | {mae} | {fp} | {sfp} | {fn} | {smin} | {smax} | {smean} |".format(
                scope=row["scope"],
                dataset=row["dataset"],
                n=row["n"],
                srcc=format_float(row["srcc"]),
                plcc=format_float(row["plcc"]),
                mae=format_float(row["mae"]),
                fp=row["false_positive_count"],
                sfp=row["severe_false_positive_count"],
                fn=row["false_negative_count"],
                smin=format_float(row["score_min"]),
                smax=format_float(row["score_max"]),
                smean=format_float(row["score_mean"]),
            )
        )
    lines.extend([
        "",
        "## Guarded TOPIQ Holdout Metrics",
        "",
        "| scope | dataset | n | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in guarded_rows:
        lines.append(
            "| {scope} | {dataset} | {n} | {srcc} | {plcc} | {mae} | {fp} | {sfp} | {fn} | {smin} | {smax} | {smean} |".format(
                scope=row["scope"],
                dataset=row["dataset"],
                n=row["n"],
                srcc=format_float(row["srcc"]),
                plcc=format_float(row["plcc"]),
                mae=format_float(row["mae"]),
                fp=row["false_positive_count"],
                sfp=row["severe_false_positive_count"],
                fn=row["false_negative_count"],
                smin=format_float(row["score_min"]),
                smax=format_float(row["score_max"]),
                smean=format_float(row["score_mean"]),
            )
        )
    lines.extend([
        "",
        "## Pass Criteria",
        "",
        "- severe FP count must be 0.",
        "- FP count should be as low as possible, preferably 0.",
        "- No major dataset collapse, especially FLIVE.",
        "- Dataset-wise metrics are reported separately above.",
        "",
        "## Case Files",
        "",
        f"- false positive cases: {len(fp_cases)}",
        f"- false negative cases: {len(fn_cases)}",
        "",
        "## Boundary",
        "",
        "This is still a bounded holdout/dry-run evaluation. Do not train a student or generate full teacher labels from this report alone.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = read_predictions(Path(args.validation_predictions_csv))
    holdout_rows = read_predictions(Path(args.holdout_predictions_csv))
    calibrators = fit_datasetwise_calibrators(validation_rows)
    guarded_rows = apply_guarded_target(holdout_rows, calibrators)
    metrics = compute_metrics(guarded_rows)
    fp_cases = collect_cases(guarded_rows, "false_positive")
    fn_cases = collect_cases(guarded_rows, "false_negative")
    recommendation, reason = choose_recommendation(metrics)

    predictions_csv = out_dir / "guarded_holdout_predictions.csv"
    metrics_csv = out_dir / "guarded_holdout_metrics.csv"
    fp_csv = out_dir / "false_positive_cases.csv"
    fn_csv = out_dir / "false_negative_cases.csv"
    report_md = out_dir / "report.md"

    prediction_fields = [
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
        "candidate",
        "scope",
        "dataset",
        "n",
        "srcc",
        "plcc",
        "mae",
        "false_positive_count",
        "severe_false_positive_count",
        "false_negative_count",
        "score_min",
        "score_max",
        "score_mean",
    ]
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
    write_csv(predictions_csv, guarded_rows, prediction_fields)
    write_csv(metrics_csv, metrics, metric_fields)
    write_csv(fp_csv, fp_cases, case_fields)
    write_csv(fn_csv, fn_cases, case_fields)
    write_report(report_md, validation_rows, guarded_rows, metrics, fp_cases, fn_cases, recommendation, reason)

    print(f"Wrote {predictions_csv}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {fp_csv}")
    print(f"Wrote {fn_csv}")
    print(f"Wrote {report_md}")
    print(f"recommendation={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
