# 검증 보정으로 TOPIQ 50k 확장 예측에 가드 교사 라벨을 적용하는 스크립트
from __future__ import annotations

import argparse
import bisect
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_VALIDATION = "outputs/topiq_teacher_validation_20260531/predictions.csv"
DEFAULT_EXPANSION = "outputs/topiq_teacher_label_expansion_50k_20260601/predictions.csv"
DEFAULT_OUT_DIR = "outputs/topiq_teacher_label_expansion_50k_20260601"
SCORE_COL = "quality_score_0_100"
SCORE_COLUMNS = ["topiq_raw_x100", "calibrated_topiq_0_100", "guarded_target_0_100"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply dataset-wise isotonic guarded TOPIQ labels to 50k expansion predictions.")
    parser.add_argument("--validation_predictions_csv", default=DEFAULT_VALIDATION)
    parser.add_argument("--expansion_predictions_csv", default=DEFAULT_EXPANSION)
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


def source_group_from_path(image_path: str) -> str:
    filename = Path(image_path).name
    normalized = image_path.replace("\\", "/")
    if "/patches/" in normalized and "_patch_" in filename:
        return filename.split("_patch_", 1)[0]
    return filename


def read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = safe_float(row.get(SCORE_COL))
            raw = safe_float(row.get("topiq_raw"))
            if truth is None or raw is None:
                continue
            item = dict(row)
            item[SCORE_COL] = truth
            item["topiq_raw"] = raw
            item["source_group"] = source_group_from_path(str(row.get("image_path", "")))
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


def apply_guarded_labels(rows: list[dict[str, Any]], calibrators: dict[str, IsotonicStepCalibrator]) -> list[dict[str, Any]]:
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


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": "", "max": "", "mean": "", "std": ""}
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def metric_row(scope: str, dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [float(row[SCORE_COL]) for row in rows]
    y_pred = [float(row["guarded_target_0_100"]) for row in rows]
    errors = [abs(pred - truth) for truth, pred in zip(y_true, y_pred)]
    source_counts = Counter(str(row.get("source_group", "")) for row in rows)
    base = {
        "scope": scope,
        "dataset": dataset,
        "n": len(rows),
        "source_group_count": len(source_counts),
        "multi_row_source_group_count": sum(1 for count in source_counts.values() if count > 1),
        "srcc": spearman(y_true, y_pred) if len(rows) >= 2 else "",
        "plcc": pearson(y_true, y_pred) if len(rows) >= 2 else "",
        "mae": statistics.fmean(errors) if errors else "",
        "false_positive_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth < 50.0 and pred > 65.0),
        "severe_false_positive_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth < 40.0 and pred > 65.0),
        "false_negative_count": sum(1 for truth, pred in zip(y_true, y_pred) if truth > 75.0 and pred < 45.0),
    }
    for score_col in SCORE_COLUMNS:
        stats = describe([float(row[score_col]) for row in rows])
        for key, value in stats.items():
            base[f"{score_col}_{key}"] = value
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
                "source_group": row.get("source_group", ""),
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
                count = sum(1 for value in values if low <= value <= high) if high == 100 else sum(1 for value in values if low <= value < high)
                result.append({"dataset": dataset, "score_column": score_col, "bin_low": low, "bin_high": high, "count": count})
    return result


def choose_recommendation(metrics: list[dict[str, Any]]) -> tuple[str, str]:
    overall = next(row for row in metrics if row["scope"] == "overall")
    dataset_rows = [row for row in metrics if row["scope"] == "dataset"]
    if overall["severe_false_positive_count"] != 0:
        return "NEEDS_GUARD_REVISION", "Severe false positives are nonzero."
    if overall["false_positive_count"] > 100:
        return "NEEDS_GUARD_REVISION", "False positives are high for the 50k expansion."
    for row in dataset_rows:
        if row["srcc"] == "" or float(row["srcc"]) < 0.5 or row["plcc"] == "" or float(row["plcc"]) < 0.5:
            return "DO_NOT_USE_EXPANDED_LABELS", f"Dataset metric collapse for {row['dataset']}."
    guarded_min = float(overall["guarded_target_0_100_min"])
    guarded_max = float(overall["guarded_target_0_100_max"])
    guarded_mean = float(overall["guarded_target_0_100_mean"])
    if guarded_min < 0.0 or guarded_max > 100.0 or not 20.0 <= guarded_mean <= 85.0:
        return "NEEDS_GUARD_REVISION", "Expanded label distribution is outside sanity bounds."
    return "READY_FOR_50K_STUDENT_MANIFEST", "Expanded guarded labels passed basic safety, dataset metric, and distribution checks."


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value):.6f}"


def write_report(
    path: Path,
    validation_rows: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    fp_cases: list[dict[str, Any]],
    fn_cases: list[dict[str, Any]],
    hist_csv: Path,
    recommendation: str,
    reason: str,
) -> None:
    lines = [
        "# TOPIQ Guarded Label 50k Expansion Report",
        "",
        "## Executive Summary",
        f"- final recommendation: {recommendation}",
        f"- reason: {reason}",
        "- selected target: E_datasetwise_isotonic__blended_50_50",
        "- calibration fit source: validation predictions only",
        "- this target is for teacher-label design/evaluation only, not a Flutter runtime formula.",
        "",
        "## Inputs",
        f"- validation prediction rows loaded: {len(validation_rows)}",
        f"- expansion prediction rows loaded: {len(expansion_rows)}",
        "",
        "## Metrics",
        "| scope | dataset | n | source groups | multi-row groups | SRCC | PLCC | MAE | FP | severe FP | FN | raw mean | calibrated mean | guarded mean | guarded std | guarded min | guarded max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {scope} | {dataset} | {n} | {groups} | {multi} | {srcc} | {plcc} | {mae} | {fp} | {sfp} | {fn} | {raw_mean} | {cal_mean} | {guard_mean} | {guard_std} | {guard_min} | {guard_max} |".format(
                scope=row["scope"],
                dataset=row["dataset"],
                n=row["n"],
                groups=row["source_group_count"],
                multi=row["multi_row_source_group_count"],
                srcc=format_float(row["srcc"]),
                plcc=format_float(row["plcc"]),
                mae=format_float(row["mae"]),
                fp=row["false_positive_count"],
                sfp=row["severe_false_positive_count"],
                fn=row["false_negative_count"],
                raw_mean=format_float(row["topiq_raw_x100_mean"]),
                cal_mean=format_float(row["calibrated_topiq_0_100_mean"]),
                guard_mean=format_float(row["guarded_target_0_100_mean"]),
                guard_std=format_float(row["guarded_target_0_100_std"]),
                guard_min=format_float(row["guarded_target_0_100_min"]),
                guard_max=format_float(row["guarded_target_0_100_max"]),
            )
        )
    lines.extend([
        "",
        "## False Case Summary",
        f"- false positive cases: {len(fp_cases)}",
        f"- severe false positive cases: {sum(1 for row in fp_cases if row['severity'] == 'severe')}",
        f"- false negative cases: {len(fn_cases)}",
        f"- score histogram CSV: {hist_csv}",
        "",
        "## Final Recommendation",
        f"- {recommendation}",
        f"- {reason}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = read_predictions(Path(args.validation_predictions_csv))
    expansion_rows = read_predictions(Path(args.expansion_predictions_csv))
    calibrators = fit_datasetwise_calibrators(validation_rows)
    guarded_rows = apply_guarded_labels(expansion_rows, calibrators)
    metrics = compute_metrics(guarded_rows)
    fp_cases = collect_cases(guarded_rows, "false_positive")
    fn_cases = collect_cases(guarded_rows, "false_negative")
    hist_rows = histogram_rows(guarded_rows)
    recommendation, reason = choose_recommendation(metrics)

    guarded_fields = list(guarded_rows[0].keys()) if guarded_rows else [
        "dataset", "split", "image_path", SCORE_COL, "topiq_raw", "source_group",
        "topiq_raw_x100", "calibrated_topiq_0_100", "guarded_target_0_100",
    ]
    write_csv(out_dir / "guarded_labels.csv", guarded_rows, guarded_fields)
    write_csv(out_dir / "guarded_label_metrics.csv", metrics, list(metrics[0].keys()) if metrics else [])
    write_csv(out_dir / "score_histograms.csv", hist_rows, ["dataset", "score_column", "bin_low", "bin_high", "count"])
    case_fields = ["case_type", "severity", "dataset", "split", "source_group", "image_path", SCORE_COL, "topiq_raw", "topiq_raw_x100", "calibrated_topiq_0_100", "guarded_target_0_100", "gap"]
    write_csv(out_dir / "false_positive_cases.csv", fp_cases, case_fields)
    write_csv(out_dir / "false_negative_cases.csv", fn_cases, case_fields)
    write_report(out_dir / "report.md", validation_rows, expansion_rows, metrics, fp_cases, fn_cases, out_dir / "score_histograms.csv", recommendation, reason)
    print(f"Wrote guarded labels and report under {out_dir}")
    print(f"Final recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
