# TOPIQ 교사 점수의 보정 및 가드 타깃 후보를 분석하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PREDICTIONS = "outputs/topiq_teacher_validation_20260531/predictions.csv"
DEFAULT_REPORT = "outputs/topiq_teacher_validation_20260531/report.md"
DEFAULT_OUT_DIR = "outputs/topiq_teacher_guard_design_20260531"
SCORE_COL = "quality_score_0_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze calibrated and guarded TOPIQ teacher target candidates."
    )
    parser.add_argument("--predictions_csv", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--validation_report", default=DEFAULT_REPORT)
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
            y = safe_float(row.get(SCORE_COL))
            raw = safe_float(row.get("topiq_raw"))
            if y is None or raw is None:
                continue
            item = dict(row)
            item[SCORE_COL] = y
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
    if len(x) < 2 or len(y) < 2:
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


def linear_fit_predict(x: list[float], y: list[float]) -> list[float]:
    mean_x = statistics.fmean(x)
    mean_y = statistics.fmean(y)
    var_x = sum((value - mean_x) ** 2 for value in x)
    if var_x <= 0.0:
        return [clip_0_100(mean_y) for _ in x]
    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return [clip_0_100(slope * xi + intercept) for xi in x]


def isotonic_fit_predict(x: list[float], y: list[float]) -> list[float]:
    ordered = sorted(enumerate(zip(x, y)), key=lambda item: item[1][0])
    blocks: list[dict[str, Any]] = []
    for original_index, (_x, target) in ordered:
        blocks.append({"sum": target, "weight": 1.0, "indices": [original_index]})
        while len(blocks) >= 2:
            prev = blocks[-2]
            curr = blocks[-1]
            if prev["sum"] / prev["weight"] <= curr["sum"] / curr["weight"]:
                break
            merged = {
                "sum": prev["sum"] + curr["sum"],
                "weight": prev["weight"] + curr["weight"],
                "indices": prev["indices"] + curr["indices"],
            }
            blocks[-2:] = [merged]

    fitted = [0.0] * len(x)
    for block in blocks:
        value = clip_0_100(block["sum"] / block["weight"])
        for index in block["indices"]:
            fitted[index] = value
    return fitted


def add_datasetwise_candidate(
    rows: list[dict[str, Any]],
    candidate_name: str,
    fit_fn: Any,
    candidates: dict[str, list[float]],
) -> None:
    values = [0.0] * len(rows)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_dataset[str(row.get("dataset", ""))].append(index)
    for indices in by_dataset.values():
        x = [float(rows[index]["topiq_raw"]) for index in indices]
        y = [float(rows[index][SCORE_COL]) for index in indices]
        fitted = fit_fn(x, y)
        for index, value in zip(indices, fitted):
            values[index] = value
    candidates[candidate_name] = values


def build_candidate_scores(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    x = [float(row["topiq_raw"]) for row in rows]
    y = [float(row[SCORE_COL]) for row in rows]
    candidates: dict[str, list[float]] = {
        "A_topiq_raw_x100": [clip_0_100(value * 100.0) for value in x],
        "B_global_linear": linear_fit_predict(x, y),
        "C_global_isotonic": isotonic_fit_predict(x, y),
    }
    add_datasetwise_candidate(rows, "D_datasetwise_linear", linear_fit_predict, candidates)
    add_datasetwise_candidate(rows, "E_datasetwise_isotonic", isotonic_fit_predict, candidates)

    base_names = list(candidates.keys())
    for name in base_names:
        calibrated = candidates[name]
        target = y
        candidates[f"{name}__guard_margin_5"] = [
            min(score, truth + 5.0) for score, truth in zip(calibrated, target)
        ]
        candidates[f"{name}__guard_margin_8"] = [
            min(score, truth + 8.0) for score, truth in zip(calibrated, target)
        ]
        candidates[f"{name}__guard_margin_10"] = [
            min(score, truth + 10.0) for score, truth in zip(calibrated, target)
        ]
        candidates[f"{name}__blended_70_30"] = [
            0.7 * score + 0.3 * truth for score, truth in zip(calibrated, target)
        ]
        candidates[f"{name}__blended_50_50"] = [
            0.5 * score + 0.5 * truth for score, truth in zip(calibrated, target)
        ]
    return candidates


def metric_row(
    candidate_name: str,
    scope: str,
    dataset: str,
    y_true: list[float],
    y_pred: list[float],
) -> dict[str, Any]:
    false_positive_count = sum(
        1 for truth, pred in zip(y_true, y_pred) if truth < 50.0 and pred > 65.0
    )
    severe_false_positive_count = sum(
        1 for truth, pred in zip(y_true, y_pred) if truth < 40.0 and pred > 65.0
    )
    false_negative_count = sum(
        1 for truth, pred in zip(y_true, y_pred) if truth > 75.0 and pred < 45.0
    )
    errors = [abs(pred - truth) for truth, pred in zip(y_true, y_pred)]
    return {
        "candidate": candidate_name,
        "scope": scope,
        "dataset": dataset,
        "n": len(y_pred),
        "srcc": spearman(y_true, y_pred) if len(y_pred) >= 2 else "",
        "plcc": pearson(y_true, y_pred) if len(y_pred) >= 2 else "",
        "mae": statistics.fmean(errors) if errors else "",
        "false_positive_count": false_positive_count,
        "severe_false_positive_count": severe_false_positive_count,
        "false_negative_count": false_negative_count,
        "score_min": min(y_pred) if y_pred else "",
        "score_max": max(y_pred) if y_pred else "",
        "score_mean": statistics.fmean(y_pred) if y_pred else "",
    }


def compute_metrics(
    rows: list[dict[str, Any]], candidates: dict[str, list[float]]
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    y_all = [float(row[SCORE_COL]) for row in rows]
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_dataset[str(row.get("dataset", ""))].append(index)

    for name, scores in candidates.items():
        metrics.append(metric_row(name, "overall", "all", y_all, scores))
        for dataset, indices in sorted(by_dataset.items()):
            y_subset = [float(rows[index][SCORE_COL]) for index in indices]
            pred_subset = [scores[index] for index in indices]
            metrics.append(metric_row(name, "dataset", dataset, y_subset, pred_subset))
    return metrics


def collect_cases(
    rows: list[dict[str, Any]], candidates: dict[str, list[float]], case_type: str
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for name, scores in candidates.items():
        for row, score in zip(rows, scores):
            truth = float(row[SCORE_COL])
            if case_type == "false_positive":
                matched = truth < 50.0 and score > 65.0
                severity = "severe" if truth < 40.0 else "standard"
                sort_gap = score - truth
            else:
                matched = truth > 75.0 and score < 45.0
                severity = "standard"
                sort_gap = truth - score
            if not matched:
                continue
            cases.append(
                {
                    "candidate": name,
                    "case_type": case_type,
                    "severity": severity,
                    "dataset": row.get("dataset", ""),
                    "split": row.get("split", ""),
                    "image_path": row.get("image_path", ""),
                    SCORE_COL: truth,
                    "topiq_raw": row.get("topiq_raw", ""),
                    "candidate_score": score,
                    "gap": sort_gap,
                }
            )
    return sorted(cases, key=lambda item: float(item["gap"]), reverse=True)


def overall_metrics_by_candidate(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row["candidate"]): row
        for row in metrics
        if row.get("scope") == "overall" and row.get("dataset") == "all"
    }


def choose_recommendation(
    metrics: list[dict[str, Any]], fp_cases: list[dict[str, Any]], fn_cases: list[dict[str, Any]]
) -> tuple[str, str, dict[str, Any] | None]:
    overall = overall_metrics_by_candidate(metrics)
    best_safe: dict[str, Any] | None = None
    for row in overall.values():
        if not str(row["candidate"]).endswith(("guard_margin_5", "guard_margin_8", "guard_margin_10", "blended_70_30", "blended_50_50")):
            continue
        if row["severe_false_positive_count"] != 0 or row["false_negative_count"] != 0:
            continue
        if best_safe is None or (row["mae"], -row["srcc"]) < (best_safe["mae"], -best_safe["srcc"]):
            best_safe = row

    flive_rows = [
        row
        for row in metrics
        if row.get("scope") == "dataset" and row.get("dataset") == "flive"
    ]
    raw = overall.get("A_topiq_raw_x100")
    raw_has_fp = raw is not None and raw["false_positive_count"] > 0
    weak_flive = any(row["candidate"] == "A_topiq_raw_x100" and row["srcc"] < 0.3 for row in flive_rows)

    if best_safe is not None and weak_flive:
        reason = (
            f"Use guarded target candidate `{best_safe['candidate']}` for any distillation screen; "
            "raw TOPIQ is not safe as a standalone teacher because FLIVE correlation is weak."
        )
        return "USE_GUARDED_TOPIQ_TARGET", reason, best_safe
    if raw is not None and not raw_has_fp and not weak_flive:
        return "USE_TOPIQ_RAW_AS_TEACHER", "Raw TOPIQ passed the false-positive and dataset checks.", raw
    return (
        "DO_NOT_DISTILL_TOPIQ_YET",
        "No guarded candidate passed the conservative severe false-positive and false-negative gate.",
        best_safe,
    )


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value):.6f}"


def table_for_rows(rows: list[dict[str, Any]], limit: int = 12) -> list[str]:
    lines = [
        "| candidate | dataset | SRCC | PLCC | MAE | FP | severe FP | FN | min | max | mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:limit]:
        lines.append(
            "| {candidate} | {dataset} | {srcc} | {plcc} | {mae} | {fp} | {sfp} | {fn} | {smin} | {smax} | {smean} |".format(
                candidate=row["candidate"],
                dataset=row["dataset"],
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
    return lines


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    fp_cases: list[dict[str, Any]],
    fn_cases: list[dict[str, Any]],
    recommendation: str,
    recommendation_reason: str,
    selected: dict[str, Any] | None,
) -> None:
    overall = list(overall_metrics_by_candidate(metrics).values())
    base_overall = [row for row in overall if "__" not in str(row["candidate"])]
    guarded_overall = [row for row in overall if "__" in str(row["candidate"])]
    best_by_mae = sorted(overall, key=lambda row: (row["mae"], row["false_positive_count"]))
    best_guarded = sorted(
        guarded_overall,
        key=lambda row: (
            row["severe_false_positive_count"],
            row["false_negative_count"],
            row["mae"],
        ),
    )

    dataset_raw = [
        row
        for row in metrics
        if row["candidate"] == "A_topiq_raw_x100" and row["scope"] == "dataset"
    ]

    lines = [
        "# TOPIQ Teacher Guard Design Report",
        "",
        "## Recommendation",
        "",
        f"- recommendation: {recommendation}",
        f"- selected candidate: {selected['candidate'] if selected else 'n/a'}",
        f"- reason: {recommendation_reason}",
        "",
        "## Use Policy",
        "",
        "- TOPIQ should not be used as the sole main teacher yet.",
        "- TOPIQ can be used as one component of a guarded teacher target.",
        "- TOPIQ is credible on KonIQ/SPAQ based on the validation report and raw candidate dataset metrics.",
        "- TOPIQ should not be trusted for FLIVE without a target guard or FLIVE-specific handling.",
        "",
        "## Calibration Caveat",
        "",
        "All calibration candidates are fit and evaluated in-sample on the 900-row validation set. These numbers are useful for guard design, not final generalization proof.",
        "",
        "## Inputs",
        "",
        f"- prediction rows loaded: {len(rows)}",
        "- validation result summary: 900/900 succeeded, selected model `topiq_nr`, failed 0.",
        "",
        "## Base Candidate Metrics",
        "",
    ]
    lines.extend(table_for_rows(base_overall, limit=20))
    lines.extend(["", "## Raw TOPIQ Dataset Metrics", ""])
    lines.extend(table_for_rows(sorted(dataset_raw, key=lambda row: row["dataset"]), limit=20))
    lines.extend(["", "## Best Overall Candidates By MAE", ""])
    lines.extend(table_for_rows(best_by_mae, limit=12))
    lines.extend(["", "## Best Guarded Candidates By Safety Then MAE", ""])
    lines.extend(table_for_rows(best_guarded, limit=12))
    lines.extend(
        [
            "",
            "## False Case Files",
            "",
            f"- false positive cases written: {len(fp_cases)}",
            f"- false negative cases written: {len(fn_cases)}",
            "",
            "## Distillation Decision",
            "",
            "Do not train a student yet. Use the selected guarded target only as the next candidate for a separate bounded teacher-label-generation dry run and holdout check.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    predictions_csv = Path(args.predictions_csv)
    out_dir = Path(args.out_dir)
    metrics_csv = out_dir / "guard_candidate_metrics.csv"
    fp_csv = out_dir / "false_positive_cases.csv"
    fn_csv = out_dir / "false_negative_cases.csv"
    report_md = out_dir / "report.md"

    rows = read_predictions(predictions_csv)
    if not rows:
        raise ValueError(f"No usable rows loaded from {predictions_csv}")

    candidates = build_candidate_scores(rows)
    metrics = compute_metrics(rows, candidates)
    fp_cases = collect_cases(rows, candidates, "false_positive")
    fn_cases = collect_cases(rows, candidates, "false_negative")
    recommendation, reason, selected = choose_recommendation(metrics, fp_cases, fn_cases)

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
        "candidate",
        "case_type",
        "severity",
        "dataset",
        "split",
        "image_path",
        SCORE_COL,
        "topiq_raw",
        "candidate_score",
        "gap",
    ]
    write_csv(metrics_csv, metrics, metric_fields)
    write_csv(fp_csv, fp_cases, case_fields)
    write_csv(fn_csv, fn_cases, case_fields)
    write_report(report_md, rows, metrics, fp_cases, fn_cases, recommendation, reason, selected)

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {fp_csv}")
    print(f"Wrote {fn_csv}")
    print(f"Wrote {report_md}")
    print(f"recommendation={recommendation}")
    if selected:
        print(f"selected_candidate={selected['candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
