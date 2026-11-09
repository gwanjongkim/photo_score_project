# TOPIQ PyIQA 교사 모델의 검증 추론과 리포트를 생성하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

try:
    import pyiqa

    PYIQA_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent
    pyiqa = None
    PYIQA_IMPORT_ERROR = repr(exc)


DEFAULT_MANIFEST = "outputs/topiq_teacher_validation_20260531/validation_manifest.csv"
DEFAULT_OUT_DIR = "outputs/topiq_teacher_validation_20260531"
DEFAULT_SCORE_COL = "quality_score_0_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TOPIQ NR teacher validation on a bounded manifest."
    )
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--score_col", default=DEFAULT_SCORE_COL)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def read_manifest(path: Path, max_images: int | None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if max_images is not None:
        rows = rows[:max_images]
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def tensor_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


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


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8f}"


def select_topiq_model() -> str:
    assert pyiqa is not None
    model_names = list(pyiqa.list_models())
    topiq_names = sorted(name for name in model_names if "topiq" in name.lower())
    print("Available TOPIQ model names:")
    if topiq_names:
        for name in topiq_names:
            print(f"  - {name}")
    else:
        print("  (none)")

    if "topiq_nr" not in model_names:
        available = ", ".join(topiq_names) if topiq_names else "(none)"
        raise RuntimeError(
            f'Required PyIQA model "topiq_nr" is not available. '
            f"Available TOPIQ-like names: {available}"
        )
    return "topiq_nr"


def metric_pair(rows: list[dict[str, Any]], score_col: str) -> tuple[list[float], list[float]]:
    y_true: list[float] = []
    y_pred: list[float] = []
    for row in rows:
        score = safe_float(row.get(score_col))
        pred = safe_float(row.get("topiq_raw"))
        if score is None or pred is None:
            continue
        y_true.append(score)
        y_pred.append(pred)
    return y_true, y_pred


def compute_stats(rows: list[dict[str, Any]], score_col: str) -> dict[str, float | int | None]:
    y_true, y_pred = metric_pair(rows, score_col)
    return {
        "n": len(y_pred),
        "srcc": spearman(y_true, y_pred) if len(y_pred) >= 2 else None,
        "plcc": pearson(y_true, y_pred) if len(y_pred) >= 2 else None,
    }


def score_direction(overall_srcc: float | None) -> str:
    if overall_srcc is None:
        return "undetermined"
    if overall_srcc > 0.1:
        return "higher topiq_raw appears higher-is-better"
    if overall_srcc < -0.1:
        return "higher topiq_raw appears lower-is-better"
    return "weak or ambiguous"


def recommendation(
    failed: int, overall: dict[str, float | int | None], dataset_stats: dict[str, dict[str, float | int | None]]
) -> str:
    overall_srcc = overall.get("srcc")
    overall_plcc = overall.get("plcc")
    if failed > 0 or overall_srcc is None or overall_plcc is None:
        return "NEEDS_TARGET_GUARD_DESIGN_FIRST"
    if overall_srcc < 0.5 or overall_plcc < 0.5:
        return "NEEDS_TARGET_GUARD_DESIGN_FIRST"
    for stats in dataset_stats.values():
        srcc = stats.get("srcc")
        plcc = stats.get("plcc")
        if srcc is None or plcc is None or srcc < 0.3 or plcc < 0.3:
            return "NEEDS_TARGET_GUARD_DESIGN_FIRST"
    return "READY_FOR_TEACHER_LABEL_GENERATION"


def rows_with_rank_gap(rows: list[dict[str, Any]], score_col: str) -> list[dict[str, Any]]:
    scored = []
    for row in rows:
        score = safe_float(row.get(score_col))
        pred = safe_float(row.get("topiq_raw"))
        if score is None or pred is None:
            continue
        scored.append((row, score, pred))
    if not scored:
        return []

    true_ranks = rankdata([item[1] for item in scored])
    pred_ranks = rankdata([item[2] for item in scored])
    denom = max(1, len(scored) - 1)
    enriched = []
    for (row, _score, _pred), true_rank, pred_rank in zip(scored, true_ranks, pred_ranks):
        true_pct = (true_rank - 1.0) / denom
        pred_pct = (pred_rank - 1.0) / denom
        item = dict(row)
        item["rank_gap"] = pred_pct - true_pct
        enriched.append(item)
    return enriched


def ranked_false_positives(
    rows: list[dict[str, Any]], score_col: str, limit: int
) -> list[dict[str, Any]]:
    scored = rows_with_rank_gap(rows, score_col)
    return sorted(scored, key=lambda row: row["rank_gap"], reverse=True)[:limit]


def ranked_false_negatives(
    rows: list[dict[str, Any]], score_col: str, limit: int
) -> list[dict[str, Any]]:
    scored = rows_with_rank_gap(rows, score_col)
    return sorted(scored, key=lambda row: row["rank_gap"])[:limit]


def append_examples(
    lines: list[str], title: str, rows: list[dict[str, Any]], score_col: str
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            f"| rank | dataset | split | {score_col} | topiq_raw | rank_gap | image_path |",
            "|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for index, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {dataset} | {split} | {score} | {topiq} | {gap} | {path} |".format(
                rank=index,
                dataset=row.get("dataset", ""),
                split=row.get("split", ""),
                score=format_optional_float(safe_float(row.get(score_col))),
                topiq=format_optional_float(safe_float(row.get("topiq_raw"))),
                gap=format_optional_float(safe_float(row.get("rank_gap"))),
                path=row.get("image_path", ""),
            )
        )
    lines.append("")


def write_report(
    path: Path,
    selected_model: str | None,
    attempted: int,
    predictions: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    score_col: str,
) -> None:
    topiq_values = [safe_float(row.get("topiq_raw")) for row in predictions]
    topiq_values = [value for value in topiq_values if value is not None]
    overall = compute_stats(predictions, score_col)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_dataset[row.get("dataset", "")].append(row)
    dataset_stats = {
        dataset: compute_stats(rows, score_col) for dataset, rows in sorted(by_dataset.items())
    }

    direction = score_direction(overall.get("srcc"))
    rec = recommendation(len(errors), overall, dataset_stats)

    lines = [
        "# TOPIQ PyIQA Teacher Validation Report",
        "",
        f"- selected TOPIQ model: {selected_model or 'n/a'}",
        f"- torch version: {torch.__version__}",
        f"- cuda availability: {torch.cuda.is_available()}",
        f"- total attempted: {attempted}",
        f"- total succeeded: {len(predictions)}",
        f"- total failed: {len(errors)}",
        f"- topiq_raw min: {format_optional_float(min(topiq_values) if topiq_values else None)}",
        f"- topiq_raw max: {format_optional_float(max(topiq_values) if topiq_values else None)}",
        f"- topiq_raw mean: {format_optional_float(statistics.fmean(topiq_values) if topiq_values else None)}",
        f"- overall SRCC vs {score_col}: {format_optional_float(overall.get('srcc'))}",
        f"- overall PLCC vs {score_col}: {format_optional_float(overall.get('plcc'))}",
        f"- score direction conclusion: {direction}",
        f"- recommendation: {rec}",
        "",
        "## Dataset Metrics",
        "",
        f"| dataset | n | SRCC vs {score_col} | PLCC vs {score_col} |",
        "|---|---:|---:|---:|",
    ]
    for dataset, stats in dataset_stats.items():
        lines.append(
            "| {dataset} | {n} | {srcc} | {plcc} |".format(
                dataset=dataset,
                n=stats["n"],
                srcc=format_optional_float(stats["srcc"]),
                plcc=format_optional_float(stats["plcc"]),
            )
        )
    lines.append("")
    lines.extend(
        [
            "## Score Direction",
            "",
            "Higher-is-better is considered plausible only if correlations remain positive at overall and dataset level.",
            "",
        ]
    )
    append_examples(
        lines,
        "Top 20 Possible False Positives",
        ranked_false_positives(predictions, score_col, 20),
        score_col,
    )
    append_examples(
        lines,
        "Top 20 Possible False Negatives",
        ranked_false_negatives(predictions, score_col, 20),
        score_col,
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = out_dir / "predictions.csv"
    errors_csv = out_dir / "errors.csv"
    report_md = out_dir / "report.md"

    if PYIQA_IMPORT_ERROR is not None:
        write_csv(
            predictions_csv,
            [],
            ["dataset", "split", "image_path", args.score_col, "topiq_raw"],
        )
        write_csv(errors_csv, [], ["dataset", "split", "image_path", "error"])
        write_report(report_md, None, 0, [], [], args.score_col)
        print(f"pyiqa import failed: {PYIQA_IMPORT_ERROR}", file=sys.stderr)
        return 2

    rows = read_manifest(manifest_csv, args.max_images)
    selected_model: str | None = None
    try:
        selected_model = select_topiq_model()
        metric = pyiqa.create_metric(selected_model, device=args.device)
    except Exception:
        write_csv(
            predictions_csv,
            [],
            ["dataset", "split", "image_path", args.score_col, "topiq_raw"],
        )
        write_csv(errors_csv, [], ["dataset", "split", "image_path", "error"])
        write_report(report_md, selected_model, len(rows), [], [], args.score_col)
        raise

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row.get(args.image_col, ""))
        dataset = row.get("dataset", "")
        split = row.get("split", "")

        if not image_path.exists():
            errors.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "image_path": str(image_path),
                    "error": "missing image path",
                }
            )
            print(f"[{index}/{len(rows)}] skip missing: {image_path}")
            continue

        try:
            with torch.no_grad():
                raw_score = tensor_to_float(metric(str(image_path)))
        except Exception as exc:
            errors.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "image_path": str(image_path),
                    "error": repr(exc),
                }
            )
            print(f"[{index}/{len(rows)}] failed: {image_path}: {exc!r}")
            continue

        predictions.append(
            {
                "dataset": dataset,
                "split": split,
                "image_path": str(image_path),
                args.score_col: row.get(args.score_col, ""),
                "topiq_raw": raw_score,
            }
        )
        print(f"[{index}/{len(rows)}] ok: topiq_raw={raw_score:.8f} {image_path}")

    write_csv(
        predictions_csv,
        predictions,
        ["dataset", "split", "image_path", args.score_col, "topiq_raw"],
    )
    write_csv(errors_csv, errors, ["dataset", "split", "image_path", "error"])
    write_report(report_md, selected_model, len(rows), predictions, errors, args.score_col)

    print(f"Wrote {predictions_csv}")
    print(f"Wrote {errors_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
