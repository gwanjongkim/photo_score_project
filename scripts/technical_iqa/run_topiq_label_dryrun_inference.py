# TOPIQ 교사 모델을 라벨 드라이런 매니페스트에 실행하는 스크립트
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
except Exception as exc:
    pyiqa = None
    PYIQA_IMPORT_ERROR = repr(exc)


DEFAULT_MANIFEST = "outputs/topiq_teacher_label_dryrun_20260601/dryrun_manifest.csv"
DEFAULT_OUT_DIR = "outputs/topiq_teacher_label_dryrun_20260601"
SCORE_COL = "quality_score_0_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TOPIQ NR for bounded label dry run.")
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--score_col", default=SCORE_COL)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="topiq_nr")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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


def format_float(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.8f}"


def tensor_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)[0]
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def read_manifest(path: Path, max_images: int | None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows if max_images is None else rows[:max_images]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_topiq_model(requested_model: str) -> tuple[str, list[str]]:
    assert pyiqa is not None
    model_names = list(pyiqa.list_models())
    topiq_names = sorted(name for name in model_names if "topiq" in name.lower())
    print("Available TOPIQ model names:")
    if topiq_names:
        for name in topiq_names:
            print(f"  - {name}")
    else:
        print("  (none)")
    if requested_model == "topiq_nr" and "topiq_nr" not in model_names:
        available = ", ".join(topiq_names) if topiq_names else "(none)"
        raise RuntimeError(f'Required PyIQA model "topiq_nr" is not available. Available TOPIQ-like names: {available}')
    if requested_model not in model_names:
        available = ", ".join(topiq_names) if topiq_names else "(none)"
        raise RuntimeError(f'Requested PyIQA model "{requested_model}" is not available. Available TOPIQ-like names: {available}')
    return requested_model, topiq_names


def stats_for(rows: list[dict[str, Any]], score_col: str) -> dict[str, Any]:
    y_true: list[float] = []
    y_pred: list[float] = []
    for row in rows:
        truth = safe_float(row.get(score_col))
        pred = safe_float(row.get("topiq_raw"))
        if truth is not None and pred is not None:
            y_true.append(truth)
            y_pred.append(pred)
    return {
        "n": len(y_pred),
        "srcc": spearman(y_true, y_pred) if len(y_pred) >= 2 else None,
        "plcc": pearson(y_true, y_pred) if len(y_pred) >= 2 else None,
    }


def write_report(
    path: Path,
    selected_model: str | None,
    available_topiq_names: list[str],
    attempted: int,
    predictions: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    score_col: str,
) -> None:
    topiq_values = [safe_float(row.get("topiq_raw")) for row in predictions]
    topiq_values = [value for value in topiq_values if value is not None]
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_dataset[str(row.get("dataset", ""))].append(row)
    overall = stats_for(predictions, score_col)
    lines = [
        "# TOPIQ Label Dry-Run Inference Report",
        "",
        f"- selected TOPIQ model: {selected_model or 'n/a'}",
        f"- available TOPIQ model names: {', '.join(available_topiq_names) if available_topiq_names else '(none)'}",
        f"- torch version: {torch.__version__}",
        f"- cuda availability: {torch.cuda.is_available()}",
        f"- total attempted: {attempted}",
        f"- total succeeded: {len(predictions)}",
        f"- total failed: {len(errors)}",
        f"- topiq_raw min: {format_float(min(topiq_values) if topiq_values else None)}",
        f"- topiq_raw max: {format_float(max(topiq_values) if topiq_values else None)}",
        f"- topiq_raw mean: {format_float(statistics.fmean(topiq_values) if topiq_values else None)}",
        f"- overall SRCC vs {score_col}: {format_float(overall['srcc'])}",
        f"- overall PLCC vs {score_col}: {format_float(overall['plcc'])}",
        "",
        "## Dataset Metrics",
        "",
        f"| dataset | n | SRCC vs {score_col} | PLCC vs {score_col} |",
        "|---|---:|---:|---:|",
    ]
    for dataset, dataset_rows in sorted(by_dataset.items()):
        stats = stats_for(dataset_rows, score_col)
        lines.append(f"| {dataset} | {stats['n']} | {format_float(stats['srcc'])} | {format_float(stats['plcc'])} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = out_dir / "predictions.csv"
    errors_csv = out_dir / "errors.csv"
    report_md = out_dir / "inference_report.md"
    available_topiq_names: list[str] = []

    if PYIQA_IMPORT_ERROR is not None:
        write_csv(predictions_csv, [], ["dataset", "split", "image_path", args.score_col, "topiq_raw"])
        write_csv(errors_csv, [], ["dataset", "split", "image_path", "error"])
        write_report(report_md, None, available_topiq_names, 0, [], [], args.score_col)
        print(f"pyiqa import failed: {PYIQA_IMPORT_ERROR}", file=sys.stderr)
        return 2

    rows = read_manifest(Path(args.manifest_csv), args.max_images)
    selected_model: str | None = None
    try:
        selected_model, available_topiq_names = select_topiq_model(args.model_name)
        metric = pyiqa.create_metric(selected_model, device=args.device)
    except Exception:
        write_csv(predictions_csv, [], ["dataset", "split", "image_path", args.score_col, "topiq_raw"])
        write_csv(errors_csv, [], ["dataset", "split", "image_path", "error"])
        write_report(report_md, selected_model, available_topiq_names, len(rows), [], [], args.score_col)
        raise

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        image_path = Path(row.get(args.image_col, ""))
        if not image_path.exists():
            errors.append({"dataset": row.get("dataset", ""), "split": row.get("split", ""), "image_path": str(image_path), "error": "missing image path"})
            print(f"[{index}/{len(rows)}] skip missing: {image_path}")
            continue
        try:
            with torch.no_grad():
                raw_score = tensor_to_float(metric(str(image_path)))
        except Exception as exc:
            errors.append({"dataset": row.get("dataset", ""), "split": row.get("split", ""), "image_path": str(image_path), "error": repr(exc)})
            print(f"[{index}/{len(rows)}] failed: {image_path}: {exc!r}")
            continue
        predictions.append({
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "image_path": str(image_path),
            args.score_col: row.get(args.score_col, ""),
            "topiq_raw": raw_score,
        })
        print(f"[{index}/{len(rows)}] ok: topiq_raw={raw_score:.8f} {image_path}")

    write_csv(predictions_csv, predictions, ["dataset", "split", "image_path", args.score_col, "topiq_raw"])
    write_csv(errors_csv, errors, ["dataset", "split", "image_path", "error"])
    write_report(report_md, selected_model, available_topiq_names, len(rows), predictions, errors, args.score_col)
    print(f"Wrote {predictions_csv}")
    print(f"Wrote {errors_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
