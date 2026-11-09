# TOPIQ PyIQA 교사 모델의 제한된 스모크 테스트를 실행하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import pyiqa

    PYIQA_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover - exercised only when dependency is absent
    pyiqa = None
    PYIQA_IMPORT_ERROR = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a capped TOPIQ NR teacher smoke test with PyIQA."
    )
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--score_col", default="quality_score_0_100")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_images", type=int, default=30)
    return parser.parse_args()


def read_manifest(path: Path, max_images: int) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for index, row in enumerate(reader) if index < max_images]


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


def select_topiq_model() -> tuple[str, list[str]]:
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
    return "topiq_nr", topiq_names


def write_report(
    path: Path,
    import_status: str,
    selected_model: str | None,
    attempted: int,
    successful: int,
    failed: int,
    topiq_values: list[float],
    spearman_value: float | None,
    errors_csv: Path,
) -> None:
    raw_min = min(topiq_values) if topiq_values else None
    raw_max = max(topiq_values) if topiq_values else None
    raw_mean = statistics.fmean(topiq_values) if topiq_values else None

    lines = [
        "# TOPIQ PyIQA Teacher Smoke Report",
        "",
        f"- package import status: {import_status}",
        f"- torch version: {torch.__version__}",
        f"- cuda availability: {torch.cuda.is_available()}",
        f"- selected TOPIQ model name: {selected_model or 'n/a'}",
        f"- number of attempted images: {attempted}",
        f"- number of successful images: {successful}",
        f"- number of failed images: {failed}",
        f"- topiq_raw min: {format_optional_float(raw_min)}",
        f"- topiq_raw max: {format_optional_float(raw_max)}",
        f"- topiq_raw mean: {format_optional_float(raw_mean)}",
        f"- Spearman correlation against quality_score_0_100: {format_optional_float(spearman_value)}",
        f"- per-image errors: {errors_csv}",
        "",
        "Note: TOPIQ raw score direction and scale are not yet trusted until inspected.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    manifest_csv = Path(args.manifest_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = out_dir / "predictions.csv"
    errors_csv = out_dir / "errors.csv"
    report_md = out_dir / "report.md"

    import_status = "torch import ok; pyiqa import ok"
    if PYIQA_IMPORT_ERROR is not None:
        import_status = f"torch import ok; pyiqa import failed: {PYIQA_IMPORT_ERROR}"
        write_csv(
            predictions_csv,
            [],
            ["dataset", "split", "image_path", args.score_col, "topiq_raw"],
        )
        write_csv(errors_csv, [], ["dataset", "split", "image_path", "error"])
        write_report(report_md, import_status, None, 0, 0, 0, [], None, errors_csv)
        print(import_status, file=sys.stderr)
        return 2

    rows = read_manifest(manifest_csv, args.max_images)
    selected_model: str | None = None

    try:
        selected_model, _ = select_topiq_model()
        metric = pyiqa.create_metric(selected_model, device=args.device)
    except Exception as exc:
        import_status = f"{import_status}; model load failed: {exc!r}"
        write_csv(
            predictions_csv,
            [],
            ["dataset", "split", "image_path", args.score_col, "topiq_raw"],
        )
        write_csv(errors_csv, [], ["dataset", "split", "image_path", "error"])
        write_report(report_md, import_status, selected_model, len(rows), 0, len(rows), [], None, errors_csv)
        raise

    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    y_true: list[float] = []
    y_pred: list[float] = []

    for index, row in enumerate(rows, start=1):
        image_path = Path(row.get(args.image_col, ""))
        dataset = row.get("dataset", "")
        split = row.get("split", "")
        score_value = row.get(args.score_col, "")

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
                args.score_col: score_value,
                "topiq_raw": raw_score,
            }
        )
        numeric_score = safe_float(score_value)
        if numeric_score is not None:
            y_true.append(numeric_score)
            y_pred.append(raw_score)
        print(f"[{index}/{len(rows)}] ok: topiq_raw={raw_score:.8f} {image_path}")

    spearman_value = spearman(y_true, y_pred) if len(y_true) >= 2 else None
    write_csv(
        predictions_csv,
        predictions,
        ["dataset", "split", "image_path", args.score_col, "topiq_raw"],
    )
    write_csv(errors_csv, errors, ["dataset", "split", "image_path", "error"])
    write_report(
        report_md,
        import_status,
        selected_model,
        len(rows),
        len(predictions),
        len(errors),
        y_pred,
        spearman_value,
        errors_csv,
    )

    print(f"Wrote {predictions_csv}")
    print(f"Wrote {errors_csv}")
    print(f"Wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
