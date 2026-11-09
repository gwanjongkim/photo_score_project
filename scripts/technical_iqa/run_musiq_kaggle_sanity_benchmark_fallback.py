# 공식 Google MUSIQ Kaggle 모델의 100장 단위 sanity benchmark를 실행한다. (Fallback 포함)
from __future__ import annotations

import csv
import importlib.metadata
import io
import math
import random
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = (
    ROOT
    / "outputs"
    / "research_protocol_iqa_manifests_20260530"
    / "benchmark_all_research_protocol_iqa.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "musiq_kaggle_sanity_fallback_20260601"
SANITY_MANIFEST_CSV = OUTPUT_DIR / "sanity_manifest.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
METRICS_CSV = OUTPUT_DIR / "dataset_metrics.csv"
FAILURES_CSV = OUTPUT_DIR / "failures.csv"
REPORT_MD = OUTPUT_DIR / "report.md"

SEED = 20260601
SAMPLE_PER_DATASET = 100
DATASETS = {
    "flive": "FLIVE",
    "koniq10k": "KonIQ-10k",
    "spaq": "SPAQ",
}
VARIANTS = {
    "koniq-10k": "https://www.kaggle.com/models/google/musiq/TensorFlow2/koniq-10k/1",
    "spaq": "https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1",
    "paq2piq": "https://www.kaggle.com/models/google/musiq/TensorFlow2/paq2piq/1",
}
PRODUCTION_BASELINES = {
    "flive": 0.574,
    "koniq10k": 0.934,
    "spaq": 0.873,
}


@dataclass
class VariantLoad:
    variant: str
    handle: str
    loaded: bool
    load_seconds: float
    error: str = ""


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def build_sanity_manifest() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    unique_by_dataset: dict[str, dict[str, dict[str, str]]] = {
        dataset: {} for dataset in DATASETS
    }
    failures: list[dict[str, str]] = []

    with MANIFEST_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = row.get("dataset", "")
            if dataset not in DATASETS:
                continue
            image_path = row.get("image_path", "")
            if not image_path or image_path in unique_by_dataset[dataset]:
                continue
            unique_by_dataset[dataset][image_path] = {
                "dataset": dataset,
                "dataset_label": DATASETS[dataset],
                "split": row.get("split", ""),
                "image_path": image_path,
                "quality_score_0_100": row.get("quality_score_0_100", ""),
            }

    rng = random.Random(SEED)
    sampled: list[dict[str, str]] = []
    for dataset in DATASETS:
        candidates = list(unique_by_dataset[dataset].values())
        rng.shuffle(candidates)
        selected_count = 0
        for row in candidates:
            path = Path(row["image_path"])
            if not path.exists():
                failures.append(
                    failure_row("sampling", "", dataset, row["image_path"], "missing image")
                )
                continue
            if not is_readable_image(path):
                failures.append(
                    failure_row("sampling", "", dataset, row["image_path"], "unreadable image")
                )
                continue
            try:
                float(row["quality_score_0_100"])
            except ValueError:
                failures.append(
                    failure_row("sampling", "", dataset, row["image_path"], "invalid target")
                )
                continue
            sampled.append(row)
            selected_count += 1
            if selected_count == SAMPLE_PER_DATASET:
                break
        if selected_count < SAMPLE_PER_DATASET:
            failures.append(
                failure_row(
                    "sampling",
                    "",
                    dataset,
                    "",
                    f"selected only {selected_count} readable images",
                )
            )

    sampled.sort(key=lambda row: (row["dataset"], row["image_path"]))
    return sampled, failures


def failure_row(
    stage: str,
    variant: str,
    dataset: str,
    image_path: str,
    error: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "variant": variant,
        "dataset": dataset,
        "image_path": image_path,
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model(variant: str, handle: str) -> tuple[Any | None, VariantLoad]:
    start = time.perf_counter()
    try:
        model = hub.load(handle)
        return model, VariantLoad(
            variant=variant,
            handle=handle,
            loaded=True,
            load_seconds=time.perf_counter() - start,
        )
    except Exception:
        return None, VariantLoad(
            variant=variant,
            handle=handle,
            loaded=False,
            load_seconds=time.perf_counter() - start,
            error=traceback.format_exc(limit=10),
        )


def flatten_output(output: Any) -> list[float]:
    if isinstance(output, dict):
        values: list[float] = []
        for key in sorted(output):
            values.extend(flatten_output(output[key]))
        return values
    if isinstance(output, (tuple, list)):
        values = []
        for item in output:
            values.extend(flatten_output(item))
        return values
    tensor = tf.convert_to_tensor(output)
    return [float(value) for value in tf.reshape(tensor, [-1]).numpy().tolist()]


def get_fallback_bytes(image_path: str) -> bytes:
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()


def predict_one(model: Any, image_path: str) -> tuple[float, str]:
    signatures = getattr(model, "signatures", {})
    if not signatures:
        raise RuntimeError("Loaded model has no SavedModel signatures.")
    signature = signatures.get("serving_default") or next(iter(signatures.values()))
    kwargs = signature.structured_input_signature[1]
    if not kwargs:
        raise RuntimeError(f"Serving signature has no keyword inputs: {signature}")
    input_name = next(iter(kwargs))
    
    # Try original
    try:
        image_bytes = tf.io.read_file(image_path)
        values = flatten_output(signature(**{input_name: image_bytes}))
        if len(values) == 1 and math.isfinite(values[0]):
            return values[0], f"signature_{input_name}_bytes_original"
    except Exception:
        pass
        
    # Try fallback
    try:
        fallback_bytes = get_fallback_bytes(image_path)
        values = flatten_output(signature(**{input_name: fallback_bytes}))
        if len(values) == 1 and math.isfinite(values[0]):
            return values[0], f"signature_{input_name}_bytes_fallback_rgb_reencode"
    except Exception as e:
        raise RuntimeError(f"Fallback also failed: {e}") from e

    raise RuntimeError("Could not produce a valid score even with fallback.")


def rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson(rank_average(x), rank_average(y))


def aligned_mae(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    if len(pred) < 2 or float(np.std(pred)) == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(pred, target, 1)
    aligned = slope * pred + intercept
    return float(np.mean(np.abs(aligned - target))), float(slope), float(intercept)


def compute_metrics(
    predictions: list[dict[str, Any]],
    failures: list[dict[str, str]],
) -> list[dict[str, Any]]:
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_pair[(row["variant"], row["dataset"])].append(row)

    failure_counts = defaultdict(int)
    for row in failures:
        key_variant = row.get("variant", "")
        key_dataset = row.get("dataset", "")
        if key_variant and key_dataset:
            failure_counts[(key_variant, key_dataset)] += 1

    metrics: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for dataset in DATASETS:
            rows = by_pair.get((variant, dataset), [])
            preds = np.array([float(row["prediction"]) for row in rows], dtype=np.float64)
            targets = np.array(
                [float(row["quality_score_0_100"]) for row in rows], dtype=np.float64
            )
            mae, slope, intercept = aligned_mae(preds, targets)
            
            fallback_count = sum(1 for row in rows if "fallback" in row.get("call_method", ""))
            
            metrics.append(
                {
                    "variant": variant,
                    "dataset": dataset,
                    "dataset_label": DATASETS[dataset],
                    "n": len(rows),
                    "score_min": float(np.min(preds)) if len(preds) else float("nan"),
                    "score_max": float(np.max(preds)) if len(preds) else float("nan"),
                    "score_mean": float(np.mean(preds)) if len(preds) else float("nan"),
                    "score_std": float(np.std(preds, ddof=1)) if len(preds) > 1 else float("nan"),
                    "srcc": spearman(preds, targets) if len(preds) else float("nan"),
                    "plcc": pearson(preds, targets) if len(preds) else float("nan"),
                    "mae_linear_aligned": mae,
                    "linear_slope": slope,
                    "linear_intercept": intercept,
                    "load_inference_failures": failure_counts[(variant, dataset)],
                    "fallback_count": fallback_count,
                }
            )
    return metrics


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, str):
        return value
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value_float):
        return "nan"
    return f"{value_float:.{digits}f}"


def decide(metrics: list[dict[str, Any]], failures: list[dict[str, str]]) -> str:
    valid = [row for row in metrics if int(row["n"]) >= 80 and math.isfinite(float(row["srcc"]))]
    if not valid:
        return "MUSIQ_BENCHMARK_FAILED"
    if any(row["stage"] == "model_load" for row in failures):
        return "MUSIQ_BENCHMARK_FAILED"
    total_inference_failures = sum(
        1 for row in failures if row["stage"] in {"inference", "metrics"}
    )
    if total_inference_failures:
        return "MUSIQ_BLOCKED_BY_INFERENCE_FAILURE"

    by_variant = defaultdict(dict)
    for row in metrics:
        by_variant[row["variant"]][row["dataset"]] = float(row["srcc"])

    for variant, dataset_scores in by_variant.items():
        if all(
            dataset in dataset_scores
            and math.isfinite(dataset_scores[dataset])
            and dataset_scores[dataset] >= 0.70 * PRODUCTION_BASELINES[dataset]
            for dataset in DATASETS
        ):
            return "MUSIQ_FAILURE_FIXED_AND_READY_FOR_FULL_BENCHMARK"

    best_flive = max(
        (float(row["srcc"]) for row in metrics if row["dataset"] == "flive"),
        default=float("nan"),
    )
    best_ratios = []
    for dataset, baseline in PRODUCTION_BASELINES.items():
        best = max(
            (float(row["srcc"]) for row in metrics if row["dataset"] == dataset),
            default=float("nan"),
        )
        if math.isfinite(best):
            best_ratios.append(best / baseline)

    if math.isfinite(best_flive) and best_flive < 0.35:
        return "MUSIQ_NOT_PROMISING_DUE_TO_PERFORMANCE"
    if best_ratios and all(ratio < 0.60 for ratio in best_ratios):
        return "MUSIQ_NOT_PROMISING_DUE_TO_PERFORMANCE"
    return "MUSIQ_READY_FOR_SANITY_RERUN_WITH_FALLBACK"


def write_report(
    sanity_manifest: list[dict[str, str]],
    load_statuses: list[VariantLoad],
    predictions: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    failures: list[dict[str, str]],
    decision: str,
    elapsed_seconds: float,
) -> None:
    counts = defaultdict(int)
    split_counts = defaultdict(int)
    for row in sanity_manifest:
        counts[row["dataset"]] += 1
        split_counts[(row["dataset"], row["split"])] += 1

    lines = [
        "# Official Kaggle MUSIQ 100-image Sanity Benchmark Report (With Fallback)",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        f"- Evaluated {len(sanity_manifest)} images and wrote {len(predictions)} predictions.",
        f"- Elapsed seconds: {elapsed_seconds:.1f}.",
        "- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, or local repo-practical MUSIQ.",
        "",
        "## 2. Files Inspected",
        f"- Input manifest: `{MANIFEST_CSV}`.",
        f"- Output directory: `{OUTPUT_DIR}`.",
        f"- Seed: `{SEED}`.",
        f"- Sample target: `{SAMPLE_PER_DATASET}` readable images per dataset.",
        f"- TensorFlow: `{tf.__version__}`.",
        f"- TensorFlow Hub: `{hub.__version__}`.",
        f"- setuptools: `{importlib.metadata.version('setuptools')}`.",
        "",
        "## 3. Failed Image Inspection",
        "Refer to `outputs/musiq_kaggle_failure_diagnosis_20260601/failed_image_inspection.csv`.",
        "",
        "## 4. Fallback Tests",
        "Refer to `outputs/musiq_kaggle_failure_diagnosis_20260601/fallback_test_results.csv`.",
        "",
        "## 5. Patch Decision",
        "Patched with EXIF-transposed RGB re-encode fallback in `predict_one`.",
        "",
        "## 6. Optional Rerun Results",
        "This report is the result of the optional rerun.",
        "",
        "## 7. Production Baseline Implication",
    ]
    
    lines.append("| Dataset | Production SRCC | Best MUSIQ Variant | Best MUSIQ SRCC | Delta |")
    lines.append("|---|---:|---|---:|---:|")
    for dataset, baseline in PRODUCTION_BASELINES.items():
        dataset_rows = [row for row in metrics if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        best = max(dataset_rows, key=lambda row: float(row["srcc"]) if math.isfinite(float(row["srcc"])) else -999)
        best_srcc = float(best["srcc"])
        lines.append(
            f"| {DATASETS[dataset]} | {baseline:.3f} | {best['variant']} | "
            f"{fmt(best_srcc)} | {fmt(best_srcc - baseline)} |"
        )

    lines.extend(["", "## 8. Final Decision", f"`{decision}`", ""])
    
    lines.extend(["## 9. Next Step"])
    if decision == "MUSIQ_FAILURE_FIXED_AND_READY_FOR_FULL_BENCHMARK":
        lines.append("- Run the full benchmark for the best-performing variant(s).")
    elif decision == "MUSIQ_NOT_PROMISING_DUE_TO_PERFORMANCE":
        lines.append("- Consider other teacher candidates as MUSIQ is not outperforming current production on most datasets.")
    else:
        lines.append("- Investigate remaining blockers.")

    lines.extend(["", "## Detailed Metrics", ""])
    lines.append("| Variant | Dataset | n | SRCC | PLCC | Aligned MAE | Fallbacks | Failures |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in metrics:
        lines.append(
            f"| {row['variant']} | {row['dataset_label']} | {row['n']} | "
            f"{fmt(row['srcc'])} | {fmt(row['plcc'])} | "
            f"{fmt(row['mae_linear_aligned'])} | {row['fallback_count']} | {row['load_inference_failures']} |"
        )

    lines.extend(["", "## Failure Cases", ""])
    if failures:
        for row in failures[:50]:
            lines.append(
                f"- {row['stage']} | {row['variant']} | {row['dataset']} | "
                f"{row['image_path']} | {row['error']}"
            )
    else:
        lines.append("- None.")
    lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sanity_manifest, failures = build_sanity_manifest()
    write_csv(
        SANITY_MANIFEST_CSV,
        sanity_manifest,
        ["dataset", "dataset_label", "split", "image_path", "quality_score_0_100"],
    )

    predictions: list[dict[str, Any]] = []
    load_statuses: list[VariantLoad] = []

    for variant, handle in VARIANTS.items():
        print(f"\n=== Loading {variant}: {handle}")
        model, load_status = load_model(variant, handle)
        load_statuses.append(load_status)
        if model is None:
            failures.append(failure_row("model_load", variant, "", "", load_status.error))
            print(load_status.error)
            continue

        for idx, row in enumerate(sanity_manifest, start=1):
            try:
                prediction, call_method = predict_one(model, row["image_path"])
                predictions.append(
                    {
                        "variant": variant,
                        "load_method": "tensorflow_hub_direct",
                        "call_method": call_method,
                        "dataset": row["dataset"],
                        "dataset_label": row["dataset_label"],
                        "split": row["split"],
                        "image_path": row["image_path"],
                        "quality_score_0_100": row["quality_score_0_100"],
                        "prediction": prediction,
                    }
                )
            except Exception:
                failures.append(
                    failure_row(
                        "inference",
                        variant,
                        row["dataset"],
                        row["image_path"],
                        traceback.format_exc(limit=8),
                    )
                )
            if idx % 25 == 0:
                print(f"{variant}: scored {idx}/{len(sanity_manifest)} images")

    metrics = compute_metrics(predictions, failures)
    decision = decide(metrics, failures)

    write_csv(
        PREDICTIONS_CSV,
        predictions,
        [
            "variant",
            "load_method",
            "call_method",
            "dataset",
            "dataset_label",
            "split",
            "image_path",
            "quality_score_0_100",
            "prediction",
        ],
    )
    write_csv(
        METRICS_CSV,
        metrics,
        [
            "variant",
            "dataset",
            "dataset_label",
            "n",
            "score_min",
            "score_max",
            "score_mean",
            "score_std",
            "srcc",
            "plcc",
            "mae_linear_aligned",
            "linear_slope",
            "linear_intercept",
            "load_inference_failures",
            "fallback_count",
        ],
    )
    write_csv(FAILURES_CSV, failures, ["stage", "variant", "dataset", "image_path", "error"])
    elapsed = time.perf_counter() - start
    write_report(sanity_manifest, load_statuses, predictions, metrics, failures, decision, elapsed)

    print(f"\nWrote {SANITY_MANIFEST_CSV}")
    print(f"Wrote {PREDICTIONS_CSV}")
    print(f"Wrote {METRICS_CSV}")
    print(f"Wrote {FAILURES_CSV}")
    print(f"Wrote {REPORT_MD}")
    print(f"Final decision: {decision}")


if __name__ == "__main__":
    main()
