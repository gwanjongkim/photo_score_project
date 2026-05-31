# ICAA Dynamic PTQ 후보를 FP16 기준과 확장 비교하는 스크립트
from __future__ import annotations

from pathlib import Path
import csv
import json
import statistics
import sys
import time
from typing import Any

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = ROOT / "outputs/icaa_ptq_candidates_20260601/review_20260601"
FP16_PATH = ROOT / "outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite"
DYNAMIC_PATH = ROOT / "outputs/icaa_ptq_candidates_20260601/icaa_dat_tf_native_dynamic_range_ptq.tflite"
ICAA_ROOT = ROOT / "data/raw/icaa17k"
ICAA_TEST_CSV = ROOT / "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"

INPUT_SHAPE = [1, 224, 224, 3]
OUTPUT_SHAPE = [1, 2]
IMAGE_COUNT = 50
LATENCY_WARMUP = 8
LATENCY_RUNS = 50
THRESHOLDS = [0.001, 0.005, 0.01]
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, Path):
        return str(value)
    return value


def preprocess_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - IMAGENET_MEAN) / IMAGENET_STD)[np.newaxis, ...].astype(np.float32)


def append_unique(paths: list[Path], seen: set[Path], path: Path) -> None:
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
        return
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    paths.append(path)


def path_from_text(value: str) -> Path | None:
    text = value.strip().strip("\"'")
    if not text or not any(text.lower().endswith(suffix) for suffix in IMAGE_SUFFIXES):
        return None
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([ROOT / path, ICAA_ROOT / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def image_paths_from_csv(path: Path) -> list[Path]:
    rows: list[Path] = []
    seen: set[Path] = set()
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            has_header = csv.Sniffer().has_header(sample) if sample else False
            if has_header:
                reader = csv.DictReader(handle)
                for row in reader:
                    for value in row.values():
                        image_path = path_from_text(str(value))
                        if image_path is not None:
                            append_unique(rows, seen, image_path)
                            break
            else:
                reader = csv.reader(handle)
                for row in reader:
                    for value in row:
                        image_path = path_from_text(value)
                        if image_path is not None:
                            append_unique(rows, seen, image_path)
                            break
    except Exception:
        return []
    return rows


def image_paths_from_json(path: Path) -> list[Path]:
    rows: list[Path] = []
    seen: set[Path] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            image_path = path_from_text(value)
            if image_path is not None:
                append_unique(rows, seen, image_path)

    try:
        walk(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return []
    return rows


def image_paths_from_text_file(path: Path) -> list[Path]:
    rows: list[Path] = []
    seen: set[Path] = set()
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            image_path = path_from_text(line)
            if image_path is not None:
                append_unique(rows, seen, image_path)
    except Exception:
        return []
    return rows


def find_acut_50_image_set() -> tuple[list[Path], dict[str, Any]]:
    roots = [ROOT / "outputs", ROOT / "data", ROOT / "test_images", ROOT / "test_samples"]
    candidates: list[tuple[Path, list[Path]]] = []
    for search_root in roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file():
                continue
            lower = str(path).lower()
            if "acut" not in lower and "a-cut" not in lower:
                continue
            if path.suffix.lower() == ".csv":
                image_paths = image_paths_from_csv(path)
            elif path.suffix.lower() == ".json":
                image_paths = image_paths_from_json(path)
            elif path.suffix.lower() in {".txt", ".lst", ".md"}:
                image_paths = image_paths_from_text_file(path)
            else:
                continue
            if len(image_paths) >= IMAGE_COUNT:
                candidates.append((path, image_paths))

    exact = [(path, rows) for path, rows in candidates if len(rows) == IMAGE_COUNT]
    selected = exact[0] if exact else candidates[0] if candidates else None
    if selected is None:
        return [], {"source": "not_found", "candidate_manifest_count": len(candidates)}
    manifest_path, rows = selected
    return rows[:IMAGE_COUNT], {
        "source": "acut_manifest",
        "manifest_path": str(manifest_path),
        "manifest_image_count": len(rows),
        "candidate_manifest_count": len(candidates),
    }


def deterministic_icaa_subset() -> tuple[list[Path], dict[str, Any]]:
    paths: list[Path] = []
    seen: set[Path] = set()
    if ICAA_TEST_CSV.is_file():
        with ICAA_TEST_CSV.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row.get("ID")
                if image_id:
                    append_unique(paths, seen, ICAA_ROOT / image_id)
                if len(paths) >= IMAGE_COUNT:
                    break
    if len(paths) < IMAGE_COUNT:
        for path in sorted(ICAA_ROOT.rglob("*")):
            append_unique(paths, seen, path)
            if len(paths) >= IMAGE_COUNT:
                break
    return paths, {
        "source": "deterministic_icaa17k",
        "official_test_csv_present": ICAA_TEST_CSV.is_file(),
        "image_root": str(ICAA_ROOT),
    }


def validate_image_paths(paths: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    from PIL import Image

    valid: list[Path] = []
    invalid: list[dict[str, str]] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except Exception as exc:
            invalid.append({"path": str(path), "error": str(exc)})
    return valid, invalid


def select_images() -> tuple[list[Path], dict[str, Any]]:
    acut_paths, acut_info = find_acut_50_image_set()
    if len(acut_paths) >= IMAGE_COUNT:
        valid_acut_paths, invalid_acut_paths = validate_image_paths(acut_paths)
        acut_info["invalid_image_count"] = len(invalid_acut_paths)
        acut_info["invalid_images"] = invalid_acut_paths
        if len(valid_acut_paths) >= IMAGE_COUNT:
            return valid_acut_paths[:IMAGE_COUNT], acut_info
    fallback_paths, fallback_info = deterministic_icaa_subset()
    valid_fallback_paths, invalid_fallback_paths = validate_image_paths(fallback_paths)
    fallback_info["acut_search"] = acut_info
    fallback_info["invalid_image_count"] = len(invalid_fallback_paths)
    fallback_info["invalid_images"] = invalid_fallback_paths
    return valid_fallback_paths[:IMAGE_COUNT], fallback_info


def input_for_interpreter(input_detail: dict[str, Any], sample: np.ndarray) -> np.ndarray:
    dtype = np.dtype(input_detail["dtype"])
    if dtype == np.float32:
        return sample.astype(np.float32)
    if dtype == np.float16:
        return sample.astype(np.float16)
    raise TypeError(f"Unsupported input dtype for this comparison: {dtype}")


def output_from_interpreter(output_detail: dict[str, Any], raw_output: np.ndarray) -> np.ndarray:
    dtype = np.dtype(output_detail["dtype"])
    if dtype in {np.dtype("float32"), np.dtype("float16")}:
        return raw_output.astype(np.float32)
    raise TypeError(f"Unsupported output dtype for this comparison: {dtype}")


def detail_summary(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": item.get("name"),
            "index": int(item.get("index")),
            "shape": item.get("shape").tolist(),
            "dtype": np.dtype(item.get("dtype")).name,
            "quantization": tuple(item.get("quantization", ())),
        }
        for item in details
    ]


class TFLiteRunner:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_detail = self.input_details[0]
        self.output_detail = self.output_details[0]

    def run_one(self, sample: np.ndarray) -> np.ndarray:
        self.interpreter.set_tensor(self.input_detail["index"], input_for_interpreter(self.input_detail, sample))
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_detail["index"])
        return output_from_interpreter(self.output_detail, raw_output)

    def summary(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "input_details": detail_summary(self.input_details),
            "output_details": detail_summary(self.output_details),
        }


def run_outputs(model_path: Path, samples: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    runner = TFLiteRunner(model_path)
    outputs = [runner.run_one(sample) for sample in samples]
    return np.concatenate(outputs, axis=0), runner.summary()


def range_error_count(values: np.ndarray) -> int:
    return int(np.logical_or(values < 0.0, values > 1.0).sum())


def count_over(values: np.ndarray, threshold: float) -> int:
    return int(np.sum(values > threshold))


def metric_block(values: np.ndarray) -> dict[str, Any]:
    values = values.astype(np.float64).reshape(-1)
    if values.size == 0:
        return {
            "mae": None,
            "maxae": None,
            "median_ae": None,
            "p90_ae": None,
            "p95_ae": None,
            "count_over_0_001": 0,
            "count_over_0_005": 0,
            "count_over_0_01": 0,
        }
    return {
        "mae": float(np.nanmean(values)),
        "maxae": float(np.nanmax(values)),
        "median_ae": float(np.nanmedian(values)),
        "p90_ae": float(np.nanpercentile(values, 90)),
        "p95_ae": float(np.nanpercentile(values, 95)),
        "count_over_0_001": count_over(values, 0.001),
        "count_over_0_005": count_over(values, 0.005),
        "count_over_0_01": count_over(values, 0.01),
    }


def build_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    metrics = {
        "overall": metric_block(diff),
        "output_0": metric_block(diff[:, 0]),
        "output_1": metric_block(diff[:, 1]),
        "image_count": int(diff.shape[0]),
        "output_count": int(diff.size),
        "candidate_nan_count": int(np.isnan(candidate).sum()),
        "reference_nan_count": int(np.isnan(reference).sum()),
        "diff_nan_count": int(np.isnan(diff).sum()),
        "candidate_range_error_count": range_error_count(candidate),
        "reference_range_error_count": range_error_count(reference),
        "candidate_input_output_float32": True,
    }
    metrics["passes_acut_model_abs_diff_threshold"] = (
        metrics["overall"]["maxae"] is not None
        and metrics["overall"]["maxae"] <= 0.001
        and metrics["candidate_nan_count"] == 0
        and metrics["candidate_range_error_count"] == 0
    )
    return metrics


def write_comparison_csv(path: Path, image_paths: list[Path], reference: np.ndarray, candidate: np.ndarray) -> None:
    diff = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    fieldnames = [
        "image_index",
        "image_path",
        "reference_0",
        "reference_1",
        "dynamic_ptq_0",
        "dynamic_ptq_1",
        "abs_error_0",
        "abs_error_1",
        "row_mae",
        "row_maxae",
        "row_count_over_0_001",
        "row_count_over_0_005",
        "row_count_over_0_01",
        "reference_nan_count",
        "dynamic_nan_count",
        "reference_range_error_count",
        "dynamic_range_error_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            row_diff = diff[index]
            writer.writerow(
                {
                    "image_index": index,
                    "image_path": str(image_path),
                    "reference_0": float(reference[index, 0]),
                    "reference_1": float(reference[index, 1]),
                    "dynamic_ptq_0": float(candidate[index, 0]),
                    "dynamic_ptq_1": float(candidate[index, 1]),
                    "abs_error_0": float(row_diff[0]),
                    "abs_error_1": float(row_diff[1]),
                    "row_mae": float(np.nanmean(row_diff)),
                    "row_maxae": float(np.nanmax(row_diff)),
                    "row_count_over_0_001": count_over(row_diff, 0.001),
                    "row_count_over_0_005": count_over(row_diff, 0.005),
                    "row_count_over_0_01": count_over(row_diff, 0.01),
                    "reference_nan_count": int(np.isnan(reference[index]).sum()),
                    "dynamic_nan_count": int(np.isnan(candidate[index]).sum()),
                    "reference_range_error_count": range_error_count(reference[index]),
                    "dynamic_range_error_count": range_error_count(candidate[index]),
                }
            )


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def latency_stats(values_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.fmean(values_ms)) if values_ms else 0.0,
        "median_ms": float(statistics.median(values_ms)) if values_ms else 0.0,
        "p90_ms": percentile(values_ms, 90),
        "max_ms": max(values_ms) if values_ms else 0.0,
        "runs": len(values_ms),
    }


def measure_latency(model_path: Path, samples: list[np.ndarray]) -> dict[str, Any]:
    runner = TFLiteRunner(model_path)
    for index in range(LATENCY_WARMUP):
        runner.run_one(samples[index % len(samples)])

    values_ms: list[float] = []
    for index in range(LATENCY_RUNS):
        sample = samples[index % len(samples)]
        start = time.perf_counter()
        runner.run_one(sample)
        values_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "model_path": str(model_path),
        "warmup_runs": LATENCY_WARMUP,
        **latency_stats(values_ms),
    }


def write_dynamic_report(
    path: Path,
    metrics: dict[str, Any],
    selection_info: dict[str, Any],
    fp16_details: dict[str, Any],
    dynamic_details: dict[str, Any],
) -> None:
    lines = [
        "# Dynamic PTQ Extended Parity Report",
        "",
        "## Summary",
        f"- **Reference FP16**: `{FP16_PATH}`.",
        f"- **Dynamic PTQ**: `{DYNAMIC_PATH}`.",
        f"- **Image count**: {metrics['image_count']}.",
        f"- **Image selection source**: `{selection_info.get('source')}`.",
        f"- **A-cut 50-image set found**: {selection_info.get('source') == 'acut_manifest'}.",
        f"- **Preprocessing**: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.",
        f"- **Passes model-level abs diff <= 0.001 with no NaN/range errors**: {metrics['passes_acut_model_abs_diff_threshold']}.",
        "",
        "## Overall Error Metrics",
        "| Scope | MAE | MaxAE | Median AE | P90 AE | P95 AE | >0.001 | >0.005 | >0.01 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in ["overall", "output_0", "output_1"]:
        item = metrics[scope]
        lines.append(
            f"| {scope} | {item['mae']} | {item['maxae']} | {item['median_ae']} | "
            f"{item['p90_ae']} | {item['p95_ae']} | {item['count_over_0_001']} | "
            f"{item['count_over_0_005']} | {item['count_over_0_01']} |"
        )
    lines.extend(
        [
            "",
            "## NaN And Range Checks",
            f"- **Candidate NaN count**: {metrics['candidate_nan_count']}.",
            f"- **Reference NaN count**: {metrics['reference_nan_count']}.",
            f"- **Diff NaN count**: {metrics['diff_nan_count']}.",
            f"- **Candidate range errors outside [0,1]**: {metrics['candidate_range_error_count']}.",
            f"- **Reference range errors outside [0,1]**: {metrics['reference_range_error_count']}.",
            "",
            "## Interpreter Details",
            f"- **FP16 input details**: `{fp16_details.get('input_details')}`.",
            f"- **FP16 output details**: `{fp16_details.get('output_details')}`.",
            f"- **Dynamic PTQ input details**: `{dynamic_details.get('input_details')}`.",
            f"- **Dynamic PTQ output details**: `{dynamic_details.get('output_details')}`.",
            "",
            "## Selection Details",
            f"- **Selection info**: `{selection_info}`.",
            "",
            "## Threshold Interpretation",
            "- The user-provided A-cut parity gate is stricter than the observed Dynamic PTQ errors if MaxAE exceeds `0.001`.",
            "- This report does not evaluate final A-cut ranking because only ICAA model outputs are compared here.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latency_report(path: Path, latency: dict[str, Any]) -> None:
    fp16 = latency["fp16"]
    dynamic = latency["dynamic_ptq"]
    lines = [
        "# Dynamic PTQ WSL Latency Report",
        "",
        "## Scope",
        "- This is a local WSL TensorFlow Lite interpreter measurement.",
        "- It is not Android/iOS latency and must not be used as a production speed claim.",
        f"- Warmup runs per model: {LATENCY_WARMUP}.",
        f"- Timed runs per model: {LATENCY_RUNS}.",
        "",
        "| Model | Mean ms | Median ms | P90 ms | Max ms | Runs |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
        f"| FP16 reference | {fp16['mean_ms']} | {fp16['median_ms']} | {fp16['p90_ms']} | {fp16['max_ms']} | {fp16['runs']} |",
        f"| Dynamic PTQ | {dynamic['mean_ms']} | {dynamic['median_ms']} | {dynamic['p90_ms']} | {dynamic['max_ms']} | {dynamic['runs']} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    require_path(FP16_PATH, "FP16 reference")
    require_path(DYNAMIC_PATH, "Dynamic PTQ candidate")
    require_path(ICAA_ROOT, "ICAA image root")

    image_paths, selection_info = select_images()
    if len(image_paths) < IMAGE_COUNT:
        raise RuntimeError(f"Expected at least {IMAGE_COUNT} images, found {len(image_paths)}")

    samples = [preprocess_image(path) for path in image_paths]
    reference, fp16_details = run_outputs(FP16_PATH, samples)
    candidate, dynamic_details = run_outputs(DYNAMIC_PATH, samples)
    metrics = build_metrics(reference, candidate)

    comparison_path = REVIEW_DIR / "dynamic_ptq_extended_comparison.csv"
    report_path = REVIEW_DIR / "dynamic_ptq_extended_report.md"
    latency_path = REVIEW_DIR / "dynamic_ptq_latency_report.md"
    summary_json_path = REVIEW_DIR / "dynamic_ptq_extended_summary.json"

    write_comparison_csv(comparison_path, image_paths, reference, candidate)
    write_dynamic_report(report_path, metrics, selection_info, fp16_details, dynamic_details)

    latency = {
        "fp16": measure_latency(FP16_PATH, samples),
        "dynamic_ptq": measure_latency(DYNAMIC_PATH, samples),
    }
    write_latency_report(latency_path, latency)

    summary = {
        "metrics": metrics,
        "selection_info": selection_info,
        "image_paths": [str(path) for path in image_paths],
        "fp16_details": fp16_details,
        "dynamic_details": dynamic_details,
        "latency": latency,
        "outputs": {
            "comparison_csv": str(comparison_path),
            "report_md": str(report_path),
            "latency_report_md": str(latency_path),
        },
    }
    summary_json_path.write_text(json.dumps(jsonable(summary), indent=2), encoding="utf-8")
    print(json.dumps(jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
