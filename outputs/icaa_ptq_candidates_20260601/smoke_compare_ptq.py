# ICAA PTQ 후보 TFLite 모델의 FP16 기준 대비 출력 차이를 확인하는 스크립트
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import sys
from typing import Any

import numpy as np

import tensorflow as tf

OUTPUT_DIR_REL_PATH = "outputs/icaa_ptq_candidates_20260601"
FP16_REFERENCE_REL_PATH = "outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
CANDIDATE_FILENAMES = [
    "icaa_dat_tf_native_dynamic_range_ptq.tflite",
    "icaa_dat_tf_native_full_integer_int8_float_io.tflite",
    "icaa_dat_tf_native_full_integer_int8_int_io_fallback.tflite",
]
INPUT_SHAPE = [1, 224, 224, 3]
OUTPUT_SHAPE = [1, 2]
SMOKE_IMAGE_COUNT = 16
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def preprocess_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - IMAGENET_MEAN) / IMAGENET_STD)[np.newaxis, ...].astype(np.float32)


def append_unique(paths: list[Path], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen or not path.is_file():
        return
    seen.add(resolved)
    paths.append(path)


def find_images(root: Path, limit: int) -> list[Path]:
    image_root = root / IMAGE_ROOT_REL_PATH
    csv_path = root / CSV_REL_PATH
    paths: list[Path] = []
    seen: set[Path] = set()
    if csv_path.is_file():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row.get("ID")
                if not image_id:
                    continue
                append_unique(paths, seen, image_root / image_id)
                if len(paths) >= limit:
                    return paths
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for path in sorted(image_root.rglob("*")):
        if path.suffix.lower() in suffixes:
            append_unique(paths, seen, path)
            if len(paths) >= limit:
                break
    return paths


def detail_summary(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in details:
        rows.append(
            {
                "name": item.get("name"),
                "index": int(item.get("index")),
                "shape": item.get("shape").tolist(),
                "dtype": np.dtype(item.get("dtype")).name,
                "quantization": tuple(item.get("quantization", ())),
            }
        )
    return rows


def input_for_interpreter(input_detail: dict[str, Any], sample: np.ndarray) -> np.ndarray:
    dtype = np.dtype(input_detail["dtype"])
    if dtype == np.float32:
        return sample.astype(np.float32)
    if dtype == np.float16:
        return sample.astype(np.float16)
    if dtype in {np.dtype("int8"), np.dtype("uint8")}:
        scale, zero_point = input_detail.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError(f"Quantized input has invalid scale: {input_detail}")
        quantized = np.round(sample / scale + zero_point)
        info = np.iinfo(dtype)
        return np.clip(quantized, info.min, info.max).astype(dtype)
    raise TypeError(f"Unsupported input dtype: {dtype}")


def output_from_interpreter(output_detail: dict[str, Any], raw_output: np.ndarray) -> np.ndarray:
    dtype = np.dtype(output_detail["dtype"])
    if dtype in {np.dtype("float32"), np.dtype("float16")}:
        return raw_output.astype(np.float32)
    if dtype in {np.dtype("int8"), np.dtype("uint8")}:
        scale, zero_point = output_detail.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError(f"Quantized output has invalid scale: {output_detail}")
        return ((raw_output.astype(np.float32) - zero_point) * scale).astype(np.float32)
    raise TypeError(f"Unsupported output dtype: {dtype}")


def run_model(model_path: Path, samples: list[np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    outputs = []
    for sample in samples:
        interpreter.set_tensor(input_details[0]["index"], input_for_interpreter(input_details[0], sample))
        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details[0]["index"])
        outputs.append(output_from_interpreter(output_details[0], raw_output))
    return np.concatenate(outputs, axis=0), {
        "input_details": detail_summary(input_details),
        "output_details": detail_summary(output_details),
    }


def range_error_count(values: np.ndarray) -> int:
    return int(np.logical_or(values < 0.0, values > 1.0).sum())


def write_report(output_dir: Path, summary_rows: list[dict[str, Any]], reference_path: Path, image_paths: list[Path]) -> None:
    lines = [
        "# ICAA PTQ Smoke Comparison",
        "",
        f"- **Reference path**: `{reference_path}`.",
        f"- **Image count**: {len(image_paths)}.",
        "- **Preprocessing**: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.",
        "",
        "| Candidate | Status | MAE | MaxAE | NaN Count | Range Errors | Input DType | Output DType | Error |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- | :--- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {candidate} | `{status}` | {mae} | {maxae} | {nan_count} | {range_errors} | `{input_dtype}` | `{output_dtype}` | {error} |".format(
                candidate=row.get("candidate", ""),
                status=row.get("status", ""),
                mae=row.get("mae", ""),
                maxae=row.get("maxae", ""),
                nan_count=row.get("nan_count", ""),
                range_errors=row.get("range_errors", ""),
                input_dtype=row.get("input_dtype", ""),
                output_dtype=row.get("output_dtype", ""),
                error=row.get("error", ""),
            )
        )
    lines.extend(["", "No speed or latency claim is made by this smoke comparison.", ""])
    (output_dir / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = project_root()
    output_dir = root / OUTPUT_DIR_REL_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = root / FP16_REFERENCE_REL_PATH
    if not reference_path.is_file():
        (output_dir / "smoke_report.md").write_text(
            f"# ICAA PTQ Smoke Comparison\n\nFP16 reference not found: `{reference_path}`.\n",
            encoding="utf-8",
        )
        return 1

    image_paths = find_images(root, SMOKE_IMAGE_COUNT)
    if len(image_paths) < 1:
        (output_dir / "smoke_report.md").write_text("# ICAA PTQ Smoke Comparison\n\nNo smoke images found.\n", encoding="utf-8")
        return 1

    samples = [preprocess_image(path) for path in image_paths]
    reference_output, reference_details = run_model(reference_path, samples)
    candidate_paths = [output_dir / name for name in CANDIDATE_FILENAMES if (output_dir / name).is_file()]

    summary_rows: list[dict[str, Any]] = []
    per_image_rows: list[dict[str, Any]] = []
    for candidate_path in candidate_paths:
        candidate = candidate_path.stem
        try:
            candidate_output, candidate_details = run_model(candidate_path, samples)
            diff = np.abs(reference_output.astype(np.float32) - candidate_output.astype(np.float32))
            nan_count = int(np.isnan(candidate_output).sum() + np.isnan(diff).sum())
            range_errors = range_error_count(candidate_output)
            summary = {
                "candidate": candidate,
                "status": "ok",
                "mae": float(np.nanmean(diff)),
                "maxae": float(np.nanmax(diff)),
                "nan_count": nan_count,
                "range_errors": range_errors,
                "input_dtype": candidate_details["input_details"][0]["dtype"],
                "output_dtype": candidate_details["output_details"][0]["dtype"],
                "input_shape": candidate_details["input_details"][0]["shape"],
                "output_shape": candidate_details["output_details"][0]["shape"],
                "error": "",
            }
            summary_rows.append(summary)
            for index, image_path in enumerate(image_paths):
                row_diff = diff[index]
                per_image_rows.append(
                    {
                        "candidate": candidate,
                        "image_index": index,
                        "image_path": str(image_path),
                        "reference_0": float(reference_output[index, 0]),
                        "reference_1": float(reference_output[index, 1]),
                        "candidate_0": float(candidate_output[index, 0]),
                        "candidate_1": float(candidate_output[index, 1]),
                        "abs_error_0": float(row_diff[0]),
                        "abs_error_1": float(row_diff[1]),
                        "mae": float(np.nanmean(row_diff)),
                        "maxae": float(np.nanmax(row_diff)),
                        "candidate_nan_count": int(np.isnan(candidate_output[index]).sum()),
                        "candidate_range_error_count": range_error_count(candidate_output[index]),
                        "reference_nan_count": int(np.isnan(reference_output[index]).sum()),
                        "reference_range_error_count": range_error_count(reference_output[index]),
                    }
                )
        except Exception as exc:
            summary_rows.append(
                {
                    "candidate": candidate,
                    "status": "failed",
                    "mae": "",
                    "maxae": "",
                    "nan_count": "",
                    "range_errors": "",
                    "input_dtype": "",
                    "output_dtype": "",
                    "error": str(exc),
                }
            )

    with (output_dir / "smoke_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "candidate",
            "image_index",
            "image_path",
            "reference_0",
            "reference_1",
            "candidate_0",
            "candidate_1",
            "abs_error_0",
            "abs_error_1",
            "mae",
            "maxae",
            "candidate_nan_count",
            "candidate_range_error_count",
            "reference_nan_count",
            "reference_range_error_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    (output_dir / "smoke_summary.json").write_text(
        json.dumps(
            {
                "reference_path": str(reference_path),
                "reference_details": reference_details,
                "image_paths": [str(path) for path in image_paths],
                "summary": summary_rows,
                "candidate_count": len(candidate_paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(output_dir, summary_rows, reference_path, image_paths)
    print(json.dumps({"summary": summary_rows, "candidate_count": len(candidate_paths)}, indent=2))
    return 0 if summary_rows and all(row.get("status") == "ok" for row in summary_rows) else 1


if __name__ == "__main__":
    sys.exit(main())
