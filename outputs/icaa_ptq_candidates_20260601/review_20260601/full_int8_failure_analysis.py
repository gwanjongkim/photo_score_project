# ICAA Full INT8 PTQ 후보의 TFLite DIV 실패를 조사하는 스크립트
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv
import json
import re
import sys
import traceback
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as tflite_schema

ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = ROOT / "outputs/icaa_ptq_candidates_20260601/review_20260601"
INT8_PATH = ROOT / "outputs/icaa_ptq_candidates_20260601/icaa_dat_tf_native_full_integer_int8_float_io.tflite"
ICAA_ROOT = ROOT / "data/raw/icaa17k"
ICAA_TEST_CSV = ROOT / "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"

INPUT_SHAPE = [1, 224, 224, 3]
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


def dtype_name(value: Any) -> str:
    try:
        return np.dtype(value).name
    except TypeError:
        return str(value)


def detail_summary(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in details:
        rows.append(
            {
                "name": item.get("name"),
                "index": int(item.get("index")),
                "shape": item.get("shape").tolist(),
                "shape_signature": item.get("shape_signature").tolist(),
                "dtype": dtype_name(item.get("dtype")),
                "quantization": tuple(item.get("quantization", ())),
                "quantization_parameters": jsonable(item.get("quantization_parameters", {})),
            }
        )
    return rows


def preprocess_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return ((array - IMAGENET_MEAN) / IMAGENET_STD)[np.newaxis, ...].astype(np.float32)


def first_icaa_image() -> Path:
    if ICAA_TEST_CSV.is_file():
        with ICAA_TEST_CSV.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row.get("ID")
                if image_id:
                    path = ICAA_ROOT / image_id
                    if path.is_file():
                        return path
    for path in sorted(ICAA_ROOT.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            return path
    raise FileNotFoundError(f"No ICAA image found under {ICAA_ROOT}")


def quant_summary(item: dict[str, Any]) -> dict[str, Any]:
    params = item.get("quantization_parameters", {})
    scales = np.asarray(params.get("scales", []), dtype=np.float64)
    zero_points = np.asarray(params.get("zero_points", []), dtype=np.int64)
    return {
        "quantization": tuple(item.get("quantization", ())),
        "scale_count": int(scales.size),
        "zero_point_count": int(zero_points.size),
        "scale_min": float(np.min(scales)) if scales.size else None,
        "scale_max": float(np.max(scales)) if scales.size else None,
        "zero_scale_count": int(np.sum(scales == 0.0)) if scales.size else 0,
        "nan_or_inf_scale_count": int(np.sum(~np.isfinite(scales))) if scales.size else 0,
        "zero_point_min": int(np.min(zero_points)) if zero_points.size else None,
        "zero_point_max": int(np.max(zero_points)) if zero_points.size else None,
        "quantized_dimension": params.get("quantized_dimension"),
    }


def tensor_row(tensor: dict[str, Any]) -> dict[str, Any]:
    q = quant_summary(tensor)
    return {
        "tensor_index": int(tensor.get("index")),
        "tensor_name": tensor.get("name", ""),
        "dtype": dtype_name(tensor.get("dtype")),
        "shape": json.dumps(tensor.get("shape", np.asarray([])).tolist()),
        "scale_count": q["scale_count"],
        "scale_min": q["scale_min"],
        "scale_max": q["scale_max"],
        "zero_scale_count": q["zero_scale_count"],
        "nan_or_inf_scale_count": q["nan_or_inf_scale_count"],
        "zero_point_count": q["zero_point_count"],
        "zero_point_min": q["zero_point_min"],
        "zero_point_max": q["zero_point_max"],
    }


def tflite_builtin_name_map() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name in dir(tflite_schema.BuiltinOperator):
        if name.startswith("_") or name in {"Name", "Value"}:
            continue
        value = getattr(tflite_schema.BuiltinOperator, name)
        if isinstance(value, int):
            mapping[value] = name
    return mapping


def flatbuffer_operator_codes(model_path: Path) -> dict[str, Any]:
    name_by_code = tflite_builtin_name_map()
    model = tflite_schema.Model.GetRootAsModel(model_path.read_bytes(), 0)
    builtin_codes: list[str] = []
    custom_codes: list[str] = []
    for index in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(index)
        custom_code = op_code.CustomCode()
        if custom_code:
            custom_codes.append(custom_code.decode("utf-8"))
        else:
            builtin_codes.append(name_by_code.get(op_code.BuiltinCode(), str(op_code.BuiltinCode())))
    return {
        "builtin_operator_codes": sorted(set(builtin_codes)),
        "custom_operator_codes": sorted(set(custom_codes)),
        "flex_or_select_tf_ops_found": any(code.startswith("Flex") for code in custom_codes),
    }


def make_interpreter(preserve_tensors: bool = True) -> tf.lite.Interpreter:
    return tf.lite.Interpreter(
        model_path=str(INT8_PATH),
        experimental_preserve_all_tensors=preserve_tensors,
    )


def tensor_brief(tensor_index: int, tensor_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    tensor = tensor_by_index.get(tensor_index)
    if tensor is None:
        return {"tensor_index": tensor_index, "missing": True}
    q = quant_summary(tensor)
    return {
        "tensor_index": tensor_index,
        "name": tensor.get("name", ""),
        "dtype": dtype_name(tensor.get("dtype")),
        "shape": tensor.get("shape", np.asarray([])).tolist(),
        "scale_count": q["scale_count"],
        "scale_min": q["scale_min"],
        "scale_max": q["scale_max"],
        "zero_scale_count": q["zero_scale_count"],
        "nan_or_inf_scale_count": q["nan_or_inf_scale_count"],
        "zero_point_min": q["zero_point_min"],
        "zero_point_max": q["zero_point_max"],
    }


def op_row(op_index: int, op: dict[str, Any], tensor_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    input_indices = [int(index) for index in op.get("inputs", [])]
    output_indices = [int(index) for index in op.get("outputs", [])]
    input_tensors = [tensor_by_index.get(index, {}) for index in input_indices]
    output_tensors = [tensor_by_index.get(index, {}) for index in output_indices]
    dtypes = [dtype_name(tensor.get("dtype")) for tensor in input_tensors + output_tensors if tensor]
    quantized = any(dtype in {"int8", "uint8", "int16"} for dtype in dtypes)
    scale_issues = []
    for tensor in input_tensors + output_tensors:
        if not tensor:
            continue
        q = quant_summary(tensor)
        dtype = dtype_name(tensor.get("dtype"))
        if dtype in {"int8", "uint8", "int16"} and q["scale_count"] == 0:
            scale_issues.append(f"tensor_{tensor.get('index')}_missing_scale")
        if q["zero_scale_count"]:
            scale_issues.append(f"tensor_{tensor.get('index')}_zero_scale")
        if q["nan_or_inf_scale_count"]:
            scale_issues.append(f"tensor_{tensor.get('index')}_nan_or_inf_scale")
    return {
        "op_index": op_index,
        "op_name": op.get("op_name", ""),
        "is_div": op.get("op_name", "") == "DIV",
        "is_quantized_by_tensor_dtype": quantized,
        "input_indices": json.dumps(input_indices),
        "output_indices": json.dumps(output_indices),
        "input_dtypes": json.dumps([dtype_name(tensor.get("dtype")) for tensor in input_tensors]),
        "output_dtypes": json.dumps([dtype_name(tensor.get("dtype")) for tensor in output_tensors]),
        "input_names": json.dumps([tensor.get("name", "") for tensor in input_tensors]),
        "output_names": json.dumps([tensor.get("name", "") for tensor in output_tensors]),
        "input_quant": json.dumps([quant_summary(tensor) for tensor in input_tensors]),
        "output_quant": json.dumps([quant_summary(tensor) for tensor in output_tensors]),
        "scale_issues": json.dumps(scale_issues),
    }


def write_op_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "op_index",
        "op_name",
        "is_div",
        "is_quantized_by_tensor_dtype",
        "input_indices",
        "output_indices",
        "input_dtypes",
        "output_dtypes",
        "input_names",
        "output_names",
        "input_quant",
        "output_quant",
        "scale_issues",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def array_stats(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "nan_count": int(np.isnan(array).sum()) if np.issubdtype(array.dtype, np.floating) else 0,
        "finite_count": int(np.isfinite(array).sum()) if np.issubdtype(array.dtype, np.floating) else int(array.size),
        "zero_count": int(np.sum(array == 0)),
    }
    if array.size:
        result.update(
            {
                "min": float(np.nanmin(array.astype(np.float64))),
                "max": float(np.nanmax(array.astype(np.float64))),
                "mean": float(np.nanmean(array.astype(np.float64))),
            }
        )
    return result


def inspect_failed_op_tensors(
    interpreter: tf.lite.Interpreter,
    op_index: int | None,
    ops: list[dict[str, Any]],
    tensor_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if op_index is None or op_index < 0 or op_index >= len(ops):
        return {"available": False, "reason": "failed op index unavailable"}
    op = ops[op_index]
    details: dict[str, Any] = {
        "available": True,
        "op_index": op_index,
        "op_name": op.get("op_name", ""),
        "inputs": [],
        "outputs": [],
    }
    for direction in ["inputs", "outputs"]:
        for tensor_index in [int(index) for index in op.get(direction, [])]:
            item = tensor_brief(tensor_index, tensor_by_index)
            try:
                value = interpreter.get_tensor(tensor_index)
                item["runtime_stats"] = array_stats(value)
            except Exception as exc:
                item["runtime_stats_error"] = str(exc)
            details[direction].append(item)
    return details


def extract_failed_node(error_text: str) -> int | None:
    match = re.search(r"Node number (\d+)", error_text)
    return int(match.group(1)) if match else None


def invoke_case(case_name: str, sample: np.ndarray) -> dict[str, Any]:
    interpreter = make_interpreter(preserve_tensors=True)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()
    tensor_by_index = {int(tensor["index"]): tensor for tensor in tensors}
    result: dict[str, Any] = {
        "case": case_name,
        "status": "failed",
        "input_details": detail_summary(input_details),
        "output_details": detail_summary(output_details),
    }
    try:
        interpreter.set_tensor(input_details[0]["index"], sample.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        result.update({"status": "ok", "output": output.astype(np.float32).tolist()})
    except Exception as exc:
        error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        failed_node = extract_failed_node(error_text)
        result.update(
            {
                "error": error_text,
                "failed_node": failed_node,
                "failed_op_tensors": inspect_failed_op_tensors(interpreter, failed_node, ops, tensor_by_index),
            }
        )
    return result


def write_markdown_report(path: Path, analysis: dict[str, Any]) -> None:
    failed_cases = [case for case in analysis["invoke_cases"] if case.get("status") != "ok"]
    div_summary = analysis["div_summary"]
    quant = analysis["quantization_summary"]
    failed_node = failed_cases[0].get("failed_node") if failed_cases else None
    failed_op = analysis["failed_node_inventory"].get(str(failed_node), {}) if failed_node is not None else {}
    lines = [
        "# Full INT8 Failure Analysis",
        "",
        "## Confirmed Facts",
        f"- **Model path**: `{INT8_PATH}`.",
        f"- **Input details**: `{analysis['input_details']}`.",
        f"- **Output details**: `{analysis['output_details']}`.",
        f"- **Total ops**: {analysis['op_count']}.",
        f"- **DIV op count**: {div_summary['div_op_count']}.",
        f"- **Custom/Flex ops found**: {analysis['flatbuffer_operator_codes']['custom_operator_codes']}.",
        f"- **Failed invoke cases**: {len(failed_cases)} of {len(analysis['invoke_cases'])}.",
        "",
        "## Minimal Invoke Failure",
    ]
    for case in analysis["invoke_cases"]:
        lines.append(f"- **{case['case']}**: `{case['status']}`" + (f" - {case.get('error')}" if case.get("error") else "."))
    lines.extend(
        [
            "",
            "## Failed Node",
            f"- **Failed node from exception**: {failed_node}.",
            f"- **Failed op inventory row**: `{failed_op}`.",
            f"- **Failed op tensor runtime details**: `{failed_cases[0].get('failed_op_tensors') if failed_cases else {}}`.",
            "",
            "## DIV Quantization",
            f"- **DIV ops quantized by tensor dtype**: {div_summary['quantized_div_op_count']}.",
            f"- **DIV op indices**: `{div_summary['div_op_indices']}`.",
            f"- **DIV ops with scale issues**: `{div_summary['div_ops_with_scale_issues']}`.",
            "",
            "## Quantization Parameter Scan",
            f"- **Tensor dtype counts**: `{analysis['tensor_dtype_counts']}`.",
            f"- **Quantized tensor count**: {quant['quantized_tensor_count']}.",
            f"- **Quantized tensors missing scales**: {quant['quantized_tensors_missing_scales']}.",
            f"- **Quantized tensors with zero scales**: {quant['quantized_tensors_with_zero_scales']}.",
            f"- **Quantized tensors with NaN/Inf scales**: {quant['quantized_tensors_with_nan_or_inf_scales']}.",
            "",
            "## Conservative Interpretation",
            "- The TFLite flatbuffer is valid enough to allocate and expose float32 input/output, but it is not runtime-safe in this local interpreter test.",
            "- The preprocessing normalization is external to this TFLite model; the failing DIV is inside the model graph, not the Python preprocessing step.",
            "- The failed DIV node uses int8 input and output tensors; the denominator tensor has valid quantization parameters but contains literal zero values at runtime.",
            "- This is more consistent with a quantized graph numeric failure around attention normalization than with invalid external input dtype.",
            "- Representative calibration remains a plausible contributor if the calibrated dynamic ranges allow a denominator tensor to collapse to zero, but this script does not prove calibration is the sole cause.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    require_path(INT8_PATH, "Full INT8 float-IO candidate")
    require_path(ICAA_ROOT, "ICAA image root")

    interpreter = make_interpreter(preserve_tensors=True)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ops = interpreter._get_ops_details()
    tensors = interpreter.get_tensor_details()
    tensor_by_index = {int(tensor["index"]): tensor for tensor in tensors}

    inventory_rows = [op_row(index, op, tensor_by_index) for index, op in enumerate(ops)]
    inventory_path = REVIEW_DIR / "full_int8_op_inventory.csv"
    write_op_inventory(inventory_path, inventory_rows)

    tensor_dtype_counts = dict(sorted(Counter(dtype_name(tensor.get("dtype")) for tensor in tensors).items()))
    quantized_tensors = [tensor for tensor in tensors if dtype_name(tensor.get("dtype")) in {"int8", "uint8", "int16"}]
    quantization_summary = {
        "quantized_tensor_count": len(quantized_tensors),
        "quantized_tensors_missing_scales": sum(1 for tensor in quantized_tensors if quant_summary(tensor)["scale_count"] == 0),
        "quantized_tensors_with_zero_scales": sum(1 for tensor in quantized_tensors if quant_summary(tensor)["zero_scale_count"] > 0),
        "quantized_tensors_with_nan_or_inf_scales": sum(1 for tensor in quantized_tensors if quant_summary(tensor)["nan_or_inf_scale_count"] > 0),
    }
    div_rows = [row for row in inventory_rows if row["is_div"]]
    div_summary = {
        "div_op_count": len(div_rows),
        "div_op_indices": [row["op_index"] for row in div_rows],
        "quantized_div_op_count": sum(1 for row in div_rows if row["is_quantized_by_tensor_dtype"]),
        "div_ops_with_scale_issues": [row["op_index"] for row in div_rows if json.loads(row["scale_issues"])],
    }

    image_path = first_icaa_image()
    real_sample = preprocess_image(image_path)
    rng = np.random.default_rng(123)
    invoke_cases = [
        invoke_case("zeros", np.zeros(INPUT_SHAPE, dtype=np.float32)),
        invoke_case("ones", np.ones(INPUT_SHAPE, dtype=np.float32)),
        invoke_case("random_normal", rng.normal(size=INPUT_SHAPE).astype(np.float32)),
        invoke_case("first_icaa_image", real_sample),
    ]
    failed_nodes = sorted(
        {
            int(case["failed_node"])
            for case in invoke_cases
            if case.get("failed_node") is not None
        }
    )
    failed_node_inventory = {
        str(index): inventory_rows[index]
        for index in failed_nodes
        if 0 <= index < len(inventory_rows)
    }

    analysis = {
        "model_path": str(INT8_PATH),
        "model_size_bytes": INT8_PATH.stat().st_size,
        "input_details": detail_summary(input_details),
        "output_details": detail_summary(output_details),
        "op_count": len(ops),
        "op_name_counts": dict(sorted(Counter(row["op_name"] for row in inventory_rows).items())),
        "tensor_dtype_counts": tensor_dtype_counts,
        "quantization_summary": quantization_summary,
        "div_summary": div_summary,
        "flatbuffer_operator_codes": flatbuffer_operator_codes(INT8_PATH),
        "first_icaa_image": str(image_path),
        "invoke_cases": invoke_cases,
        "failed_node_inventory": failed_node_inventory,
        "div_inventory": div_rows,
        "outputs": {
            "op_inventory_csv": str(inventory_path),
            "report_md": str(REVIEW_DIR / "full_int8_failure_analysis.md"),
            "analysis_json": str(REVIEW_DIR / "full_int8_failure_analysis.json"),
        },
    }

    (REVIEW_DIR / "full_int8_failure_analysis.json").write_text(
        json.dumps(jsonable(analysis), indent=2),
        encoding="utf-8",
    )
    write_markdown_report(REVIEW_DIR / "full_int8_failure_analysis.md", analysis)
    console_summary = {
        "model_path": analysis["model_path"],
        "op_count": analysis["op_count"],
        "div_summary": analysis["div_summary"],
        "quantization_summary": analysis["quantization_summary"],
        "invoke_cases": analysis["invoke_cases"],
        "outputs": analysis["outputs"],
    }
    print(json.dumps(jsonable(console_summary), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
