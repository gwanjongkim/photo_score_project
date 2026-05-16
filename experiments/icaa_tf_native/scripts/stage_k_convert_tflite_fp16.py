# ICAA17K DAT SavedModel을 FP16 TFLite로 변환하고 검증하는 스크립트
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import json
import os
import sys
import traceback
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as tflite_schema

sys.dont_write_bytecode = True

OUTPUT_PREFIX = "icaa_tf_native_tflite_fp16"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_k_convert_tflite_fp16.py"
SAVED_MODEL_REL_PATH = "outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model"
FP32_TFLITE_REL_PATH = "outputs/icaa_tf_native_tflite_fp32_20260516_140027/icaa_dat_tf_native_fp32.tflite"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_k_tflite_fp16_conversion.md"
SEED = 123
REAL_IMAGE_COUNT = 16
OPTIONAL_REAL_IMAGE_COUNT = 64
INPUT_SHAPE = [1, 224, 224, 3]
OUTPUT_SHAPE = [1, 2]
SENSITIVITY_EPS = 1e-6

PREFERRED_COLOR_MAX_ABS_DIFF = 1e-3
PREFERRED_FULL_MAX_ABS_DIFF = 1e-3
PREFERRED_FULL_MEAN_ABS_DIFF = 1e-4
ACCEPTABLE_COLOR_MAX_ABS_DIFF = 5e-3
ACCEPTABLE_FULL_MAX_ABS_DIFF = 5e-3
ACCEPTABLE_FULL_MEAN_ABS_DIFF = 5e-4
COLOR_HARD_FAIL_ABS_DIFF = 1e-2

log_lines: list[str] = []


def log(message: str) -> None:
    print(message)
    log_lines.append(message)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


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
    return value


def detail_summary(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for item in details:
        summary.append(
            {
                "name": item.get("name"),
                "index": int(item.get("index")),
                "shape": item.get("shape").tolist(),
                "shape_signature": item.get("shape_signature").tolist(),
                "dtype": np.dtype(item.get("dtype")).name,
                "quantization": tuple(item.get("quantization", ())),
                "quantization_parameters": jsonable(item.get("quantization_parameters", {})),
            }
        )
    return summary


def preprocess_real_image(path: Path) -> np.ndarray:
    from PIL import Image

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std


def load_real_images(root: Path, limit: int) -> tuple[np.ndarray | None, list[str]]:
    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    if not csv_path.is_file() or not image_root.is_dir():
        return None, []

    arrays: list[np.ndarray] = []
    paths: list[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("ID")
            if not image_id:
                continue
            image_path = image_root / image_id
            if not image_path.is_file():
                continue
            arrays.append(preprocess_real_image(image_path))
            paths.append(str(image_path))
            if len(arrays) >= limit:
                break

    if not arrays:
        return None, paths
    return np.stack(arrays, axis=0).astype(np.float32), paths


def savedmodel_scores(infer: Any, input_data: np.ndarray) -> np.ndarray:
    outputs = infer(image=tf.constant(input_data.astype(np.float32)))
    if "scores" in outputs:
        return outputs["scores"].numpy().astype(np.float32)
    if len(outputs) == 1:
        return next(iter(outputs.values())).numpy().astype(np.float32)
    raise KeyError(f"Could not identify SavedModel score output keys: {list(outputs.keys())}")


def input_for_interpreter(input_details: list[dict[str, Any]], sample: np.ndarray) -> np.ndarray:
    dtype = np.dtype(input_details[0]["dtype"])
    if dtype == np.float32:
        return sample.astype(np.float32)
    if dtype == np.float16:
        return sample.astype(np.float16)
    raise TypeError(f"Unsupported TFLite input dtype for preprocessing pipeline: {dtype}")


def run_tflite_inference(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    outputs = []
    for index in range(input_data.shape[0]):
        sample = input_for_interpreter(input_details, input_data[index : index + 1])
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        outputs.append(interpreter.get_tensor(output_details[0]["index"]).astype(np.float32))
    return np.concatenate(outputs, axis=0)


def compare_outputs(name: str, reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    diff = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    mos_diff = diff[:, 0] if diff.shape[1] > 0 else np.asarray([], dtype=np.float32)
    color_diff = diff[:, 1] if diff.shape[1] > 1 else np.asarray([], dtype=np.float32)

    full_max = float(diff.max()) if diff.size else 0.0
    full_mean = float(diff.mean()) if diff.size else 0.0
    mos_max = float(mos_diff.max()) if mos_diff.size else 0.0
    mos_mean = float(mos_diff.mean()) if mos_diff.size else 0.0
    color_max = float(color_diff.max()) if color_diff.size else 0.0
    color_mean = float(color_diff.mean()) if color_diff.size else 0.0

    preferred = (
        color_max <= PREFERRED_COLOR_MAX_ABS_DIFF
        and full_max <= PREFERRED_FULL_MAX_ABS_DIFF
        and full_mean <= PREFERRED_FULL_MEAN_ABS_DIFF
    )
    acceptable = (
        color_max <= ACCEPTABLE_COLOR_MAX_ABS_DIFF
        and full_max <= ACCEPTABLE_FULL_MAX_ABS_DIFF
        and full_mean <= ACCEPTABLE_FULL_MEAN_ABS_DIFF
    )
    hard_color_fail = color_max > COLOR_HARD_FAIL_ABS_DIFF
    status = "pass_preferred" if preferred else "pass_acceptable" if acceptable else "fail"

    return {
        "name": name,
        "status": status,
        "pass_preferred": bool(preferred),
        "pass_acceptable": bool(acceptable),
        "hard_color_fail": bool(hard_color_fail),
        "full_max_abs_diff": full_max,
        "full_mean_abs_diff": full_mean,
        "mos_max_abs_diff": mos_max,
        "mos_mean_abs_diff": mos_mean,
        "color_max_abs_diff": color_max,
        "color_mean_abs_diff": color_mean,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
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


def inspect_tflite_model(tflite_path: Path) -> dict[str, Any]:
    name_by_code = tflite_builtin_name_map()
    data = tflite_path.read_bytes()
    model = tflite_schema.Model.GetRootAsModel(data, 0)

    builtin_operator_codes = []
    custom_operator_codes = []
    for index in range(model.OperatorCodesLength()):
        op_code = model.OperatorCodes(index)
        custom_code = op_code.CustomCode()
        if custom_code:
            custom_operator_codes.append(custom_code.decode("utf-8"))
        else:
            builtin_operator_codes.append(name_by_code.get(op_code.BuiltinCode(), str(op_code.BuiltinCode())))

    ref_interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    )
    ops = [item.get("op_name", "") for item in ref_interpreter._get_ops_details()]

    return {
        "builtin_operator_codes": sorted(set(builtin_operator_codes)),
        "custom_operator_codes": sorted(set(custom_operator_codes)),
        "flex_or_select_tf_ops_found": any(code.startswith("Flex") for code in custom_operator_codes),
        "op_count_builtin_ref": len(ops),
        "op_counts_builtin_ref": dict(sorted(Counter(ops).items())),
    }


def build_fp16_converter(sm_model: Any, infer: Any, use_select_tf_ops: bool) -> tf.lite.TFLiteConverter:
    @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32, name="image")])
    def model_func(image: tf.Tensor) -> dict[str, tf.Tensor]:
        return infer(image)

    concrete_func = model_func.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], sm_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    if use_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter


def error_requires_select_tf_ops(error_text: str) -> bool:
    markers = [
        "SELECT_TF_OPS",
        "Select TF",
        "Flex",
        "not a builtin",
        "neither a custom op nor a flex op",
        "failed to legalize operation",
    ]
    return any(marker in error_text for marker in markers)


def write_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case",
        "path",
        "sm_mos",
        "sm_color",
        "fp32_mos",
        "fp32_color",
        "fp16_mos",
        "fp16_color",
        "sm_vs_fp16_mos_abs_diff",
        "sm_vs_fp16_color_abs_diff",
        "fp32_vs_fp16_mos_abs_diff",
        "fp32_vs_fp16_color_abs_diff",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_prediction_rows(
    rows: list[dict[str, Any]],
    case: str,
    paths: list[str],
    sm_out: np.ndarray,
    fp32_out: np.ndarray,
    fp16_out: np.ndarray,
) -> None:
    for index in range(fp16_out.shape[0]):
        rows.append(
            {
                "case": case,
                "path": paths[index] if index < len(paths) else "",
                "sm_mos": float(sm_out[index, 0]),
                "sm_color": float(sm_out[index, 1]),
                "fp32_mos": float(fp32_out[index, 0]),
                "fp32_color": float(fp32_out[index, 1]),
                "fp16_mos": float(fp16_out[index, 0]),
                "fp16_color": float(fp16_out[index, 1]),
                "sm_vs_fp16_mos_abs_diff": float(abs(sm_out[index, 0] - fp16_out[index, 0])),
                "sm_vs_fp16_color_abs_diff": float(abs(sm_out[index, 1] - fp16_out[index, 1])),
                "fp32_vs_fp16_mos_abs_diff": float(abs(fp32_out[index, 0] - fp16_out[index, 0])),
                "fp32_vs_fp16_color_abs_diff": float(abs(fp32_out[index, 1] - fp16_out[index, 1])),
            }
        )


def format_result_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Test Case | Status | Full Max | Full Mean | MOS Max | MOS Mean | Color Max | Color Mean |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            "| {name} | `{status}` | {full_max_abs_diff:.9g} | {full_mean_abs_diff:.9g} | "
            "{mos_max_abs_diff:.9g} | {mos_mean_abs_diff:.9g} | "
            "{color_max_abs_diff:.9g} | {color_mean_abs_diff:.9g} |".format(**item)
        )
    return "\n".join(lines)


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}%"
    return "n/a"


def write_markdown_report(root: Path, report: dict[str, Any]) -> None:
    report_path = root / REPORT_REL_PATH
    sensitivity = report.get("input_sensitivity", {})
    inspection = report.get("tflite_inspection", {})
    savedmodel_results = report.get("savedmodel_vs_fp16_results", [])
    fp32_results = report.get("fp32_vs_fp16_results", [])
    unresolved = report.get("unresolved_issues", [])
    unresolved_text = "\n".join(f"- {item}" for item in unresolved) if unresolved else "- None."

    lines = [
        "# Stage K: FP16 TFLite Conversion and Verification",
        "",
        "## Summary",
        f"- **Status**: `{report.get('status')}`.",
        f"- **Overall pass**: {report.get('overall_pass')}.",
        f"- **Exact command run**: `{COMMAND_USED}`.",
        f"- **SavedModel path**: `{report.get('saved_model_path')}`.",
        f"- **FP32 TFLite reference path**: `{report.get('fp32_reference_tflite_path')}`.",
        f"- **FP16 TFLite path**: `{report.get('tflite_path')}`.",
        "",
        "## Size",
        f"- **FP32 file size**: {report.get('fp32_tflite_size_bytes')} bytes.",
        f"- **FP16 file size**: {report.get('fp16_tflite_size_bytes')} bytes.",
        f"- **Size reduction**: {format_percent(report.get('size_reduction_percent'))}.",
        "",
        "## Conversion",
        f"- **Builtin FP16 conversion succeeded**: {report.get('builtin_success')}.",
        f"- **SELECT_TF_OPS / Flex required**: {report.get('select_tf_ops_required')}.",
        f"- **Custom ops found**: {bool(inspection.get('custom_operator_codes'))}.",
        f"- **Flex/SELECT_TF_OPS operator codes found**: {inspection.get('flex_or_select_tf_ops_found')}.",
        "- **INT8 / representative dataset / full integer quantization**: Not used.",
        "",
        "## Interpreter Details",
        f"- **Input details**: `{report.get('input_details')}`.",
        f"- **Output details**: `{report.get('output_details')}`.",
        f"- **Builtin op count**: {inspection.get('op_count_builtin_ref')}.",
        f"- **Builtin operator types**: `{inspection.get('builtin_operator_codes')}`.",
        f"- **Custom operator codes**: `{inspection.get('custom_operator_codes')}`.",
        "",
        "## Input Sensitivity",
        f"- **Zero vs One max diff**: {sensitivity.get('zero_vs_one_max_diff')}.",
        f"- **Zero vs Random max diff**: {sensitivity.get('zero_vs_random_max_diff')}.",
        f"- **Random vs Real max diff**: {sensitivity.get('random_vs_real_max_diff')}.",
        f"- **Input-sensitive**: {sensitivity.get('input_sensitive')}.",
        "",
        "## SavedModel vs FP16 TFLite Parity",
        format_result_table(savedmodel_results),
        "",
        "## FP32 TFLite vs FP16 TFLite Parity",
        format_result_table(fp32_results),
        "",
        "## Real Image Counts",
        f"- **Required real-image count**: {report.get('real_image_count')}.",
        f"- **Optional real-image count**: {report.get('optional_real_image_count')}.",
        "",
        "## Unresolved Issues",
        unresolved_text,
        "",
        "## Recommendation",
        f"- **Safe to proceed to Android/Flutter smoke testing**: {report.get('safe_to_plan_android_flutter_smoke')}.",
        f"- **Safe to proceed to INT8 exploration later**: {report.get('safe_to_explore_int8_later')}.",
        "- This report does not claim mobile deployment success.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    root = project_root()
    saved_model_path = root / SAVED_MODEL_REL_PATH
    fp32_tflite_path = root / FP32_TFLITE_REL_PATH
    require_path(saved_model_path, "SavedModel")
    require_path(fp32_tflite_path, "FP32 TFLite reference")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"{OUTPUT_PREFIX}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log(f"Output directory: {output_dir}")
    log(f"SavedModel path: {saved_model_path}")
    log(f"FP32 TFLite reference: {fp32_tflite_path}")

    report: dict[str, Any] = {
        "status": "failed",
        "overall_pass": False,
        "command": COMMAND_USED,
        "saved_model_path": str(saved_model_path),
        "fp32_reference_tflite_path": str(fp32_tflite_path),
        "output_dir": str(output_dir),
        "log_lines": log_lines,
        "unresolved_issues": [],
    }
    prediction_rows: list[dict[str, Any]] = []
    sensitivity_report: dict[str, Any] = {}
    fp32_vs_fp16_comparison: dict[str, Any] = {}

    try:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        log("Loading SavedModel")
        sm_model = tf.saved_model.load(str(saved_model_path))
        infer = sm_model.signatures["serving_default"]
        log(f"SavedModel serving keys: inputs={list(infer.structured_input_signature)}, outputs={list(infer.structured_outputs.keys())}")

        log("Converting SavedModel to FP16 TFLite with fixed shape [1, 224, 224, 3]")
        builtin_success = False
        select_tf_ops_required = False
        conversion_errors: list[str] = []
        tflite_model: bytes | None = None
        try:
            converter = build_fp16_converter(sm_model, infer, use_select_tf_ops=False)
            tflite_model = converter.convert()
            builtin_success = True
            log("Builtin FP16 TFLite conversion succeeded.")
        except Exception as exc:
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            conversion_errors.append(error_text)
            log(f"Builtin FP16 TFLite conversion failed: {error_text}")
            if error_requires_select_tf_ops(error_text):
                log("Error appears to require SELECT_TF_OPS; retrying with explicit fallback.")
                converter = build_fp16_converter(sm_model, infer, use_select_tf_ops=True)
                tflite_model = converter.convert()
                select_tf_ops_required = True
                log("FP16 TFLite conversion with SELECT_TF_OPS succeeded.")
            else:
                report["conversion_errors"] = conversion_errors
                report["unresolved_issues"].append("Builtin FP16 conversion failed and the error did not clearly require SELECT_TF_OPS fallback.")
                return report, prediction_rows, sensitivity_report, fp32_vs_fp16_comparison

        tflite_filename = "icaa_dat_tf_native_fp16_select_tf_ops.tflite" if select_tf_ops_required else "icaa_dat_tf_native_fp16.tflite"
        tflite_path = output_dir / tflite_filename
        tflite_path.write_bytes(tflite_model)
        fp32_size = fp32_tflite_path.stat().st_size
        fp16_size = tflite_path.stat().st_size
        size_reduction = (1.0 - (fp16_size / fp32_size)) * 100.0
        log(f"FP16 TFLite model saved to {tflite_path}")
        log(f"FP32 size: {fp32_size} bytes; FP16 size: {fp16_size} bytes; reduction: {size_reduction:.2f}%")

        log("Creating FP16 TFLite Interpreter")
        fp16_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter_creation_success = True
        log("Allocating FP16 TFLite tensors")
        fp16_interpreter.allocate_tensors()
        allocate_success = True
        input_details = fp16_interpreter.get_input_details()
        output_details = fp16_interpreter.get_output_details()
        input_summary = detail_summary(input_details)
        output_summary = detail_summary(output_details)
        log(f"FP16 input details: {input_summary}")
        log(f"FP16 output details: {output_summary}")

        zero_input = np.zeros(INPUT_SHAPE, dtype=np.float32)
        one_input = np.ones(INPUT_SHAPE, dtype=np.float32)
        random_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        real_inputs_all, real_paths_all = load_real_images(root, OPTIONAL_REAL_IMAGE_COUNT)
        if real_inputs_all is None or len(real_paths_all) < REAL_IMAGE_COUNT:
            raise RuntimeError(f"Expected at least {REAL_IMAGE_COUNT} real ICAA17K images, found {len(real_paths_all)}")
        real16 = real_inputs_all[:REAL_IMAGE_COUNT]
        real16_paths = real_paths_all[:REAL_IMAGE_COUNT]
        real1 = real_inputs_all[:1]

        log("Running FP16 invoke smoke and input sensitivity probe")
        zero_out = run_tflite_inference(fp16_interpreter, zero_input)
        invoke_success = True
        one_out = run_tflite_inference(fp16_interpreter, one_input)
        random_fp16_out = run_tflite_inference(fp16_interpreter, random_input)
        real1_fp16_out = run_tflite_inference(fp16_interpreter, real1)
        zero_vs_one = float(np.abs(zero_out - one_out).max())
        zero_vs_random = float(np.abs(zero_out - random_fp16_out).max())
        random_vs_real = float(np.abs(random_fp16_out - real1_fp16_out).max())
        input_sensitive = (
            zero_vs_one > SENSITIVITY_EPS
            and zero_vs_random > SENSITIVITY_EPS
            and random_vs_real > SENSITIVITY_EPS
        )
        sensitivity_report = {
            "zero_output": zero_out.tolist(),
            "one_output": one_out.tolist(),
            "random_output": random_fp16_out.tolist(),
            "real_output": real1_fp16_out.tolist(),
            "real_path": real_paths_all[0],
            "zero_vs_one_max_diff": zero_vs_one,
            "zero_vs_random_max_diff": zero_vs_random,
            "random_vs_real_max_diff": random_vs_real,
            "sensitivity_epsilon": SENSITIVITY_EPS,
            "input_sensitive": bool(input_sensitive),
        }
        log(f"Input sensitivity zero_vs_one={zero_vs_one:.9g}, zero_vs_random={zero_vs_random:.9g}, random_vs_real={random_vs_real:.9g}")

        log("Creating FP32 reference TFLite Interpreter")
        fp32_interpreter = tf.lite.Interpreter(model_path=str(fp32_tflite_path))
        fp32_interpreter.allocate_tensors()

        savedmodel_vs_fp16_results = []
        fp32_vs_fp16_results = []

        def evaluate_case(name: str, input_data: np.ndarray, paths: list[str]) -> bool:
            sm_out = savedmodel_scores(infer, input_data)
            fp32_out = run_tflite_inference(fp32_interpreter, input_data)
            fp16_out = run_tflite_inference(fp16_interpreter, input_data)
            sm_cmp = compare_outputs(name, sm_out, fp16_out)
            fp32_cmp = compare_outputs(name, fp32_out, fp16_out)
            savedmodel_vs_fp16_results.append(sm_cmp)
            fp32_vs_fp16_results.append(fp32_cmp)
            add_prediction_rows(prediction_rows, name, paths, sm_out, fp32_out, fp16_out)
            log(
                f"{name}: SavedModel-vs-FP16 {sm_cmp['status']} "
                f"full_max={sm_cmp['full_max_abs_diff']:.9g}; "
                f"FP32-vs-FP16 {fp32_cmp['status']} full_max={fp32_cmp['full_max_abs_diff']:.9g}"
            )
            return bool(sm_cmp["pass_acceptable"] and fp32_cmp["pass_acceptable"])

        log("Running parity checks")
        random_ok = evaluate_case("random_normalized_input", random_input, [""])
        real16_ok = evaluate_case(f"{REAL_IMAGE_COUNT}_real_images", real16, real16_paths)

        optional_real_count = 0
        optional_ok = True
        if random_ok and real16_ok and len(real_paths_all) >= OPTIONAL_REAL_IMAGE_COUNT:
            log(f"Running optional {OPTIONAL_REAL_IMAGE_COUNT}-real-image parity check")
            optional_real_count = OPTIONAL_REAL_IMAGE_COUNT
            optional_ok = evaluate_case(
                f"{OPTIONAL_REAL_IMAGE_COUNT}_real_images",
                real_inputs_all[:OPTIONAL_REAL_IMAGE_COUNT],
                real_paths_all[:OPTIONAL_REAL_IMAGE_COUNT],
            )

        inspection = inspect_tflite_model(tflite_path)
        input_shape_ok = input_summary[0]["shape"] == INPUT_SHAPE
        output_shape_ok = output_summary[0]["shape"] == OUTPUT_SHAPE
        input_dtype_compatible = input_summary[0]["dtype"] in {"float32", "float16"}
        output_readable = output_summary[0]["dtype"] in {"float32", "float16"}
        preferred_external_io = input_summary[0]["dtype"] == "float32" and output_summary[0]["dtype"] == "float32"
        no_custom_or_flex = (
            not select_tf_ops_required
            and not inspection["custom_operator_codes"]
            and not inspection["flex_or_select_tf_ops_found"]
        )
        all_sm_parity = all(item["pass_acceptable"] for item in savedmodel_vs_fp16_results)
        all_fp32_parity = all(item["pass_acceptable"] for item in fp32_vs_fp16_results)
        hard_color_fail = any(item["hard_color_fail"] for item in savedmodel_vs_fp16_results + fp32_vs_fp16_results)
        overall_pass = bool(
            builtin_success
            and interpreter_creation_success
            and allocate_success
            and invoke_success
            and input_sensitive
            and input_shape_ok
            and output_shape_ok
            and input_dtype_compatible
            and output_readable
            and all_sm_parity
            and all_fp32_parity
            and optional_ok
            and no_custom_or_flex
            and not hard_color_fail
        )

        unresolved_issues: list[str] = []
        if not input_sensitive:
            unresolved_issues.append("FP16 TFLite outputs are effectively constant under the input sensitivity probe.")
        if not preferred_external_io:
            unresolved_issues.append("External input/output dtype is not the preferred float32-to-float32 contract.")
        if not no_custom_or_flex:
            unresolved_issues.append("SELECT_TF_OPS, Flex, or custom operators were required or found.")
        if not all_sm_parity:
            unresolved_issues.append("SavedModel-vs-FP16 parity exceeded acceptable thresholds.")
        if not all_fp32_parity:
            unresolved_issues.append("FP32-vs-FP16 parity exceeded acceptable thresholds.")
        if hard_color_fail:
            unresolved_issues.append("Color max_abs_diff exceeded the hard 1e-2 failure threshold.")

        fp32_vs_fp16_comparison = {
            "fp32_reference_tflite_path": str(fp32_tflite_path),
            "fp16_tflite_path": str(tflite_path),
            "results": fp32_vs_fp16_results,
        }

        report.update(
            {
                "status": "ok" if overall_pass else "failed",
                "overall_pass": overall_pass,
                "tflite_path": str(tflite_path),
                "tflite_filename": tflite_filename,
                "fp32_tflite_size_bytes": fp32_size,
                "fp16_tflite_size_bytes": fp16_size,
                "size_reduction_percent": size_reduction,
                "builtin_success": builtin_success,
                "select_tf_ops_required": select_tf_ops_required,
                "conversion_errors": conversion_errors,
                "interpreter_creation_success": interpreter_creation_success,
                "allocate_tensors_success": allocate_success,
                "invoke_success": invoke_success,
                "input_details": input_summary,
                "output_details": output_summary,
                "input_shape_ok": input_shape_ok,
                "output_shape_ok": output_shape_ok,
                "input_dtype_compatible": input_dtype_compatible,
                "output_readable": output_readable,
                "preferred_external_io": preferred_external_io,
                "input_sensitivity": sensitivity_report,
                "savedmodel_vs_fp16_results": savedmodel_vs_fp16_results,
                "fp32_vs_fp16_results": fp32_vs_fp16_results,
                "real_image_count": REAL_IMAGE_COUNT,
                "optional_real_image_count": optional_real_count,
                "tflite_inspection": inspection,
                "no_custom_or_flex": no_custom_or_flex,
                "hard_color_fail": hard_color_fail,
                "unresolved_issues": unresolved_issues,
                "safe_to_plan_android_flutter_smoke": overall_pass,
                "safe_to_explore_int8_later": overall_pass,
            }
        )
        return report, prediction_rows, sensitivity_report, fp32_vs_fp16_comparison

    except Exception as exc:
        traceback.print_exc()
        report["errors"] = [str(exc)] + traceback.format_exc().splitlines()
        report["unresolved_issues"] = report.get("unresolved_issues", []) + [str(exc)]
        return report, prediction_rows, sensitivity_report, fp32_vs_fp16_comparison


def main() -> int:
    report, prediction_rows, sensitivity_report, fp32_vs_fp16_comparison = run()
    output_dir = Path(report["output_dir"])

    (output_dir / "stage_k_tflite_fp16_report.json").write_text(
        json.dumps(jsonable(report), indent=2),
        encoding="utf-8",
    )
    (output_dir / "stage_k_tflite_fp16_log.txt").write_text(
        "\n".join(report.get("log_lines", [])),
        encoding="utf-8",
    )
    if sensitivity_report:
        (output_dir / "tflite_fp16_input_sensitivity_report.json").write_text(
            json.dumps(jsonable(sensitivity_report), indent=2),
            encoding="utf-8",
        )
    if fp32_vs_fp16_comparison:
        (output_dir / "optional_fp32_vs_fp16_comparison.json").write_text(
            json.dumps(jsonable(fp32_vs_fp16_comparison), indent=2),
            encoding="utf-8",
        )
    if prediction_rows:
        write_predictions_csv(output_dir / "stage_k_tflite_fp16_predictions.csv", prediction_rows)

    write_markdown_report(project_root(), report)
    if report["status"] == "ok":
        print("Stage K completed successfully.")
        return 0
    print(f"Stage K failed with status: {report['status']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
