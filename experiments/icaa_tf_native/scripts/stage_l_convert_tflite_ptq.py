# ICAA17K DAT SavedModel을 PTQ TFLite 후보로 변환하고 검증하는 스크립트
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import csv
import json
import os
import sys
import traceback
from typing import Any, Callable

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as tflite_schema

sys.dont_write_bytecode = True

COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_l_convert_tflite_ptq.py"
OUTPUT_DIR_REL_PATH = "outputs/icaa_ptq_candidates_20260601"
SAVED_MODEL_REL_PATH = "outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model"
FP16_REFERENCE_REL_PATH = "outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"

DYNAMIC_RANGE_FILENAME = "icaa_dat_tf_native_dynamic_range_ptq.tflite"
FULL_INT8_FLOAT_IO_FILENAME = "icaa_dat_tf_native_full_integer_int8_float_io.tflite"
FULL_INT8_INT_IO_FALLBACK_FILENAME = "icaa_dat_tf_native_full_integer_int8_int_io_fallback.tflite"

INPUT_SHAPE = [1, 224, 224, 3]
OUTPUT_SHAPE = [1, 2]
SEED = 123
REPRESENTATIVE_IMAGE_COUNT = 64
SMOKE_IMAGE_COUNT = 16

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

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
    if isinstance(value, Path):
        return str(value)
    return value


def dtype_name(value: Any) -> str:
    try:
        return np.dtype(value).name
    except TypeError:
        return str(value)


def detail_summary(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for item in details:
        summary.append(
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
    return summary


def preprocess_image(path: Path) -> np.ndarray:
    from PIL import Image

    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - IMAGENET_MEAN) / IMAGENET_STD


def append_unique(paths: list[Path], seen: set[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in seen or not path.is_file():
        return
    seen.add(resolved)
    paths.append(path)


def find_icaa_images(root: Path, limit: int) -> list[Path]:
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


def representative_dataset(paths: list[Path]) -> Callable[[], Any]:
    def generator() -> Any:
        for path in paths:
            sample = preprocess_image(path)[np.newaxis, ...].astype(np.float32)
            yield [sample]

    return generator


def build_converter(sm_model: Any, concrete_func: Any) -> tf.lite.TFLiteConverter:
    return tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], sm_model)


def tflite_builtin_name_map() -> dict[int, str]:
    mapping: dict[int, str] = {}
    for name in dir(tflite_schema.BuiltinOperator):
        if name.startswith("_") or name in {"Name", "Value"}:
            continue
        value = getattr(tflite_schema.BuiltinOperator, name)
        if isinstance(value, int):
            mapping[value] = name
    return mapping


def load_interpreter(path: Path) -> tuple[tf.lite.Interpreter, str | None]:
    try:
        interpreter = tf.lite.Interpreter(
            model_path=str(path),
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        )
        return interpreter, None
    except Exception as exc:
        warning = f"BUILTIN_REF interpreter failed, retried default interpreter: {exc}"
        interpreter = tf.lite.Interpreter(model_path=str(path))
        return interpreter, warning


def inspect_tflite_model(path: Path) -> dict[str, Any]:
    name_by_code = tflite_builtin_name_map()
    data = path.read_bytes()
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

    interpreter_warning = None
    allocate_error = None
    input_summary: list[dict[str, Any]] = []
    output_summary: list[dict[str, Any]] = []
    op_counts: dict[str, int] = {}
    tensor_dtype_counts: dict[str, int] = {}
    internal_float32_tensor_count = None
    try:
        interpreter, interpreter_warning = load_interpreter(path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_summary = detail_summary(input_details)
        output_summary = detail_summary(output_details)
        ops = [item.get("op_name", "") for item in interpreter._get_ops_details()]
        op_counts = dict(sorted(Counter(ops).items()))

        tensor_details = interpreter.get_tensor_details()
        tensor_dtype_counts = dict(sorted(Counter(dtype_name(item.get("dtype")) for item in tensor_details).items()))
        boundary_indices = {int(item["index"]) for item in input_details + output_details}
        internal_float32_tensor_count = sum(
            1
            for item in tensor_details
            if dtype_name(item.get("dtype")) == "float32" and int(item.get("index")) not in boundary_indices
        )
    except Exception as exc:
        allocate_error = "".join(traceback.format_exception_only(type(exc), exc)).strip()

    flex_or_select = any(code.startswith("Flex") for code in custom_operator_codes)
    builtins_only = not custom_operator_codes and not flex_or_select
    input_shape = input_summary[0]["shape"] if input_summary else []
    output_shape = output_summary[0]["shape"] if output_summary else []
    input_dtype = input_summary[0]["dtype"] if input_summary else ""
    output_dtype = output_summary[0]["dtype"] if output_summary else ""

    return {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "input_details": input_summary,
        "output_details": output_summary,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "input_dtype": input_dtype,
        "output_dtype": output_dtype,
        "input_shape_ok": input_shape == INPUT_SHAPE,
        "output_shape_ok": output_shape == OUTPUT_SHAPE,
        "input_float32": input_dtype == "float32",
        "output_float32": output_dtype == "float32",
        "builtin_operator_codes": sorted(set(builtin_operator_codes)),
        "custom_operator_codes": sorted(set(custom_operator_codes)),
        "flex_or_select_tf_ops_found": flex_or_select,
        "tflite_builtins_only": builtins_only,
        "op_count": sum(op_counts.values()),
        "op_counts": op_counts,
        "tensor_dtype_counts": tensor_dtype_counts,
        "internal_float32_tensor_count": internal_float32_tensor_count,
        "interpreter_warning": interpreter_warning,
        "interpreter_or_allocate_error": allocate_error,
    }


def write_candidate(path: Path, model_bytes: bytes) -> bool:
    if path.exists():
        log(f"Preserving existing candidate, not overwriting: {path}")
        return False
    path.write_bytes(model_bytes)
    return True


def convert_dynamic_range(sm_model: Any, concrete_func: Any) -> bytes:
    converter = build_converter(sm_model, concrete_func)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()


def convert_full_integer_float_io(sm_model: Any, concrete_func: Any, rep_paths: list[Path]) -> bytes:
    converter = build_converter(sm_model, concrete_func)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(rep_paths)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    return converter.convert()


def convert_full_integer_int_io_fallback(sm_model: Any, concrete_func: Any, rep_paths: list[Path]) -> bytes:
    converter = build_converter(sm_model, concrete_func)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(rep_paths)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def attempt_conversion(
    name: str,
    path: Path,
    convert_func: Callable[[], bytes],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "filename": path.name,
        "status": "failed",
        "started_at": started_at,
        "notes": notes or [],
    }
    log(f"Converting {name}")
    try:
        model_bytes = convert_func()
        write_performed = write_candidate(path, model_bytes)
        validation = inspect_tflite_model(path)
        result.update(
            {
                "status": "succeeded",
                "write_performed": write_performed,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "validation": validation,
            }
        )
        log(
            f"{name} succeeded: size={validation.get('size_bytes')} "
            f"input={validation.get('input_dtype')} {validation.get('input_shape')} "
            f"output={validation.get('output_dtype')} {validation.get('output_shape')}"
        )
    except Exception as exc:
        error_text = traceback.format_exc()
        result.update(
            {
                "status": "failed",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
                "traceback": error_text.splitlines(),
            }
        )
        log(f"{name} failed: {exc}")
    return result


def format_bool(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def format_op_counts(op_counts: dict[str, int], limit: int = 24) -> str:
    if not op_counts:
        return "n/a"
    items = sorted(op_counts.items(), key=lambda item: (-item[1], item[0]))
    text = ", ".join(f"{name}={count}" for name, count in items[:limit])
    if len(items) > limit:
        text += f", ... (+{len(items) - limit} more)"
    return text


def write_inventory(output_dir: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model_name",
        "status",
        "filename",
        "path",
        "exists",
        "size_bytes",
        "input_dtype",
        "output_dtype",
        "input_shape",
        "output_shape",
        "input_float32",
        "output_float32",
        "flex_ops_exist",
        "tflite_builtins_only",
        "op_count",
        "internal_float32_tensor_count",
        "tensor_dtype_counts",
        "op_counts",
        "custom_operator_codes",
        "notes",
        "error",
    ]
    rows = []
    for result in results:
        validation = result.get("validation", {})
        rows.append(
            {
                "model_name": result.get("name", ""),
                "status": result.get("status", ""),
                "filename": result.get("filename", ""),
                "path": result.get("path", ""),
                "exists": validation.get("exists", Path(result.get("path", "")).is_file()),
                "size_bytes": validation.get("size_bytes", ""),
                "input_dtype": validation.get("input_dtype", ""),
                "output_dtype": validation.get("output_dtype", ""),
                "input_shape": json.dumps(validation.get("input_shape", [])),
                "output_shape": json.dumps(validation.get("output_shape", [])),
                "input_float32": validation.get("input_float32", ""),
                "output_float32": validation.get("output_float32", ""),
                "flex_ops_exist": validation.get("flex_or_select_tf_ops_found", ""),
                "tflite_builtins_only": validation.get("tflite_builtins_only", ""),
                "op_count": validation.get("op_count", ""),
                "internal_float32_tensor_count": validation.get("internal_float32_tensor_count", ""),
                "tensor_dtype_counts": json.dumps(validation.get("tensor_dtype_counts", {}), sort_keys=True),
                "op_counts": json.dumps(validation.get("op_counts", {}), sort_keys=True),
                "custom_operator_codes": json.dumps(validation.get("custom_operator_codes", [])),
                "notes": " | ".join(result.get("notes", [])),
                "error": result.get("error", ""),
            }
        )

    with (output_dir / "model_inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validation_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Status | Size Bytes | Input | Output | Shape OK | Flex Ops | Builtins Only | Internal Float32 Tensors |",
        "| :--- | :--- | ---: | :--- | :--- | :--- | :--- | :--- | ---: |",
    ]
    for result in results:
        validation = result.get("validation", {})
        input_text = f"{validation.get('input_dtype', 'n/a')} {validation.get('input_shape', [])}"
        output_text = f"{validation.get('output_dtype', 'n/a')} {validation.get('output_shape', [])}"
        shape_ok = bool(validation.get("input_shape_ok") and validation.get("output_shape_ok"))
        lines.append(
            "| {name} | `{status}` | {size} | `{input}` | `{output}` | {shape_ok} | {flex} | {builtins} | {internal_float32} |".format(
                name=result.get("name", ""),
                status=result.get("status", ""),
                size=validation.get("size_bytes", ""),
                input=input_text,
                output=output_text,
                shape_ok=format_bool(shape_ok) if validation else "n/a",
                flex=format_bool(validation.get("flex_or_select_tf_ops_found")) if validation else "n/a",
                builtins=format_bool(validation.get("tflite_builtins_only")) if validation else "n/a",
                internal_float32=validation.get("internal_float32_tensor_count", ""),
            )
        )
    return "\n".join(lines)


def ops_summary_section(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for result in results:
        validation = result.get("validation", {})
        if not validation:
            continue
        lines.extend(
            [
                f"### {result.get('name')}",
                "",
                f"- **TFLITE_BUILTINS only**: {validation.get('tflite_builtins_only')}.",
                f"- **Flex ops exist**: {validation.get('flex_or_select_tf_ops_found')}.",
                f"- **Builtin operator codes**: `{validation.get('builtin_operator_codes')}`.",
                f"- **Custom operator codes**: `{validation.get('custom_operator_codes')}`.",
                f"- **Tensor dtype counts**: `{validation.get('tensor_dtype_counts')}`.",
                f"- **Ops summary**: `{format_op_counts(validation.get('op_counts', {}))}`.",
                "",
            ]
        )
    return lines


def write_conversion_report(output_dir: Path, report: dict[str, Any]) -> None:
    results = report.get("candidate_results", [])
    dynamic = next((item for item in results if item.get("name") == "dynamic_range_ptq"), {})
    full_int8 = next((item for item in results if item.get("name") == "full_integer_int8_float_io"), {})
    fallback = next((item for item in results if item.get("name") == "full_integer_int8_int_io_fallback"), {})
    full_validation = full_int8.get("validation", {})

    lines = [
        "# ICAA Stage L PTQ Conversion Report",
        "",
        "## Summary",
        f"- **Status**: `{report.get('status')}`.",
        f"- **Exact command run**: `{COMMAND_USED}`.",
        f"- **SavedModel path**: `{report.get('saved_model_path')}`.",
        f"- **Output directory**: `{output_dir}`.",
        f"- **Representative image count requested**: {REPRESENTATIVE_IMAGE_COUNT}.",
        f"- **Representative image count used**: {report.get('representative_image_count')}.",
        f"- **Preprocessing contract**: RGB resize to `224x224`, `pixel / 255.0`, then ImageNet mean/std.",
        "- **Speed claim**: none; no benchmark was run.",
        "",
        "## Conversion Outcomes",
        f"- **Dynamic range PTQ succeeded**: {dynamic.get('status') == 'succeeded'}.",
        f"- **Full integer INT8 float-IO PTQ succeeded**: {full_int8.get('status') == 'succeeded'}.",
        f"- **Full integer INT8 input/output stayed float32**: {bool(full_validation.get('input_float32') and full_validation.get('output_float32')) if full_validation else False}.",
        f"- **INT8 int-IO fallback produced**: {fallback.get('status') == 'succeeded'}.",
        "",
        "## Validation Inventory",
        validation_table(results),
        "",
        "## Ops Summary",
        "",
    ]
    lines.extend(ops_summary_section(results) or ["- No model ops summary was available.", ""])

    failed = [item for item in results if item.get("status") != "succeeded"]
    lines.extend(["## Conversion Errors", ""])
    if failed:
        for item in failed:
            lines.append(f"- **{item.get('name')}**: {item.get('error', 'unknown error')}.")
    else:
        lines.append("- None.")

    lines.extend(
        [
            "",
            "## Generated Smoke Script",
            f"- `{output_dir / 'smoke_compare_ptq.py'}`.",
            f"- It compares candidates against `{report.get('fp16_reference_path')}` when that FP16 reference exists.",
            "",
            "## Next Step",
            "- Run a device-side or representative desktop latency benchmark before making any speed or deployment claim.",
            "",
        ]
    )

    (output_dir / "conversion_report.md").write_text("\n".join(lines), encoding="utf-8")


def smoke_script_text() -> str:
    return r'''# ICAA PTQ 후보 TFLite 모델의 FP16 기준 대비 출력 차이를 확인하는 스크립트
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
'''


def write_smoke_script(output_dir: Path) -> None:
    path = output_dir / "smoke_compare_ptq.py"
    if path.exists():
        log(f"Preserving existing smoke script, not overwriting: {path}")
        return
    path.write_text(smoke_script_text(), encoding="utf-8")


def run() -> dict[str, Any]:
    root = project_root()
    output_dir = root / OUTPUT_DIR_REL_PATH
    saved_model_path = root / SAVED_MODEL_REL_PATH
    fp16_reference_path = root / FP16_REFERENCE_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH

    require_path(saved_model_path, "SavedModel")
    require_path(image_root, "ICAA17K image root")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_smoke_script(output_dir)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    rep_paths = find_icaa_images(root, REPRESENTATIVE_IMAGE_COUNT)
    report: dict[str, Any] = {
        "status": "failed",
        "command": COMMAND_USED,
        "saved_model_path": str(saved_model_path),
        "fp16_reference_path": str(fp16_reference_path),
        "output_dir": str(output_dir),
        "representative_image_count": len(rep_paths),
        "representative_image_paths": [str(path) for path in rep_paths],
        "input_shape": INPUT_SHAPE,
        "output_shape": OUTPUT_SHAPE,
        "preprocessing": {
            "resize": "RGB bilinear resize to 224x224",
            "normalization": "pixel / 255.0, then ImageNet mean/std",
            "mean": IMAGENET_MEAN.tolist(),
            "std": IMAGENET_STD.tolist(),
        },
        "candidate_results": [],
        "log_lines": log_lines,
    }

    log(f"Output directory: {output_dir}")
    log(f"SavedModel path: {saved_model_path}")
    log(f"Representative images found: {len(rep_paths)}")

    try:
        log("Loading SavedModel")
        sm_model = tf.saved_model.load(str(saved_model_path))
        infer = sm_model.signatures["serving_default"]

        @tf.function(input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32, name="image")])
        def model_func(image: tf.Tensor) -> dict[str, tf.Tensor]:
            return infer(image)

        concrete_func = model_func.get_concrete_function()

        dynamic_result = attempt_conversion(
            "dynamic_range_ptq",
            output_dir / DYNAMIC_RANGE_FILENAME,
            lambda: convert_dynamic_range(sm_model, concrete_func),
            notes=["converter.optimizations=[tf.lite.Optimize.DEFAULT]", "No representative dataset used."],
        )
        report["candidate_results"].append(dynamic_result)

        if not rep_paths:
            report["candidate_results"].append(
                {
                    "name": "full_integer_int8_float_io",
                    "path": str(output_dir / FULL_INT8_FLOAT_IO_FILENAME),
                    "filename": FULL_INT8_FLOAT_IO_FILENAME,
                    "status": "failed",
                    "error": "No representative ICAA17K images were available.",
                    "notes": ["Full integer PTQ requires a representative dataset."],
                }
            )
        else:
            full_int8_result = attempt_conversion(
                "full_integer_int8_float_io",
                output_dir / FULL_INT8_FLOAT_IO_FILENAME,
                lambda: convert_full_integer_float_io(sm_model, concrete_func, rep_paths),
                notes=[
                    "converter.optimizations=[tf.lite.Optimize.DEFAULT]",
                    "representative_dataset uses data/raw/icaa17k with unchanged preprocessing.",
                    "converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]",
                    "converter.inference_input_type=tf.float32",
                    "converter.inference_output_type=tf.float32",
                ],
            )
            report["candidate_results"].append(full_int8_result)

            if full_int8_result.get("status") != "succeeded":
                fallback_result = attempt_conversion(
                    "full_integer_int8_int_io_fallback",
                    output_dir / FULL_INT8_INT_IO_FALLBACK_FILENAME,
                    lambda: convert_full_integer_int_io_fallback(sm_model, concrete_func, rep_paths),
                    notes=[
                        "Fallback only: this does not preserve the current Flutter float32 IO contract.",
                        "Produced only because float32 input/output full-integer INT8 conversion failed.",
                    ],
                )
                report["candidate_results"].append(fallback_result)

        produced = [item for item in report["candidate_results"] if item.get("status") == "succeeded"]
        dynamic_ok = dynamic_result.get("status") == "succeeded"
        full_float_ok = any(
            item.get("name") == "full_integer_int8_float_io" and item.get("status") == "succeeded"
            for item in report["candidate_results"]
        )
        if dynamic_ok and full_float_ok:
            report["status"] = "ok"
        elif produced:
            report["status"] = "partial"
        else:
            report["status"] = "failed"
        return report
    except Exception as exc:
        traceback.print_exc()
        report["errors"] = [str(exc)] + traceback.format_exc().splitlines()
        report["status"] = "failed"
        return report


def main() -> int:
    report = run()
    output_dir = Path(report["output_dir"])
    write_inventory(output_dir, report.get("candidate_results", []))
    write_conversion_report(output_dir, report)
    (output_dir / "conversion_report.json").write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    (output_dir / "conversion_log.txt").write_text("\n".join(report.get("log_lines", [])) + "\n", encoding="utf-8")

    if report["status"] in {"ok", "partial"}:
        print(f"Stage L completed with status: {report['status']}")
        return 0
    print(f"Stage L failed with status: {report['status']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
