# ICAA17K DAT SoftHistogram 모듈의 PyTorch-TensorFlow 동등성을 검증하는 스크립트
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import csv
import importlib
import importlib.util
import io
import json
import os
import sys
import traceback
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
EXPORT_SAFE_MODEL_REL_DIR = "experiments/icaa_export_safe_v3/export_safe_models"
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_d_soft_histogram_parity.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_d"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_d_soft_histogram_parity.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
PREFERRED_MAX_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-4

SOFT_HISTOGRAM_ARGS = {
    "n_features": 36,
    "n_examples": 6,
    "num_bins": 6,
    "quantiles": False,
}

MODEL_KWARGS = {
    "img_size": 224,
    "patch_size": 4,
    "num_classes": 1,
    "expansion": 4,
    "dim_stem": 128,
    "dims": [128, 256, 512, 1024],
    "depths": [2, 2, 18, 2],
    "stage_spec": [
        ["L", "S"],
        ["L", "S"],
        ["L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D"],
        ["L", "D"],
    ],
    "heads": [4, 8, 16, 32],
    "window_sizes": [7, 7, 7, 7],
    "groups": [-1, -1, 4, 8],
    "use_pes": [False, False, True, True],
    "dwc_pes": [False, False, False, False],
    "strides": [-1, -1, 1, 1],
    "sr_ratios": [-1, -1, -1, -1],
    "offset_range_factor": [-1, -1, 2, 2],
    "no_offs": [False, False, False, False],
    "fixed_pes": [False, False, False, False],
    "use_dwc_mlps": [False, False, False, False],
    "use_conv_patches": False,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.5,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dirs(root: Path) -> None:
    for rel_path in [
        "experiments/icaa_tf_native/tf_models",
        "experiments/icaa_tf_native/scripts",
        "experiments/icaa_tf_native/reports",
    ]:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def import_export_safe_modules(package_dir: Path):
    package_name = "icaa_export_safe_v3_export_safe_models_stage_d"
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    dat_module = importlib.import_module(f"{package_name}.dat")
    soft_histogram_module = importlib.import_module(f"{package_name}.SoftHistogram")
    return dat_module.DAT, soft_histogram_module.SoftHistogram


def import_tf_soft_histogram(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.soft_histogram import TFSoftHistogram

    return TFSoftHistogram


def load_checkpoint(torch, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")


def is_tensor_like(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "numel") and hasattr(value, "dtype")


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, OrderedDict) and all(is_tensor_like(v) for v in checkpoint_obj.values()):
        return checkpoint_obj, "checkpoint object is an OrderedDict state_dict"
    if isinstance(checkpoint_obj, Mapping):
        for key in ["state_dict", "model", "model_state_dict"]:
            value = checkpoint_obj.get(key)
            if isinstance(value, Mapping) and all(is_tensor_like(v) for v in value.values()):
                return OrderedDict(value.items()), f"checkpoint object contains a {key!r} state_dict"
        if all(is_tensor_like(v) for v in checkpoint_obj.values()):
            return OrderedDict(checkpoint_obj.items()), "checkpoint object is a mapping of tensor state entries"
    raise RuntimeError(f"Checkpoint is not a recognized tensor state_dict: {type(checkpoint_obj)!r}")


def tensor_np(tensor):
    return tensor.detach().cpu().numpy()


def histogram_state_dict(state_dict):
    prefix = "hist_feature."
    return OrderedDict((name[len(prefix):], value) for name, value in state_dict.items() if name.startswith(prefix))


def shape_of(value) -> list[int]:
    return [int(dim) for dim in tuple(value.shape)]


def inspect_pytorch_soft_histogram(module) -> dict:
    try:
        named_parameters = list(module.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError:
        named_parameters = list(module.named_parameters(recurse=True))

    seen_ids = {}
    parameter_rows = []
    for name, parameter in named_parameters:
        parameter_id = id(parameter)
        alias_of = seen_ids.get(parameter_id)
        if alias_of is None:
            seen_ids[parameter_id] = name
        parameter_rows.append(
            {
                "name": name,
                "shape": shape_of(parameter),
                "requires_grad": bool(parameter.requires_grad),
                "alias_of": alias_of,
            }
        )

    state_rows = []
    for name, value in module.state_dict().items():
        state_rows.append(
            {
                "name": name,
                "shape": shape_of(value),
                "dtype": str(value.dtype),
            }
        )

    return {
        "class_name": module.__class__.__name__,
        "constructor_args": SOFT_HISTOGRAM_ARGS,
        "has_trainable_parameters": any(row["requires_grad"] for row in parameter_rows),
        "parameters": parameter_rows,
        "state_dict_entries": state_rows,
        "buffers": [
            {"name": name, "shape": shape_of(buffer), "dtype": str(buffer.dtype)}
            for name, buffer in module.named_buffers(recurse=True)
        ],
        "aliases": {
            "centers_is_bin_centers_conv_bias": module.centers is module.bin_centers_conv.bias,
            "widths_is_bin_widths_conv_weight": module.widths is module.bin_widths_conv.weight,
        },
        "conv1d": {
            "bin_centers_conv": {
                "in_channels": module.bin_centers_conv.in_channels,
                "out_channels": module.bin_centers_conv.out_channels,
                "kernel_size": list(module.bin_centers_conv.kernel_size),
                "groups": module.bin_centers_conv.groups,
                "bias": module.bin_centers_conv.bias is not None,
            },
            "bin_widths_conv": {
                "in_channels": module.bin_widths_conv.in_channels,
                "out_channels": module.bin_widths_conv.out_channels,
                "kernel_size": list(module.bin_widths_conv.kernel_size),
                "groups": module.bin_widths_conv.groups,
                "bias": module.bin_widths_conv.bias is not None,
            },
        },
        "activation_math": [
            "input.transpose(0, 1).unsqueeze(0)",
            "grouped Conv1d feature-to-bin expansion",
            "torch.abs",
            "grouped Conv1d per-bin width scaling",
            "torch.nn.functional.relu",
        ],
    }


def compare_outputs(name: str, torch_output, tf_output) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_output)
    tf_np = tf_output.numpy()
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    preferred_pass = max_abs_diff <= PREFERRED_MAX_ABS_DIFF
    acceptable_pass = max_abs_diff <= ACCEPTABLE_MAX_ABS_DIFF
    status = "pass_preferred" if preferred_pass else "pass_acceptable" if acceptable_pass else "fail"
    return {
        "name": name,
        "status": status,
        "pass": bool(acceptable_pass),
        "preferred_pass": bool(preferred_pass),
        "acceptable_pass": bool(acceptable_pass),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "torch_output_shape": list(torch_np.shape),
        "tf_output_shape": list(tf_np.shape),
        "thresholds": {
            "preferred_max_abs_diff": PREFERRED_MAX_ABS_DIFF,
            "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_DIFF,
        },
        "likely_cause_if_failed": None if acceptable_pass else "Grouped Conv1d emulation or copied SoftHistogram state needs inspection.",
    }


def run_soft_histogram_case(torch, tf, soft_hist_pt, soft_hist_tf, name: str, input_np):
    x_pt = torch.from_numpy(input_np.astype("float32"))
    x_tf = tf.convert_to_tensor(input_np.astype("float32"), dtype=tf.float32)
    with torch.no_grad():
        y_pt = soft_hist_pt(x_pt)
    y_tf = soft_hist_tf(x_tf, training=False)
    return compare_outputs(name, y_pt, y_tf)


def preprocess_real_image(path: Path):
    import numpy as np
    from PIL import Image

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std


def first_real_image_input(root: Path):
    import numpy as np

    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    if not csv_path.is_file() or not image_root.is_dir():
        return None, None
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("ID")
            if not image_id:
                continue
            image_path = image_root / image_id
            if not image_path.is_file():
                continue
            image_nhwc = preprocess_real_image(image_path)
            image_nchw = np.transpose(image_nhwc, (2, 0, 1))[None, ...].astype(np.float32)
            return image_nchw, str(image_path)
    return None, None


def capture_soft_histogram_io(torch, model, image_nchw):
    captured = {}

    def hook(_module, inputs, output):
        captured["input"] = inputs[0].detach().cpu()
        captured["output"] = output.detach().cpu()

    handle = model.hist_feature.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(torch.from_numpy(image_nchw.astype("float32")))
    finally:
        handle.remove()
    if "input" not in captured:
        raise RuntimeError("SoftHistogram hook did not capture an input tensor.")
    return captured


def tensor_summary(name: str, tensor) -> dict:
    import numpy as np

    array = tensor_np(tensor) if hasattr(tensor, "detach") else np.asarray(tensor)
    return {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def result_line(result: dict) -> str:
    return (
        f"{result['name']}: status={result['status']} "
        f"max_abs_diff={result['max_abs_diff']:.9g} "
        f"mean_abs_diff={result['mean_abs_diff']:.9g} "
        f"torch_shape={result['torch_output_shape']} "
        f"tf_shape={result['tf_output_shape']}"
    )


def markdown_result_table(results: list[dict]) -> str:
    lines = [
        "| Test | Status | Max abs diff | Mean abs diff | PyTorch shape | TensorFlow shape |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['status']} | {result['max_abs_diff']:.9g} | "
            f"{result['mean_abs_diff']:.9g} | {result['torch_output_shape']} | {result['tf_output_shape']} |"
        )
    return "\n".join(lines)


def write_markdown_report(report: dict, report_path: Path) -> None:
    unresolved = report["unresolved_issues"] or ["none"]
    warning_lines = report["warnings"] or ["none"]
    error_lines = report["errors"] or ["none"]
    runtime_notices = [
        line for line in report.get("captured_output", [])
        if line.startswith(("WARNING:", "E", "W"))
    ] or ["none"]

    body = f"""# Stage D SoftHistogram Parity

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## PyTorch SoftHistogram Inspection
- Class: `{report['pytorch_soft_histogram']['class_name']}`
- Constructor arguments: `{json.dumps(report['pytorch_soft_histogram']['constructor_args'])}`
- Has trainable parameters: {report['pytorch_soft_histogram']['has_trainable_parameters']}
- Buffers: {report['pytorch_soft_histogram']['buffers']}
- `centers` aliases `bin_centers_conv.bias`: {report['pytorch_soft_histogram']['aliases']['centers_is_bin_centers_conv_bias']}
- `widths` aliases `bin_widths_conv.weight`: {report['pytorch_soft_histogram']['aliases']['widths_is_bin_widths_conv_weight']}
- Conv1D/groups: `{json.dumps(report['pytorch_soft_histogram']['conv1d'])}`

## Expected Shapes From DAT Path
- Dummy SoftHistogram input shape: {report['dummy_capture']['input']['shape']}
- Dummy SoftHistogram output shape: {report['dummy_capture']['output']['shape']}
- Real captured SoftHistogram input shape: {report['real_capture']['input']['shape'] if report.get('real_capture') else None}
- Real captured SoftHistogram output shape: {report['real_capture']['output']['shape'] if report.get('real_capture') else None}

## TensorFlow Implementation Summary
```json
{json.dumps(report['tf_soft_histogram_structure'], indent=2)}
```

## Weight and State Mapping
- Copied `hist_feature.bin_centers_conv.weight` to the TensorFlow feature-to-bin weight vector.
- Copied `hist_feature.bin_centers_conv.bias` to the TensorFlow feature-to-bin bias vector.
- Copied `hist_feature.bin_widths_conv.weight` to the TensorFlow per-bin width weight vector.
- Copied `hist_feature.bin_widths_conv.bias` to the TensorFlow per-bin width bias vector.
- Copied `hist_feature.centers` for state completeness, but PyTorch `forward` does not reference it.
- `hist_feature.widths` aliases `bin_widths_conv.weight` in PyTorch; the TensorFlow layer aliases the same variable.

## Parity Results
{markdown_result_table(report['results'])}

## Captured Real Tensor
- Performed: {report['real_capture_performed']}
- Image path: `{report.get('real_image_path')}`

## Unresolved Issues
""" + "\n".join(f"- {line}" for line in unresolved) + f"""

## Stage E Local Attention / Block Parity Decision
- Safe to proceed to Stage E local attention / block parity: {report['safe_to_proceed_to_stage_e_local_attention_block_parity']}
- Scope of this decision: SoftHistogram parity only. This does not establish full TensorFlow DAT feasibility.

## Warnings
""" + "\n".join(f"- {line}" for line in warning_lines) + """

## Runtime Log Notices
""" + "\n".join(f"- {line}" for line in runtime_notices) + """

## Errors
""" + "\n".join(f"- {line}" for line in error_lines) + "\n"

    report_path.write_text(body, encoding="utf-8")


def run() -> tuple[dict, dict]:
    root = project_root()
    ensure_dirs(root)
    checkpoint_path = root / CHECKPOINT_REL_PATH
    export_safe_model_dir = root / EXPORT_SAFE_MODEL_REL_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"{OUTPUT_PREFIX}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = root / REPORT_REL_PATH

    log_lines: list[str] = []
    warnings_seen: list[str] = []
    errors: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            import numpy as np
            import tensorflow as tf
            import torch

            torch.manual_seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            checkpoint_obj = load_checkpoint(torch, checkpoint_path)
            state_dict, state_dict_confirmation = extract_state_dict(checkpoint_obj)
            DAT, SoftHistogram = import_export_safe_modules(export_safe_model_dir)
            model = DAT(**MODEL_KWARGS)
            strict_load = model.load_state_dict(state_dict, strict=True)
            model.eval()

            soft_hist_pt = SoftHistogram(**SOFT_HISTOGRAM_ARGS)
            soft_hist_load = soft_hist_pt.load_state_dict(histogram_state_dict(state_dict), strict=True)
            soft_hist_pt.eval()

            TFSofthistogram = import_tf_soft_histogram(root)
            soft_hist_tf = TFSofthistogram(**SOFT_HISTOGRAM_ARGS)
            soft_hist_tf.load_from_pytorch_state_dict(state_dict)

            pytorch_inspection = inspect_pytorch_soft_histogram(soft_hist_pt)

            dummy_input_nchw = np.random.randn(1, 3, 224, 224).astype(np.float32)
            dummy_capture = capture_soft_histogram_io(torch, model, dummy_input_nchw)
            expected_input_shape = shape_of(dummy_capture["input"])

            random_np = np.random.randn(*expected_input_shape).astype(np.float32)
            zeros_np = np.zeros(expected_input_shape, dtype=np.float32)
            ones_np = np.ones(expected_input_shape, dtype=np.float32)

            results = [
                run_soft_histogram_case(torch, tf, soft_hist_pt, soft_hist_tf, "random expected-shape input", random_np),
                run_soft_histogram_case(torch, tf, soft_hist_pt, soft_hist_tf, "zeros expected-shape input", zeros_np),
                run_soft_histogram_case(torch, tf, soft_hist_pt, soft_hist_tf, "ones expected-shape input", ones_np),
            ]

            real_image_input, real_image_path = first_real_image_input(root)
            real_capture = None
            real_capture_performed = False
            if real_image_input is not None:
                real_capture = capture_soft_histogram_io(torch, model, real_image_input)
                real_input_np = tensor_np(real_capture["input"])
                results.append(
                    run_soft_histogram_case(
                        torch,
                        tf,
                        soft_hist_pt,
                        soft_hist_tf,
                        "captured real pre-SoftHistogram tensor",
                        real_input_np,
                    )
                )
                real_capture_performed = True

            for result in results:
                log(result_line(result))

            log(f"checkpoint path: {checkpoint_path}")
            log(f"state_dict confirmation: {state_dict_confirmation}")
            log(f"strict load missing keys: {list(getattr(strict_load, 'missing_keys', []))}")
            log(f"strict load unexpected keys: {list(getattr(strict_load, 'unexpected_keys', []))}")
            log(f"SoftHistogram strict load missing keys: {list(getattr(soft_hist_load, 'missing_keys', []))}")
            log(f"SoftHistogram strict load unexpected keys: {list(getattr(soft_hist_load, 'unexpected_keys', []))}")
            log(f"torch version: {torch.__version__}")
            log(f"tensorflow version: {tf.__version__}")
            log(f"numpy version: {np.__version__}")

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        unresolved = []
        for result in results:
            if not result["pass"]:
                unresolved.append(f"{result['name']} failed: {result['likely_cause_if_failed']}")

        captured_summary = {
            "dummy_capture": {
                "input": tensor_summary("dummy_soft_histogram_input", dummy_capture["input"]),
                "output": tensor_summary("dummy_soft_histogram_output", dummy_capture["output"]),
            },
            "real_capture": None,
            "real_image_path": real_image_path,
        }
        if real_capture is not None:
            captured_summary["real_capture"] = {
                "input": tensor_summary("real_soft_histogram_input", real_capture["input"]),
                "output": tensor_summary("real_soft_histogram_output", real_capture["output"]),
            }

        safe_to_stage_e = bool(all(result["pass"] for result in results))

        report = {
            "status": "ok",
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "pytorch_reference_model_path": str(export_safe_model_dir),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "seed": SEED,
            "thresholds": {
                "preferred_max_abs_diff": PREFERRED_MAX_ABS_DIFF,
                "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_DIFF,
            },
            "state_dict_confirmation": state_dict_confirmation,
            "strict_load": {
                "success": True,
                "missing_keys": list(getattr(strict_load, "missing_keys", [])),
                "unexpected_keys": list(getattr(strict_load, "unexpected_keys", [])),
            },
            "soft_histogram_strict_load": {
                "success": True,
                "missing_keys": list(getattr(soft_hist_load, "missing_keys", [])),
                "unexpected_keys": list(getattr(soft_hist_load, "unexpected_keys", [])),
            },
            "pytorch_soft_histogram": pytorch_inspection,
            "tf_soft_histogram_structure": soft_hist_tf.structure_summary(),
            "dummy_capture": captured_summary["dummy_capture"],
            "real_capture": captured_summary["real_capture"],
            "real_capture_performed": real_capture_performed,
            "real_image_path": real_image_path,
            "results": results,
            "safe_to_proceed_to_stage_e_local_attention_block_parity": safe_to_stage_e,
            "unresolved_issues": unresolved,
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }
        return report, captured_summary

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.extend(traceback.format_exc().splitlines())
        report = {
            "status": "failed",
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "pytorch_reference_model_path": str(export_safe_model_dir),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "seed": SEED,
            "thresholds": {
                "preferred_max_abs_diff": PREFERRED_MAX_ABS_DIFF,
                "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_DIFF,
            },
            "state_dict_confirmation": None,
            "strict_load": {"success": False, "missing_keys": [], "unexpected_keys": []},
            "soft_histogram_strict_load": {"success": False, "missing_keys": [], "unexpected_keys": []},
            "pytorch_soft_histogram": None,
            "tf_soft_histogram_structure": None,
            "dummy_capture": None,
            "real_capture": None,
            "real_capture_performed": False,
            "real_image_path": None,
            "results": [],
            "safe_to_proceed_to_stage_e_local_attention_block_parity": False,
            "unresolved_issues": ["Stage D script failed before completing SoftHistogram parity."],
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }
        return report, {}


def main() -> int:
    root = project_root()
    stdout_stderr = io.StringIO()
    with redirect_stdout(stdout_stderr), redirect_stderr(stdout_stderr):
        report, captured_summary = run()

    output_dir = Path(report["output_dir"])
    json_path = output_dir / "stage_d_soft_histogram_report.json"
    log_path = output_dir / "stage_d_soft_histogram_log.txt"
    captured_summary_path = output_dir / "optional_captured_tensors_summary.json"
    markdown_path = root / REPORT_REL_PATH

    report["json_path"] = str(json_path)
    report["log_path"] = str(log_path)
    report["captured_tensors_summary_path"] = str(captured_summary_path)
    captured_output = stdout_stderr.getvalue().splitlines()
    report["captured_output"] = captured_output

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    captured_summary_path.write_text(json.dumps(captured_summary, indent=2), encoding="utf-8")
    if report["status"] == "ok":
        write_markdown_report(report, markdown_path)
    else:
        markdown_path.write_text(
            "# Stage D SoftHistogram Parity\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Stage D failed before SoftHistogram parity completed. See JSON and log artifacts for details.\n",
            encoding="utf-8",
        )
    log_path.write_text("\n".join(captured_output) + "\n", encoding="utf-8")

    for line in captured_output:
        print(line)
    print(f"json report: {json_path}")
    print(f"log: {log_path}")
    print(f"captured tensor summary: {captured_summary_path}")
    print(f"markdown report: {markdown_path}")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

