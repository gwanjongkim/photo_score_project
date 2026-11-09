# ICAA17K DAT 패치 투영 블록의 PyTorch-TensorFlow 동등성을 검증하는 스크립트
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_c_patch_projection_parity.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_c"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_c_patch_projection_parity.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
PREFERRED_MAX_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-4

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


def import_dat_from_package_dir(package_dir: Path):
    package_name = "icaa_export_safe_v3_export_safe_models_stage_c"
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
    return importlib.import_module(f"{package_name}.dat").DAT


def import_tf_patch_projection(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.patch_projection import TFPatchProjection

    return TFPatchProjection


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


def compare_outputs(name: str, torch_output, tf_output_nhwc) -> dict:
    import numpy as np
    import tensorflow as tf

    torch_np = tensor_np(torch_output)
    tf_nhwc_np = tf_output_nhwc.numpy()
    tf_nchw_np = tf.transpose(tf_output_nhwc, (0, 3, 1, 2)).numpy()
    diff = np.abs(torch_np.astype(np.float32) - tf_nchw_np.astype(np.float32))
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
        "tf_output_shape_nhwc": list(tf_nhwc_np.shape),
        "tf_output_shape_nchw": list(tf_nchw_np.shape),
        "thresholds": {
            "preferred_max_abs_diff": PREFERRED_MAX_ABS_DIFF,
            "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_DIFF,
        },
        "likely_cause_if_failed": None if acceptable_pass else "Patch projection mapping or LayerNorm semantics need inspection.",
    }


def preprocess_real_image(path: Path):
    import numpy as np
    from PIL import Image

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std


def load_real_image_batch(root: Path, limit: int = 3):
    import numpy as np

    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    if not csv_path.is_file() or not image_root.is_dir():
        return None, []

    arrays = []
    paths = []
    try:
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
    except Exception:
        return None, paths

    if not arrays:
        return None, paths
    return np.stack(arrays, axis=0).astype(np.float32), paths


def run_patch_projection(torch, tf, np, patch_proj_pt, patch_proj_tf, x_nhwc):
    x_pt = torch.from_numpy(np.transpose(x_nhwc, (0, 3, 1, 2)).astype(np.float32))
    x_tf = tf.convert_to_tensor(x_nhwc.astype(np.float32), dtype=tf.float32)
    with torch.no_grad():
        y_pt = patch_proj_pt(x_pt)
    y_tf = patch_proj_tf(x_tf, training=False)
    return y_pt, y_tf


def result_line(result: dict) -> str:
    return (
        f"{result['name']}: status={result['status']} "
        f"max_abs_diff={result['max_abs_diff']:.9g} "
        f"mean_abs_diff={result['mean_abs_diff']:.9g} "
        f"torch_shape={result['torch_output_shape']} "
        f"tf_nhwc_shape={result['tf_output_shape_nhwc']} "
        f"tf_nchw_shape={result['tf_output_shape_nchw']}"
    )


def write_markdown_report(report: dict, report_path: Path) -> None:
    random_result = report["random_input_result"]
    real_result = report.get("real_image_result")
    unresolved = report["unresolved_issues"] or ["none"]
    warning_lines = report["warnings"] or ["none"]
    error_lines = report["errors"] or ["none"]
    runtime_notices = [
        line for line in report.get("captured_output", [])
        if line.startswith(("WARNING:", "E", "W"))
    ] or ["none"]

    real_section = "- Real-image parity was skipped or no local images were available."
    if real_result:
        real_section = (
            f"- Status: {real_result['status']}\n"
            f"- PyTorch output shape: {real_result['torch_output_shape']}\n"
            f"- TensorFlow output shape before transpose: {real_result['tf_output_shape_nhwc']}\n"
            f"- TensorFlow output shape after transpose: {real_result['tf_output_shape_nchw']}\n"
            f"- max_abs_diff: {real_result['max_abs_diff']:.9g}\n"
            f"- mean_abs_diff: {real_result['mean_abs_diff']:.9g}\n"
            f"- Image paths: {report['real_image_paths']}"
        )

    body = f"""# Stage C Patch Projection Parity

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## Checkpoint Tensors Used
- `patch_proj.0.weight`
- `patch_proj.0.bias`
- `patch_proj.1.norm.weight`
- `patch_proj.1.norm.bias`

## TensorFlow Patch Projection Structure
```json
{json.dumps(report['tf_patch_projection_structure'], indent=2)}
```

## Weight Mapping Rules Used
- Conv2D kernel: PyTorch `[out_c, in_c, kH, kW]` -> TensorFlow `[kH, kW, in_c, out_c]`.
- Conv2D bias: direct copy.
- LayerNorm gamma/beta: direct copy.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow output is transposed back to NCHW before comparison.

## Random Input Result
- PyTorch output shape: {random_result['torch_output_shape']}
- TensorFlow output shape before transpose: {random_result['tf_output_shape_nhwc']}
- TensorFlow output shape after transpose: {random_result['tf_output_shape_nchw']}
- max_abs_diff: {random_result['max_abs_diff']:.9g}
- mean_abs_diff: {random_result['mean_abs_diff']:.9g}
- pass/fail: {random_result['status']}

## Real Image Result
{real_section}

## Unresolved Issues
""" + "\n".join(f"- {line}" for line in unresolved) + f"""

## Stage D SoftHistogram Parity Decision
- Safe to proceed to Stage D SoftHistogram parity: {report['safe_to_proceed_to_stage_d_softhistogram_parity']}
- Scope of this decision: patch projection parity only. This does not establish full TensorFlow DAT feasibility.

## Warnings
""" + "\n".join(f"- {line}" for line in warning_lines) + """

## Runtime Log Notices
""" + "\n".join(f"- {line}" for line in runtime_notices) + """

## Errors
""" + "\n".join(f"- {line}" for line in error_lines) + "\n"

    report_path.write_text(body, encoding="utf-8")


def run() -> dict:
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
            DAT = import_dat_from_package_dir(export_safe_model_dir)
            model = DAT(**MODEL_KWARGS)
            strict_load = model.load_state_dict(state_dict, strict=True)
            model.eval()
            patch_proj_pt = model.patch_proj
            patch_proj_pt.eval()

            TFPatchProjection = import_tf_patch_projection(root)
            patch_proj_tf = TFPatchProjection(dim_stem=128, patch_size=4, epsilon=1e-5)
            patch_proj_tf.build_for_input_shape((1, 224, 224, 3))
            patch_proj_tf.load_from_pytorch_state_dict(state_dict)

            log(f"checkpoint path: {checkpoint_path}")
            log(f"state_dict confirmation: {state_dict_confirmation}")
            log(f"strict load missing keys: {list(getattr(strict_load, 'missing_keys', []))}")
            log(f"strict load unexpected keys: {list(getattr(strict_load, 'unexpected_keys', []))}")
            log(f"torch version: {torch.__version__}")
            log(f"tensorflow version: {tf.__version__}")
            log(f"numpy version: {np.__version__}")

            random_input_nhwc = np.random.randn(1, 224, 224, 3).astype(np.float32)
            random_y_pt, random_y_tf = run_patch_projection(torch, tf, np, patch_proj_pt, patch_proj_tf, random_input_nhwc)
            random_result = compare_outputs("Random input patch projection", random_y_pt, random_y_tf)
            log(result_line(random_result))

            real_input_nhwc, real_image_paths = load_real_image_batch(root, limit=3)
            real_result = None
            if real_input_nhwc is not None:
                real_y_pt, real_y_tf = run_patch_projection(torch, tf, np, patch_proj_pt, patch_proj_tf, real_input_nhwc)
                real_result = compare_outputs("Real ICAA17K image patch projection", real_y_pt, real_y_tf)
                log(result_line(real_result))
            else:
                log("real image parity: skipped; no loadable local ICAA17K images found")

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        unresolved = []
        if not random_result["pass"]:
            unresolved.append(f"Random input parity failed: {random_result['likely_cause_if_failed']}")
        if real_result is not None and not real_result["pass"]:
            unresolved.append(f"Real image parity failed: {real_result['likely_cause_if_failed']}")

        safe_to_stage_d = bool(random_result["pass"] and (real_result is None or real_result["pass"]))

        return {
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
            "checkpoint_tensors_used": [
                "patch_proj.0.weight",
                "patch_proj.0.bias",
                "patch_proj.1.norm.weight",
                "patch_proj.1.norm.bias",
            ],
            "state_dict_confirmation": state_dict_confirmation,
            "strict_load": {
                "success": True,
                "missing_keys": list(getattr(strict_load, "missing_keys", [])),
                "unexpected_keys": list(getattr(strict_load, "unexpected_keys", [])),
            },
            "tf_patch_projection_structure": patch_proj_tf.structure_summary(),
            "random_input_result": random_result,
            "real_image_result": real_result,
            "real_image_paths": real_image_paths,
            "safe_to_proceed_to_stage_d_softhistogram_parity": safe_to_stage_d,
            "unresolved_issues": unresolved,
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.extend(traceback.format_exc().splitlines())
        return {
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
            "checkpoint_tensors_used": [
                "patch_proj.0.weight",
                "patch_proj.0.bias",
                "patch_proj.1.norm.weight",
                "patch_proj.1.norm.bias",
            ],
            "state_dict_confirmation": None,
            "strict_load": {"success": False, "missing_keys": [], "unexpected_keys": []},
            "tf_patch_projection_structure": None,
            "random_input_result": None,
            "real_image_result": None,
            "real_image_paths": [],
            "safe_to_proceed_to_stage_d_softhistogram_parity": False,
            "unresolved_issues": ["Stage C script failed before completing patch projection parity."],
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }


def main() -> int:
    root = project_root()
    stdout_stderr = io.StringIO()
    with redirect_stdout(stdout_stderr), redirect_stderr(stdout_stderr):
        report = run()

    output_dir = Path(report["output_dir"])
    json_path = output_dir / "stage_c_patch_projection_report.json"
    log_path = output_dir / "stage_c_patch_projection_log.txt"
    markdown_path = root / REPORT_REL_PATH

    report["json_path"] = str(json_path)
    report["log_path"] = str(log_path)
    captured_output = stdout_stderr.getvalue().splitlines()
    report["captured_output"] = captured_output

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] == "ok":
        write_markdown_report(report, markdown_path)
    else:
        markdown_path.write_text(
            "# Stage C Patch Projection Parity\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Stage C failed before patch projection parity completed. See JSON and log artifacts for details.\n",
            encoding="utf-8",
        )
    log_path.write_text("\n".join(captured_output) + "\n", encoding="utf-8")

    for line in captured_output:
        print(line)
    print(f"json report: {json_path}")
    print(f"log: {log_path}")
    print(f"markdown report: {markdown_path}")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

