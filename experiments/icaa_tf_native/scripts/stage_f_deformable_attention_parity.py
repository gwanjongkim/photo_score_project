# ICAA17K DAT deformable attention 모듈의 PyTorch-TensorFlow 동등성을 검증하는 스크립트
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_f_deformable_attention_parity.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_f"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_f_deformable_attention_parity.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
PREFERRED_MAX_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-4
SELECTED_PREFIX = "stages.2.attns.1"
RANDOM_MODULE_INPUT_SCALE = 0.5

CHECKPOINT_TENSORS_USED = [
    "stages.2.attns.1.rpe_table",
    "stages.2.attns.1.conv_offset.0.weight",
    "stages.2.attns.1.conv_offset.0.bias",
    "stages.2.attns.1.conv_offset.1.norm.weight",
    "stages.2.attns.1.conv_offset.1.norm.bias",
    "stages.2.attns.1.conv_offset.3.weight",
    "stages.2.attns.1.proj_q.weight",
    "stages.2.attns.1.proj_q.bias",
    "stages.2.attns.1.proj_k.weight",
    "stages.2.attns.1.proj_k.bias",
    "stages.2.attns.1.proj_v.weight",
    "stages.2.attns.1.proj_v.bias",
    "stages.2.attns.1.proj_out.weight",
    "stages.2.attns.1.proj_out.bias",
    "stages.2.attns.1.ref_point14.weight",
    "stages.2.attns.1.ref_point14.bias",
    "stages.2.attns.1.ref_gate.weight",
    "stages.2.attns.1.ref_gate.bias",
]

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
    package_name = "icaa_export_safe_v3_export_safe_models_stage_f"
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


def import_tf_dattention(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.deformable_attention import TFDAttentionBaseline

    return TFDAttentionBaseline


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


def first_dattention_input_from_image(torch, model, image_nchw):
    with torch.no_grad():
        x = model.patch_proj(torch.from_numpy(image_nchw.astype("float32")))
        x, _pos, _ref = model.stages[0](x)
        x = model.down_projs[0](x)
        x, _pos, _ref = model.stages[1](x)
        x = model.down_projs[1](x)
        stage = model.stages[2]
        x = stage.proj(x)

        x0 = x
        attn0_out, _pos, _ref = stage.attns[0](stage.layer_norms[0](x))
        x = stage.drop_path[0](attn0_out) + x0
        x0 = x
        x = stage.mlps[0](stage.layer_norms[1](x))
        x = stage.drop_path[0](x) + x0

        selected_input = stage.layer_norms[2](x)
        selected_output = stage.attns[1](selected_input)
    return selected_input.detach().cpu(), selected_output


def compare_nchw_to_nhwc(name: str, torch_tensor, tf_tensor_nhwc) -> dict:
    import numpy as np
    import tensorflow as tf

    torch_np = tensor_np(torch_tensor)
    tf_nhwc_np = tf_tensor_nhwc.numpy()
    tf_nchw_np = tf.transpose(tf_tensor_nhwc, (0, 3, 1, 2)).numpy()
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
    }


def compare_bghw2(name: str, torch_tensor, tf_tensor, batch: int, groups: int) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_tensor)
    tf_np = tf_tensor.numpy().reshape(batch, groups, torch_np.shape[2], torch_np.shape[3], 2)
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    preferred_pass = max_abs_diff <= PREFERRED_MAX_ABS_DIFF
    acceptable_pass = max_abs_diff <= ACCEPTABLE_MAX_ABS_DIFF
    return {
        "name": name,
        "status": "pass_preferred" if preferred_pass else "pass_acceptable" if acceptable_pass else "fail",
        "pass": bool(acceptable_pass),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "torch_output_shape": list(torch_np.shape),
        "tf_output_shape": list(tf_np.shape),
    }


def likely_failed_subcomponent(result, pos_diff, reference_diff) -> str | None:
    if result["pass"]:
        return None
    if not reference_diff["pass"]:
        return "reference projection"
    if not pos_diff["pass"]:
        return "offset branch / reference grid / gate"
    return "bilinear sampler, q/k/v projection, attention logits, softmax, relative position bias, output projection, or layout"


def run_case(torch, tf, np, attn_pt, attn_tf, name: str, input_nchw):
    x_pt = torch.from_numpy(input_nchw.astype(np.float32))
    x_tf = tf.convert_to_tensor(np.transpose(input_nchw, (0, 2, 3, 1)).astype(np.float32), dtype=tf.float32)
    with torch.no_grad():
        y_pt, pos_pt, reference_pt = attn_pt(x_pt)
    y_tf, debug_tf = attn_tf.call_with_debug(x_tf, training=False)
    result = compare_nchw_to_nhwc(name, y_pt, y_tf)
    batch = int(input_nchw.shape[0])
    pos_diff = compare_bghw2(f"{name} pos", pos_pt, debug_tf["pos"], batch, attn_tf.n_groups)
    reference_diff = compare_bghw2(f"{name} reference", reference_pt, debug_tf["reference"], batch, attn_tf.n_groups)
    result["likely_failed_subcomponent"] = likely_failed_subcomponent(result, pos_diff, reference_diff)
    return result, {
        "pos_diff": pos_diff,
        "reference_diff": reference_diff,
        "input_summary": tensor_summary(f"{name} input_nchw", input_nchw),
        "output_summary": tensor_summary(f"{name} output_nchw", y_pt),
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
    nhwc = np.stack(arrays, axis=0).astype(np.float32)
    return np.transpose(nhwc, (0, 3, 1, 2)).astype(np.float32), paths


def tensor_summary(name: str, tensor_or_array) -> dict:
    import numpy as np

    array = tensor_np(tensor_or_array) if hasattr(tensor_or_array, "detach") else np.asarray(tensor_or_array)
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
        f"tf_nhwc_shape={result['tf_output_shape_nhwc']} "
        f"tf_nchw_shape={result['tf_output_shape_nchw']}"
    )


def markdown_result_table(results: list[dict]) -> str:
    lines = [
        "| Test | Status | Max abs diff | Mean abs diff | PyTorch shape | TensorFlow shape before transpose |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['status']} | {result['max_abs_diff']:.9g} | "
            f"{result['mean_abs_diff']:.9g} | {result['torch_output_shape']} | {result['tf_output_shape_nhwc']} |"
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
    tensor_names = "\n".join(f"- `{name}`" for name in report["checkpoint_tensors_used"])

    body = f"""# Stage F Deformable Attention Parity

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## PyTorch Module Inspection
```json
{json.dumps(report['pytorch_module_inspection'], indent=2)}
```

## Checkpoint Tensors Used
{tensor_names}

## DAttentionBaseline Shapes
- PyTorch input shape: {report['module_input_shape_nchw']}
- PyTorch output shape: {report['module_output_shape_nchw']}
- TensorFlow input shape: {report['module_input_shape_nhwc']}
- TensorFlow output shape: {report['module_output_shape_nhwc']}

## Test Input Generation
```json
{json.dumps(report['test_input_generation'], indent=2)}
```

## TensorFlow Implementation Summary
```json
{json.dumps(report['tf_module_structure'], indent=2)}
```

## Manual Bilinear Sampler Summary
- Normalized grid coordinates are mapped with `align_corners=True`: `(coord + 1) * 0.5 * (size - 1)`.
- Out-of-range corner weights are zeroed to match `padding_mode='zeros'`.
- Corner indices are clamped only for safe `tf.gather` indexing.
- Implemented with TensorFlow tensor ops, no NumPy or pixel loops in the sampler.

## Weight Mapping Rules Used
- 1x1 Conv2D projections: PyTorch `[out_c, in_c, 1, 1]` -> `PytorchLikeConv1x1` kernel `[out_c, in_c]`, applied as NHWC flat matrix multiplication with `kernel^T`.
- Depthwise offset Conv2D: PyTorch `[channel, 1, kH, kW]` -> TensorFlow `[kH, kW, channel, 1]`.
- Conv biases and LayerNorm gamma/beta: direct copy.
- `rpe_table`: direct copy as `[heads, 2H-1, 2W-1]`.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow output is transposed back to NCHW before comparison.

## Parity Results
{markdown_result_table(report['results'])}

## Debug Tensor Summary
- Debug summary artifact: `{report['debug_tensors_summary_path']}`
- Debug diffs include returned `pos` and `reference` tensors for each test case.

## Real Image Inputs
- Performed: {report['real_images_performed']}
- Image paths: {report['real_image_paths']}

## Unresolved Issues
""" + "\n".join(f"- {line}" for line in unresolved) + f"""

## Stage G Full TransformerStage Decision
- Safe to proceed to Stage G full TransformerStage parity: {report['safe_to_proceed_to_stage_g_full_transformerstage_parity']}
- Scope of this decision: `DAttentionBaseline` module parity only. This does not establish full TensorFlow DAT feasibility.

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
            DAT = import_dat_from_package_dir(export_safe_model_dir)
            model = DAT(**MODEL_KWARGS)
            strict_load = model.load_state_dict(state_dict, strict=True)
            model.eval()
            attn_pt = model.stages[2].attns[1]
            attn_pt.eval()

            TFDAttention = import_tf_dattention(root)
            attn_tf = TFDAttention()
            attn_tf.build_for_input_shape((1, 14, 14, 512))
            attn_tf.load_from_pytorch_state_dict(state_dict, prefix=SELECTED_PREFIX)

            pytorch_module_inspection = {
                "class_name": attn_pt.__class__.__name__,
                "selected_stage_index": 2,
                "selected_attention_index": 1,
                "first_d_in_stage_spec": "stage_spec[2][1]",
                "q_size": [int(attn_pt.q_h), int(attn_pt.q_w)],
                "kv_size": [int(attn_pt.kv_h), int(attn_pt.kv_w)],
                "n_heads": int(attn_pt.n_heads),
                "n_head_channels": int(attn_pt.n_head_channels),
                "n_groups": int(attn_pt.n_groups),
                "n_group_channels": int(attn_pt.n_group_channels),
                "n_group_heads": int(attn_pt.n_group_heads),
                "scale": float(attn_pt.scale),
                "offset_range_factor": int(attn_pt.offset_range_factor),
                "use_pe": bool(attn_pt.use_pe),
                "dwc_pe": bool(attn_pt.dwc_pe),
                "fixed_pe": bool(attn_pt.fixed_pe),
                "no_off": bool(attn_pt.no_off),
                "offset_branch": [
                    attn_pt.conv_offset[0].__class__.__name__,
                    attn_pt.conv_offset[1].__class__.__name__,
                    attn_pt.conv_offset[2].__class__.__name__,
                    attn_pt.conv_offset[3].__class__.__name__,
                ],
                "qkv_projection_classes": {
                    "proj_q": attn_pt.proj_q.__class__.__name__,
                    "proj_k": attn_pt.proj_k.__class__.__name__,
                    "proj_v": attn_pt.proj_v.__class__.__name__,
                    "proj_out": attn_pt.proj_out.__class__.__name__,
                },
                "rpe_table_shape": list(attn_pt.rpe_table.shape),
                "dropout_eval_behavior": "proj_drop and attn_drop are Dropout(0.0) in this config; eval mode disables dropout randomness.",
            }

            random_input = (np.random.randn(1, 512, 14, 14) * RANDOM_MODULE_INPUT_SCALE).astype(np.float32)
            random_result, random_debug = run_case(torch, tf, np, attn_pt, attn_tf, "random DAttentionBaseline input", random_input)

            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
            random_image_nhwc = np.random.rand(1, 224, 224, 3).astype(np.float32)
            random_image = np.transpose((random_image_nhwc - mean) / std, (0, 3, 1, 2)).astype(np.float32)
            captured_random_input, _captured_random_output = first_dattention_input_from_image(torch, model, random_image)
            captured_random_np = tensor_np(captured_random_input)
            captured_random_result, captured_random_debug = run_case(
                torch,
                tf,
                np,
                attn_pt,
                attn_tf,
                "captured random-image DAttentionBaseline input",
                captured_random_np,
            )

            real_image_nchw, real_image_paths = load_real_image_batch(root, limit=3)
            real_result = None
            real_debug = None
            captured_real_np = None
            if real_image_nchw is not None:
                captured_real_input, _captured_real_output = first_dattention_input_from_image(torch, model, real_image_nchw)
                captured_real_np = tensor_np(captured_real_input)
                real_result, real_debug = run_case(
                    torch,
                    tf,
                    np,
                    attn_pt,
                    attn_tf,
                    "captured 3-real-image DAttentionBaseline input",
                    captured_real_np,
                )

            results = [random_result, captured_random_result]
            if real_result is not None:
                results.append(real_result)

            for result in results:
                log(result_line(result))

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        unresolved = []
        for result in results:
            if not result["pass"]:
                unresolved.append(
                    f"{result['name']} failed; likely subcomponent: {result.get('likely_failed_subcomponent') or 'unknown'}"
                )

        safe_to_stage_g = bool(all(result["pass"] for result in results))
        module_input_shape_nchw = list(random_input.shape)
        module_input_shape_nhwc = [random_input.shape[0], random_input.shape[2], random_input.shape[3], random_input.shape[1]]
        module_output_shape_nchw = random_result["torch_output_shape"]
        module_output_shape_nhwc = random_result["tf_output_shape_nhwc"]

        debug_summary = {
            "random_module_input": {
                "input": tensor_summary("random_module_input_nchw", random_input),
                "output_result": random_result,
                "pos_reference_diffs": random_debug,
            },
            "captured_random_image_input": {
                "input": tensor_summary("captured_random_image_input_nchw", captured_random_np),
                "output_result": captured_random_result,
                "pos_reference_diffs": captured_random_debug,
            },
            "captured_real_image_input": None,
            "real_image_paths": real_image_paths,
        }
        if real_result is not None:
            debug_summary["captured_real_image_input"] = {
                "input": tensor_summary("captured_real_image_input_nchw", captured_real_np),
                "output_result": real_result,
                "pos_reference_diffs": real_debug,
            }

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
            "pytorch_module_inspection": pytorch_module_inspection,
            "checkpoint_tensors_used": CHECKPOINT_TENSORS_USED,
            "tf_module_structure": attn_tf.structure_summary(),
            "test_input_generation": {
                "random_module_input": {
                    "distribution": "numpy randn scaled by RANDOM_MODULE_INPUT_SCALE",
                    "scale": RANDOM_MODULE_INPUT_SCALE,
                    "reason": "keeps the synthetic block input near observed captured DAttention tensor scale while still using deterministic random values",
                    "seed": SEED,
                },
                "captured_random_image_input": {
                    "distribution": "numpy uniform [0, 1) RGB image with ImageNet mean/std preprocessing",
                    "seed": SEED,
                },
                "real_image_input": {
                    "source": CSV_REL_PATH,
                    "preprocessing": "PIL bilinear resize to 224x224, RGB [0,1], ImageNet mean/std normalization",
                },
            },
            "module_input_shape_nchw": module_input_shape_nchw,
            "module_input_shape_nhwc": module_input_shape_nhwc,
            "module_output_shape_nchw": module_output_shape_nchw,
            "module_output_shape_nhwc": module_output_shape_nhwc,
            "results": results,
            "real_images_performed": real_result is not None,
            "real_image_paths": real_image_paths,
            "safe_to_proceed_to_stage_g_full_transformerstage_parity": safe_to_stage_g,
            "unresolved_issues": unresolved,
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }
        return report, debug_summary

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
            "pytorch_module_inspection": None,
            "checkpoint_tensors_used": CHECKPOINT_TENSORS_USED,
            "tf_module_structure": None,
            "module_input_shape_nchw": None,
            "module_input_shape_nhwc": None,
            "module_output_shape_nchw": None,
            "module_output_shape_nhwc": None,
            "results": [],
            "real_images_performed": False,
            "real_image_paths": [],
            "safe_to_proceed_to_stage_g_full_transformerstage_parity": False,
            "unresolved_issues": ["Stage F script failed before completing deformable-attention parity."],
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }
        return report, {}


def main() -> int:
    root = project_root()
    stdout_stderr = io.StringIO()
    with redirect_stdout(stdout_stderr), redirect_stderr(stdout_stderr):
        report, debug_summary = run()

    output_dir = Path(report["output_dir"])
    json_path = output_dir / "stage_f_deformable_attention_report.json"
    log_path = output_dir / "stage_f_deformable_attention_log.txt"
    debug_summary_path = output_dir / "optional_debug_tensors_summary.json"
    markdown_path = root / REPORT_REL_PATH

    report["json_path"] = str(json_path)
    report["log_path"] = str(log_path)
    report["debug_tensors_summary_path"] = str(debug_summary_path)
    captured_output = stdout_stderr.getvalue().splitlines()
    report["captured_output"] = captured_output

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    debug_summary_path.write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")
    if report["status"] == "ok":
        write_markdown_report(report, markdown_path)
    else:
        markdown_path.write_text(
            "# Stage F Deformable Attention Parity\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Stage F failed before deformable-attention parity completed. See JSON and log artifacts for details.\n",
            encoding="utf-8",
        )
    log_path.write_text("\n".join(captured_output) + "\n", encoding="utf-8")

    for line in captured_output:
        print(line)
    print(f"json report: {json_path}")
    print(f"log: {log_path}")
    print(f"debug tensor summary: {debug_summary_path}")
    print(f"markdown report: {markdown_path}")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
