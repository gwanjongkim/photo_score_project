# ICAA17K DAT Stage 2 블록 3/4 MLP 원인 분리를 위한 진단 스크립트
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_g2_mlp_surgical_debug.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_g2_mlp_debug"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/debug_stage_g2_block_mlp.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
PREFERRED_MAX_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-4
TARGET_BLOCKS = [3, 4]

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
    package_name = "icaa_export_safe_v3_export_safe_models_g2_mlp_debug"
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


def import_tf_stage2(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.transformer_stage import TFTransformerStage2

    return TFTransformerStage2


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


def result_status(max_abs_diff: float) -> tuple[str, bool, bool, bool]:
    preferred_pass = max_abs_diff <= PREFERRED_MAX_ABS_DIFF
    acceptable_pass = max_abs_diff <= ACCEPTABLE_MAX_ABS_DIFF
    status = "pass_preferred" if preferred_pass else "pass_acceptable" if acceptable_pass else "fail"
    return status, bool(acceptable_pass), bool(preferred_pass), bool(acceptable_pass)


def compare_arrays(name: str, torch_tensor, tf_tensor_or_array) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_tensor) if hasattr(torch_tensor, "detach") else np.asarray(torch_tensor)
    tf_np = tf_tensor_or_array.numpy() if hasattr(tf_tensor_or_array, "numpy") else np.asarray(tf_tensor_or_array)
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    status, acceptable, preferred, acceptable_pass = result_status(max_abs_diff)
    return {
        "name": name,
        "status": status,
        "pass": acceptable,
        "preferred_pass": preferred,
        "acceptable_pass": acceptable_pass,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "torch_shape": list(torch_np.shape),
        "tf_shape": list(tf_np.shape),
    }


def compare_nchw_to_nhwc(name: str, torch_tensor, tf_tensor_nhwc) -> dict:
    import numpy as np
    import tensorflow as tf

    torch_np = tensor_np(torch_tensor)
    tf_nhwc_np = tf_tensor_nhwc.numpy()
    tf_nchw_np = tf.transpose(tf_tensor_nhwc, (0, 3, 1, 2)).numpy()
    diff = np.abs(torch_np.astype(np.float32) - tf_nchw_np.astype(np.float32))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    status, acceptable, preferred, acceptable_pass = result_status(max_abs_diff)
    return {
        "name": name,
        "status": status,
        "pass": acceptable,
        "preferred_pass": preferred,
        "acceptable_pass": acceptable_pass,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "torch_shape": list(torch_np.shape),
        "tf_shape_nhwc": list(tf_nhwc_np.shape),
        "tf_shape_nchw": list(tf_nchw_np.shape),
    }


def torch_mlp_intermediates(torch, mlp, norm2_nchw):
    with torch.no_grad():
        b, c, h, w = norm2_nchw.shape
        tokens = norm2_nchw.permute(0, 2, 3, 1).reshape(b, h * w, c)
        linear1 = mlp.chunk.linear1(tokens)
        gelu = mlp.chunk.act(linear1)
        after_drop1 = mlp.chunk.drop1(gelu)
        linear2 = mlp.chunk.linear2(after_drop1)
        after_drop2 = mlp.chunk.drop2(linear2)
        output = after_drop2.reshape(b, h, w, c).permute(0, 3, 1, 2)
    return {
        "linear1_output": linear1.detach().cpu(),
        "gelu_output": gelu.detach().cpu(),
        "linear2_output": linear2.detach().cpu(),
        "mlp_output": output.detach().cpu(),
    }


def tf_mlp_intermediates(tf, mlp, norm2_nhwc):
    x = tf.convert_to_tensor(norm2_nhwc, dtype=tf.float32)
    shape = tf.shape(x)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    tokens = tf.reshape(x, [batch, height * width, channels])
    linear1 = mlp.linear1(tokens)
    gelu = tf.nn.gelu(linear1, approximate=False)
    linear2 = mlp.linear2(gelu)
    output = tf.reshape(linear2, [batch, height, width, mlp.channels])
    return {
        "linear1_output": linear1,
        "gelu_output": gelu,
        "linear2_output": linear2,
        "mlp_output": output,
    }


def torch_stage_debug(torch, stage, stage_input, target_blocks: list[int]):
    debug = {}
    x = stage.proj(stage_input)
    with torch.no_grad():
        for block_idx in range(stage.depths):
            x0 = x
            if block_idx in target_blocks:
                debug.setdefault(block_idx, {})["block_input"] = x.detach().cpu()
            norm1 = stage.layer_norms[2 * block_idx](x)
            if block_idx in target_blocks:
                debug[block_idx]["norm1_output"] = norm1.detach().cpu()
            attn_out, _pos, _ref = stage.attns[block_idx](norm1)
            if block_idx in target_blocks:
                debug[block_idx]["attention_output"] = attn_out.detach().cpu()
            after_attn = stage.drop_path[block_idx](attn_out) + x0
            if block_idx in target_blocks:
                debug[block_idx]["after_attention_residual"] = after_attn.detach().cpu()
            x0 = after_attn
            norm2 = stage.layer_norms[2 * block_idx + 1](after_attn)
            mlp_debug = torch_mlp_intermediates(torch, stage.mlps[block_idx], norm2)
            mlp_out = mlp_debug["mlp_output"]
            after_mlp = stage.drop_path[block_idx](mlp_out) + x0
            if block_idx in target_blocks:
                debug[block_idx]["norm2_output"] = norm2.detach().cpu()
                debug[block_idx].update(mlp_debug)
                debug[block_idx]["after_mlp_residual"] = after_mlp.detach().cpu()
            x = after_mlp
    return x, debug


def tf_stage_debug(tf, stage_tf, stage_input_nhwc, target_blocks: list[int]):
    debug = {}
    x = tf.convert_to_tensor(stage_input_nhwc, dtype=tf.float32)
    for block_idx, spec in enumerate(stage_tf.stage_spec):
        x0 = x
        if block_idx in target_blocks:
            debug.setdefault(block_idx, {})["block_input"] = x
        norm1 = stage_tf.layer_norms[2 * block_idx](x)
        if block_idx in target_blocks:
            debug[block_idx]["norm1_output"] = norm1
        if spec == "D":
            attn_out, _attn_debug = stage_tf.attns[block_idx].call_with_debug(norm1, training=False)
        else:
            attn_out = stage_tf.attns[block_idx](norm1, training=False)
        if block_idx in target_blocks:
            debug[block_idx]["attention_output"] = attn_out
        after_attn = attn_out + x0
        if block_idx in target_blocks:
            debug[block_idx]["after_attention_residual"] = after_attn
        x0 = after_attn
        norm2 = stage_tf.layer_norms[2 * block_idx + 1](after_attn)
        mlp_debug = tf_mlp_intermediates(tf, stage_tf.mlps[block_idx], norm2)
        mlp_out = mlp_debug["mlp_output"]
        after_mlp = mlp_out + x0
        if block_idx in target_blocks:
            debug[block_idx]["norm2_output"] = norm2
            debug[block_idx].update(mlp_debug)
            debug[block_idx]["after_mlp_residual"] = after_mlp
        x = after_mlp
    return x, debug


def direct_same_input_mlp_debug(tf, torch, stage_pt, stage_tf, block_idx: int, norm2_nchw):
    import numpy as np

    pt_debug = torch_mlp_intermediates(torch, stage_pt.mlps[block_idx], norm2_nchw)
    norm2_nhwc = np.transpose(tensor_np(norm2_nchw), (0, 2, 3, 1)).astype(np.float32)
    tf_debug = tf_mlp_intermediates(tf, stage_tf.mlps[block_idx], tf.convert_to_tensor(norm2_nhwc, dtype=tf.float32))
    comparisons = [
        compare_arrays("direct_same_input_linear1_output", pt_debug["linear1_output"], tf_debug["linear1_output"]),
        compare_arrays("direct_same_input_gelu_output", pt_debug["gelu_output"], tf_debug["gelu_output"]),
        compare_arrays("direct_same_input_linear2_output", pt_debug["linear2_output"], tf_debug["linear2_output"]),
        compare_nchw_to_nhwc("direct_same_input_mlp_output", pt_debug["mlp_output"], tf_debug["mlp_output"]),
    ]
    overall = next((row for row in comparisons if not row["pass"]), None)
    return {
        "block_index": block_idx,
        "overall_pass": overall is None,
        "overall_status": "pass" if overall is None else "fail",
        "first_failure": None if overall is None else overall["name"],
        "comparisons": comparisons,
    }


def full_flow_block_comparisons(block_idx: int, pt_block_debug: dict, tf_block_debug: dict) -> list[dict]:
    comparisons = []
    image_keys = [
        "block_input",
        "norm1_output",
        "attention_output",
        "after_attention_residual",
        "norm2_output",
        "mlp_output",
        "after_mlp_residual",
    ]
    token_keys = ["linear1_output", "gelu_output", "linear2_output"]
    for key in image_keys:
        comparisons.append(compare_nchw_to_nhwc(f"block{block_idx}_{key}", pt_block_debug[key], tf_block_debug[key]))
    for key in token_keys:
        comparisons.append(compare_arrays(f"block{block_idx}_{key}", pt_block_debug[key], tf_block_debug[key]))
    return comparisons


def first_failing_comparison(comparisons: list[dict]) -> dict | None:
    for row in comparisons:
        if not row["pass"]:
            return row
    return None


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


def stage2_input_from_image(torch, model, image_nchw):
    with torch.no_grad():
        x = model.patch_proj(torch.from_numpy(image_nchw.astype("float32")))
        x, _pos, _ref = model.stages[0](x)
        x = model.down_projs[0](x)
        x, _pos, _ref = model.stages[1](x)
        x = model.down_projs[1](x)
        return tensor_np(x)


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


def run_case(tf, torch, np, stage_pt, stage_tf, name: str, stage_input_nchw):
    x_pt = torch.from_numpy(stage_input_nchw.astype(np.float32))
    x_tf = tf.convert_to_tensor(np.transpose(stage_input_nchw, (0, 2, 3, 1)).astype(np.float32), dtype=tf.float32)
    _pt_output, pt_debug = torch_stage_debug(torch, stage_pt, x_pt, TARGET_BLOCKS)
    _tf_output, tf_debug = tf_stage_debug(tf, stage_tf, x_tf, TARGET_BLOCKS)

    block_results = {}
    flat_full_flow = []
    flat_direct = []
    for block_idx in TARGET_BLOCKS:
        full_flow = full_flow_block_comparisons(block_idx, pt_debug[block_idx], tf_debug[block_idx])
        direct = direct_same_input_mlp_debug(tf, torch, stage_pt, stage_tf, block_idx, pt_debug[block_idx]["norm2_output"])
        block_results[str(block_idx)] = {
            "full_flow_first_failure": None if first_failing_comparison(full_flow) is None else first_failing_comparison(full_flow)["name"],
            "full_flow_comparisons": full_flow,
            "direct_same_input_mlp": direct,
        }
        flat_full_flow.extend(full_flow)
        flat_direct.extend(direct["comparisons"])

    direct_pass_by_block = {
        str(block_idx): block_results[str(block_idx)]["direct_same_input_mlp"]["overall_pass"]
        for block_idx in TARGET_BLOCKS
    }
    first_full_flow_failure = first_failing_comparison(flat_full_flow)
    first_direct_failure = first_failing_comparison(flat_direct)
    return {
        "name": name,
        "stage_input_shape_nchw": list(stage_input_nchw.shape),
        "stage_input_shape_nhwc": list(x_tf.shape),
        "direct_same_input_pass_by_block": direct_pass_by_block,
        "all_direct_same_input_mlp_pass": all(direct_pass_by_block.values()),
        "first_full_flow_failure": None if first_full_flow_failure is None else first_full_flow_failure["name"],
        "first_direct_same_input_failure": None if first_direct_failure is None else first_direct_failure["name"],
        "block_results": block_results,
    }


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Case | Block | Direct same-input MLP | First full-flow failure | First direct failure |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        for block_idx in TARGET_BLOCKS:
            block_result = row["block_results"][str(block_idx)]
            lines.append(
                f"| {row['name']} | {block_idx} | "
                f"{block_result['direct_same_input_mlp']['overall_status']} | "
                f"{block_result['full_flow_first_failure']} | "
                f"{block_result['direct_same_input_mlp']['first_failure']} |"
            )
    return "\n".join(lines)


def write_markdown_report(report: dict, report_path: Path) -> None:
    warning_lines = report["warnings"] or ["none"]
    error_lines = report["errors"] or ["none"]
    runtime_notices = [
        line for line in report.get("captured_output", [])
        if line.startswith(("WARNING:", "E", "W"))
    ] or ["none"]
    tensor_names = "\n".join(f"- `{name}`" for name in report["checkpoint_keys_used"])

    body = f"""# Stage G-2 MLP Surgical Debug

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Target blocks: {report['target_blocks']}
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## Checkpoint Keys Used
{tensor_names}

## Direct Same-Input MLP Decision
{markdown_table(report['case_results'])}

## Interpretation
- Block 3 direct same-input MLP parity passed: {report['block3_direct_same_input_mlp_pass']}
- Block 4 direct same-input MLP parity passed: {report['block4_direct_same_input_mlp_pass']}
- First diverging subcomponent in full-flow comparisons: {report['first_full_flow_failure']}
- First diverging subcomponent in direct same-input comparisons: {report['first_direct_same_input_failure']}
- Supports numeric drift: {report['supports_numeric_drift']}
- Supports implementation bug: {report['supports_implementation_bug']}
- Code modification recommended: {report['code_modification_recommended']}
- Stage G-2 rerun recommendation: {report['stage_g2_rerun_recommendation']}

## Debug Tensor Summary
- Debug summary artifact: `{report['debug_tensors_summary_path']}`
- Captured components per target block: block input, norm1 output, attention output, after-attention residual, norm2 output, MLP linear1 output, GELU output, MLP linear2 output, and after-MLP residual.
- Direct same-input MLP tests use PyTorch-captured `norm2_output` as the input to both MLP implementations.

## Real Image Inputs
- Performed: {report['real_images_performed']}
- Image paths: {report['real_image_paths']}

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
            checkpoint_keys_used = []
            for block_idx in TARGET_BLOCKS:
                for suffix in [
                    "chunk.linear1.weight",
                    "chunk.linear1.bias",
                    "chunk.linear2.weight",
                    "chunk.linear2.bias",
                ]:
                    checkpoint_keys_used.append(f"stages.2.mlps.{block_idx}.{suffix}")

            DAT = import_dat_from_package_dir(export_safe_model_dir)
            model = DAT(**MODEL_KWARGS)
            strict_load = model.load_state_dict(state_dict, strict=True)
            model.eval()
            stage_pt = model.stages[2]
            stage_pt.eval()

            TFStage2 = import_tf_stage2(root)
            stage_tf = TFStage2(dim=512, heads=16, window_size=7, expansion=4, groups=4, epsilon=1e-5)
            stage_tf.build_for_input_shape((1, 14, 14, 512))
            stage_tf.load_from_pytorch_state_dict(state_dict, prefix="stages.2")

            random_stage_input = np.random.randn(1, 512, 14, 14).astype(np.float32)
            random_result = run_case(tf, torch, np, stage_pt, stage_tf, "deterministic random stage-2 input", random_stage_input)

            random_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
            random_captured_input = stage2_input_from_image(torch, model, random_image)
            captured_result = run_case(
                tf,
                torch,
                np,
                stage_pt,
                stage_tf,
                "captured stage-2 input from deterministic random image",
                random_captured_input,
            )

            real_image_nchw, real_image_paths = load_real_image_batch(root, limit=3)
            real_result = None
            real_captured_input = None
            if real_image_nchw is not None:
                real_captured_input = stage2_input_from_image(torch, model, real_image_nchw)
                real_result = run_case(
                    tf,
                    torch,
                    np,
                    stage_pt,
                    stage_tf,
                    "captured stage-2 input from 3 real ICAA17K images",
                    real_captured_input,
                )

            case_results = [random_result, captured_result]
            if real_result is not None:
                case_results.append(real_result)

            for row in case_results:
                for block_idx in TARGET_BLOCKS:
                    block_result = row["block_results"][str(block_idx)]
                    direct = block_result["direct_same_input_mlp"]
                    log(
                        f"{row['name']} block{block_idx}: "
                        f"direct_same_input_mlp={direct['overall_status']} "
                        f"full_flow_first_failure={block_result['full_flow_first_failure']} "
                        f"direct_first_failure={direct['first_failure']}"
                    )

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        block3_direct = all(row["block_results"]["3"]["direct_same_input_mlp"]["overall_pass"] for row in case_results)
        block4_direct = all(row["block_results"]["4"]["direct_same_input_mlp"]["overall_pass"] for row in case_results)
        all_direct = block3_direct and block4_direct

        first_full_flow_failure = None
        first_direct_failure = None
        for row in case_results:
            if first_full_flow_failure is None and row["first_full_flow_failure"] is not None:
                first_full_flow_failure = row["first_full_flow_failure"]
            if first_direct_failure is None and row["first_direct_same_input_failure"] is not None:
                first_direct_failure = row["first_direct_same_input_failure"]

        supports_numeric_drift = bool(all_direct and first_full_flow_failure is not None)
        supports_implementation_bug = bool(first_direct_failure is not None)
        code_modification_recommended = bool(supports_implementation_bug)
        stage_g2_rerun_recommendation = (
            "Do not change MLP implementation; investigate accumulated numeric drift or explicitly approve a relaxed full-stage gate before rerun."
            if supports_numeric_drift
            else "Investigate direct MLP implementation or mapping failure before rerunning Stage G-2."
        )

        debug_summary = {
            "random_stage_input": {
                "input": tensor_summary("random_stage_input_nchw", random_stage_input),
                "result": random_result,
            },
            "captured_random_image_stage2_input": {
                "input": tensor_summary("captured_random_image_stage2_input_nchw", random_captured_input),
                "result": captured_result,
            },
            "captured_real_image_stage2_input": None,
            "real_image_paths": real_image_paths,
        }
        if real_result is not None:
            debug_summary["captured_real_image_stage2_input"] = {
                "input": tensor_summary("captured_real_image_stage2_input_nchw", real_captured_input),
                "result": real_result,
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
            "target_blocks": TARGET_BLOCKS,
            "checkpoint_keys_used": checkpoint_keys_used,
            "case_results": case_results,
            "block3_direct_same_input_mlp_pass": block3_direct,
            "block4_direct_same_input_mlp_pass": block4_direct,
            "first_full_flow_failure": first_full_flow_failure,
            "first_direct_same_input_failure": first_direct_failure,
            "supports_numeric_drift": supports_numeric_drift,
            "supports_implementation_bug": supports_implementation_bug,
            "code_modification_recommended": code_modification_recommended,
            "stage_g2_rerun_recommendation": stage_g2_rerun_recommendation,
            "real_images_performed": real_result is not None,
            "real_image_paths": real_image_paths,
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
            "target_blocks": TARGET_BLOCKS,
            "checkpoint_keys_used": [],
            "case_results": [],
            "block3_direct_same_input_mlp_pass": False,
            "block4_direct_same_input_mlp_pass": False,
            "first_full_flow_failure": None,
            "first_direct_same_input_failure": None,
            "supports_numeric_drift": False,
            "supports_implementation_bug": False,
            "code_modification_recommended": False,
            "stage_g2_rerun_recommendation": "Diagnostic failed before producing a recommendation.",
            "real_images_performed": False,
            "real_image_paths": [],
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
    json_path = output_dir / "debug_stage_g2_block_mlp_report.json"
    log_path = output_dir / "debug_stage_g2_block_mlp_log.txt"
    debug_summary_path = output_dir / "optional_debug_tensors_summary.json"
    markdown_path = root / REPORT_REL_PATH

    report["json_path"] = str(json_path)
    report["log_path"] = str(log_path)
    report["debug_tensors_summary_path"] = str(debug_summary_path)
    captured_output = stdout_stderr.getvalue().splitlines()
    report["captured_output"] = captured_output

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    debug_summary_path.write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")
    write_markdown_report(report, markdown_path)
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
