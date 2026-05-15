# ICAA17K DAT 2번 TransformerStage의 PyTorch-TensorFlow 동등성을 검증하는 스크립트
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_g2_transformer_stage2_parity.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_g2"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_g2_transformer_stage2_parity.py"
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
    package_name = "icaa_export_safe_v3_export_safe_models_stage_g2"
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


def stage_torch_with_debug(torch, stage, stage_input):
    debug = {}
    with torch.no_grad():
        x = stage.proj(stage_input)
        debug["stage_input_after_proj"] = x.detach().cpu()
        positions = []
        references = []
        for block_idx in range(stage.depths):
            x0 = x
            norm_attn = stage.layer_norms[2 * block_idx](x)
            debug[f"block{block_idx}_norm_attn"] = norm_attn.detach().cpu()
            attn_out, pos, ref = stage.attns[block_idx](norm_attn)
            debug[f"block{block_idx}_attn_out"] = attn_out.detach().cpu()
            x = stage.drop_path[block_idx](attn_out) + x0
            debug[f"block{block_idx}_after_attn_residual"] = x.detach().cpu()

            x0 = x
            norm_mlp = stage.layer_norms[2 * block_idx + 1](x)
            debug[f"block{block_idx}_norm_mlp"] = norm_mlp.detach().cpu()
            mlp_out = stage.mlps[block_idx](norm_mlp)
            debug[f"block{block_idx}_mlp_out"] = mlp_out.detach().cpu()
            x = stage.drop_path[block_idx](mlp_out) + x0
            debug[f"block{block_idx}_output"] = x.detach().cpu()
            positions.append(pos.detach().cpu() if pos is not None else None)
            references.append(ref.detach().cpu() if ref is not None else None)
        debug["stage_output"] = x.detach().cpu()
    return x, positions, references, debug


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


def compare_dattention_bghw2(name: str, torch_tensor, tf_tensor, batch: int, groups: int) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_tensor)
    tf_np = tf_tensor.numpy().reshape(batch, groups, torch_np.shape[2], torch_np.shape[3], 2)
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
    }


def first_divergence(debug_diffs: list[dict], dattention_diffs: list[dict]) -> dict | None:
    by_name = {row["name"]: row for row in debug_diffs}
    dattn_by_name = {row["name"]: row for row in dattention_diffs}
    for block_idx in range(18):
        ordered_names = [
            f"block{block_idx}_norm_attn",
            f"block{block_idx}_dattention_pos",
            f"block{block_idx}_dattention_reference",
            f"block{block_idx}_attn_out",
            f"block{block_idx}_after_attn_residual",
            f"block{block_idx}_norm_mlp",
            f"block{block_idx}_mlp_out",
            f"block{block_idx}_output",
        ]
        for name in ordered_names:
            row = by_name.get(name) or dattn_by_name.get(name)
            if row is not None and row["max_abs_diff"] > ACCEPTABLE_MAX_ABS_DIFF:
                return row
    final_row = by_name.get("stage_output")
    if final_row is not None and final_row["max_abs_diff"] > ACCEPTABLE_MAX_ABS_DIFF:
        return final_row
    return None


def likely_failed_subcomponent(row: dict | None) -> str | None:
    if row is None:
        return None
    name = row["name"]
    if "_dattention_pos" in name:
        return f"{name.split('_')[0]} DAttention sampler / offset / gate"
    if "_dattention_reference" in name:
        return f"{name.split('_')[0]} DAttention reference projection"
    if "_norm_attn" in name or "_norm_mlp" in name:
        return f"{name.split('_')[0]} LayerNorm"
    if "_attn_out" in name:
        block_idx = int(name.split("_")[0].replace("block", ""))
        attn_type = "D" if block_idx % 2 else "L"
        return f"block {block_idx} attention type {attn_type}"
    if "_after_attn_residual" in name or "_output" in name:
        return f"{name.split('_')[0]} residual add, MLP, layout, or drop path"
    if "_mlp_out" in name:
        return f"{name.split('_')[0]} MLP"
    if name == "stage_output":
        return "final stage output, layout, or accumulated drift"
    return name


def run_case(torch, tf, np, stage_pt, stage_tf, name: str, stage_input_nchw):
    x_pt = torch.from_numpy(stage_input_nchw.astype(np.float32))
    x_tf = tf.convert_to_tensor(np.transpose(stage_input_nchw, (0, 2, 3, 1)).astype(np.float32), dtype=tf.float32)
    y_pt, positions, references, debug_pt = stage_torch_with_debug(torch, stage_pt, x_pt)
    y_tf, debug_tf = stage_tf.call_with_debug(x_tf, training=False)
    result = compare_nchw_to_nhwc(name, y_pt, y_tf)
    result["torch_input_shape"] = list(stage_input_nchw.shape)
    result["tf_input_shape_nhwc"] = list(x_tf.shape)
    result["positions_returned"] = [None if item is None else list(item.shape) for item in positions]
    result["references_returned"] = [None if item is None else list(item.shape) for item in references]

    debug_diffs = []
    for block_idx in range(stage_pt.depths):
        for key in [
            f"block{block_idx}_norm_attn",
            f"block{block_idx}_attn_out",
            f"block{block_idx}_after_attn_residual",
            f"block{block_idx}_norm_mlp",
            f"block{block_idx}_mlp_out",
            f"block{block_idx}_output",
        ]:
            debug_diffs.append(compare_nchw_to_nhwc(key, debug_pt[key], debug_tf[key]))
    debug_diffs.append(compare_nchw_to_nhwc("stage_output", debug_pt["stage_output"], debug_tf["stage_output"]))

    dattention_diffs = []
    batch = int(stage_input_nchw.shape[0])
    for block_idx in range(stage_pt.depths):
        if positions[block_idx] is None:
            continue
        dattention_diffs.append(
            compare_dattention_bghw2(
                f"block{block_idx}_dattention_pos",
                positions[block_idx],
                debug_tf[f"block{block_idx}_dattention_pos"],
                batch,
                stage_tf.groups,
            )
        )
        dattention_diffs.append(
            compare_dattention_bghw2(
                f"block{block_idx}_dattention_reference",
                references[block_idx],
                debug_tf[f"block{block_idx}_dattention_reference"],
                batch,
                stage_tf.groups,
            )
        )

    first_bad = first_divergence(debug_diffs, dattention_diffs)
    result["first_diverging_block"] = None if first_bad is None else first_bad["name"]
    result["likely_failed_subcomponent"] = likely_failed_subcomponent(first_bad)
    return result, debug_diffs, dattention_diffs


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


def result_line(result: dict) -> str:
    return (
        f"{result['name']}: status={result['status']} "
        f"max_abs_diff={result['max_abs_diff']:.9g} "
        f"mean_abs_diff={result['mean_abs_diff']:.9g} "
        f"torch_input_shape={result['torch_input_shape']} "
        f"torch_output_shape={result['torch_output_shape']} "
        f"tf_input_shape={result['tf_input_shape_nhwc']} "
        f"tf_output_shape={result['tf_output_shape_nhwc']} "
        f"first_diverging_block={result['first_diverging_block']}"
    )


def markdown_result_table(results: list[dict]) -> str:
    lines = [
        "| Test | Status | Max abs diff | Mean abs diff | PyTorch input | PyTorch output | TensorFlow input | TensorFlow output before transpose | First divergence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['status']} | {result['max_abs_diff']:.9g} | "
            f"{result['mean_abs_diff']:.9g} | {result['torch_input_shape']} | {result['torch_output_shape']} | "
            f"{result['tf_input_shape_nhwc']} | {result['tf_output_shape_nhwc']} | {result['first_diverging_block']} |"
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
    first_divergence_lines = [
        f"- {result['name']}: {result['first_diverging_block'] or 'none'}"
        for result in report["results"]
    ] or ["- none"]

    body = f"""# Stage G-2 TransformerStage 2 Parity

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## PyTorch Classes Inspected
```json
{json.dumps(report['pytorch_stage_inspection'], indent=2)}
```

## Checkpoint Tensors Used
{tensor_names}

## Stage 2 Shapes
- PyTorch input shape: {report['stage_input_shape_nchw']}
- PyTorch output shape: {report['stage_output_shape_nchw']}
- TensorFlow input shape: {report['stage_input_shape_nhwc']}
- TensorFlow output shape: {report['stage_output_shape_nhwc']}

## TensorFlow Implementation Summary
```json
{json.dumps(report['tf_stage_structure'], indent=2)}
```

## Weight Mapping Rules Used
- Stage projection: PyTorch `Identity`, so no projection tensor is mapped.
- LocalAttention linear kernels: PyTorch `[out, in]` -> TensorFlow Dense `[in, out]`.
- LocalAttention relative position bias table and index: direct copy / integer gather indices.
- DAttention 1x1 Conv2D projections: PyTorch `[out_c, in_c, 1, 1]` -> explicit NHWC matrix multiply in `PytorchLikeConv1x1`.
- DAttention depthwise offset Conv2D: PyTorch `[channel, 1, kH, kW]` -> TensorFlow `[kH, kW, channel, 1]`.
- DAttention `rpe_table`: direct copy as `[heads, 2H-1, 2W-1]`.
- LayerNorm gamma/beta and all biases: direct copy.
- GELU: TensorFlow exact GELU to match PyTorch default `nn.GELU()`.
- Layout: PyTorch NCHW input/output, TensorFlow NHWC input/output; TensorFlow outputs are transposed back to NCHW before comparison.

## Parity Results
{markdown_result_table(report['results'])}

## First Diverging Block
""" + "\n".join(first_divergence_lines) + f"""

## Debug Tensor Summary
- Debug summary artifact: `{report['debug_tensors_summary_path']}`
- Per-block diffs are recorded for attention output, after-attention residual, MLP output, after-MLP residual, and final stage output.
- DAttention `pos` and `reference` diffs are recorded for each deformable block.

## Real Image Inputs
- Performed: {report['real_images_performed']}
- Image paths: {report['real_image_paths']}

## Unresolved Issues
""" + "\n".join(f"- {line}" for line in unresolved) + f"""

## Stage G-3 TransformerStage 3 Decision
- Safe to proceed to Stage G-3 TransformerStage 3 parity: {report['safe_to_proceed_to_stage_g3_transformerstage3_parity']}
- Scope of this decision: `TransformerStage` stage `2` only. This does not establish full TensorFlow DAT feasibility.

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
    checkpoint_tensors_used: list[str] = []

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
            checkpoint_tensors_used = [name for name in state_dict.keys() if name.startswith("stages.2.")]

            DAT = import_dat_from_package_dir(export_safe_model_dir)
            model = DAT(**MODEL_KWARGS)
            strict_load = model.load_state_dict(state_dict, strict=True)
            model.eval()
            stage = model.stages[2]
            stage.eval()

            TFStage2 = import_tf_stage2(root)
            stage_tf = TFStage2(dim=512, heads=16, window_size=7, expansion=4, groups=4, epsilon=1e-5)
            stage_tf.build_for_input_shape((1, 14, 14, 512))
            stage_tf.load_from_pytorch_state_dict(state_dict, prefix="stages.2")

            pytorch_stage_inspection = {
                "class_name": stage.__class__.__name__,
                "stage_index": 2,
                "stage_input_shape_nchw": [None, 512, 14, 14],
                "stage_output_shape_nchw": [None, 512, 14, 14],
                "depth": int(stage.depths),
                "stage_spec": MODEL_KWARGS["stage_spec"][2],
                "proj_class": stage.proj.__class__.__name__,
                "attention_blocks": [
                    {
                        "index": idx,
                        "class_name": attn.__class__.__name__,
                        "attention_type": MODEL_KWARGS["stage_spec"][2][idx],
                        "window_size": None if not hasattr(attn, "window_size") else list(attn.window_size),
                        "heads": int(attn.heads if hasattr(attn, "heads") else attn.n_heads),
                        "scale": float(attn.scale),
                    }
                    for idx, attn in enumerate(stage.attns)
                ],
                "deformable_attention_details": {
                    "q_size": [int(stage.attns[1].q_h), int(stage.attns[1].q_w)],
                    "kv_size": [int(stage.attns[1].kv_h), int(stage.attns[1].kv_w)],
                    "n_heads": int(stage.attns[1].n_heads),
                    "n_head_channels": int(stage.attns[1].n_head_channels),
                    "n_groups": int(stage.attns[1].n_groups),
                    "offset_range_factor": int(stage.attns[1].offset_range_factor),
                    "use_pe": bool(stage.attns[1].use_pe),
                    "dwc_pe": bool(stage.attns[1].dwc_pe),
                    "fixed_pe": bool(stage.attns[1].fixed_pe),
                    "no_off": bool(stage.attns[1].no_off),
                    "rpe_table_shape": list(stage.attns[1].rpe_table.shape),
                },
                "mlp_blocks": [
                    {
                        "index": idx,
                        "class_name": mlp.__class__.__name__,
                        "hidden_dim": int(mlp.dim2),
                    }
                    for idx, mlp in enumerate(stage.mlps)
                ],
                "layer_norms": [
                    {
                        "index": idx,
                        "class_name": norm.__class__.__name__,
                        "inner_class_name": norm.norm.__class__.__name__,
                        "normalized_shape": list(norm.norm.normalized_shape),
                        "eps": float(norm.norm.eps),
                    }
                    for idx, norm in enumerate(stage.layer_norms)
                ],
                "drop_path": [
                    {
                        "index": idx,
                        "class_name": drop.__class__.__name__,
                        "drop_prob": None if not hasattr(drop, "drop_prob") else float(drop.drop_prob),
                    }
                    for idx, drop in enumerate(stage.drop_path)
                ],
                "drop_path_eval_behavior": "All stage-2 drop_path modules are DropPath with nonzero configured probabilities, but eval mode disables them and makes them identity.",
                "residual_order": [
                    "x = proj(x)",
                    "x0 = x",
                    "attn_out = attns[d](layer_norms[2*d](x))",
                    "x = drop_path[d](attn_out) + x0",
                    "x0 = x",
                    "mlp_out = mlps[d](layer_norms[2*d+1](x))",
                    "x = drop_path[d](mlp_out) + x0",
                ],
                "down_projection_relationship": {
                    "is_part_of_stage2": False,
                    "dat_forward_order": "DAT.forward calls model.stages[2](x), then applies model.down_projs[2](x) outside the stage when i < 3.",
                    "down_proj2_class": model.down_projs[2].__class__.__name__,
                },
            }

            random_stage_input = np.random.randn(1, 512, 14, 14).astype(np.float32)
            random_result, random_debug, random_dattention = run_case(
                torch, tf, np, stage, stage_tf, "deterministic random stage-2 input", random_stage_input
            )

            random_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
            random_captured_input = stage2_input_from_image(torch, model, random_image)
            captured_result, captured_debug, captured_dattention = run_case(
                torch,
                tf,
                np,
                stage,
                stage_tf,
                "captured stage-2 input from deterministic random image",
                random_captured_input,
            )

            real_image_nchw, real_image_paths = load_real_image_batch(root, limit=3)
            real_result = None
            real_debug = None
            real_dattention = None
            real_captured_input = None
            if real_image_nchw is not None:
                real_captured_input = stage2_input_from_image(torch, model, real_image_nchw)
                real_result, real_debug, real_dattention = run_case(
                    torch,
                    tf,
                    np,
                    stage,
                    stage_tf,
                    "captured stage-2 input from 3 real ICAA17K images",
                    real_captured_input,
                )

            results = [random_result, captured_result]
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
                    f"{result['name']} failed; first diverging block: {result.get('first_diverging_block')}; likely subcomponent: {result.get('likely_failed_subcomponent') or 'unknown'}"
                )

        safe_to_stage_g3 = bool(all(result["pass"] for result in results))
        stage_input_shape_nchw = list(random_stage_input.shape)
        stage_input_shape_nhwc = [
            random_stage_input.shape[0],
            random_stage_input.shape[2],
            random_stage_input.shape[3],
            random_stage_input.shape[1],
        ]
        stage_output_shape_nchw = random_result["torch_output_shape"]
        stage_output_shape_nhwc = random_result["tf_output_shape_nhwc"]

        debug_summary = {
            "random_stage_input": {
                "input": tensor_summary("random_stage_input_nchw", random_stage_input),
                "output_result": random_result,
                "per_block_diffs": random_debug,
                "dattention_position_reference_diffs": random_dattention,
            },
            "captured_random_image_stage2_input": {
                "input": tensor_summary("captured_random_image_stage2_input_nchw", random_captured_input),
                "output_result": captured_result,
                "per_block_diffs": captured_debug,
                "dattention_position_reference_diffs": captured_dattention,
            },
            "captured_real_image_stage2_input": None,
            "real_image_paths": real_image_paths,
        }
        if real_result is not None:
            debug_summary["captured_real_image_stage2_input"] = {
                "input": tensor_summary("captured_real_image_stage2_input_nchw", real_captured_input),
                "output_result": real_result,
                "per_block_diffs": real_debug,
                "dattention_position_reference_diffs": real_dattention,
            }

        report = {
            "status": "ok" if safe_to_stage_g3 else "parity_failed",
            "overall_pass": safe_to_stage_g3,
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
            "pytorch_stage_inspection": pytorch_stage_inspection,
            "checkpoint_tensors_used": checkpoint_tensors_used,
            "tf_stage_structure": stage_tf.structure_summary(),
            "test_input_generation": {
                "random_stage_input": {"distribution": "numpy randn", "seed": SEED},
                "captured_random_image_stage2_input": {
                    "distribution": "numpy randn image passed through PyTorch patch_proj, stage 0, down_projs[0], stage 1, and down_projs[1]",
                    "seed": SEED,
                },
                "captured_real_image_stage2_input": {
                    "source": CSV_REL_PATH,
                    "preprocessing": "PIL bilinear resize to 224x224, RGB [0,1], ImageNet mean/std normalization, then PyTorch patch_proj, stage 0, down_projs[0], stage 1, and down_projs[1]",
                },
            },
            "stage_input_shape_nchw": stage_input_shape_nchw,
            "stage_input_shape_nhwc": stage_input_shape_nhwc,
            "stage_output_shape_nchw": stage_output_shape_nchw,
            "stage_output_shape_nhwc": stage_output_shape_nhwc,
            "results": results,
            "real_images_performed": real_result is not None,
            "real_image_paths": real_image_paths,
            "safe_to_proceed_to_stage_g3_transformerstage3_parity": safe_to_stage_g3,
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
            "overall_pass": False,
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
            "pytorch_stage_inspection": None,
            "checkpoint_tensors_used": checkpoint_tensors_used,
            "tf_stage_structure": None,
            "test_input_generation": None,
            "stage_input_shape_nchw": None,
            "stage_input_shape_nhwc": None,
            "stage_output_shape_nchw": None,
            "stage_output_shape_nhwc": None,
            "results": [],
            "real_images_performed": False,
            "real_image_paths": [],
            "safe_to_proceed_to_stage_g3_transformerstage3_parity": False,
            "unresolved_issues": ["Stage G-2 script failed before completing TransformerStage 2 parity."],
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
    json_path = output_dir / "stage_g2_transformer_stage2_report.json"
    log_path = output_dir / "stage_g2_transformer_stage2_log.txt"
    debug_summary_path = output_dir / "optional_debug_tensors_summary.json"
    markdown_path = root / REPORT_REL_PATH

    report["json_path"] = str(json_path)
    report["log_path"] = str(log_path)
    report["debug_tensors_summary_path"] = str(debug_summary_path)
    captured_output = stdout_stderr.getvalue().splitlines()
    report["captured_output"] = captured_output

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    debug_summary_path.write_text(json.dumps(debug_summary, indent=2), encoding="utf-8")
    if report["status"] in {"ok", "parity_failed"}:
        write_markdown_report(report, markdown_path)
    else:
        markdown_path.write_text(
            "# Stage G-2 TransformerStage 2 Parity\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Stage G-2 failed before TransformerStage 2 parity completed. See JSON and log artifacts for details.\n",
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
