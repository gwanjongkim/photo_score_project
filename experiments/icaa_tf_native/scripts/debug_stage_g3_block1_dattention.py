# ICAA17K DAT 3번 TransformerStage 블록 1 DAttention 원인 분리를 위한 진단 스크립트
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_g3_block1_dattention_surgical_debug.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_g3_dattention_debug"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/debug_stage_g3_block1_dattention.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
PREFERRED_MAX_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-4
TARGET_STAGE = 3
TARGET_BLOCK = 1
TARGET_PREFIX = "stages.3.attns.1"

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


def import_dat_package(package_dir: Path):
    package_name = "icaa_export_safe_v3_export_safe_models_g3_dattention_debug"
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
    dat_blocks = importlib.import_module(f"{package_name}.dat_blocks")
    return dat_module.DAT, dat_blocks


def import_tf_stage3(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.transformer_stage import TFTransformerStage3

    return TFTransformerStage3


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


def status_for(max_abs_diff: float) -> tuple[str, bool, bool]:
    preferred_pass = max_abs_diff <= PREFERRED_MAX_ABS_DIFF
    acceptable_pass = max_abs_diff <= ACCEPTABLE_MAX_ABS_DIFF
    status = "pass_preferred" if preferred_pass else "pass_acceptable" if acceptable_pass else "fail"
    return status, bool(acceptable_pass), bool(preferred_pass)


def compare_same_layout(name: str, torch_tensor, tf_tensor_or_array) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_tensor) if hasattr(torch_tensor, "detach") else np.asarray(torch_tensor)
    tf_np = tf_tensor_or_array.numpy() if hasattr(tf_tensor_or_array, "numpy") else np.asarray(tf_tensor_or_array)
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    status, acceptable_pass, preferred_pass = status_for(max_abs_diff)
    return {
        "name": name,
        "status": status,
        "pass": acceptable_pass,
        "preferred_pass": preferred_pass,
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
    status, acceptable_pass, preferred_pass = status_for(max_abs_diff)
    return {
        "name": name,
        "status": status,
        "pass": acceptable_pass,
        "preferred_pass": preferred_pass,
        "acceptable_pass": acceptable_pass,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "torch_shape": list(torch_np.shape),
        "tf_shape_nhwc": list(tf_nhwc_np.shape),
        "tf_shape_nchw": list(tf_nchw_np.shape),
    }


def capture_stage3_block1_norm_attn(torch, stage, stage_input_nchw):
    x = torch.from_numpy(stage_input_nchw.astype("float32"))
    with torch.no_grad():
        x = stage.proj(x)
        block0_input = x
        norm0 = stage.layer_norms[0](x)
        attn0, _pos0, _ref0 = stage.attns[0](norm0)
        x = stage.drop_path[0](attn0) + block0_input
        block0_mlp_input = x
        norm_mlp0 = stage.layer_norms[1](x)
        mlp0 = stage.mlps[0](norm_mlp0)
        block1_input = stage.drop_path[0](mlp0) + block0_mlp_input
        block1_norm_attn = stage.layer_norms[2](block1_input)
    return {
        "block1_input": block1_input.detach().cpu(),
        "block1_norm_attn": block1_norm_attn.detach().cpu(),
    }


def torch_dattention_with_debug(torch, torch_nn_functional, dat_blocks, attn, x):
    with torch.no_grad():
        b, c, height, width = x.size()
        dtype = x.dtype
        device = x.device

        q = attn.proj_q(x)
        q_off = q.reshape(b, attn.n_groups, attn.n_group_channels, height, width)
        q_off = q_off.reshape(b * attn.n_groups, attn.n_group_channels, height, width)
        offset_raw = attn.conv_offset[0](q_off)
        offset_norm = attn.conv_offset[1](offset_raw)
        offset_gelu = attn.conv_offset[2](offset_norm)
        offset_conv = attn.conv_offset[3](offset_gelu)
        h_key = offset_conv.size(2)
        w_key = offset_conv.size(3)
        n_sample = h_key * w_key

        if attn.offset_range_factor > 0:
            h_inv = 1.0 / torch.as_tensor(h_key, device=device, dtype=dtype)
            w_inv = 1.0 / torch.as_tensor(w_key, device=device, dtype=dtype)
            offset_range = torch.stack([h_inv, w_inv]).view(1, 2, 1, 1)
            offset_conv = offset_conv.tanh().mul(offset_range).mul(attn.offset_range_factor)

        offset = offset_conv.permute(0, 2, 3, 1)
        referencek = attn.ref_point14(x).permute(0, 2, 3, 1).tanh()
        reference = referencek.repeat_interleave(attn.n_groups, dim=0)
        gate = (attn.ref_gate(x).permute(0, 2, 3, 1) * -99999).sigmoid()
        gate = gate.repeat_interleave(attn.n_groups, dim=0)

        if attn.no_off:
            offset = offset.fill(0.0)

        offset_x, offset_y = torch.chunk(offset, 2, dim=3)
        reference_x, reference_y = torch.chunk(reference, 2, dim=3)
        temp_x = offset_x * reference_x
        temp_y = offset_y * reference_y
        temp_x = torch.where(temp_x <= 0.0, torch.zeros_like(temp_x), torch.ones_like(temp_x))
        temp_y = torch.where(temp_y <= 0.0, torch.zeros_like(temp_y), torch.ones_like(temp_y))
        offset_temp = temp_x * temp_y
        manu_offset = torch.where(
            offset_temp <= 0.0,
            torch.full_like(offset_temp, 0.25),
            torch.ones_like(offset_temp),
        )
        offset = offset * torch.cat((manu_offset, manu_offset), 3)

        if attn.offset_range_factor >= 0:
            pos = torch.mul((offset + reference), gate)
        else:
            pos = torch.mul((offset + reference), gate).tanh()

        pos_y, pos_x = torch.chunk(pos, 2, dim=-1)
        grid = torch.cat([pos_x, pos_y], dim=-1)

        x_grouped = x.reshape(b, attn.n_groups, attn.n_group_channels, height, width)
        x_grouped = x_grouped.reshape(b * attn.n_groups, attn.n_group_channels, height, width)
        x_sampled_grouped = dat_blocks.manual_bilinear_grid_sample(x_grouped, grid)
        x_sampled_for_conv = x_sampled_grouped.reshape(b, c, 1, n_sample)

        q_heads = q.reshape(b * attn.n_heads, attn.n_head_channels, height * width)
        k_projected = attn.proj_k(x_sampled_for_conv)
        v_projected = attn.proj_v(x_sampled_for_conv)
        k_heads = k_projected.reshape(b * attn.n_heads, attn.n_head_channels, n_sample)
        v_heads = v_projected.reshape(b * attn.n_heads, attn.n_head_channels, n_sample)

        logits_before_bias = torch.einsum("b c m, b c n -> b m n", q_heads, k_heads).mul(attn.scale)

        rpe_table = attn.rpe_table
        rpe_bias = rpe_table[None, ...].expand(b, -1, -1, -1)
        q_grid = attn._get_ref_points(height, width, b, dtype, device)
        displacement = (
            q_grid.reshape(b * attn.n_groups, height * width, 2).unsqueeze(2)
            - pos.reshape(b * attn.n_groups, n_sample, 2).unsqueeze(1)
        ).mul(0.5)
        disp_y, disp_x = torch.chunk(displacement, 2, dim=-1)
        disp_grid = torch.cat([disp_x, disp_y], dim=-1)
        attn_bias_grouped = dat_blocks.manual_bilinear_grid_sample(
            rpe_bias.reshape(b * attn.n_groups, attn.n_group_heads, 2 * height - 1, 2 * width - 1),
            disp_grid,
        )
        attn_bias = attn_bias_grouped.reshape(b * attn.n_heads, height * width, n_sample)
        logits_after_bias = logits_before_bias + attn_bias
        softmax_attention = torch_nn_functional.softmax(logits_after_bias, dim=2)
        weighted_attention_heads = torch.einsum("b m n, b c n -> b c m", softmax_attention, v_heads)
        weighted_attention_output = weighted_attention_heads.reshape(b, c, height, width)
        output_projection = attn.proj_out(weighted_attention_output)
        final_output = attn.proj_drop(output_projection)

        module_output, module_pos, module_reference = attn(x)

    return {
        "input": x.detach().cpu(),
        "q": q.detach().cpu(),
        "q_off": q_off.detach().cpu(),
        "offset_raw": offset_raw.detach().cpu(),
        "offset_norm": offset_norm.detach().cpu(),
        "offset_gelu": offset_gelu.detach().cpu(),
        "offset": offset.detach().cpu(),
        "reference": reference.detach().cpu(),
        "gate": gate.detach().cpu(),
        "pos": pos.detach().cpu(),
        "grid": grid.detach().cpu(),
        "sampled_feature": x_sampled_grouped.detach().cpu(),
        "sampled_feature_for_projection": x_sampled_for_conv.detach().cpu(),
        "k": k_heads.detach().cpu(),
        "v": v_heads.detach().cpu(),
        "attention_logits_before_bias": logits_before_bias.detach().cpu(),
        "relative_position_bias": attn_bias.detach().cpu(),
        "attention_logits_after_bias": logits_after_bias.detach().cpu(),
        "softmax_attention": softmax_attention.detach().cpu(),
        "weighted_attention_heads": weighted_attention_heads.detach().cpu(),
        "weighted_attention_output_before_projection": weighted_attention_output.detach().cpu(),
        "output_projection": output_projection.detach().cpu(),
        "final_dattention_output": final_output.detach().cpu(),
        "module_output": module_output.detach().cpu(),
        "module_pos": module_pos.detach().cpu(),
        "module_reference": module_reference.detach().cpu(),
    }


def tf_dattention_with_debug(tf, attn, x_nhwc):
    output, debug = attn.call_with_debug(x_nhwc, training=False)
    logits_before_bias = tf.einsum("bcm,bcn->bmn", debug["q_heads"], debug["k_heads"]) * tf.cast(attn.scale, output.dtype)
    result = {
        "input": x_nhwc,
        "q": debug["q"],
        "q_off": debug["q_off"],
        "offset_raw": debug["offset_raw"],
        "offset_norm": debug["offset_norm"],
        "offset_gelu": debug["offset_gelu"],
        "offset": debug["offset"],
        "reference": debug["reference"],
        "gate": debug["gate"],
        "pos": debug["pos"],
        "grid": debug["grid"],
        "sampled_feature": debug["x_sampled_grouped"],
        "sampled_feature_for_projection": debug["x_sampled_for_conv"],
        "k": debug["k_heads"],
        "v": debug["v_heads"],
        "attention_logits_before_bias": logits_before_bias,
        "relative_position_bias": debug["attn_bias"],
        "attention_logits_after_bias": debug["attn_logits"],
        "softmax_attention": debug["attn_softmax"],
        "weighted_attention_heads": debug["out_heads"],
        "weighted_attention_output_before_projection": debug["out_map"],
        "output_projection": debug["output"],
        "final_dattention_output": output,
    }
    return result


def compare_dattention_debug(torch_debug: dict, tf_debug: dict) -> list[dict]:
    return [
        compare_nchw_to_nhwc("block1_dattention_input", torch_debug["input"], tf_debug["input"]),
        compare_nchw_to_nhwc("q", torch_debug["q"], tf_debug["q"]),
        compare_same_layout("offset", torch_debug["offset"], tf_debug["offset"]),
        compare_same_layout("pos", torch_debug["pos"], tf_debug["pos"]),
        compare_same_layout("reference", torch_debug["reference"], tf_debug["reference"]),
        compare_nchw_to_nhwc("sampled_feature", torch_debug["sampled_feature"], tf_debug["sampled_feature"]),
        compare_same_layout("k", torch_debug["k"], tf_debug["k"]),
        compare_same_layout("v", torch_debug["v"], tf_debug["v"]),
        compare_same_layout("attention_logits_before_bias", torch_debug["attention_logits_before_bias"], tf_debug["attention_logits_before_bias"]),
        compare_same_layout("relative_position_bias", torch_debug["relative_position_bias"], tf_debug["relative_position_bias"]),
        compare_same_layout("attention_logits_after_bias", torch_debug["attention_logits_after_bias"], tf_debug["attention_logits_after_bias"]),
        compare_same_layout("softmax_attention", torch_debug["softmax_attention"], tf_debug["softmax_attention"]),
        compare_nchw_to_nhwc(
            "weighted_attention_output_before_projection",
            torch_debug["weighted_attention_output_before_projection"],
            tf_debug["weighted_attention_output_before_projection"],
        ),
        compare_nchw_to_nhwc("output_projection", torch_debug["output_projection"], tf_debug["output_projection"]),
        compare_nchw_to_nhwc("final_dattention_output", torch_debug["final_dattention_output"], tf_debug["final_dattention_output"]),
    ]


def first_failure(comparisons: list[dict]) -> dict | None:
    for row in comparisons:
        if row["max_abs_diff"] > ACCEPTABLE_MAX_ABS_DIFF:
            return row
    return None


def likely_component(row: dict | None) -> str | None:
    if row is None:
        return None
    name = row["name"]
    if name in {"block1_dattention_input", "q"}:
        return "input projection or layout"
    if name == "offset":
        return "offset branch, offset range, manual offset gate, or layout"
    if name in {"pos", "reference"}:
        return "reference projection, gate, offset application, or coordinate layout"
    if name == "sampled_feature":
        return "bilinear sampler or grid coordinate ordering"
    if name in {"k", "v"}:
        return "sampled feature projection or 1x1 weight mapping"
    if name == "attention_logits_before_bias":
        return "q/k matmul accumulation, scale, or head layout"
    if name == "relative_position_bias":
        return "RPE table reshape, RPE sampler, or displacement grid"
    if name == "attention_logits_after_bias":
        return "attention logits plus relative position bias"
    if name == "softmax_attention":
        return "softmax axis or accumulated logits drift"
    if name == "weighted_attention_output_before_projection":
        return "attention/value weighted sum or head-to-map layout"
    if name in {"output_projection", "final_dattention_output"}:
        return "output projection, final layout, or accumulated DAttention drift"
    return name


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


def stage3_input_from_image(torch, model, image_nchw):
    with torch.no_grad():
        x = model.patch_proj(torch.from_numpy(image_nchw.astype("float32")))
        x, _pos, _ref = model.stages[0](x)
        x = model.down_projs[0](x)
        x, _pos, _ref = model.stages[1](x)
        x = model.down_projs[1](x)
        x, _pos, _ref = model.stages[2](x)
        x = model.down_projs[2](x)
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


def run_case(torch, torch_nn_functional, tf, np, dat_blocks, stage_pt, stage_tf, name: str, stage_input_nchw):
    captures = capture_stage3_block1_norm_attn(torch, stage_pt, stage_input_nchw)
    norm_attn_nchw = captures["block1_norm_attn"]
    norm_attn_tf = tf.convert_to_tensor(
        np.transpose(norm_attn_nchw.numpy(), (0, 2, 3, 1)).astype(np.float32),
        dtype=tf.float32,
    )
    attn_pt = stage_pt.attns[TARGET_BLOCK]
    attn_tf = stage_tf.attns[TARGET_BLOCK]
    torch_debug = torch_dattention_with_debug(torch, torch_nn_functional, dat_blocks, attn_pt, norm_attn_nchw)
    tf_debug = tf_dattention_with_debug(tf, attn_tf, norm_attn_tf)
    comparisons = compare_dattention_debug(torch_debug, tf_debug)
    manual_vs_module = compare_same_layout(
        "pytorch_manual_final_vs_module_final",
        torch_debug["final_dattention_output"],
        torch_debug["module_output"],
    )
    first_bad = first_failure(comparisons)
    final_row = comparisons[-1]
    return {
        "name": name,
        "stage_input_shape_nchw": list(stage_input_nchw.shape),
        "direct_input_shape_nchw": list(norm_attn_nchw.shape),
        "direct_input_shape_nhwc": list(norm_attn_tf.shape),
        "direct_same_input_dattention_pass": bool(final_row["acceptable_pass"]),
        "direct_same_input_dattention_preferred_pass": bool(final_row["preferred_pass"]),
        "final_output_max_abs_diff": final_row["max_abs_diff"],
        "final_output_mean_abs_diff": final_row["mean_abs_diff"],
        "first_diverging_subcomponent": None if first_bad is None else first_bad["name"],
        "likely_failed_subcomponent": likely_component(first_bad),
        "manual_pytorch_reconstruction": manual_vs_module,
        "comparisons": comparisons,
    }, {
        "stage_input": tensor_summary(f"{name}_stage_input_nchw", stage_input_nchw),
        "block1_input": tensor_summary(f"{name}_block1_input_nchw", captures["block1_input"]),
        "direct_input": tensor_summary(f"{name}_block1_norm_attn_nchw", norm_attn_nchw),
        "manual_pytorch_reconstruction": manual_vs_module,
        "comparisons": comparisons,
    }


def markdown_case_table(case_results: list[dict]) -> str:
    lines = [
        "| Case | Direct same-input DAttention | Max abs diff | Mean abs diff | First diverging subcomponent | Likely component |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in case_results:
        lines.append(
            f"| {case['name']} | {case['direct_same_input_dattention_pass']} | "
            f"{case['final_output_max_abs_diff']:.9g} | {case['final_output_mean_abs_diff']:.9g} | "
            f"{case['first_diverging_subcomponent']} | {case['likely_failed_subcomponent']} |"
        )
    return "\n".join(lines)


def markdown_first_failure_rows(case_results: list[dict]) -> str:
    lines = []
    for case in case_results:
        first_name = case["first_diverging_subcomponent"]
        if first_name is None:
            lines.append(f"- {case['name']}: none")
            continue
        first_row = next(row for row in case["comparisons"] if row["name"] == first_name)
        lines.append(
            f"- {case['name']}: `{first_name}` max_abs_diff={first_row['max_abs_diff']:.9g}, "
            f"mean_abs_diff={first_row['mean_abs_diff']:.9g}"
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
    body = f"""# Stage G-3 Block1 DAttention Surgical Debug

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Exact block tested: Stage `{TARGET_STAGE}` block `{TARGET_BLOCK}`
- PyTorch module: `{TARGET_PREFIX}`
- TensorFlow module: `TFTransformerStage3.attns[1]` / `TFDAttentionBaseline`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## Checkpoint Keys Used
{tensor_names}

## Direct Same-Input DAttention Decision
{markdown_case_table(report['case_results'])}

## First Diverging Subcomponent
{markdown_first_failure_rows(report['case_results'])}

## Interpretation
- Direct same-input DAttention parity passed for all cases: {report['all_direct_same_input_dattention_pass']}
- Supports numeric drift: {report['supports_numeric_drift']}
- Supports implementation bug: {report['supports_implementation_bug']}
- Code modification recommended: {report['code_modification_recommended']}
- Full DAT assembly should remain blocked: {report['full_dat_assembly_should_remain_blocked']}
- Decision note: {report['decision_note']}

## Captured Subcomponents
- block1 DAttention input
- q
- offset
- pos
- reference
- sampled feature
- k
- v
- attention logits before relative position bias
- relative position bias / sampled RPE bias
- attention logits after bias
- softmax attention
- weighted attention output before projection
- output projection
- final DAttention output

## Debug Tensor Summary
- Debug summary artifact: `{report['debug_tensors_summary_path']}`
- PyTorch manual reconstruction of DAttention is compared against the PyTorch module output for each case.

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
    checkpoint_keys_used: list[str] = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            import numpy as np
            import tensorflow as tf
            import torch
            import torch.nn.functional as torch_nn_functional

            torch.manual_seed(SEED)
            np.random.seed(SEED)
            tf.random.set_seed(SEED)

            checkpoint_obj = load_checkpoint(torch, checkpoint_path)
            state_dict, state_dict_confirmation = extract_state_dict(checkpoint_obj)
            checkpoint_keys_used = [name for name in state_dict.keys() if name.startswith(f"{TARGET_PREFIX}.")]

            DAT, dat_blocks = import_dat_package(export_safe_model_dir)
            model = DAT(**MODEL_KWARGS)
            strict_load = model.load_state_dict(state_dict, strict=True)
            model.eval()
            stage_pt = model.stages[TARGET_STAGE]
            stage_pt.eval()

            TFStage3 = import_tf_stage3(root)
            stage_tf = TFStage3(dim=1024, heads=32, window_size=7, expansion=4, groups=8, epsilon=1e-5)
            stage_tf.build_for_input_shape((1, 7, 7, 1024))
            stage_tf.load_from_pytorch_state_dict(state_dict, prefix="stages.3")

            random_stage_input = np.random.randn(1, 1024, 7, 7).astype(np.float32)
            random_case, random_debug = run_case(
                torch,
                torch_nn_functional,
                tf,
                np,
                dat_blocks,
                stage_pt,
                stage_tf,
                "deterministic random stage-3 input",
                random_stage_input,
            )

            random_image = np.random.randn(1, 3, 224, 224).astype(np.float32)
            random_captured_input = stage3_input_from_image(torch, model, random_image)
            captured_case, captured_debug = run_case(
                torch,
                torch_nn_functional,
                tf,
                np,
                dat_blocks,
                stage_pt,
                stage_tf,
                "captured stage-3 input from deterministic random image",
                random_captured_input,
            )

            real_image_nchw, real_image_paths = load_real_image_batch(root, limit=3)
            real_case = None
            real_debug = None
            real_captured_input = None
            if real_image_nchw is not None:
                real_captured_input = stage3_input_from_image(torch, model, real_image_nchw)
                real_case, real_debug = run_case(
                    torch,
                    torch_nn_functional,
                    tf,
                    np,
                    dat_blocks,
                    stage_pt,
                    stage_tf,
                    "captured stage-3 input from 3 real ICAA17K images",
                    real_captured_input,
                )

            case_results = [random_case, captured_case]
            if real_case is not None:
                case_results.append(real_case)

            for case in case_results:
                log(
                    f"{case['name']}: direct_same_input_dattention={case['direct_same_input_dattention_pass']} "
                    f"max_abs_diff={case['final_output_max_abs_diff']:.9g} "
                    f"mean_abs_diff={case['final_output_mean_abs_diff']:.9g} "
                    f"first_diverging_subcomponent={case['first_diverging_subcomponent']}"
                )

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        all_direct_pass = bool(all(case["direct_same_input_dattention_pass"] for case in case_results))
        any_direct_fail = not all_direct_pass
        supports_numeric_drift = all_direct_pass
        supports_implementation_bug = any_direct_fail
        code_modification_recommended = False
        if any_direct_fail:
            decision_note = (
                "Direct same-input DAttention failed; this supports a DAttention implementation, sampler, RPE, "
                "projection, layout, or weight-mapping issue. No speculative fix was applied."
            )
        else:
            decision_note = (
                "Direct same-input DAttention passed; this supports accumulated numeric drift before or around Stage G-3 "
                "rather than a structural Stage 3 block1 DAttention implementation bug."
            )

        debug_summary = {
            "random_stage_input": random_debug,
            "captured_random_image_stage3_input": captured_debug,
            "captured_real_image_stage3_input": real_debug,
            "real_image_paths": real_image_paths,
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
            "target": {
                "stage": TARGET_STAGE,
                "block": TARGET_BLOCK,
                "pytorch_module": TARGET_PREFIX,
                "tensorflow_module": "TFTransformerStage3.attns[1]",
            },
            "checkpoint_keys_used": checkpoint_keys_used,
            "case_results": case_results,
            "all_direct_same_input_dattention_pass": all_direct_pass,
            "supports_numeric_drift": supports_numeric_drift,
            "supports_implementation_bug": supports_implementation_bug,
            "code_modification_recommended": code_modification_recommended,
            "full_dat_assembly_should_remain_blocked": True,
            "decision_note": decision_note,
            "real_images_performed": real_case is not None,
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
            "target": {
                "stage": TARGET_STAGE,
                "block": TARGET_BLOCK,
                "pytorch_module": TARGET_PREFIX,
                "tensorflow_module": "TFTransformerStage3.attns[1]",
            },
            "checkpoint_keys_used": checkpoint_keys_used,
            "case_results": [],
            "all_direct_same_input_dattention_pass": False,
            "supports_numeric_drift": False,
            "supports_implementation_bug": False,
            "code_modification_recommended": False,
            "full_dat_assembly_should_remain_blocked": True,
            "decision_note": "Diagnostic failed before the same-input DAttention comparison completed.",
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
    json_path = output_dir / "debug_stage_g3_block1_dattention_report.json"
    log_path = output_dir / "debug_stage_g3_block1_dattention_log.txt"
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
            "# Stage G-3 Block1 DAttention Surgical Debug\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Diagnostic failed before same-input DAttention parity completed. See JSON and log artifacts for details.\n",
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
