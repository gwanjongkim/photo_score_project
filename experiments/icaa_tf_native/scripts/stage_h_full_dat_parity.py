# ICAA17K DAT 전체 TensorFlow 모델의 최종 출력 동등성을 검증하는 스크립트
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_h_full_dat_parity.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_h_full_dat"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_h_full_dat_parity.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
REAL_IMAGE_COUNT = 16
OPTIONAL_REAL_IMAGE_COUNT = 64
PREFERRED_COLOR_MAX_ABS_DIFF = 1e-3
PREFERRED_FULL_MEAN_ABS_DIFF = 1e-4
ACCEPTABLE_COLOR_MAX_ABS_DIFF = 5e-3
ACCEPTABLE_FULL_MEAN_ABS_DIFF = 1e-3
DIAGNOSE_COLOR_MAX_ABS_DIFF = 1e-2

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

FORWARD_ORDER = [
    "patch_proj",
    "stages[0]",
    "down_projs[0]",
    "stages[1]",
    "down_projs[1]",
    "stages[2]",
    "down_projs[2]",
    "stages[3]",
    "cls_norm",
    "reshape NCHW to [B, C, H*W] and mean over spatial axis",
    "hst_head",
    "hist_feature",
    "rearrange 'b p w -> w p b'",
    "squeeze dim 2",
    "class_head + sigmoid",
    "class_head2 + sigmoid",
    "concat [MOS, color] to [B, 2]",
]


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
    package_name = "icaa_export_safe_v3_export_safe_models_stage_h"
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


def import_tf_dat(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.tf_dat_model import TFICAA17KDAT

    return TFICAA17KDAT


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


def torch_forward_with_debug(torch, model, image_nchw):
    x = torch.from_numpy(image_nchw.astype("float32"))
    debug = {}
    with torch.no_grad():
        x = model.patch_proj(x)
        debug["patch_proj"] = x.detach().cpu()
        x, _pos, _ref = model.stages[0](x)
        debug["stage0"] = x.detach().cpu()
        x = model.down_projs[0](x)
        debug["down_proj0"] = x.detach().cpu()
        x, _pos, _ref = model.stages[1](x)
        debug["stage1"] = x.detach().cpu()
        x = model.down_projs[1](x)
        debug["down_proj1"] = x.detach().cpu()
        x, _pos, _ref = model.stages[2](x)
        debug["stage2"] = x.detach().cpu()
        x = model.down_projs[2](x)
        debug["down_proj2"] = x.detach().cpu()
        x, _pos, _ref = model.stages[3](x)
        debug["stage3"] = x.detach().cpu()
        x = model.cls_norm(x)
        debug["cls_norm"] = x.detach().cpu()

        batch, channels, height, width = x.shape
        pooled = x.reshape(batch, channels, height * width).mean(dim=2)
        debug["pooled_feature"] = pooled.detach().cpu()
        hist_logits = model.hst_head(pooled)
        debug["hst_head"] = hist_logits.detach().cpu()
        hist_raw = model.hist_feature(hist_logits)
        debug["soft_histogram_raw"] = hist_raw.detach().cpu()
        hist_vector = hist_raw.permute(2, 1, 0).squeeze(dim=2)
        debug["soft_histogram"] = hist_vector.detach().cpu()
        mos = torch.sigmoid(model.class_head(hist_vector))
        color = torch.sigmoid(model.class_head2(hist_vector))
        debug["mos"] = mos.detach().cpu()
        debug["color"] = color.detach().cpu()
        final_manual = torch.cat([mos, color], dim=1)
        debug["final_output"] = final_manual.detach().cpu()
        final_module, _positions, _references = model(torch.from_numpy(image_nchw.astype("float32")))
        debug["module_output"] = final_module.detach().cpu()
    return final_module.detach().cpu(), debug


def compare_same_layout(name: str, torch_tensor, tf_tensor_or_array) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_tensor) if hasattr(torch_tensor, "detach") else np.asarray(torch_tensor)
    tf_np = tf_tensor_or_array.numpy() if hasattr(tf_tensor_or_array, "numpy") else np.asarray(tf_tensor_or_array)
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    return {
        "name": name,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
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
    return {
        "name": name,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "torch_shape": list(torch_np.shape),
        "tf_shape_nhwc": list(tf_nhwc_np.shape),
        "tf_shape_nchw": list(tf_nchw_np.shape),
    }


def final_output_metrics(torch_output, tf_output) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_output) if hasattr(torch_output, "detach") else np.asarray(torch_output)
    tf_np = tf_output.numpy() if hasattr(tf_output, "numpy") else np.asarray(tf_output)
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    mos_diff = diff[:, 0]
    color_diff = diff[:, 1]
    full_max = float(diff.max()) if diff.size else 0.0
    full_mean = float(diff.mean()) if diff.size else 0.0
    color_max = float(color_diff.max()) if color_diff.size else 0.0
    color_mean = float(color_diff.mean()) if color_diff.size else 0.0
    mos_max = float(mos_diff.max()) if mos_diff.size else 0.0
    mos_mean = float(mos_diff.mean()) if mos_diff.size else 0.0
    preferred = color_max <= PREFERRED_COLOR_MAX_ABS_DIFF and full_mean <= PREFERRED_FULL_MEAN_ABS_DIFF
    acceptable = color_max <= ACCEPTABLE_COLOR_MAX_ABS_DIFF and full_mean <= ACCEPTABLE_FULL_MEAN_ABS_DIFF
    needs_diagnosis = color_max > DIAGNOSE_COLOR_MAX_ABS_DIFF
    status = "pass_preferred" if preferred else "pass_acceptable" if acceptable else "fail"
    return {
        "status": status,
        "pass_preferred": bool(preferred),
        "pass_acceptable": bool(acceptable),
        "needs_diagnosis": bool(needs_diagnosis),
        "full_max_abs_diff": full_max,
        "full_mean_abs_diff": full_mean,
        "mos_max_abs_diff": mos_max,
        "mos_mean_abs_diff": mos_mean,
        "color_max_abs_diff": color_max,
        "color_mean_abs_diff": color_mean,
        "torch_output_shape": list(torch_np.shape),
        "tf_output_shape": list(tf_np.shape),
    }


def intermediate_diffs(debug_pt: dict, debug_tf: dict) -> list[dict]:
    image_like = [
        "patch_proj",
        "stage0",
        "down_proj0",
        "stage1",
        "down_proj1",
        "stage2",
        "down_proj2",
        "stage3",
        "cls_norm",
    ]
    same_layout = [
        "pooled_feature",
        "hst_head",
        "soft_histogram_raw",
        "soft_histogram",
        "mos",
        "color",
        "final_output",
    ]
    rows = [compare_nchw_to_nhwc(name, debug_pt[name], debug_tf[name]) for name in image_like]
    rows.extend(compare_same_layout(name, debug_pt[name], debug_tf[name]) for name in same_layout)
    rows.append(compare_same_layout("pytorch_manual_final_vs_module_final", debug_pt["final_output"], debug_pt["module_output"]))
    return rows


def preprocess_real_image(path: Path):
    import numpy as np
    from PIL import Image

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std


def load_real_image_batch(root: Path, limit: int):
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


def run_case(torch, tf, np, model_pt, model_tf, name: str, image_nchw, image_paths: list[str]):
    torch_output, debug_pt = torch_forward_with_debug(torch, model_pt, image_nchw)
    image_nhwc = np.transpose(image_nchw, (0, 2, 3, 1)).astype(np.float32)
    tf_output, debug_tf = model_tf.call_with_debug(tf.convert_to_tensor(image_nhwc, dtype=tf.float32), training=False)
    metrics = final_output_metrics(torch_output, tf_output)
    diffs = intermediate_diffs(debug_pt, debug_tf)
    metrics.update(
        {
            "name": name,
            "batch_size": int(image_nchw.shape[0]),
            "image_paths": image_paths,
            "torch_input_shape": list(image_nchw.shape),
            "tf_input_shape_nhwc": list(image_nhwc.shape),
            "intermediate_max_abs_diff": max(row["max_abs_diff"] for row in diffs),
            "intermediate_mean_abs_diff_max": max(row["mean_abs_diff"] for row in diffs),
        }
    )
    return metrics, diffs, tensor_np(torch_output), tf_output.numpy()


def prediction_rows(case_name: str, image_paths: list[str], torch_output, tf_output) -> list[dict]:
    import numpy as np

    torch_np = np.asarray(torch_output)
    tf_np = np.asarray(tf_output)
    rows = []
    for idx in range(torch_np.shape[0]):
        image_path = image_paths[idx] if idx < len(image_paths) else ""
        rows.append(
            {
                "case": case_name,
                "index": idx,
                "image_path": image_path,
                "torch_mos": float(torch_np[idx, 0]),
                "tf_mos": float(tf_np[idx, 0]),
                "mos_abs_diff": float(abs(torch_np[idx, 0] - tf_np[idx, 0])),
                "torch_color": float(torch_np[idx, 1]),
                "tf_color": float(tf_np[idx, 1]),
                "color_abs_diff": float(abs(torch_np[idx, 1] - tf_np[idx, 1])),
            }
        )
    return rows


def checkpoint_group_counts(state_dict) -> dict:
    groups = [
        "patch_proj.",
        "stages.0.",
        "down_projs.0.",
        "stages.1.",
        "down_projs.1.",
        "stages.2.",
        "down_projs.2.",
        "stages.3.",
        "cls_norm.",
        "hst_head.",
        "hist_feature.",
        "class_head.",
        "class_head2.",
    ]
    return {group.rstrip("."): sum(1 for key in state_dict.keys() if key.startswith(group)) for group in groups}


def mapped_checkpoint_keys(state_dict) -> list[str]:
    prefixes = [
        "patch_proj.",
        "stages.0.",
        "down_projs.0.",
        "stages.1.",
        "down_projs.1.",
        "stages.2.",
        "down_projs.2.",
        "stages.3.",
        "cls_norm.",
        "hst_head.",
        "hist_feature.",
        "class_head.",
        "class_head2.",
    ]
    return [key for key in state_dict.keys() if any(key.startswith(prefix) for prefix in prefixes)]


def markdown_result_table(results: list[dict]) -> str:
    lines = [
        "| Case | Status | B | Full max | Full mean | MOS max | MOS mean | Color max | Color mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['name']} | {result['status']} | {result['batch_size']} | "
            f"{result['full_max_abs_diff']:.9g} | {result['full_mean_abs_diff']:.9g} | "
            f"{result['mos_max_abs_diff']:.9g} | {result['mos_mean_abs_diff']:.9g} | "
            f"{result['color_max_abs_diff']:.9g} | {result['color_mean_abs_diff']:.9g} |"
        )
    return "\n".join(lines)


def markdown_intermediate_summary(intermediate_summary: dict) -> str:
    lines = [
        "| Case | Largest intermediate tensor diff | Max abs diff | Mean abs diff |",
        "| --- | --- | --- | --- |",
    ]
    for case_name, rows in intermediate_summary.items():
        if not rows:
            continue
        worst = max(rows, key=lambda row: row["max_abs_diff"])
        lines.append(
            f"| {case_name} | {worst['name']} | {worst['max_abs_diff']:.9g} | {worst['mean_abs_diff']:.9g} |"
        )
    return "\n".join(lines)


def write_markdown_report(report: dict, report_path: Path) -> None:
    warning_lines = report["warnings"] or ["none"]
    error_lines = report["errors"] or ["none"]
    runtime_notices = [
        line for line in report.get("captured_output", [])
        if line.startswith(("WARNING:", "E", "W"))
    ] or ["none"]
    unresolved = report["unresolved_issues"] or ["none"]
    mapped_groups = "\n".join(
        f"- `{group}`: {count} tensors"
        for group, count in report["checkpoint_group_counts"].items()
    )
    body = f"""# Stage H Full DAT Parity

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- PyTorch reference model path: `{report['pytorch_reference_model_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}
- SavedModel export: not performed
- TFLite conversion: not performed
- Flutter changes: none

## PyTorch Forward Order Confirmed
{chr(10).join(f"{idx + 1}. `{step}`" for idx, step in enumerate(report['pytorch_forward_order_confirmed']))}

## TensorFlow Full DAT Structure Summary
```json
{json.dumps(report['tf_model_structure'], indent=2)}
```

## Checkpoint Groups Mapped
{mapped_groups}

## Final Output Parity
{markdown_result_table(report['results'])}

## Intermediate Drift Summary
{markdown_intermediate_summary(report['intermediate_diff_summary'])}

## Real Image Inputs
- Requested count: {REAL_IMAGE_COUNT}
- Performed count: {report['real_image_input_count']}
- Optional 64-image run performed: {report['optional_64_real_images_performed']}

## Stage G-2/G-3 Drift Impact
- Significant final-score impact detected: {report['stage_g2_g3_drift_significantly_affects_final_scores']}
- Interpretation: {report['drift_impact_interpretation']}

## Decisions
- Safe to proceed to Stage I TensorFlow SavedModel export: {report['safe_to_proceed_to_stage_i_savedmodel_export']}
- Safe to proceed to TFLite conversion later: {report['safe_to_proceed_to_tflite_conversion_later']}

## Unresolved Issues
""" + "\n".join(f"- {line}" for line in unresolved) + f"""

## Artifacts
- JSON report: `{report['json_path']}`
- Predictions CSV: `{report['predictions_csv_path']}`
- Log: `{report['log_path']}`
- Intermediate diff summary: `{report['intermediate_diff_summary_path']}`

## Warnings
""" + "\n".join(f"- {line}" for line in warning_lines) + """

## Runtime Log Notices
""" + "\n".join(f"- {line}" for line in runtime_notices) + """

## Errors
""" + "\n".join(f"- {line}" for line in error_lines) + "\n"
    report_path.write_text(body, encoding="utf-8")


def run() -> tuple[dict, dict, list[dict]]:
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
            model_pt = DAT(**MODEL_KWARGS)
            strict_load = model_pt.load_state_dict(state_dict, strict=True)
            model_pt.eval()

            TFICAA17KDAT = import_tf_dat(root)
            model_tf = TFICAA17KDAT(epsilon=1e-5)
            model_tf.build_for_input_shape((1, 224, 224, 3))
            model_tf.load_from_pytorch_state_dict(state_dict)

            random_image_nchw = np.random.randn(1, 3, 224, 224).astype(np.float32)
            random_result, random_intermediates, random_pt, random_tf = run_case(
                torch,
                tf,
                np,
                model_pt,
                model_tf,
                "deterministic random normalized image input",
                random_image_nchw,
                ["random_normalized_image"],
            )

            real_image_nchw, real_image_paths = load_real_image_batch(root, REAL_IMAGE_COUNT)
            real_result = None
            real_intermediates = None
            real_pt = None
            real_tf = None
            if real_image_nchw is not None:
                real_result, real_intermediates, real_pt, real_tf = run_case(
                    torch,
                    tf,
                    np,
                    model_pt,
                    model_tf,
                    f"{REAL_IMAGE_COUNT} real ICAA17K test images",
                    real_image_nchw,
                    real_image_paths,
                )

            results = [random_result]
            if real_result is not None:
                results.append(real_result)

            optional_64_performed = False
            optional_result = None
            optional_intermediates = None
            optional_pt = None
            optional_tf = None
            optional_paths: list[str] = []
            if real_result is not None and real_result["pass_preferred"]:
                optional_image_nchw, optional_paths = load_real_image_batch(root, OPTIONAL_REAL_IMAGE_COUNT)
                if optional_image_nchw is not None and optional_image_nchw.shape[0] >= OPTIONAL_REAL_IMAGE_COUNT:
                    optional_64_performed = True
                    optional_result, optional_intermediates, optional_pt, optional_tf = run_case(
                        torch,
                        tf,
                        np,
                        model_pt,
                        model_tf,
                        f"{OPTIONAL_REAL_IMAGE_COUNT} real ICAA17K test images",
                        optional_image_nchw,
                        optional_paths,
                    )
                    results.append(optional_result)

            for result in results:
                log(
                    f"{result['name']}: status={result['status']} "
                    f"full_max_abs_diff={result['full_max_abs_diff']:.9g} "
                    f"full_mean_abs_diff={result['full_mean_abs_diff']:.9g} "
                    f"mos_max_abs_diff={result['mos_max_abs_diff']:.9g} "
                    f"color_max_abs_diff={result['color_max_abs_diff']:.9g}"
                )

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        required_results = [random_result]
        if real_result is not None:
            required_results.append(real_result)
        overall_acceptable = bool(required_results and all(result["pass_acceptable"] for result in required_results))
        overall_preferred = bool(required_results and all(result["pass_preferred"] for result in required_results))
        needs_diagnosis = any(result["needs_diagnosis"] for result in required_results)
        unresolved = []
        if real_result is None:
            unresolved.append("16 real ICAA17K test images could not be loaded; only random normalized input was tested.")
        for result in required_results:
            if not result["pass_acceptable"]:
                unresolved.append(
                    f"{result['name']} failed score-level acceptable threshold: color max {result['color_max_abs_diff']:.9g}, full mean {result['full_mean_abs_diff']:.9g}."
                )
        if needs_diagnosis:
            unresolved.append("At least one required case has color max_abs_diff > 1e-2 and needs diagnosis.")

        stage_g2_g3_drift_significant = not overall_acceptable
        if overall_preferred:
            interpretation = "Accumulated Stage G-2/G-3 feature drift did not materially affect final scores under preferred thresholds."
        elif overall_acceptable:
            interpretation = "Accumulated Stage G-2/G-3 feature drift affected internals but final scores stayed within acceptable Stage H thresholds."
        else:
            interpretation = "Accumulated Stage G-2/G-3 feature drift or full-model assembly drift caused unacceptable final-score differences."

        intermediate_summary = {
            "deterministic random normalized image input": random_intermediates,
        }
        if real_intermediates is not None:
            intermediate_summary[f"{REAL_IMAGE_COUNT} real ICAA17K test images"] = real_intermediates
        if optional_intermediates is not None:
            intermediate_summary[f"{OPTIONAL_REAL_IMAGE_COUNT} real ICAA17K test images"] = optional_intermediates

        prediction_rows_all = prediction_rows("deterministic random normalized image input", ["random_normalized_image"], random_pt, random_tf)
        if real_pt is not None and real_tf is not None:
            prediction_rows_all.extend(prediction_rows(f"{REAL_IMAGE_COUNT} real ICAA17K test images", real_image_paths, real_pt, real_tf))
        if optional_pt is not None and optional_tf is not None:
            prediction_rows_all.extend(prediction_rows(f"{OPTIONAL_REAL_IMAGE_COUNT} real ICAA17K test images", optional_paths, optional_pt, optional_tf))

        report = {
            "status": "ok" if overall_acceptable else "parity_failed",
            "overall_preferred": overall_preferred,
            "overall_acceptable": overall_acceptable,
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "pytorch_reference_model_path": str(export_safe_model_dir),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "seed": SEED,
            "thresholds": {
                "preferred": {
                    "color_max_abs_diff": PREFERRED_COLOR_MAX_ABS_DIFF,
                    "full_mean_abs_diff": PREFERRED_FULL_MEAN_ABS_DIFF,
                },
                "acceptable": {
                    "color_max_abs_diff": ACCEPTABLE_COLOR_MAX_ABS_DIFF,
                    "full_mean_abs_diff": ACCEPTABLE_FULL_MEAN_ABS_DIFF,
                },
                "diagnose_if_color_max_abs_diff_gt": DIAGNOSE_COLOR_MAX_ABS_DIFF,
            },
            "state_dict_confirmation": state_dict_confirmation,
            "strict_load": {
                "success": True,
                "missing_keys": list(getattr(strict_load, "missing_keys", [])),
                "unexpected_keys": list(getattr(strict_load, "unexpected_keys", [])),
            },
            "pytorch_forward_order_confirmed": FORWARD_ORDER,
            "tf_model_structure": model_tf.structure_summary(),
            "checkpoint_group_counts": checkpoint_group_counts(state_dict),
            "mapped_checkpoint_keys": mapped_checkpoint_keys(state_dict),
            "results": results,
            "real_image_input_count": 0 if real_result is None else real_result["batch_size"],
            "real_image_paths": real_image_paths if real_result is not None else [],
            "optional_64_real_images_performed": optional_64_performed,
            "stage_g2_g3_drift_significantly_affects_final_scores": stage_g2_g3_drift_significant,
            "drift_impact_interpretation": interpretation,
            "safe_to_proceed_to_stage_i_savedmodel_export": overall_acceptable,
            "safe_to_proceed_to_tflite_conversion_later": overall_acceptable,
            "unresolved_issues": unresolved,
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }
        return report, intermediate_summary, prediction_rows_all

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.extend(traceback.format_exc().splitlines())
        report = {
            "status": "failed",
            "overall_preferred": False,
            "overall_acceptable": False,
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "pytorch_reference_model_path": str(export_safe_model_dir),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "seed": SEED,
            "thresholds": {
                "preferred": {
                    "color_max_abs_diff": PREFERRED_COLOR_MAX_ABS_DIFF,
                    "full_mean_abs_diff": PREFERRED_FULL_MEAN_ABS_DIFF,
                },
                "acceptable": {
                    "color_max_abs_diff": ACCEPTABLE_COLOR_MAX_ABS_DIFF,
                    "full_mean_abs_diff": ACCEPTABLE_FULL_MEAN_ABS_DIFF,
                },
                "diagnose_if_color_max_abs_diff_gt": DIAGNOSE_COLOR_MAX_ABS_DIFF,
            },
            "state_dict_confirmation": None,
            "strict_load": {"success": False, "missing_keys": [], "unexpected_keys": []},
            "pytorch_forward_order_confirmed": FORWARD_ORDER,
            "tf_model_structure": None,
            "checkpoint_group_counts": {},
            "mapped_checkpoint_keys": [],
            "results": [],
            "real_image_input_count": 0,
            "real_image_paths": [],
            "optional_64_real_images_performed": False,
            "stage_g2_g3_drift_significantly_affects_final_scores": True,
            "drift_impact_interpretation": "Stage H failed before full-model parity completed.",
            "safe_to_proceed_to_stage_i_savedmodel_export": False,
            "safe_to_proceed_to_tflite_conversion_later": False,
            "unresolved_issues": ["Stage H failed before full-model parity completed."],
            "warnings": warnings_seen,
            "errors": errors,
            "log_lines": log_lines,
        }
        return report, {}, []


def main() -> int:
    root = project_root()
    stdout_stderr = io.StringIO()
    with redirect_stdout(stdout_stderr), redirect_stderr(stdout_stderr):
        report, intermediate_summary, predictions = run()

    output_dir = Path(report["output_dir"])
    json_path = output_dir / "stage_h_full_dat_report.json"
    predictions_csv_path = output_dir / "stage_h_full_dat_predictions.csv"
    log_path = output_dir / "stage_h_full_dat_log.txt"
    intermediate_path = output_dir / "optional_intermediate_diff_summary.json"
    markdown_path = root / REPORT_REL_PATH

    report["json_path"] = str(json_path)
    report["predictions_csv_path"] = str(predictions_csv_path)
    report["log_path"] = str(log_path)
    report["intermediate_diff_summary_path"] = str(intermediate_path)
    report["intermediate_diff_summary"] = intermediate_summary
    captured_output = stdout_stderr.getvalue().splitlines()
    report["captured_output"] = captured_output

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    intermediate_path.write_text(json.dumps(intermediate_summary, indent=2), encoding="utf-8")
    with predictions_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case",
            "index",
            "image_path",
            "torch_mos",
            "tf_mos",
            "mos_abs_diff",
            "torch_color",
            "tf_color",
            "color_abs_diff",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    if report["status"] in {"ok", "parity_failed"}:
        write_markdown_report(report, markdown_path)
    else:
        markdown_path.write_text(
            "# Stage H Full DAT Parity\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Stage H failed before full-model parity completed. See JSON and log artifacts for details.\n",
            encoding="utf-8",
        )
    log_path.write_text("\n".join(captured_output) + "\n", encoding="utf-8")

    for line in captured_output:
        print(line)
    print(f"json report: {json_path}")
    print(f"predictions csv: {predictions_csv_path}")
    print(f"log: {log_path}")
    print(f"intermediate diff summary: {intermediate_path}")
    print(f"markdown report: {markdown_path}")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
