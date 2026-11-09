# ICAA17K DAT 기본 레이어의 PyTorch-TensorFlow 가중치 매핑 동등성을 검증하는 스크립트
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
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
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_b_basic_layer_parity.md"
OUTPUT_PREFIX = "icaa_tf_native_stage_b"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_b_basic_layer_parity.py"
SEED = 123
PREFERRED_MAX_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-4


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dirs(root: Path) -> None:
    for rel_path in [
        "experiments/icaa_tf_native/tf_models",
        "experiments/icaa_tf_native/scripts",
        "experiments/icaa_tf_native/reports",
    ]:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


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


def compare_arrays(name: str, torch_output, tf_output, notes: list[str] | None = None) -> dict:
    import numpy as np

    torch_np = tensor_np(torch_output) if hasattr(torch_output, "detach") else np.asarray(torch_output)
    tf_np = tf_output.numpy() if hasattr(tf_output, "numpy") else np.asarray(tf_output)
    diff = np.abs(torch_np.astype(np.float32) - tf_np.astype(np.float32))
    max_abs_diff = float(diff.max()) if diff.size else 0.0
    mean_abs_diff = float(diff.mean()) if diff.size else 0.0
    preferred_pass = max_abs_diff <= PREFERRED_MAX_ABS_DIFF
    acceptable_pass = max_abs_diff <= ACCEPTABLE_MAX_ABS_DIFF
    status = "pass_preferred" if preferred_pass else "pass_acceptable" if acceptable_pass else "fail"
    likely_cause = None if acceptable_pass else "Cross-framework operation semantics or mapping rule need inspection."
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
        "likely_cause_if_failed": likely_cause,
        "notes": notes or [],
    }


def make_tf_conv2d(tf, state_dict):
    import numpy as np

    weight = tensor_np(state_dict["patch_proj.0.weight"])
    bias = tensor_np(state_dict["patch_proj.0.bias"])
    kernel = np.transpose(weight, (2, 3, 1, 0))
    layer = tf.keras.layers.Conv2D(
        filters=weight.shape[0],
        kernel_size=(weight.shape[2], weight.shape[3]),
        strides=(4, 4),
        padding="valid",
        use_bias=True,
    )
    layer(tf.zeros((1, 16, 20, weight.shape[1]), dtype=tf.float32))
    layer.set_weights([kernel, bias])
    return layer


def run_conv2d_parity(torch, tf, np, state_dict) -> dict:
    conv = torch.nn.Conv2d(3, 128, kernel_size=4, stride=4, padding=0, bias=True)
    conv.weight.data.copy_(state_dict["patch_proj.0.weight"])
    conv.bias.data.copy_(state_dict["patch_proj.0.bias"])
    conv.eval()

    x_pt = torch.randn(2, 3, 16, 20, dtype=torch.float32)
    x_tf = tf.convert_to_tensor(np.transpose(tensor_np(x_pt), (0, 2, 3, 1)), dtype=tf.float32)
    tf_conv = make_tf_conv2d(tf, state_dict)

    with torch.no_grad():
        y_pt = conv(x_pt)
    y_tf_nhwc = tf_conv(x_tf)
    y_tf_nchw = tf.transpose(y_tf_nhwc, (0, 3, 1, 2))

    return compare_arrays(
        "Conv2D patch projection",
        y_pt,
        y_tf_nchw,
        [
            "checkpoint tensors: patch_proj.0.weight, patch_proj.0.bias",
            "mapping: PyTorch [out_c,in_c,kH,kW] -> TF [kH,kW,in_c,out_c]",
            "input layout: PyTorch NCHW -> TF NHWC -> compare as NCHW",
        ],
    )


def make_tf_dense(tf, weight, bias):
    import numpy as np

    weight_np = tensor_np(weight)
    bias_np = tensor_np(bias)
    layer = tf.keras.layers.Dense(weight_np.shape[0], use_bias=True)
    layer(tf.zeros((1, weight_np.shape[1]), dtype=tf.float32))
    layer.set_weights([np.transpose(weight_np, (1, 0)), bias_np])
    return layer


def run_dense_parity(torch, tf, np, state_dict) -> dict:
    weight = state_dict["hst_head.weight"]
    bias = state_dict["hst_head.bias"]
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    linear.weight.data.copy_(weight)
    linear.bias.data.copy_(bias)
    linear.eval()

    x_np = np.random.randn(4, int(weight.shape[1])).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
    tf_dense = make_tf_dense(tf, weight, bias)

    with torch.no_grad():
        y_pt = linear(x_pt)
    y_tf = tf_dense(x_tf)

    return compare_arrays(
        "Dense hst_head",
        y_pt,
        y_tf,
        [
            "checkpoint tensors: hst_head.weight, hst_head.bias",
            "mapping: PyTorch Linear [out_features,in_features] -> TF Dense [in_features,out_features]",
            "bias direct copy",
        ],
    )


def run_layernorm_parity(torch, tf, np, state_dict) -> dict:
    gamma = state_dict["stages.0.layer_norms.0.norm.weight"]
    beta = state_dict["stages.0.layer_norms.0.norm.bias"]
    channels = int(gamma.shape[0])
    eps = 1e-5

    x_pt = torch.randn(2, channels, 5, 7, dtype=torch.float32)
    x_pt_nhwc = x_pt.permute(0, 2, 3, 1)
    y_pt = torch.nn.functional.layer_norm(x_pt_nhwc, (channels,), gamma, beta, eps=eps).permute(0, 3, 1, 2)

    x_tf = tf.convert_to_tensor(np.transpose(tensor_np(x_pt), (0, 2, 3, 1)), dtype=tf.float32)
    layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=eps)
    layer(tf.zeros((1, 5, 7, channels), dtype=tf.float32))
    layer.set_weights([tensor_np(gamma), tensor_np(beta)])
    y_tf_nhwc = layer(x_tf)
    y_tf_nchw = tf.transpose(y_tf_nhwc, (0, 3, 1, 2))

    return compare_arrays(
        "LayerNormProxy channel LayerNorm",
        y_pt,
        y_tf_nchw,
        [
            "checkpoint tensors: stages.0.layer_norms.0.norm.weight, stages.0.layer_norms.0.norm.bias",
            "LayerNormProxy evidence: NCHW -> NHWC, channel-only LayerNorm axis=-1, NHWC -> NCHW",
            "epsilon: 1e-5 to match torch.nn.LayerNorm default",
        ],
    )


def run_gelu_parity(torch, tf, np) -> dict:
    x_np = np.random.randn(3, 7, 11).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)

    with torch.no_grad():
        y_pt_default = torch.nn.functional.gelu(x_pt)
    y_tf_exact = tf.nn.gelu(x_tf, approximate=False)
    y_tf_tanh = tf.nn.gelu(x_tf, approximate=True)

    exact_result = compare_arrays(
        "GELU TensorFlow exact vs PyTorch default",
        y_pt_default,
        y_tf_exact,
        ["torch.nn.functional.gelu default vs tf.nn.gelu approximate=False"],
    )
    tanh_result = compare_arrays(
        "GELU TensorFlow tanh approximate vs PyTorch default",
        y_pt_default,
        y_tf_tanh,
        ["torch.nn.functional.gelu default vs tf.nn.gelu approximate=True"],
    )
    chosen = exact_result if exact_result["max_abs_diff"] <= tanh_result["max_abs_diff"] else tanh_result
    result = dict(chosen)
    result["name"] = "GELU"
    result["chosen_mode"] = "exact" if chosen is exact_result else "tanh_approximate"
    result["mode_comparison"] = {
        "exact": {
            "max_abs_diff": exact_result["max_abs_diff"],
            "mean_abs_diff": exact_result["mean_abs_diff"],
            "pass": exact_result["pass"],
        },
        "tanh_approximate": {
            "max_abs_diff": tanh_result["max_abs_diff"],
            "mean_abs_diff": tanh_result["mean_abs_diff"],
            "pass": tanh_result["pass"],
        },
    }
    result["notes"] = [
        "No checkpoint tensor is used for GELU because it is parameter-free.",
        "PyTorch DAT uses nn.GELU() default semantics, so TensorFlow modes are compared against PyTorch default GELU.",
        f"Chosen mode: {result['chosen_mode']}",
    ]
    return result


def run_mlp_parity(torch, tf, np, state_dict, gelu_mode: str) -> dict:
    w1 = state_dict["stages.0.mlps.0.chunk.linear1.weight"]
    b1 = state_dict["stages.0.mlps.0.chunk.linear1.bias"]
    w2 = state_dict["stages.0.mlps.0.chunk.linear2.weight"]
    b2 = state_dict["stages.0.mlps.0.chunk.linear2.bias"]

    linear1 = torch.nn.Linear(w1.shape[1], w1.shape[0], bias=True)
    linear2 = torch.nn.Linear(w2.shape[1], w2.shape[0], bias=True)
    linear1.weight.data.copy_(w1)
    linear1.bias.data.copy_(b1)
    linear2.weight.data.copy_(w2)
    linear2.bias.data.copy_(b2)
    linear1.eval()
    linear2.eval()

    x_np = np.random.randn(2, 5, int(w1.shape[1])).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
    tf_dense1 = make_tf_dense(tf, w1, b1)
    tf_dense2 = make_tf_dense(tf, w2, b2)
    use_tanh = gelu_mode == "tanh_approximate"

    with torch.no_grad():
        y_pt = linear2(torch.nn.functional.gelu(linear1(x_pt)))
    y_tf = tf_dense2(tf.nn.gelu(tf_dense1(x_tf), approximate=use_tanh))

    return compare_arrays(
        "Isolated TransformerMLP linear1-GELU-linear2",
        y_pt,
        y_tf,
        [
            "checkpoint tensors: stages.0.mlps.0.chunk.linear1.{weight,bias}, stages.0.mlps.0.chunk.linear2.{weight,bias}",
            "tested isolated Linear -> GELU -> Linear path only; dropout is inactive and no spatial rearrange is included",
            f"PyTorch GELU mode: default; TensorFlow GELU mode: {gelu_mode}",
        ],
    )


def result_line(result: dict) -> str:
    return (
        f"{result['name']}: status={result['status']} "
        f"max_abs_diff={result['max_abs_diff']:.9g} "
        f"mean_abs_diff={result['mean_abs_diff']:.9g} "
        f"torch_shape={result['torch_output_shape']} tf_shape={result['tf_output_shape']}"
    )


def markdown_result_table(results: dict[str, dict]) -> str:
    lines = [
        "| Test | Status | Max abs diff | Mean abs diff | Output shape |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in ["conv2d", "dense", "layernorm", "gelu", "mlp"]:
        result = results[key]
        lines.append(
            f"| {result['name']} | {result['status']} | {result['max_abs_diff']:.9g} | "
            f"{result['mean_abs_diff']:.9g} | {result['torch_output_shape']} |"
        )
    return "\n".join(lines)


def write_markdown_report(report: dict, report_path: Path) -> None:
    results = report["results"]
    unresolved = report["unresolved_issues"] or ["none"]
    warning_lines = report["warnings"] or ["none"]
    error_lines = report["errors"] or ["none"]
    runtime_notices = [
        line for line in report.get("captured_output", [])
        if line.startswith(("WARNING:", "E", "W"))
    ] or ["none"]
    safe_line = report["safe_to_proceed_to_stage_c_patch_projection_parity"]

    body = f"""# Stage B Basic Layer Parity

## Scope
- Exact command used: `{report['command_used']}`
- Checkpoint path: `{report['checkpoint_path']}`
- Output directory: `{report['output_dir']}`
- Status: `{report['status']}`
- Seeds: torch={SEED}, numpy={SEED}, tensorflow={SEED}

## Checkpoint Tensors Used
- Conv2D: `patch_proj.0.weight`, `patch_proj.0.bias`
- Dense: `hst_head.weight`, `hst_head.bias`
- LayerNorm: `stages.0.layer_norms.0.norm.weight`, `stages.0.layer_norms.0.norm.bias`
- GELU: parameter-free function check
- MLP: `stages.0.mlps.0.chunk.linear1.weight`, `stages.0.mlps.0.chunk.linear1.bias`, `stages.0.mlps.0.chunk.linear2.weight`, `stages.0.mlps.0.chunk.linear2.bias`

## Mapping Rules Verified
- Conv2D: PyTorch `[out_c, in_c, kH, kW]` -> TensorFlow `[kH, kW, in_c, out_c]`; input NCHW -> NHWC and output back to NCHW.
- Dense: PyTorch `[out_features, in_features]` -> TensorFlow `[in_features, out_features]`; bias direct copy.
- LayerNorm: `LayerNormProxy` channel-only normalization maps to TensorFlow `LayerNormalization(axis=-1, epsilon=1e-5)` on NHWC tensors.
- GELU: selected `{results['gelu'].get('chosen_mode')}` mode by direct PyTorch-vs-TensorFlow comparison.
- MLP: isolated `linear1 -> GELU -> linear2` mapping with dropout omitted because parity is evaluated in inference form.

## Results
{markdown_result_table(results)}

## GELU Approximate Mode Check
- Exact max_abs_diff: {results['gelu']['mode_comparison']['exact']['max_abs_diff']:.9g}
- Tanh approximate max_abs_diff: {results['gelu']['mode_comparison']['tanh_approximate']['max_abs_diff']:.9g}
- Chosen mode: `{results['gelu']['chosen_mode']}`

## Unresolved Issues
""" + "\n".join(f"- {line}" for line in unresolved) + f"""

## Stage C Patch Projection Parity Decision
- Safe to proceed to Stage C patch projection parity: {safe_line}
- Scope of this decision: basic Conv2D, Dense, LayerNorm, GELU, and isolated MLP mapping only. This does not establish full TensorFlow DAT feasibility.

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

            log(f"checkpoint path: {checkpoint_path}")
            log(f"state_dict confirmation: {state_dict_confirmation}")
            log(f"checkpoint tensor count: {len(state_dict)}")
            log(f"torch version: {torch.__version__}")
            log(f"tensorflow version: {tf.__version__}")
            log(f"numpy version: {np.__version__}")

            results = {
                "conv2d": run_conv2d_parity(torch, tf, np, state_dict),
                "dense": run_dense_parity(torch, tf, np, state_dict),
                "layernorm": run_layernorm_parity(torch, tf, np, state_dict),
                "gelu": run_gelu_parity(torch, tf, np),
            }
            results["mlp"] = run_mlp_parity(torch, tf, np, state_dict, results["gelu"]["chosen_mode"])

            for result in results.values():
                log(result_line(result))

            warnings_seen.extend(
                f"{Path(item.filename).name}:{item.lineno}: {item.message}"
                for item in captured
            )

        unresolved = []
        for result in results.values():
            if not result["pass"]:
                unresolved.append(
                    f"{result['name']} failed acceptable threshold; likely cause: {result['likely_cause_if_failed']}"
                )

        safe_to_stage_c = bool(
            results["conv2d"]["pass"]
            and results["layernorm"]["pass"]
            and results["dense"]["pass"]
            and results["gelu"]["pass"]
            and results["mlp"]["pass"]
        )

        report = {
            "status": "ok",
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "seed": SEED,
            "thresholds": {
                "preferred_max_abs_diff": PREFERRED_MAX_ABS_DIFF,
                "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_DIFF,
            },
            "results": results,
            "safe_to_proceed_to_stage_c_patch_projection_parity": safe_to_stage_c,
            "unresolved_issues": unresolved,
            "warnings": warnings_seen,
            "errors": errors,
        }
        return report

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.extend(traceback.format_exc().splitlines())
        return {
            "status": "failed",
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "seed": SEED,
            "thresholds": {
                "preferred_max_abs_diff": PREFERRED_MAX_ABS_DIFF,
                "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_DIFF,
            },
            "results": {},
            "safe_to_proceed_to_stage_c_patch_projection_parity": False,
            "unresolved_issues": ["Stage B script failed before completing all parity checks."],
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
    json_path = output_dir / "stage_b_basic_layer_parity_report.json"
    log_path = output_dir / "stage_b_basic_layer_parity_log.txt"
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
            "# Stage B Basic Layer Parity\n\n"
            f"- Status: `{report['status']}`\n"
            f"- Output directory: `{report['output_dir']}`\n"
            "- Stage B failed before all parity checks completed. See JSON and log artifacts for details.\n",
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
