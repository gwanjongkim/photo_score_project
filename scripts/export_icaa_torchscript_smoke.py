# ICAA17K 공식 DAT 체크포인트의 TorchScript 추적 가능성을 확인하는 스모크 테스트
from pathlib import Path
from datetime import datetime
import argparse
import json
import sys
import traceback
import warnings

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"

OFFICIAL_DAT_CONFIG = {
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
    return Path(__file__).resolve().parents[1]


def add_official_repo_to_path(root: Path) -> Path:
    official_repo = root / "external" / "icaa_official_repo" / "ICAA17K_code"
    if not official_repo.is_dir():
        raise RuntimeError(f"Official ICAA17K repo path not found: {official_repo}")
    sys.path.insert(0, str(official_repo))
    return official_repo


def import_torch_and_dat():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("Failed to import torch.") from exc

    try:
        from models.dat import DAT
    except Exception as exc:
        raise RuntimeError(
            "Failed to import DAT from external/icaa_official_repo/ICAA17K_code/models/dat.py. "
            "Check that official repo dependencies such as timm and einops are installed."
        ) from exc

    return torch, nn, DAT


def resolve_device(torch, requested: str):
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def load_checkpoint(torch, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


def check_state_dict_compatibility(model, checkpoint):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint.keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches = []

    for key in sorted(model_keys & checkpoint_keys):
        if shape_of(model_state[key]) != shape_of(checkpoint[key]):
            shape_mismatches.append(
                {
                    "key": key,
                    "model_shape": shape_of(model_state[key]),
                    "checkpoint_shape": shape_of(checkpoint[key]),
                }
            )
    return missing, unexpected, shape_mismatches


def make_wrapper_class(nn):
    class DATOutputWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output, positions, references = self.model(x)
            return output

    return DATOutputWrapper


def tensor_values(tensor):
    return [float(value) for value in tensor.detach().cpu().flatten().tolist()]


def warning_messages(recorded_warnings):
    counts = {}
    for item in recorded_warnings:
        message = f"{item.category.__name__}: {item.message}"
        counts[message] = counts.get(message, 0) + 1
    return [f"{message} (count={count})" for message, count in counts.items()]


def parse_args():
    parser = argparse.ArgumentParser(description="TorchScript export smoke for official ICAA17K DAT checkpoint.")
    parser.add_argument("--device", default="cpu", help="cpu by default, or explicitly pass cuda / cuda:N")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    official_repo = add_official_repo_to_path(root)
    torch, nn, DAT = import_torch_and_dat()
    device = resolve_device(torch, args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_torchscript_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    traced_path = output_dir / "icaa_dat_traced.pt"
    report_path = output_dir / "export_report.json"
    log_path = output_dir / "export_log.txt"
    log_lines = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    report = {
        "status": "failed",
        "project_root": str(root),
        "official_repo": str(official_repo),
        "checkpoint_path": str(root / CHECKPOINT_REL_PATH),
        "output_dir": str(output_dir),
        "device": str(device),
        "model_config": OFFICIAL_DAT_CONFIG,
        "trace": {},
        "script": {},
        "comparison": {},
    }

    try:
        checkpoint_path = root / CHECKPOINT_REL_PATH
        checkpoint = load_checkpoint(torch, checkpoint_path)
        if not hasattr(checkpoint, "keys"):
            raise RuntimeError(f"Expected checkpoint to be state_dict-like, got {type(checkpoint).__name__}.")
        log("DAT import: ok")
        log(f"official repo path: {official_repo}")
        log(f"checkpoint path: {checkpoint_path}")
        log(f"output dir: {output_dir}")
        log(f"device: {device}")
        log(f"checkpoint key count: {len(checkpoint)}")
        log(f"patch_proj.0.weight shape: {shape_of(checkpoint.get('patch_proj.0.weight'))}")

        model = DAT(**OFFICIAL_DAT_CONFIG)
        missing, unexpected, shape_mismatches = check_state_dict_compatibility(model, checkpoint)
        log(f"missing keys: {missing}")
        log(f"unexpected keys: {unexpected}")
        report["checkpoint"] = {
            "key_count": len(checkpoint),
            "patch_proj_0_weight_shape": shape_of(checkpoint.get("patch_proj.0.weight")),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
            "strict_load": True,
        }
        if shape_mismatches:
            raise RuntimeError("Checkpoint shape mismatches found; refusing to load.")
        if missing or unexpected:
            raise RuntimeError("Missing or unexpected checkpoint keys found; refusing to load.")

        load_result = model.load_state_dict(checkpoint, strict=True)
        log(f"strict load result: {load_result}")
        log("checkpoint loaded successfully: yes")

        wrapper_class = make_wrapper_class(nn)
        wrapper = wrapper_class(model).to(device)
        wrapper.eval()
        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        report["dummy_input_shape"] = tuple(dummy_input.shape)

        with torch.no_grad():
            original_output = wrapper(dummy_input)
        log(f"original wrapper output shape: {tuple(original_output.shape)}")
        log(f"original output values: {tensor_values(original_output)}")

        traced_model = None
        trace_error = None
        trace_warnings = []
        try:
            with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                traced_model = torch.jit.trace(wrapper, dummy_input)
                trace_warnings = warning_messages(caught)
            traced_model.save(str(traced_path))
            report["trace"] = {
                "succeeded": True,
                "saved_path": str(traced_path),
                "warnings": trace_warnings,
            }
            log(f"trace succeeded: yes")
            log(f"traced model saved: {traced_path}")
            if trace_warnings:
                log("trace warnings:")
                for message in trace_warnings:
                    log(f"  {message}")
        except Exception as exc:
            trace_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            report["trace"] = {
                "succeeded": False,
                "saved_path": None,
                "warnings": trace_warnings,
                "error": trace_error,
            }
            log("trace succeeded: no")
            log(f"trace error: {exc}")

        if traced_model is None:
            script_warnings = []
            try:
                with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    scripted_model = torch.jit.script(wrapper)
                    script_warnings = warning_messages(caught)
                report["script"] = {"attempted": True, "succeeded": True, "warnings": script_warnings}
                log("script fallback succeeded: yes")
                if script_warnings:
                    log("script warnings:")
                    for message in script_warnings:
                        log(f"  {message}")
            except Exception as exc:
                script_error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                report["script"] = {
                    "attempted": True,
                    "succeeded": False,
                    "warnings": script_warnings,
                    "error": script_error,
                }
                log("script fallback succeeded: no")
                log(f"script error: {exc}")
                raise RuntimeError("TorchScript trace failed, and script fallback also failed.") from exc
        else:
            loaded_model = torch.jit.load(str(traced_path), map_location=device)
            loaded_model.eval()
            with torch.no_grad():
                traced_output = loaded_model(dummy_input)

            diff = (original_output - traced_output).abs()
            max_abs_diff = diff.max().item()
            mean_abs_diff = diff.mean().item()
            report["comparison"] = {
                "loaded_traced_inference_succeeded": True,
                "original_shape": tuple(original_output.shape),
                "traced_shape": tuple(traced_output.shape),
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "original_output": tensor_values(original_output),
                "traced_output": tensor_values(traced_output),
            }
            log(f"loaded traced inference succeeded: yes")
            log(f"traced output shape: {tuple(traced_output.shape)}")
            log(f"traced output values: {tensor_values(traced_output)}")
            log(f"max_abs_diff: {max_abs_diff:.10f}")
            log(f"mean_abs_diff: {mean_abs_diff:.10f}")
            report["status"] = "passed"

        log("final summary:")
        log(f"  pass: {report['status'] == 'passed'}")
        log(f"  traced model: {traced_path if report['status'] == 'passed' else 'not produced'}")
        log(f"  report: {report_path}")
        log(f"  log: {log_path}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log("final summary:")
        log("  pass: False")
        log(f"  error: {exc}")
        log(f"  report: {report_path}")
        log(f"  log: {log_path}")
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
