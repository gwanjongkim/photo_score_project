# PyTorch DAT 체크포인트 구조를 검사하고 매핑 인벤토리를 생성하는 스크립트
from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping
from pathlib import Path
import csv
import importlib
import importlib.util
import json
import sys
import traceback
import warnings

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
EXPORT_SAFE_MODEL_REL_DIR = "experiments/icaa_export_safe_v3/export_safe_models"
CSV_REL_PATH = "experiments/icaa_tf_native/weight_maps/pytorch_state_dict_inventory.csv"
SUMMARY_REL_PATH = "experiments/icaa_tf_native/weight_maps/pytorch_state_dict_summary.json"
REPORT_REL_PATH = "experiments/icaa_tf_native/reports/stage_a_state_dict_inspection.md"
OUTPUT_REL_DIR = "outputs/icaa_tf_native_state_dict_inspection"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/inspect_pytorch_state_dict.py"

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

CSV_COLUMNS = [
    "index",
    "name",
    "top_module",
    "shape",
    "ndim",
    "dtype",
    "numel",
    "likely_tf_layer_type",
    "likely_mapping_rule",
    "notes",
]

TOP_MODULE_ORDER = [
    "patch_proj",
    "stages",
    "down_projs",
    "cls_norm",
    "hst_head",
    "SoftHistogram-related layers",
    "class_head",
    "class_head2",
    "others",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dirs(root: Path) -> None:
    for rel_path in [
        "experiments/icaa_tf_native/scripts",
        "experiments/icaa_tf_native/reports",
        "experiments/icaa_tf_native/weight_maps",
        OUTPUT_REL_DIR,
    ]:
        (root / rel_path).mkdir(parents=True, exist_ok=True)


def import_dat_from_package_dir(package_dir: Path):
    package_name = "icaa_export_safe_v3_export_safe_models_stage_a"
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


def top_module_for_name(name: str) -> str:
    if name.startswith("patch_proj."):
        return "patch_proj"
    if name.startswith("stages."):
        return "stages"
    if name.startswith("down_projs."):
        return "down_projs"
    if name.startswith("cls_norm."):
        return "cls_norm"
    if name.startswith("hst_head."):
        return "hst_head"
    if name.startswith("hist_feature.") or "bin_centers_conv" in name or "bin_widths_conv" in name:
        return "SoftHistogram-related layers"
    if name.startswith("class_head2."):
        return "class_head2"
    if name.startswith("class_head."):
        return "class_head"
    return "others"


def module_path_for_param(name: str, module_info: dict[str, dict]) -> str:
    parts = name.split(".")
    for end in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:end])
        if candidate in module_info:
            return candidate
    return ""


def build_module_info(model) -> dict[str, dict]:
    info = {}
    for module_name, module in model.named_modules():
        entry = {
            "class_name": module.__class__.__name__,
            "in_channels": getattr(module, "in_channels", None),
            "out_channels": getattr(module, "out_channels", None),
            "in_features": getattr(module, "in_features", None),
            "out_features": getattr(module, "out_features", None),
            "groups": getattr(module, "groups", None),
            "kernel_size": getattr(module, "kernel_size", None),
        }
        info[module_name] = entry
    return info


def shape_list(tensor) -> list[int]:
    return [int(dim) for dim in tuple(tensor.shape)]


def shape_text(tensor) -> str:
    return json.dumps(shape_list(tensor))


def is_depthwise_conv2d(module_entry: dict | None) -> bool:
    if not module_entry or module_entry.get("class_name") != "Conv2d":
        return False
    groups = module_entry.get("groups")
    in_channels = module_entry.get("in_channels")
    out_channels = module_entry.get("out_channels")
    return bool(groups and in_channels and out_channels and groups == in_channels and out_channels % in_channels == 0)


def infer_layer_type_and_rule(name: str, tensor, module_info: dict[str, dict]) -> tuple[str, str, str]:
    lower_name = name.lower()
    parts = name.split(".")
    leaf = parts[-1]
    module_path = module_path_for_param(name, module_info)
    module_entry = module_info.get(module_path)
    module_class = module_entry.get("class_name") if module_entry else None
    shape = shape_list(tensor)
    ndim = len(shape)
    notes = []

    if module_entry:
        notes.append(f"module={module_path or '<root>'}:{module_class}")
        if module_entry.get("groups") not in (None, 1):
            notes.append(f"groups={module_entry.get('groups')}")

    if name.startswith("hist_feature.") or "bin_centers_conv" in name or "bin_widths_conv" in name:
        return (
            "SoftHistogram",
            "SoftHistogram grouped Conv1d parameters -> inspect manually before TensorFlow mapping",
            "; ".join(notes + ["grouped Conv1d histogram path"]),
        )

    if any(token in lower_name for token in ["relative_position_bias_table", "relative_position_index", "rpe_table", "attn_mask"]):
        return (
            "PositionalEncoding_or_Buffer",
            "Unknown -> mark unknown",
            "; ".join(notes + ["attention positional/buffer tensor; verify semantics before mapping"]),
        )

    if module_class == "LayerNorm":
        if leaf == "weight":
            return "LayerNormalization_gamma", "LayerNorm weight/bias -> direct copy", "; ".join(notes)
        if leaf == "bias":
            return "LayerNormalization_beta", "LayerNorm weight/bias -> direct copy", "; ".join(notes)

    if leaf == "bias":
        return "Bias", "Bias -> direct copy", "; ".join(notes)

    if module_class == "Linear" and leaf == "weight" and ndim == 2:
        return "Dense", "PyTorch Linear [out,in] -> TF Dense [in,out]", "; ".join(notes)

    if module_class == "Conv2d" and leaf == "weight" and ndim == 4:
        if is_depthwise_conv2d(module_entry):
            return (
                "DepthwiseConv2D",
                "Depthwise Conv if present -> inspect carefully, do not guess",
                "; ".join(notes + [f"shape={shape}; depthwise/grouped Conv2d metadata verified"]),
            )
        return "Conv2D", "PyTorch Conv2d [out_c,in_c,kH,kW] -> TF Conv2D [kH,kW,in_c,out_c]", "; ".join(notes)

    if module_class == "Conv1d":
        return (
            "SoftHistogram" if name.startswith("hist_feature.") else "Unknown",
            "Unknown -> mark unknown",
            "; ".join(notes + ["Conv1d outside an established mapping rule"]),
        )

    if ndim == 2 and leaf == "weight":
        return "Dense", "PyTorch Linear [out,in] -> TF Dense [in,out]", "; ".join(notes + ["shape/name fallback"])

    if ndim == 4 and leaf == "weight":
        return "Conv2D", "PyTorch Conv2d [out_c,in_c,kH,kW] -> TF Conv2D [kH,kW,in_c,out_c]", "; ".join(notes + ["shape/name fallback"])

    return "Unknown", "Unknown -> mark unknown", "; ".join(notes + ["no safe mapping inferred"])


def build_inventory_rows(state_dict, module_info: dict[str, dict]) -> list[dict]:
    rows = []
    for index, (name, tensor) in enumerate(state_dict.items()):
        layer_type, mapping_rule, notes = infer_layer_type_and_rule(name, tensor, module_info)
        rows.append(
            {
                "index": index,
                "name": name,
                "top_module": top_module_for_name(name),
                "shape": shape_text(tensor),
                "ndim": len(tuple(tensor.shape)),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "likely_tf_layer_type": layer_type,
                "likely_mapping_rule": mapping_rule,
                "notes": notes,
            }
        )
    return rows


def summarize_rows(rows: list[dict]) -> tuple[dict, dict, dict]:
    top_breakdown = {
        name: {"tensor_count": 0, "numel": 0}
        for name in TOP_MODULE_ORDER
    }
    layer_type_breakdown = Counter()
    mapping_rule_breakdown = Counter()

    for row in rows:
        top_entry = top_breakdown.setdefault(row["top_module"], {"tensor_count": 0, "numel": 0})
        top_entry["tensor_count"] += 1
        top_entry["numel"] += int(row["numel"])
        layer_type_breakdown[row["likely_tf_layer_type"]] += 1
        mapping_rule_breakdown[row["likely_mapping_rule"]] += 1

    return top_breakdown, dict(sorted(layer_type_breakdown.items())), dict(sorted(mapping_rule_breakdown.items()))


def write_csv(rows: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def tensor_shape_or_none(value):
    return list(value.shape) if hasattr(value, "shape") else None


def output_shape_summary(output):
    if hasattr(output, "shape"):
        return {"type": type(output).__name__, "shape": list(output.shape)}
    if isinstance(output, (tuple, list)):
        summary = {"type": type(output).__name__, "length": len(output)}
        if output:
            summary["primary_item"] = output_shape_summary(output[0])
        if len(output) > 1:
            summary["auxiliary_item_count"] = len(output) - 1
            summary["auxiliary_items"] = [nested_collection_summary(item) for item in output[1:]]
        return summary
    return {"type": type(output).__name__, "shape": None}


def nested_collection_summary(value):
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    if isinstance(value, (tuple, list)):
        summary = {"type": type(value).__name__, "length": len(value)}
        if value:
            summary["first_item"] = nested_collection_summary(value[0])
        return summary
    return {"type": type(value).__name__, "shape": None}


def compact_warnings(captured_warnings) -> list[str]:
    compact = []
    for item in captured_warnings:
        message = str(item.message)
        origin = f"{Path(item.filename).name}:{item.lineno}"
        compact.append(f"{origin}: {message}")
    return compact


def inspect_export_safe_model(torch, DAT, state_dict):
    model = DAT(**MODEL_KWARGS)
    model.eval()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    strict_load = {
        "attempted": True,
        "success": False,
        "missing_keys": [],
        "unexpected_keys": [],
        "error": None,
    }

    try:
        load_result = model.load_state_dict(state_dict, strict=True)
        strict_load["success"] = True
        strict_load["missing_keys"] = list(getattr(load_result, "missing_keys", []))
        strict_load["unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []))
    except RuntimeError as exc:
        strict_load["error"] = str(exc)
        loose_result = model.load_state_dict(state_dict, strict=False)
        strict_load["missing_keys"] = list(getattr(loose_result, "missing_keys", []))
        strict_load["unexpected_keys"] = list(getattr(loose_result, "unexpected_keys", []))

    dummy_input_shape = [1, 3, 224, 224]
    dummy_output_summary = None
    with torch.no_grad():
        dummy_output = model(torch.zeros(*dummy_input_shape))
        dummy_output_summary = output_shape_summary(dummy_output)

    return model, {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "total_parameter_count": parameter_count,
        "strict_load": strict_load,
        "dummy_input_shape": dummy_input_shape,
        "dummy_output_shape": dummy_output_summary,
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(summary: dict, report_path: Path) -> None:
    top_rows = [
        [name, data["tensor_count"], data["numel"]]
        for name, data in summary["top_level_module_breakdown"].items()
        if data["tensor_count"] or name in TOP_MODULE_ORDER
    ]
    category_rows = [
        [name, count]
        for name, count in summary["layer_type_breakdown"].items()
    ]
    unknown_rows = summary["uncertain_or_unknown_parameter_groups"]

    if unknown_rows:
        unknown_section = "\n".join(
            f"- `{item['name']}` shape={item['shape']} type={item['likely_tf_layer_type']} rule={item['likely_mapping_rule']}"
            for item in unknown_rows[:40]
        )
        if len(unknown_rows) > 40:
            unknown_section += f"\n- ... {len(unknown_rows) - 40} additional uncertain rows are listed in the CSV."
    else:
        unknown_section = "- No `Unknown` layer-type rows were found. Depthwise and SoftHistogram rows still require Stage B parity checks."

    warnings_or_none = summary["warnings"] or ["none"]
    errors_or_none = summary["errors"] or ["none"]
    strict_load = summary["export_safe_model"]["strict_load"]

    report = f"""# Stage A State Dict Inspection

## Scope
- Checkpoint path: `{summary['checkpoint_path']}`
- Exact command used: `{summary['command_used']}`
- Inspection status: `{summary['status']}`
- Checkpoint object type: `{summary['checkpoint_object_type']}`
- State dict confirmation: `{summary['state_dict_confirmation']}`

## Totals
- Total parameter tensors: {summary['total_parameter_tensors']}
- Total parameter count: {summary['total_parameter_count']}
- CSV inventory: `{summary['csv_path']}`
- JSON summary: `{summary['summary_path']}`

## Top-Level Module Breakdown
{markdown_table(['top_module', 'tensor_count', 'numel'], top_rows)}

## Discovered Layer Categories
{markdown_table(['likely_tf_layer_type', 'tensor_count'], category_rows)}

## Uncertain/Unknown Parameter Groups
{unknown_section}

## Export-Safe v3 Model Check
- Model class: `{summary['export_safe_model']['model_class']}`
- Model parameter count: {summary['export_safe_model']['total_parameter_count']}
- Strict checkpoint load success: {strict_load['success']}
- Missing keys after strict checkpoint load: {strict_load['missing_keys']}
- Unexpected keys after strict checkpoint load: {strict_load['unexpected_keys']}
- Strict load error: {strict_load['error']}
- Dummy input shape: {summary['export_safe_model']['dummy_input_shape']}
- Dummy output shape summary: `{json.dumps(summary['export_safe_model']['dummy_output_shape'])}`

## Proposed First TensorFlow Implementation Target for Stage B
Start with isolated base-layer parity before assembling DAT: `LayerNormProxy`, plain `Conv2D`/`Dense` transposition loaders, and the small `SoftHistogram` grouped-Conv1D path. Do not begin deformable attention until these lower-risk mappings have exact PyTorch-vs-TensorFlow parity evidence.

## Warnings
""" + "\n".join(f"- {line}" for line in warnings_or_none) + """

## Errors
""" + "\n".join(f"- {line}" for line in errors_or_none) + "\n"

    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    root = project_root()
    ensure_dirs(root)
    checkpoint_path = root / CHECKPOINT_REL_PATH
    export_safe_model_dir = root / EXPORT_SAFE_MODEL_REL_DIR
    csv_path = root / CSV_REL_PATH
    summary_path = root / SUMMARY_REL_PATH
    report_path = root / REPORT_REL_PATH

    captured_warning_lines = []
    errors = []

    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            import torch

            checkpoint_obj = load_checkpoint(torch, checkpoint_path)
            state_dict, state_dict_confirmation = extract_state_dict(checkpoint_obj)
            DAT = import_dat_from_package_dir(export_safe_model_dir)
            model, model_summary = inspect_export_safe_model(torch, DAT, state_dict)
            module_info = build_module_info(model)
            rows = build_inventory_rows(state_dict, module_info)
            captured_warning_lines.extend(compact_warnings(captured))

        write_csv(rows, csv_path)
        top_breakdown, layer_type_breakdown, mapping_rule_breakdown = summarize_rows(rows)
        uncertain_rows = [
            {
                "name": row["name"],
                "shape": row["shape"],
                "likely_tf_layer_type": row["likely_tf_layer_type"],
                "likely_mapping_rule": row["likely_mapping_rule"],
                "notes": row["notes"],
            }
            for row in rows
            if row["likely_tf_layer_type"] == "Unknown"
            or row["likely_mapping_rule"] == "Unknown -> mark unknown"
            or row["likely_tf_layer_type"] in {"DepthwiseConv2D", "SoftHistogram"}
        ]

        summary = {
            "status": "ok",
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_object_type": type(checkpoint_obj).__name__,
            "state_dict_confirmation": state_dict_confirmation,
            "is_ordered_dict_state_dict": isinstance(state_dict, OrderedDict),
            "total_parameter_tensors": len(state_dict),
            "total_parameter_count": int(sum(tensor.numel() for tensor in state_dict.values())),
            "top_level_module_breakdown": top_breakdown,
            "layer_type_breakdown": layer_type_breakdown,
            "mapping_rule_breakdown": mapping_rule_breakdown,
            "uncertain_or_unknown_parameter_groups": uncertain_rows,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "export_safe_model_path": str(export_safe_model_dir),
            "export_safe_model": model_summary,
            "warnings": captured_warning_lines,
            "errors": errors,
            "model_config": MODEL_KWARGS,
        }

        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_report(summary, report_path)

        print(f"checkpoint path: {checkpoint_path}")
        print(f"checkpoint object type: {type(checkpoint_obj).__name__}")
        print(f"state_dict confirmation: {state_dict_confirmation}")
        print(f"total parameter tensors: {summary['total_parameter_tensors']}")
        print(f"total parameter count: {summary['total_parameter_count']}")
        print("top-level module breakdown:")
        for name, data in top_breakdown.items():
            print(f"  {name}: tensors={data['tensor_count']} numel={data['numel']}")
        print(f"model class: {model_summary['model_class']}")
        print(f"model total parameter count: {model_summary['total_parameter_count']}")
        print(f"strict load missing keys: {model_summary['strict_load']['missing_keys']}")
        print(f"strict load unexpected keys: {model_summary['strict_load']['unexpected_keys']}")
        print(f"dummy input shape: {model_summary['dummy_input_shape']}")
        print(f"dummy output shape summary: {json.dumps(model_summary['dummy_output_shape'])}")
        print(f"csv inventory: {csv_path}")
        print(f"json summary: {summary_path}")
        print(f"report: {report_path}")
        if captured_warning_lines:
            print("warnings:")
            for warning_line in captured_warning_lines:
                print(f"  {warning_line}")
        return 0

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        failure_summary = {
            "status": "failed",
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_object_type": None,
            "state_dict_confirmation": None,
            "total_parameter_tensors": 0,
            "total_parameter_count": 0,
            "top_level_module_breakdown": {name: {"tensor_count": 0, "numel": 0} for name in TOP_MODULE_ORDER},
            "layer_type_breakdown": {},
            "mapping_rule_breakdown": {},
            "uncertain_or_unknown_parameter_groups": [],
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "export_safe_model_path": str(export_safe_model_dir),
            "export_safe_model": {
                "model_class": None,
                "total_parameter_count": 0,
                "strict_load": {
                    "attempted": False,
                    "success": False,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "error": None,
                },
                "dummy_input_shape": [1, 3, 224, 224],
                "dummy_output_shape": None,
            },
            "warnings": captured_warning_lines,
            "errors": errors + traceback.format_exc().splitlines(),
            "model_config": MODEL_KWARGS,
        }
        summary_path.write_text(json.dumps(failure_summary, indent=2), encoding="utf-8")
        write_report(failure_summary, report_path)
        print(f"inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"failure summary: {summary_path}", file=sys.stderr)
        print(f"failure report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
