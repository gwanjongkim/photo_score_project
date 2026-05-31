# ICAA QAT 적용 가능성을 보수적으로 점검하는 스크립트
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import json
import os
import sys
import traceback
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/icaa_qat_attempt_20260601"
SAVED_MODEL_PATH = ROOT / "outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model"
CHECKPOINT_PATH = ROOT / "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
TF_NATIVE_ROOT = ROOT / "experiments/icaa_tf_native"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    return value


def probe_import(module_name: str) -> dict[str, Any]:
    try:
        module = __import__(module_name)
        return {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
        }
    except Exception as exc:
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def is_tensor_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "numel") and hasattr(value, "dtype")


def extract_state_dict(checkpoint_obj: Any) -> tuple[Any, str]:
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


def saved_model_probe(tf: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(SAVED_MODEL_PATH),
        "exists": SAVED_MODEL_PATH.exists(),
        "tf_saved_model_load": {"ok": False},
        "keras_load_model": {"ok": False},
    }
    if not SAVED_MODEL_PATH.exists():
        return result

    try:
        loaded = tf.saved_model.load(str(SAVED_MODEL_PATH))
        signatures = getattr(loaded, "signatures", {})
        result["tf_saved_model_load"] = {
            "ok": True,
            "object_type": type(loaded).__name__,
            "signatures": list(signatures.keys()),
        }
        if "serving_default" in signatures:
            infer = signatures["serving_default"]
            result["tf_saved_model_load"]["structured_input_signature"] = str(infer.structured_input_signature)
            result["tf_saved_model_load"]["structured_outputs"] = str(infer.structured_outputs)
    except Exception as exc:
        result["tf_saved_model_load"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc().splitlines(),
        }

    try:
        loaded_keras = tf.keras.models.load_model(str(SAVED_MODEL_PATH))
        result["keras_load_model"] = {
            "ok": True,
            "object_type": type(loaded_keras).__name__,
            "is_keras_model": isinstance(loaded_keras, tf.keras.Model),
        }
    except Exception as exc:
        result["keras_load_model"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return result


def source_model_probe(tf: Any, torch: Any | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_exists": CHECKPOINT_PATH.exists(),
        "source_root": str(TF_NATIVE_ROOT),
        "import_ok": False,
        "instantiate_ok": False,
        "load_pytorch_weights_ok": False,
        "clone_model_ok": False,
        "model_kind": "unknown",
    }
    if torch is None:
        result["blocked_reason"] = "torch is not available in this interpreter, so the PyTorch checkpoint cannot be loaded"
        return result

    try:
        sys.path.insert(0, str(TF_NATIVE_ROOT))
        from tf_models.tf_dat_model import TFICAA17KDAT

        result["import_ok"] = True
        model = TFICAA17KDAT(epsilon=1e-5)
        model.build_for_input_shape((1, 224, 224, 3))
        result["instantiate_ok"] = True
        result["is_keras_model"] = isinstance(model, tf.keras.Model)
        result["model_class"] = type(model).__name__
        result["model_kind"] = "subclassed_keras_model"
        result["has_functional_inputs"] = bool(getattr(model, "inputs", None))
        result["has_functional_outputs"] = bool(getattr(model, "outputs", None))
        result["layer_count"] = len(getattr(model, "layers", []))
        result["layer_classes"] = sorted({type(layer).__name__ for layer in model.layers})

        checkpoint_obj = torch.load(CHECKPOINT_PATH, map_location="cpu")
        state_dict, state_dict_note = extract_state_dict(checkpoint_obj)
        model.load_from_pytorch_state_dict(state_dict)
        result["load_pytorch_weights_ok"] = True
        result["state_dict_note"] = state_dict_note
        result["state_dict_key_count"] = len(state_dict)

        try:
            cloned = tf.keras.models.clone_model(model)
            result["clone_model_ok"] = True
            result["clone_model_type"] = type(cloned).__name__
        except Exception as exc:
            result["clone_model_ok"] = False
            result["clone_model_error_type"] = type(exc).__name__
            result["clone_model_error"] = str(exc)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc().splitlines()
    return result


def tfmot_probe(tf: Any, tfmot_import: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tfmot_available": tfmot_import["available"],
        "quantize_model_available": False,
        "quantize_apply_available": False,
    }
    if not tfmot_import["available"]:
        result["blocked_reason"] = "tensorflow_model_optimization is not installed"
        return result
    try:
        import tensorflow_model_optimization as tfmot

        qat = tfmot.quantization.keras
        result["quantize_model_available"] = hasattr(qat, "quantize_model")
        result["quantize_apply_available"] = hasattr(qat, "quantize_apply")
        result["quantize_annotate_layer_available"] = hasattr(qat, "quantize_annotate_layer")
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    return result


def write_markdown_report(path: Path, audit: dict[str, Any]) -> None:
    imports = audit["imports"]
    saved = audit["saved_model"]
    source = audit["source_model"]
    tfmot = audit["tfmot"]
    direct_possible = audit["direct_qat_possible"]
    safe = audit["qat_can_be_applied_safely"]
    blockers = audit["blockers"]

    lines = [
        "# ICAA QAT Feasibility Report",
        "",
        "## Summary",
        f"- **Direct QAT possible**: {direct_possible}.",
        f"- **QAT can be applied safely in this workspace without new environment changes**: {safe}.",
        f"- **Recommended decision from feasibility**: `{audit['recommended_decision']}`.",
        "",
        "## Confirmed Facts",
        f"- **SavedModel exists**: {saved['exists']} at `{SAVED_MODEL_PATH}`.",
        f"- **`tf.saved_model.load` works**: {saved['tf_saved_model_load'].get('ok')}.",
        f"- **`tf.keras.models.load_model` works**: {saved['keras_load_model'].get('ok')}.",
        f"- **Source model import works**: {source.get('import_ok')}.",
        f"- **Source model instantiate/build works**: {source.get('instantiate_ok')}.",
        f"- **PyTorch checkpoint load into source model works**: {source.get('load_pytorch_weights_ok')}.",
        f"- **Source model kind**: `{source.get('model_kind')}`.",
        f"- **Source model has Functional inputs/outputs**: inputs={source.get('has_functional_inputs')}, outputs={source.get('has_functional_outputs')}.",
        f"- **`tf.keras.models.clone_model(source_model)` works**: {source.get('clone_model_ok')}.",
        f"- **`tensorflow_model_optimization` available**: {imports['tensorflow_model_optimization'].get('available')}.",
        f"- **`torch` available in audit interpreter**: {imports['torch'].get('available')}.",
        "",
        "## Blockers",
    ]
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("- None found by this audit.")

    lines.extend(
        [
            "",
            "## SavedModel Details",
            f"- **SavedModel probe**: `{saved}`.",
            "",
            "## Source Model Details",
            f"- **Source probe**: `{source}`.",
            "",
            "## TFMOT Details",
            f"- **TFMOT probe**: `{tfmot}`.",
            "",
            "## Conservative Conclusion",
            "- The source model is recoverable as a subclassed Keras model from code plus the PyTorch checkpoint.",
            "- The production SavedModel itself is a graph export and is not directly recoverable as a trainable Keras model with `tf.keras.models.load_model` in this environment.",
            "- Direct, standard TFMOT QAT is blocked because the required package is absent and the model is a large subclassed/custom-layer DAT rather than a Functional/Sequential model that TFMOT can safely clone and quantize without custom configs.",
            "- Creating manual fake-quant training or installing new packages would be a new unsupported experiment outside this conservative final attempt.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    imports = {
        "tensorflow": probe_import("tensorflow"),
        "tf_keras": probe_import("tf_keras"),
        "torch": probe_import("torch"),
        "tensorflow_model_optimization": probe_import("tensorflow_model_optimization"),
    }

    if not imports["tensorflow"]["available"]:
        raise RuntimeError("TensorFlow is required for the QAT feasibility audit")

    import tensorflow as tf

    torch_module = None
    if imports["torch"]["available"]:
        import torch as torch_module

    audit: dict[str, Any] = {
        "imports": imports,
        "saved_model": saved_model_probe(tf),
        "source_model": source_model_probe(tf, torch_module),
        "tfmot": tfmot_probe(tf, imports["tensorflow_model_optimization"]),
    }

    blockers: list[str] = []
    if not audit["saved_model"]["keras_load_model"].get("ok"):
        blockers.append("SavedModel is not directly loadable as a trainable Keras model with tf.keras.models.load_model.")
    if not audit["source_model"].get("load_pytorch_weights_ok"):
        blockers.append("Source Keras model could not be fully rebuilt from the PyTorch checkpoint in the audit interpreter.")
    if audit["source_model"].get("model_kind") == "subclassed_keras_model":
        blockers.append("Recoverable source model is subclassed with custom layers, not a Functional/Sequential graph model.")
    if not audit["source_model"].get("clone_model_ok"):
        blockers.append("Source model could not be cloned with tf.keras.models.clone_model, which standard TFMOT QAT commonly requires.")
    if not audit["tfmot"].get("tfmot_available"):
        blockers.append("tensorflow_model_optimization is not installed in the checked interpreters; installing it would modify the environment outside the allowed output directory.")

    direct_qat_possible = not blockers and audit["tfmot"].get("quantize_model_available")
    safe_qat = bool(direct_qat_possible)
    audit["blockers"] = blockers
    audit["direct_qat_possible"] = bool(direct_qat_possible)
    audit["qat_can_be_applied_safely"] = safe_qat
    audit["recommended_decision"] = "E" if not safe_qat else "continue_to_qat_training"

    (OUT_DIR / "qat_feasibility_audit.json").write_text(
        json.dumps(jsonable(audit), indent=2),
        encoding="utf-8",
    )
    write_markdown_report(OUT_DIR / "qat_feasibility_report.md", audit)
    print(json.dumps(jsonable(audit), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
