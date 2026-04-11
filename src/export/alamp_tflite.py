from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from src.datasets.native_size_dataset import prepare_alamp_inputs
from src.models.alamp import LayoutCueAugmentation, WeightedPatchPooling, build_alamp_model


DEFAULT_GLOBAL_SIZE = 384
DEFAULT_PATCH_SIZE = 224
DEFAULT_NUM_PATCHES = 5
DEFAULT_PATCH_SCALES = (0.245, 0.35, 0.4725)


@dataclass(frozen=True)
class AlampPaths:
    checkpoint_dir: Path
    weights_path: Path
    keras_path: Path
    saved_model_dir: Path


def resolve_alamp_paths(checkpoint_dir: str | Path) -> AlampPaths:
    checkpoint_dir = Path(checkpoint_dir)
    return AlampPaths(
        checkpoint_dir=checkpoint_dir,
        weights_path=checkpoint_dir / "best.weights.h5",
        keras_path=checkpoint_dir / "final_model.keras",
        saved_model_dir=checkpoint_dir / "saved_model",
    )


def set_export_policy() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")


def build_rebuild_model(
    global_size: int = DEFAULT_GLOBAL_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    num_patches: int = DEFAULT_NUM_PATCHES,
) -> tf.keras.Model:
    return build_alamp_model(
        global_size=global_size,
        patch_size=patch_size,
        num_patches=num_patches,
        backbone_weights=None,
    )


def load_rebuild_model(
    weights_path: str | Path,
    global_size: int = DEFAULT_GLOBAL_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    num_patches: int = DEFAULT_NUM_PATCHES,
) -> tf.keras.Model:
    set_export_policy()
    model = build_rebuild_model(
        global_size=global_size,
        patch_size=patch_size,
        num_patches=num_patches,
    )
    model.load_weights(Path(weights_path))
    return model


def load_checkpoint_model(keras_path: str | Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        str(keras_path),
        compile=False,
        safe_mode=False,
        custom_objects={
            "LayoutCueAugmentation": LayoutCueAugmentation,
            "WeightedPatchPooling": WeightedPatchPooling,
        },
    )


def summarize_exception(exc: Exception, limit: int = 1200) -> str:
    message = f"{type(exc).__name__}: {exc}"
    message = " ".join(message.split())
    if len(message) > limit:
        return f"{message[:limit]}..."
    return message


def convert_keras_model(model: tf.keras.Model, mode: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()


def convert_saved_model(saved_model_dir: str | Path, mode: str) -> bytes:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if mode == "select_tf_ops":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
    return converter.convert()


def collect_model_io(model: tf.keras.Model) -> dict[str, object]:
    def tensor_json(tensor: tf.Tensor) -> dict[str, object]:
        return {
            "name": str(tensor.name),
            "shape": list(tensor.shape),
            "dtype": tensor.dtype,
        }

    return {
        "inputs": [tensor_json(tensor) for tensor in model.inputs],
        "outputs": [tensor_json(tensor) for tensor in model.outputs],
    }


def collect_saved_model_signature(saved_model_dir: str | Path) -> dict[str, object]:
    loaded = tf.saved_model.load(str(saved_model_dir))
    signature = loaded.signatures["serving_default"]
    inputs = {
        key: {
            "shape": list(spec.shape),
            "dtype": spec.dtype.name,
        }
        for key, spec in signature.structured_input_signature[1].items()
    }
    outputs = {
        key: {
            "shape": list(spec.shape),
            "dtype": spec.dtype.name,
        }
        for key, spec in signature.structured_outputs.items()
    }
    return {"inputs": inputs, "outputs": outputs}


def load_image_tensor(image_path: str | Path) -> tf.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image, dtype="float32") / 255.0
    return tf.convert_to_tensor(image_array, dtype=tf.float32)


def prepare_sample_inputs(
    image_path: str | Path,
    global_size: int = DEFAULT_GLOBAL_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    num_patches: int = DEFAULT_NUM_PATCHES,
    patch_scales: tuple[float, ...] = DEFAULT_PATCH_SCALES,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    image = load_image_tensor(image_path)
    global_view, patches, boxes, proposal_scores = prepare_alamp_inputs(
        image=image,
        global_size=global_size,
        patch_size=patch_size,
        num_patches=num_patches,
        patch_scales=patch_scales,
    )
    inputs = {
        "global_view": global_view[None, ...].numpy().astype(np.float32),
        "patches": patches[None, ...].numpy().astype(np.float32),
    }
    debug = {
        "boxes": boxes.numpy().astype(np.float32).tolist(),
        "proposal_scores": proposal_scores.numpy().astype(np.float32).tolist(),
    }
    return inputs, debug


def tensor_detail_to_json(detail: dict[str, object]) -> dict[str, object]:
    return {
        "name": str(detail["name"]),
        "shape": [int(dim) for dim in detail["shape"]],
        "shape_signature": [int(dim) for dim in detail["shape_signature"]],
        "dtype": np.dtype(detail["dtype"]).name,
        "index": int(detail["index"]),
    }


def _match_tflite_input_key(detail: dict[str, object]) -> str:
    shape = tuple(int(dim) for dim in detail["shape"])
    if shape[-3:] == (384, 384, 3):
        return "global_view"
    if shape[-4:] == (5, 224, 224, 3):
        return "patches"
    name = str(detail["name"])
    if "global" in name:
        return "global_view"
    if "patch" in name:
        return "patches"
    raise KeyError(f"Unable to map TFLite input detail {name} with shape {shape}")


def run_tflite_inference(model_path: str | Path, inputs: dict[str, np.ndarray], num_threads: int = 1) -> dict[str, object]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=num_threads)
    input_details = interpreter.get_input_details()
    for detail in input_details:
        key = _match_tflite_input_key(detail)
        interpreter.resize_tensor_input(detail["index"], inputs[key].shape, strict=False)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        key = _match_tflite_input_key(detail)
        value = inputs[key].astype(detail["dtype"], copy=False)
        interpreter.set_tensor(detail["index"], value)

    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return {
        "output": np.asarray(output, dtype=np.float32),
        "input_details": [tensor_detail_to_json(detail) for detail in input_details],
        "output_details": [tensor_detail_to_json(detail) for detail in output_details],
    }


def run_source_prediction(model: tf.keras.Model, inputs: dict[str, np.ndarray]) -> np.ndarray:
    prediction = model.predict(inputs, verbose=0)
    return np.asarray(prediction, dtype=np.float32)


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
