from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from src.models.rgnet import GraphConvolution, RegionGraphBuilder, RegionWeightedPooling, build_rgnet_model
from src.models.nima_distribution import (
    build_nima_distribution_model,
    distribution_mean_score,
    emd_loss,
    mean_score_mae,
)


@dataclass(frozen=True)
class TFLitePreset:
    name: str
    display_name: str
    task: str
    checkpoint_path: str
    weights_path: str
    saved_model_dir: str
    default_output_name: str
    input_size: int
    input_dtype: str
    input_color_format: str
    normalization: str
    output_shape: tuple[int, ...]
    output_summary: str
    output_interpretation: str
    output_postprocess: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "preset": self.name,
            "display_name": self.display_name,
            "task": self.task,
            "input": {
                "image_size": [self.input_size, self.input_size],
                "dtype": self.input_dtype,
                "color_format": self.input_color_format,
                "tensor_layout": "NHWC",
                "normalization": self.normalization,
            },
            "output": {
                "shape": list(self.output_shape),
                "summary": self.output_summary,
                "interpretation": self.output_interpretation,
                "postprocess": self.output_postprocess,
            },
        }


PRESETS: dict[str, TFLitePreset] = {
    "koniq": TFLitePreset(
        name="koniq",
        display_name="KonIQ Technical Regressor",
        task="technical_quality_regression",
        checkpoint_path="checkpoints/technical_koniq_gpu/final_model.keras",
        weights_path="checkpoints/technical_koniq_gpu/best.weights.h5",
        saved_model_dir="checkpoints/technical_koniq_gpu/saved_model",
        default_output_name="koniq_mobile.tflite",
        input_size=224,
        input_dtype="float32",
        input_color_format="RGB",
        normalization="pixel_value / 255.0",
        output_shape=(1, 1),
        output_summary="Single scalar technical-quality score.",
        output_interpretation="Raw MOS-like score. Current repo mobile scoring normalizes with clip(score / 100.0, 0, 1).",
        output_postprocess="Squeeze to scalar float. Optional normalized score = clip(raw_score / 100.0, 0, 1).",
    ),
    "flive_image": TFLitePreset(
        name="flive_image",
        display_name="FLIVE Full-Image Regressor",
        task="technical_quality_regression",
        checkpoint_path="checkpoints/technical_flive_image_gpu/final_model.keras",
        weights_path="checkpoints/technical_flive_image_gpu/best.weights.h5",
        saved_model_dir="checkpoints/technical_flive_image_gpu/saved_model",
        default_output_name="flive_image_mobile.tflite",
        input_size=224,
        input_dtype="float32",
        input_color_format="RGB",
        normalization="pixel_value / 255.0",
        output_shape=(1, 1),
        output_summary="Single scalar image-quality score.",
        output_interpretation="Raw MOS-like score. Current repo mobile scoring normalizes with clip(score / 100.0, 0, 1).",
        output_postprocess="Squeeze to scalar float. Optional normalized score = clip(raw_score / 100.0, 0, 1).",
    ),
    "aadb": TFLitePreset(
        name="aadb",
        display_name="AADB Aesthetic Regressor",
        task="aesthetic_regression",
        checkpoint_path="checkpoints/composition_aadb_gpu/final_model.keras",
        weights_path="checkpoints/composition_aadb_gpu/best.weights.h5",
        saved_model_dir="checkpoints/composition_aadb_gpu/saved_model",
        default_output_name="aadb_mobile.tflite",
        input_size=224,
        input_dtype="float32",
        input_color_format="RGB",
        normalization="pixel_value / 255.0",
        output_shape=(1, 1),
        output_summary="Single scalar aesthetic score.",
        output_interpretation="Sigmoid output already in [0, 1]. Higher is better.",
        output_postprocess="Squeeze to scalar float and optionally clip to [0, 1].",
    ),
    "rgnet": TFLitePreset(
        name="rgnet",
        display_name="RGNet AADB Regressor",
        task="aesthetic_regression",
        checkpoint_path="checkpoints/rgnet_aadb_gpu/final_model.keras",
        weights_path="checkpoints/rgnet_aadb_gpu/best.weights.h5",
        saved_model_dir="checkpoints/rgnet_aadb_gpu/saved_model",
        default_output_name="rgnet_aadb_gpu.tflite",
        input_size=256,
        input_dtype="float32",
        input_color_format="RGB",
        normalization="pixel_value / 255.0",
        output_shape=(1, 1),
        output_summary="Single scalar RGNet aesthetic score.",
        output_interpretation="Sigmoid output already in [0, 1]. Higher is better.",
        output_postprocess="Squeeze to scalar float and optionally clip to [0, 1].",
    ),
    "nima": TFLitePreset(
        name="nima",
        display_name="NIMA Distribution Model",
        task="aesthetic_distribution_regression",
        checkpoint_path="checkpoints/nima_ava_gpu/final_model.keras",
        weights_path="checkpoints/nima_ava_gpu/best.weights.h5",
        saved_model_dir="checkpoints/nima_ava_gpu/saved_model",
        default_output_name="nima_mobile.tflite",
        input_size=224,
        input_dtype="float32",
        input_color_format="RGB",
        normalization="pixel_value / 255.0",
        output_shape=(1, 10),
        output_summary="10-bin aesthetic score distribution over scores 1..10.",
        output_interpretation="Softmax distribution. Derive mean score with sum(prob[i] * (i + 1)).",
        output_postprocess="Use distribution directly or compute mean_score = sum(distribution * [1..10]).",
    ),
}


def preset_choices() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> TFLitePreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown preset '{name}'. Available: {', '.join(preset_choices())}") from exc


def resolve_paths(preset_name: str, model_path: str | None = None) -> tuple[Path, Path, Path]:
    preset = get_preset(preset_name)
    if model_path is None:
        checkpoint_path = Path(preset.checkpoint_path)
        weights_path = Path(preset.weights_path)
        saved_model_dir = Path(preset.saved_model_dir)
    else:
        user_path = Path(model_path)
        if user_path.is_dir() and (user_path / "saved_model.pb").exists():
            saved_model_dir = user_path
            checkpoint_path = user_path.parent / "final_model.keras"
            weights_path = user_path.parent / "best.weights.h5"
        elif user_path.is_dir() and (user_path / "saved_model").is_dir():
            checkpoint_path = user_path / "final_model.keras"
            weights_path = user_path / "best.weights.h5"
            saved_model_dir = user_path / "saved_model"
        elif user_path.name == "best.weights.h5":
            checkpoint_path = user_path.parent / "final_model.keras"
            weights_path = user_path
            saved_model_dir = user_path.parent / "saved_model"
        elif user_path.suffix in {".keras", ".h5"}:
            checkpoint_path = user_path
            weights_path = user_path.parent / "best.weights.h5"
            saved_model_dir = user_path.parent / "saved_model"
        else:
            raise FileNotFoundError(
                f"Could not resolve SavedModel from {user_path}. Provide a checkpoint directory, .keras file, or SavedModel dir."
            )

    if not saved_model_dir.exists():
        raise FileNotFoundError(f"SavedModel directory not found: {saved_model_dir}")
    return checkpoint_path, weights_path, saved_model_dir


def build_export_model(preset_name: str) -> tf.keras.Model:
    preset = get_preset(preset_name)
    input_shape = (preset.input_size, preset.input_size, 3)

    if preset_name in {"koniq", "flive_image"}:
        base = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=input_shape)
        inputs = tf.keras.Input(shape=input_shape)
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="linear")(x)
        return tf.keras.Model(inputs, outputs, name="technical_regressor")

    if preset_name == "aadb":
        base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights=None, input_shape=input_shape)
        inputs = tf.keras.Input(shape=input_shape)
        x = base(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, outputs, name="aadb_regressor")

    if preset_name == "nima":
        return build_nima_distribution_model(input_shape=input_shape, backbone_weights=None)

    if preset_name == "rgnet":
        return build_rgnet_model(input_shape=input_shape, backbone_weights=None)

    raise KeyError(f"No offline rebuild path for preset '{preset_name}'.")


def load_image_array(image_path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype="float32") / 255.0


def load_nima_image_array(image_path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    resize_side = max(256, image_size)
    image = image.resize((resize_side, resize_side), Image.Resampling.BILINEAR)
    left = (resize_side - image_size) // 2
    top = (resize_side - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))
    return np.asarray(image, dtype="float32") / 255.0


def load_preset_image_array(preset_name: str, image_path: str | Path, image_size: int) -> np.ndarray:
    if preset_name == "nima":
        return load_nima_image_array(image_path, image_size=image_size)
    return load_image_array(image_path, image_size=image_size)


def load_source_model(preset_name: str, checkpoint_path: str | Path) -> tf.keras.Model:
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.name == "best.weights.h5":
        model = build_export_model(preset_name)
        model.load_weights(checkpoint_path)
        return model

    checkpoint_path_str = str(checkpoint_path)
    if preset_name == "nima":
        return tf.keras.models.load_model(
            checkpoint_path_str,
            compile=False,
            custom_objects={"emd_loss": emd_loss, "mean_score_mae": mean_score_mae},
        )
    if preset_name == "rgnet":
        return tf.keras.models.load_model(
            checkpoint_path_str,
            compile=False,
            custom_objects={
                "RegionGraphBuilder": RegionGraphBuilder,
                "GraphConvolution": GraphConvolution,
                "RegionWeightedPooling": RegionWeightedPooling,
            },
        )
    return tf.keras.models.load_model(checkpoint_path_str, compile=False)


def describe_output(preset_name: str, output: np.ndarray) -> dict[str, object]:
    output = np.asarray(output)
    squeezed = np.squeeze(output)
    summary: dict[str, object] = {
        "raw_shape": list(output.shape),
        "dtype": str(output.dtype),
    }

    if preset_name == "nima":
        distribution = np.asarray(squeezed, dtype="float32").reshape(-1)
        mean_score = float(
            distribution_mean_score(tf.convert_to_tensor(distribution[None, :], dtype=tf.float32))[0].numpy()
        )
        summary.update(
            {
                "distribution": distribution.tolist(),
                "distribution_sum": float(distribution.sum()),
                "mean_score": mean_score,
                "plausible": bool(np.all(np.isfinite(distribution)) and abs(float(distribution.sum()) - 1.0) < 0.05),
            }
        )
        return summary

    score = float(np.asarray(squeezed, dtype="float32"))
    summary.update({"score": score, "plausible": bool(np.isfinite(score))})
    if preset_name in {"aadb", "rgnet"}:
        summary["within_unit_interval"] = bool(0.0 <= score <= 1.0)
    if preset_name in {"koniq", "flive_image"}:
        summary["normalized_score"] = float(np.clip(score / 100.0, 0.0, 1.0))
    return summary
