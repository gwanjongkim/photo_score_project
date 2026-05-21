# TFLite 미적 평가 모델을 로드하고 이미지별 원점수를 계산합니다.
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from model_registry import ModelSpec


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_PATCH_SCALES = (0.245, 0.35, 0.4725)


@dataclass(frozen=True)
class ModelPrediction:
    score: float
    details: dict[str, Any]


def import_tflite_interpreter():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ModuleNotFoundError:
        import tensorflow as tf

        return tf.lite.Interpreter


def iter_image_paths(input_dir: Path, *, recursive: bool = False, extensions: set[str] | None = None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    allowed = extensions or SUPPORTED_EXTENSIONS
    pattern = "**/*" if recursive else "*"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() in allowed)


def load_image_rgb(image_path: Path) -> Image.Image:
    with Image.open(image_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
        image.load()
    if image.width < 2 or image.height < 2:
        raise ValueError(f"Image dimensions are too small: {image.width}x{image.height}")
    return image


def normalize_array(arr: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "zero_to_one":
        return arr.astype(np.float32) / np.float32(255.0)
    if normalization == "zero_to_255":
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported normalization: {normalization}")


def resize_to_model_input(image: Image.Image, *, width: int, height: int, normalization: str) -> np.ndarray:
    resized = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    arr = normalize_array(np.asarray(resized), normalization)
    return arr[None, ...].astype(np.float32)


def resize_with_pad(image: Image.Image, *, target_size: int, normalization: str) -> np.ndarray:
    target_size = int(target_size)
    scale = min(target_size / image.width, target_size / image.height)
    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_width) // 2
    top = (target_size - new_height) // 2
    canvas.paste(resized, (left, top))
    return normalize_array(np.asarray(canvas), normalization).astype(np.float32)


def _normalize_map(score_map: np.ndarray) -> np.ndarray:
    min_value = float(np.min(score_map))
    max_value = float(np.max(score_map))
    if not math.isfinite(min_value) or not math.isfinite(max_value) or max_value <= min_value:
        return np.zeros_like(score_map, dtype=np.float32)
    return ((score_map - min_value) / (max_value - min_value)).astype(np.float32)


def _box_mean(channel: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(channel.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    height, width = channel.shape
    bottom = np.arange(kernel_size, kernel_size + height)
    top = np.arange(0, height)
    right = np.arange(kernel_size, kernel_size + width)
    left = np.arange(0, width)
    sums = (
        integral[np.ix_(bottom, right)]
        - integral[np.ix_(top, right)]
        - integral[np.ix_(bottom, left)]
        + integral[np.ix_(top, left)]
    )
    return (sums / float(kernel_size * kernel_size)).astype(np.float32)


def compute_saliency_map(image_unit: np.ndarray) -> np.ndarray:
    gray = (
        0.299 * image_unit[..., 0]
        + 0.587 * image_unit[..., 1]
        + 0.114 * image_unit[..., 2]
    ).astype(np.float32)
    grad_y, grad_x = np.gradient(gray)
    edge = np.sqrt(np.square(grad_x) + np.square(grad_y)).astype(np.float32)

    gray_mean = _box_mean(gray)
    gray_sq_mean = _box_mean(np.square(gray))
    luminance_variance = np.maximum(gray_sq_mean - np.square(gray_mean), 0.0).astype(np.float32)

    color_variances = []
    for channel_index in range(3):
        channel = image_unit[..., channel_index]
        mean = _box_mean(channel)
        sq_mean = _box_mean(np.square(channel))
        color_variances.append(np.maximum(sq_mean - np.square(mean), 0.0))
    color_variance = np.mean(np.stack(color_variances, axis=-1), axis=-1).astype(np.float32)

    combined = (
        0.50 * _normalize_map(edge)
        + 0.35 * _normalize_map(luminance_variance)
        + 0.15 * _normalize_map(color_variance)
    )
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def _box_iou_xyxy(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    denom = left_area + right_area - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area / denom)


def _mean_saliency_for_box(saliency: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box_xyxy
    left = max(0, min(saliency.shape[1] - 1, int(math.floor(x1))))
    top = max(0, min(saliency.shape[0] - 1, int(math.floor(y1))))
    right = max(left + 1, min(saliency.shape[1], int(math.ceil(x2))))
    bottom = max(top + 1, min(saliency.shape[0], int(math.ceil(y2))))
    return float(np.mean(saliency[top:bottom, left:right]))


def _fixed_anchor_boxes(width: int, height: int, patch_count: int) -> list[dict[str, Any]]:
    short_side = min(width, height)
    crop_size = max(32, int(round(short_side * 0.50)))
    crop_size = min(crop_size, width, height)
    anchors = [(0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    while len(anchors) < patch_count:
        index = len(anchors)
        grid = int(math.ceil(math.sqrt(patch_count)))
        gx = (index % grid + 0.5) / grid
        gy = (index // grid + 0.5) / grid
        anchors.append((gx, min(0.95, gy)))

    boxes: list[dict[str, Any]] = []
    for center_x_ratio, center_y_ratio in anchors[:patch_count]:
        center_x = width * center_x_ratio
        center_y = height * center_y_ratio
        left = max(0.0, min(float(width - crop_size), center_x - crop_size / 2.0))
        top = max(0.0, min(float(height - crop_size), center_y - crop_size / 2.0))
        right = left + crop_size
        bottom = top + crop_size
        boxes.append(
            {
                "box_xyxy": [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))],
                "box_yxyx_normalized": [top / height, left / width, bottom / height, right / width],
                "selection_score": 0.0,
                "source": "fixed_anchor_fallback",
            }
        )
    return boxes


def select_adaptive_patch_boxes(
    image_unit: np.ndarray,
    *,
    patch_count: int,
    patch_scales: tuple[float, ...] = DEFAULT_PATCH_SCALES,
    grid_size: int = 5,
    min_crop_size: int = 64,
    iou_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    height, width = image_unit.shape[:2]
    try:
        saliency = compute_saliency_map(image_unit)
        short_side = min(width, height)
        candidates: list[tuple[float, tuple[float, float, float, float]]] = []
        for scale in patch_scales:
            crop_size = max(float(min_crop_size), float(short_side) * float(scale))
            crop_size = min(crop_size, float(short_side))
            if crop_size < 1.0:
                continue
            centers_y = np.linspace(crop_size / 2.0, float(height) - crop_size / 2.0, int(grid_size))
            centers_x = np.linspace(crop_size / 2.0, float(width) - crop_size / 2.0, int(grid_size))
            for center_y in centers_y:
                for center_x in centers_x:
                    left = max(0.0, min(float(width) - crop_size, float(center_x) - crop_size / 2.0))
                    top = max(0.0, min(float(height) - crop_size, float(center_y) - crop_size / 2.0))
                    box = (left, top, left + crop_size, top + crop_size)
                    candidates.append((_mean_saliency_for_box(saliency, box), box))

        if not candidates:
            return _fixed_anchor_boxes(width, height, patch_count)

        selected: list[tuple[float, tuple[float, float, float, float]]] = []
        for score, box in sorted(candidates, key=lambda item: item[0], reverse=True):
            if all(_box_iou_xyxy(box, existing_box) <= iou_threshold for _, existing_box in selected):
                selected.append((score, box))
            if len(selected) >= patch_count:
                break

        if len(selected) < patch_count:
            fallback = _fixed_anchor_boxes(width, height, patch_count)
            used = [box for _, box in selected]
            for fallback_box in fallback:
                x1, y1, x2, y2 = [float(value) for value in fallback_box["box_xyxy"]]
                box = (x1, y1, x2, y2)
                if all(_box_iou_xyxy(box, existing) <= iou_threshold for existing in used):
                    selected.append((0.0, box))
                    used.append(box)
                if len(selected) >= patch_count:
                    break

        boxes: list[dict[str, Any]] = []
        for score, box in selected[:patch_count]:
            left, top, right, bottom = box
            boxes.append(
                {
                    "box_xyxy": [int(round(left)), int(round(top)), int(round(right)), int(round(bottom))],
                    "box_yxyx_normalized": [top / height, left / width, bottom / height, right / width],
                    "selection_score": float(score),
                    "source": "adaptive_saliency_like",
                }
            )
        while len(boxes) < patch_count:
            boxes.extend(_fixed_anchor_boxes(width, height, patch_count - len(boxes)))
        return boxes[:patch_count]
    except Exception:
        return _fixed_anchor_boxes(width, height, patch_count)


def make_alamp_inputs(
    image: Image.Image,
    *,
    global_size: int,
    patch_size: int,
    patch_count: int,
    normalization: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    image_unit = normalize_array(np.asarray(image), normalization)
    global_view = resize_with_pad(image, target_size=global_size, normalization=normalization)
    boxes = select_adaptive_patch_boxes(image_unit, patch_count=patch_count)
    patches = []
    for box in boxes:
        left, top, right, bottom = box["box_xyxy"]
        patch = image.crop((left, top, right, bottom)).resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        patches.append(normalize_array(np.asarray(patch), normalization))
    inputs = {
        "global_view": global_view[None, ...].astype(np.float32),
        "patches": np.stack(patches, axis=0)[None, ...].astype(np.float32),
    }
    debug = {
        "global_size": int(global_size),
        "patch_size": int(patch_size),
        "patch_count": int(patch_count),
        "patch_boxes": boxes,
    }
    return inputs, debug


class TfliteModelRunner:
    def __init__(self, spec: ModelSpec, *, num_threads: int = 1):
        self.spec = spec
        Interpreter = import_tflite_interpreter()
        self.interpreter = Interpreter(model_path=str(spec.model_path), num_threads=int(num_threads))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, inputs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        if isinstance(inputs, dict):
            for name, value in inputs.items():
                detail = self._input_detail_for_name(name)
                self.interpreter.set_tensor(detail["index"], value.astype(detail["dtype"]))
        else:
            if len(self.input_details) != 1:
                raise ValueError(f"{self.spec.model_id} expects {len(self.input_details)} inputs, not one.")
            detail = self.input_details[0]
            self.interpreter.set_tensor(detail["index"], inputs.astype(detail["dtype"]))
        self.interpreter.invoke()
        return np.asarray(self.interpreter.get_tensor(self.output_details[0]["index"]))

    def _input_detail_for_name(self, logical_name: str) -> dict[str, Any]:
        matches = [detail for detail in self.input_details if logical_name in str(detail["name"])]
        if len(matches) == 1:
            return matches[0]
        if logical_name == "global_view":
            shape_len = 4
        elif logical_name == "patches":
            shape_len = 5
        else:
            raise ValueError(f"Unknown logical input name: {logical_name}")
        shape_matches = [detail for detail in self.input_details if len(detail["shape"]) == shape_len]
        if len(shape_matches) == 1:
            return shape_matches[0]
        names = ", ".join(str(detail["name"]) for detail in self.input_details)
        raise ValueError(f"Could not resolve input {logical_name!r} for {self.spec.model_id}; inputs: {names}")

    def score_image(self, image: Image.Image) -> ModelPrediction:
        if self.spec.runner == "nima_distribution":
            return self._score_nima(image)
        if self.spec.runner == "scalar_tflite":
            return self._score_scalar(image)
        if self.spec.runner == "vector_tflite":
            return self._score_vector(image)
        if self.spec.runner == "alamp_signature":
            return self._score_alamp(image)
        raise ValueError(f"Unsupported runner: {self.spec.runner}")

    def _score_nima(self, image: Image.Image) -> ModelPrediction:
        arr = resize_to_model_input(
            image,
            width=int(self.spec.config["input_width"]),
            height=int(self.spec.config["input_height"]),
            normalization=str(self.spec.config["normalization"]),
        )
        output = np.squeeze(self.predict(arr)).astype(np.float32)
        if output.shape != (10,):
            raise ValueError(f"NIMA output must be shape [1,10] or [10], got {list(output.shape)}.")
        distribution_sum = float(np.sum(output))
        if distribution_sum > 0.0 and math.isfinite(distribution_sum):
            distribution = output / distribution_sum
        else:
            distribution = output
        expected_score = float(np.sum(distribution * np.arange(1, 11, dtype=np.float32)))
        unit_score = float((expected_score - 1.0) / 9.0)
        return ModelPrediction(
            score=unit_score,
            details={
                "expected_score_1_to_10": expected_score,
                "distribution_sum": distribution_sum,
                "distribution": distribution.astype(float).tolist(),
                "raw_distribution": output.astype(float).tolist(),
            },
        )

    def _score_scalar(self, image: Image.Image) -> ModelPrediction:
        arr = resize_to_model_input(
            image,
            width=int(self.spec.config["input_width"]),
            height=int(self.spec.config["input_height"]),
            normalization=str(self.spec.config["normalization"]),
        )
        output = self.predict(arr)
        squeezed = np.squeeze(output)
        if np.asarray(squeezed).shape != ():
            raise ValueError(
                f"{self.spec.model_id} scalar_tflite output must be scalar after squeeze, "
                f"got raw shape {list(np.asarray(output).shape)} and squeezed shape {list(np.asarray(squeezed).shape)}."
            )
        score = float(squeezed)
        return ModelPrediction(score=score, details={"raw_output": np.asarray(output).astype(float).tolist()})

    def _score_vector(self, image: Image.Image) -> ModelPrediction:
        arr = resize_to_model_input(
            image,
            width=int(self.spec.config["input_width"]),
            height=int(self.spec.config["input_height"]),
            normalization=str(self.spec.config["normalization"]),
        )
        output = self.predict(arr)
        raw_output = np.asarray(output)
        squeezed = np.squeeze(raw_output)
        score_index = int(self.spec.config["score_index"])
        if np.asarray(squeezed).shape == ():
            if score_index != 0:
                raise ValueError(
                    f"{self.spec.model_id} vector_tflite output is scalar, so score_index must be 0; "
                    f"got {score_index}. Raw output shape: {list(raw_output.shape)}."
                )
            score = float(squeezed)
            vector_output = [score]
        elif squeezed.ndim == 1:
            if score_index >= int(squeezed.shape[0]):
                raise ValueError(
                    f"{self.spec.model_id} score_index {score_index} is out of range for vector_tflite "
                    f"output length {int(squeezed.shape[0])}. Raw output shape: {list(raw_output.shape)}."
                )
            score = float(squeezed[score_index])
            vector_output = squeezed.astype(float).tolist()
        else:
            raise ValueError(
                f"{self.spec.model_id} vector_tflite output must be scalar, [N], or [1,N], "
                f"got raw shape {list(raw_output.shape)} and squeezed shape {list(squeezed.shape)}."
            )
        return ModelPrediction(
            score=score,
            details={
                "score_index": score_index,
                "raw_output_shape": list(raw_output.shape),
                "vector_output": vector_output,
                "raw_output": raw_output.astype(float).tolist(),
            },
        )

    def _score_alamp(self, image: Image.Image) -> ModelPrediction:
        inputs, debug = make_alamp_inputs(
            image,
            global_size=int(self.spec.config["global_size"]),
            patch_size=int(self.spec.config["patch_size"]),
            patch_count=int(self.spec.config["patch_count"]),
            normalization=str(self.spec.config["normalization"]),
        )
        output = self.predict(inputs)
        squeezed = np.squeeze(output)
        if np.asarray(squeezed).shape != ():
            raise ValueError(
                f"{self.spec.model_id} alamp_signature output must be scalar after squeeze, "
                f"got raw shape {list(np.asarray(output).shape)} and squeezed shape {list(np.asarray(squeezed).shape)}."
            )
        score = float(squeezed)
        debug["raw_output"] = np.asarray(output).astype(float).tolist()
        return ModelPrediction(score=score, details=debug)
