# 외부 A-LAMP 패치 JSONL을 VGG16 멀티패치 입력으로 변환한다.
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
from PIL import Image


PREPROCESSING_MODE = "tf.keras.applications.vgg16.preprocess_input_rgb_float_pixels_0_255"


def load_jsonl_records(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= int(max_samples):
                break
    return records


def resolve_image_path(record: dict[str, Any], repo_root: str | Path = ".") -> Path | None:
    value = record.get("resolved_image_path") or record.get("image_path")
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path if path.exists() else None
    candidate = Path(repo_root) / path
    return candidate if candidate.exists() else None


def label_from_record(record: dict[str, Any], label_threshold: float = 5.0) -> np.ndarray:
    if "mean_score" in record and record["mean_score"] is not None:
        label = 1.0 if float(record["mean_score"]) > float(label_threshold) else 0.0
    elif "label" in record and record["label"] is not None:
        label = float(record["label"])
    else:
        raise ValueError("Record requires mean_score or label.")
    return np.asarray([label], dtype=np.float32)


def label_summary(records: list[dict[str, Any]], label_threshold: float = 5.0) -> dict[str, Any]:
    labels = [float(label_from_record(record, label_threshold=label_threshold)[0]) for record in records]
    positives = int(sum(1 for value in labels if value >= 0.5))
    negatives = int(len(labels) - positives)
    return {
        "total": int(len(labels)),
        "positive": positives,
        "negative": negatives,
        "label_threshold": float(label_threshold),
        "label_rule": "mean_score > label_threshold when mean_score is present, else existing label",
    }


def _boxes_from_record(record: dict[str, Any], width: int, height: int) -> list[list[float]]:
    boxes = record.get("patch_boxes") or record.get("boxes_abs_xyxy")
    if boxes:
        return [[float(value) for value in box] for box in boxes]

    norm_boxes = record.get("boxes_norm_xyxy")
    if norm_boxes:
        return [
            [
                float(box[0]) * width,
                float(box[1]) * height,
                float(box[2]) * width,
                float(box[3]) * height,
            ]
            for box in norm_boxes
        ]

    raise ValueError("Record requires patch_boxes, boxes_abs_xyxy, or boxes_norm_xyxy.")


def _clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"Expected 4 coordinates per patch box, got {len(box)}.")
    x1, y1, x2, y2 = box
    left = max(0, min(width, int(math.floor(x1))))
    top = max(0, min(height, int(math.floor(y1))))
    right = max(0, min(width, int(math.ceil(x2))))
    bottom = max(0, min(height, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid patch box after clipping: {box} -> {(left, top, right, bottom)}.")
    return left, top, right, bottom


def load_patch_tensor(
    record: dict[str, Any],
    *,
    image_path: Path,
    patch_size: int = 224,
    patch_count: int = 5,
) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        boxes = _boxes_from_record(record, width=width, height=height)
        if len(boxes) < patch_count:
            raise ValueError(f"Expected at least {patch_count} patch boxes, got {len(boxes)}.")

        crops: list[np.ndarray] = []
        for box in boxes[:patch_count]:
            clipped_box = _clip_box(box, width=width, height=height)
            crop = image.crop(clipped_box)
            crop = crop.resize((patch_size, patch_size), Image.BILINEAR)
            crops.append(np.asarray(crop, dtype=np.float32))

    patches = np.asarray(crops, dtype=np.float32)
    return tf.keras.applications.vgg16.preprocess_input(patches).astype(np.float32)


def iter_examples(
    records: list[dict[str, Any]],
    *,
    repo_root: str | Path = ".",
    patch_size: int = 224,
    patch_count: int = 5,
    label_threshold: float = 5.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    skipped = 0
    for record in records:
        image_path = resolve_image_path(record, repo_root=repo_root)
        if image_path is None:
            skipped += 1
            continue
        try:
            patches = load_patch_tensor(
                record,
                image_path=image_path,
                patch_size=patch_size,
                patch_count=patch_count,
            )
            label = label_from_record(record, label_threshold=label_threshold)
        except Exception as exc:
            skipped += 1
            logging.warning("Skipping %s because patch loading failed: %s", image_path, exc)
            continue
        yield patches, label
    if skipped:
        logging.info("Skipped %s unusable external-patch records.", skipped)


def make_external_patch_dataset(
    records: list[dict[str, Any]],
    *,
    repo_root: str | Path = ".",
    patch_size: int = 224,
    patch_count: int = 5,
    batch_size: int = 4,
    label_threshold: float = 5.0,
    training: bool = False,
    repeat: bool = False,
    shuffle_seed: int = 42,
    class_weights: dict[int, float] | None = None,
    random_horizontal_flip: bool = False,
) -> tf.data.Dataset:
    normalized_class_weights = (
        {int(label): float(weight) for label, weight in class_weights.items()}
        if class_weights is not None
        else None
    )
    flip_enabled = bool(training and random_horizontal_flip)
    rng = np.random.default_rng(shuffle_seed)

    def generator() -> Iterator[Any]:
        for patches, label in iter_examples(
            records,
            repo_root=repo_root,
            patch_size=patch_size,
            patch_count=patch_count,
            label_threshold=label_threshold,
        ):
            if flip_enabled and rng.random() < 0.5:
                patches = np.flip(patches, axis=2).copy()
            if normalized_class_weights is None:
                yield patches, label
                continue
            label_key = int(float(label[0]) >= 0.5)
            yield patches, label, np.asarray(normalized_class_weights[label_key], dtype=np.float32)

    output_signature: tuple[tf.TensorSpec, ...] = (
        tf.TensorSpec(shape=(patch_count, patch_size, patch_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    if normalized_class_weights is not None:
        output_signature = (
            *output_signature,
            tf.TensorSpec(shape=(), dtype=tf.float32),
        )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    )
    if training:
        dataset = dataset.shuffle(
            min(max(len(records), 1), 512),
            seed=shuffle_seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)
