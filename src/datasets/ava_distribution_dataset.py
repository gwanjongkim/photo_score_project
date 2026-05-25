"""
AVA distribution dataset utilities for paper-faithful NIMA-style training.

Based on:
- NIMA: Neural Image Assessment
- AVA: A Large-Scale Database for Aesthetic Visual Analysis

Faithful parts:
- trains on 10-bin score distributions
- supports mean score derived from the distribution
- uses AVA-style vote histograms when present

Approximated parts:
- CSV schema auto-detection is repo-practical rather than tied to one exact AVA export
- train/val split generation is handled by a lightweight preprocessing helper

Expected CSV columns:
- required: image_path
- one of:
  - dist_1..dist_10
  - score_1..score_10
  - vote_1..vote_10

Optional output/metadata columns:
- mean_score
- total_votes
- relative_path
- split
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import tensorflow as tf

from src.utils.image_utils import collect_invalid_image_paths


AUTOTUNE = tf.data.AUTOTUNE
_DIST_PREFIXES = ("dist_", "score_", "vote_")


def _decode_image(path: tf.Tensor, image_size: tuple[int, int], training: bool) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])

    resize_h = tf.maximum(256, image_size[0])
    resize_w = tf.maximum(256, image_size[1])
    img = tf.image.resize(img, [resize_h, resize_w], method="bilinear")
    if training:
        img = tf.image.random_crop(img, [image_size[0], image_size[1], 3])
        img = tf.image.random_flip_left_right(img)
    else:
        img = tf.image.resize_with_crop_or_pad(img, image_size[0], image_size[1])

    img = tf.ensure_shape(img, [image_size[0], image_size[1], 3])
    return img



def infer_distribution_columns(df: pd.DataFrame) -> list[str]:
    lowered = {col.lower(): col for col in df.columns}
    for prefix in _DIST_PREFIXES:
        cols = [lowered.get(f"{prefix}{idx}") for idx in range(1, 11)]
        if all(col is not None for col in cols):
            return [str(col) for col in cols]
    raise ValueError(
        "AVA CSV must contain one of dist_1..dist_10, score_1..score_10, or vote_1..vote_10 columns."
    )


def load_ava_distribution_frame(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    validated_cache_path = csv_path.with_name(f"{csv_path.stem}.validated.csv")
    if validated_cache_path.exists() and validated_cache_path.stat().st_mtime >= csv_path.stat().st_mtime:
        cached = pd.read_csv(validated_cache_path)
        print(f"[NIMA] Using cached validated CSV: {validated_cache_path}")
        return cached

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        raise ValueError("AVA CSV must contain an image_path column.")

    dist_cols = infer_distribution_columns(df)
    dist = df[dist_cols].astype("float32")
    row_sums = dist.sum(axis=1)
    nonzero = row_sums > 0
    if not nonzero.all():
        raise ValueError("AVA CSV contains empty distribution rows.")

    if not dist_cols[0].lower().startswith("dist_"):
        dist = dist.div(row_sums, axis=0)

    out = df.copy()
    for idx in range(10):
        out[f"dist_{idx + 1}"] = dist.iloc[:, idx]

    if "mean_score" not in out.columns:
        scores = tf.range(1, 11, dtype=tf.float32).numpy()
        out["mean_score"] = (dist.to_numpy() * scores[None, :]).sum(axis=1)

    invalid_paths = collect_invalid_image_paths(out["image_path"].astype(str).tolist())
    if invalid_paths:
        invalid_set = set(invalid_paths)
        out = out[~out["image_path"].astype(str).isin(invalid_set)].reset_index(drop=True)
        print(f"[NIMA] Filtered {len(invalid_paths)} invalid image rows from {csv_path}.")
        log_path = Path(csv_path).with_suffix(".invalid_images.txt")
        log_path.write_text("\n".join(invalid_paths) + ("\n" if invalid_paths else ""), encoding="utf-8")
        print(f"[NIMA] Wrote invalid image log to {log_path}")

    if out.empty:
        raise ValueError(f"{csv_path} has no valid images after validation.")
    out.to_csv(validated_cache_path, index=False)
    print(f"[NIMA] Wrote validated CSV cache to {validated_cache_path}")
    return out


def make_ava_distribution_dataset(
    csv_path: str | Path | None = None,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    training: bool = False,
    shuffle: bool = True,
    frame: pd.DataFrame | None = None,
    repeat: bool = False,
    drop_remainder: bool = False,
) -> tf.data.Dataset:
    if frame is None:
        if csv_path is None:
            raise ValueError("Provide either csv_path or frame.")
        df = load_ava_distribution_frame(csv_path)
    else:
        df = frame.copy()
    paths = df["image_path"].astype(str).tolist()
    dist = df[[f"dist_{idx}" for idx in range(1, 11)]].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, dist))
    if training and shuffle:
        ds = ds.shuffle(min(len(df), 10000), reshuffle_each_iteration=True)
    if repeat:
        ds = ds.repeat()

    def _map_fn(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_image(path, image_size=image_size, training=training)
        target.set_shape([10])
        return image, target

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(AUTOTUNE)
