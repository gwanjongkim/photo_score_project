# ARP resize_with_pad 기술 IQA 데이터셋을 구성한다.
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def _resolve_csv_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_image_path(value: Any, csv_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    return str(path.resolve())


def _load_manifest(
    csv_path: str | Path,
    target_col: str,
    image_col: str,
) -> pd.DataFrame:
    path = _resolve_csv_path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    if image_col not in df.columns:
        raise ValueError(
            f"{path} is missing required image column {image_col!r}; "
            f"found {df.columns.tolist()}"
        )

    if target_col not in df.columns:
        if target_col == "normalized_mos" and "mos" in df.columns:
            df = df.copy()
            df[target_col] = pd.to_numeric(df["mos"], errors="raise") / 100.0
        else:
            raise ValueError(
                f"{path} is missing target column {target_col!r}; "
                f"found {df.columns.tolist()}"
            )

    df = df.copy()
    df[image_col] = [
        _resolve_image_path(value, path.parent) for value in df[image_col].astype(str)
    ]
    df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype("float32")
    return df


def _decode_resize_with_pad(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size, image_size)
    image.set_shape([image_size, image_size, 3])
    return image


def make_arp_iqa_dataset(
    csv_path: str | Path,
    target_col: str = "normalized_mos",
    image_col: str = "image_path",
    image_size: int = 384,
    batch_size: int = 8,
    shuffle: bool = False,
    seed: int = 42,
    cache: bool = False,
    prefetch: bool = True,
) -> tf.data.Dataset:
    df = _load_manifest(
        csv_path=csv_path,
        target_col=target_col,
        image_col=image_col,
    )
    paths = df[image_col].astype(str).tolist()
    targets = df[target_col].astype("float32").tolist()

    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        dataset = dataset.shuffle(
            min(len(df), 10000),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_resize_with_pad(path, image_size=image_size)
        return image, tf.reshape(tf.cast(target, tf.float32), (1,))

    dataset = dataset.map(_map_fn, num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.batch(batch_size)
    if prefetch:
        dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def inspect_csv(
    csv_path: str | Path,
    target_col: str = "normalized_mos",
    image_col: str = "image_path",
    validate_paths: bool = False,
) -> dict[str, Any]:
    df = _load_manifest(
        csv_path=csv_path,
        target_col=target_col,
        image_col=image_col,
    )
    targets = df[target_col].astype("float32")
    summary: dict[str, Any] = {
        "csv_path": str(_resolve_csv_path(csv_path)),
        "row_count": int(len(df)),
        "image_col": image_col,
        "target_col": target_col,
        "target_min": float(targets.min()) if len(targets) else None,
        "target_max": float(targets.max()) if len(targets) else None,
        "target_mean": float(targets.mean()) if len(targets) else None,
        "target_std": float(targets.std(ddof=0)) if len(targets) else None,
        "validated_paths": bool(validate_paths),
    }
    if validate_paths:
        exists = df[image_col].map(lambda value: Path(value).is_file())
        summary["existing_paths"] = int(exists.sum())
        summary["missing_paths"] = int((~exists).sum())
    return summary


__all__ = ["inspect_csv", "make_arp_iqa_dataset"]
