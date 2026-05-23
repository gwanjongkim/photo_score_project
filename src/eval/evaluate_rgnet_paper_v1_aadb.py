# RGNet 논문 지향 v1 AADB 회귀 모델을 평가한다.
from __future__ import annotations

import argparse
import json
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.models.rgnet_paper_v1 import MODEL_VARIANT, get_rgnet_paper_v1_custom_objects


PREPROCESS_BACKENDS = ("tf", "pil_bilinear")


def _read_yaml(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping config in {path}")
    return loaded


def _cfg(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict) and key in value:
        return value[key]
    return default


def _resolve(value: Any, config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    return value if value is not None else _cfg(config, section, key, default)


def _normalize_preprocess_backend(value: str | None) -> str:
    backend = "tf" if value is None else str(value).lower()
    if backend not in PREPROCESS_BACKENDS:
        raise ValueError(f"Unsupported preprocess_backend: {value}. Expected one of {PREPROCESS_BACKENDS}")
    return backend


def _setup_tensorflow() -> dict[str, object]:
    tf.keras.mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("memory growth setup failed:", exc)
    return {
        "visible_gpus": [str(gpu) for gpu in gpus],
        "mixed_precision_policy": str(tf.keras.mixed_precision.global_policy()),
        "tensorflow_version": tf.__version__,
    }


def _load_frame(csv_path: str, image_col: str, target_col: str, max_samples: int | None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required = {image_col, target_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    frame = frame.dropna(subset=[image_col, target_col]).reset_index(drop=True)
    if max_samples is not None:
        frame = frame.head(int(max_samples)).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{csv_path} produced an empty frame")
    return frame


def _decode_image_pil_bilinear(contents: tf.Tensor, image_size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(contents.numpy())) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _decode_image(path: tf.Tensor, image_size: int, preprocess_backend: str) -> tf.Tensor:
    image = tf.io.read_file(path)
    if preprocess_backend == "tf":
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return tf.image.resize(image, [image_size, image_size])
    if preprocess_backend == "pil_bilinear":
        image = tf.py_function(
            lambda contents: _decode_image_pil_bilinear(contents, image_size),
            [image],
            Tout=tf.float32,
        )
        image.set_shape([image_size, image_size, 3])
        return image
    raise ValueError(f"Unsupported preprocess_backend: {preprocess_backend}")


def _make_dataset(
    frame: pd.DataFrame,
    image_col: str,
    target_col: str,
    image_size: int,
    batch_size: int,
    preprocess_backend: str,
) -> tf.data.Dataset:
    paths = frame[image_col].astype(str).to_numpy()
    targets = frame[target_col].astype("float32").to_numpy().reshape(-1, 1)
    dataset = tf.data.Dataset.from_tensor_slices((paths, targets))

    def _map(path: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _decode_image(path, image_size=image_size, preprocess_backend=preprocess_backend), target

    return dataset.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    error = y_pred - y_true
    mse = float(np.mean(np.square(error)))
    return {
        "num_samples": int(y_true.size),
        "target_min": float(np.min(y_true)),
        "target_max": float(np.max(y_true)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
        "srcc": _safe_corr(_rank(y_true), _rank(y_pred)),
        "plcc": _safe_corr(y_true, y_pred),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
    }


def evaluate_split(
    model: tf.keras.Model,
    split_name: str,
    csv_path: str,
    image_col: str,
    target_col: str,
    image_size: int,
    batch_size: int,
    preprocess_backend: str,
    max_samples: int | None,
    output_dir: Path,
) -> dict[str, object]:
    frame = _load_frame(csv_path, image_col, target_col, max_samples)
    dataset = _make_dataset(frame, image_col, target_col, image_size, batch_size, preprocess_backend)
    started = time.perf_counter()
    predictions = np.asarray(model.predict(dataset, verbose=1), dtype=np.float32).reshape(-1)[: len(frame)]
    elapsed = time.perf_counter() - started
    targets = frame[target_col].astype("float32").to_numpy()
    metrics = regression_metrics(targets, predictions)
    metrics.update(
        {
            "split": split_name,
            "csv": csv_path,
            "target_col": target_col,
            "image_col": image_col,
            "preprocess_backend": preprocess_backend,
            "elapsed_seconds": float(elapsed),
            "seconds_per_image": float(elapsed / max(1, len(frame))),
        }
    )

    predictions_path = output_dir / f"{split_name}_predictions.csv"
    out_frame = frame.copy()
    out_frame["prediction"] = predictions
    out_frame["signed_error"] = predictions - targets
    out_frame["abs_error"] = np.abs(out_frame["signed_error"])
    out_frame.to_csv(predictions_path, index=False)
    metrics["predictions_path"] = str(predictions_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGNet-paper-v1 AADB regression outputs.")
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--test_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--image_col")
    parser.add_argument("--target_col")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--model_name")
    parser.add_argument("--preprocess_backend", choices=PREPROCESS_BACKENDS)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _read_yaml(args.config)

    model_path = _resolve(args.model_path, config, "evaluation", "model_path", None)
    if model_path is None:
        raise ValueError("--model_path is required")
    test_csv = _resolve(args.test_csv, config, "data", "test_csv", "data/processed/aadb/test.csv")
    val_csv = _resolve(args.val_csv, config, "data", "val_csv", None)
    image_col = _resolve(args.image_col, config, "data", "image_col", "image_path")
    target_col = _resolve(args.target_col, config, "data", "target_col", "score")
    image_size = int(_resolve(args.image_size, config, "model", "image_size", 256))
    batch_size = int(_resolve(args.batch_size, config, "evaluation", "batch_size", 16))
    output_dir = Path(_resolve(args.output_dir, config, "evaluation", "output_dir", "outputs/rgnet_paper_v1_aadb_regression_20260509/eval"))
    model_name = _resolve(args.model_name, config, "evaluation", "model_name", "rgnet_paper_v1_aadb_regression")
    preprocess_backend = _normalize_preprocess_backend(
        _resolve(args.preprocess_backend, config, "evaluation", "preprocess_backend", _cfg(config, "data", "preprocess_backend", "tf"))
    )
    max_test_samples = _resolve(args.max_test_samples, config, "evaluation", "max_test_samples", None)
    max_val_samples = _resolve(args.max_val_samples, config, "evaluation", "max_val_samples", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    tf_info = _setup_tensorflow()
    print(f"preprocess_backend: {preprocess_backend}")
    model = tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False,
        custom_objects=get_rgnet_paper_v1_custom_objects(),
    )

    split_metrics = {
        "test": evaluate_split(
            model,
            "test",
            str(test_csv),
            image_col,
            target_col,
            image_size,
            batch_size,
            preprocess_backend,
            max_test_samples,
            output_dir,
        )
    }
    if val_csv:
        split_metrics["val"] = evaluate_split(
            model,
            "val",
            str(val_csv),
            image_col,
            target_col,
            image_size,
            batch_size,
            preprocess_backend,
            max_val_samples,
            output_dir,
        )

    summary = {
        "created_at_local": datetime.now().astimezone().isoformat(),
        "model_variant": MODEL_VARIANT,
        "official_reproduction": False,
        "model_name": model_name,
        "model_path": str(model_path),
        "image_size": image_size,
        "preprocess_backend": preprocess_backend,
        "preprocessing": {
            "backend": preprocess_backend,
            "normalization": "float32_rgb_0_1",
            "resize": "tf.image.resize" if preprocess_backend == "tf" else "PIL.Image.BILINEAR",
        },
        "batch_size": batch_size,
        "tensorflow": tf_info,
        "splits": split_metrics,
    }
    summary_path = output_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"evaluation_summary": str(summary_path), **summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
