# 모바일 기술 IQA 학생 모델 v1 기준선을 학습하고 평가하는 스크립트
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "student_split",
    "quality_score_0_100",
    "topiq_raw",
    "calibrated_topiq_0_100",
    "guarded_target_0_100",
]

DEFAULT_MANIFEST = (
    "outputs/student_training_manifest_design_20260601/student_training_manifest.csv"
)
DEFAULT_OUT_DIR = "outputs/mobile_student_iqa_v1_20260601"
SPLITS = ["train", "val", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the first MobileNetV3Small baseline technical IQA student."
    )
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--image_col", default="image_path")
    parser.add_argument("--target_col", default="guarded_target_0_100")
    parser.add_argument("--reference_col", default="quality_score_0_100")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["huber", "mae"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=5.0)
    parser.add_argument(
        "--backbone_weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet weights when available; falls back to random init if unavailable.",
    )
    parser.add_argument(
        "--unfreeze_backbone",
        action="store_true",
        help="Fine-tune the MobileNetV3Small backbone instead of freezing it.",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def require_columns(df: pd.DataFrame, image_col: str, target_col: str, reference_col: str) -> None:
    required = set(REQUIRED_COLUMNS)
    required.update({image_col, target_col, reference_col})
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def apply_optional_caps(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    caps = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    parts = []
    rng_seed = args.seed
    for split in SPLITS:
        split_df = df[df["student_split"] == split].copy()
        cap = caps[split]
        if cap is not None and len(split_df) > cap:
            split_df = split_df.sample(n=cap, random_state=rng_seed)
        parts.append(split_df)
    return pd.concat(parts, ignore_index=True)


def validate_image_rows(
    df: pd.DataFrame,
    image_col: str,
    target_col: str,
    reference_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable_rows = []
    error_rows = []
    numeric_cols = [target_col, reference_col, "topiq_raw", "calibrated_topiq_0_100"]

    for idx, row in df.iterrows():
        image_path = str(row[image_col])
        error: Optional[str] = None
        if not image_path or image_path.lower() == "nan":
            error = "empty image path"
        elif not Path(image_path).is_file():
            error = "missing image file"
        else:
            for col in numeric_cols:
                try:
                    value = float(row[col])
                except Exception:
                    error = f"non-numeric {col}"
                    break
                if not math.isfinite(value):
                    error = f"non-finite {col}"
                    break
        if error is None:
            try:
                with Image.open(image_path) as img:
                    img.convert("RGB")
            except Exception as exc:
                error = f"image load failed: {exc}"

        if error:
            error_rows.append(
                {
                    "row_index": idx,
                    "dataset": row.get("dataset", ""),
                    "student_split": row.get("student_split", ""),
                    "image_path": image_path,
                    "error": error,
                }
            )
        else:
            usable_rows.append(row)

    usable_df = pd.DataFrame(usable_rows).reset_index(drop=True)
    errors_df = pd.DataFrame(error_rows)
    return usable_df, errors_df


def load_rgb_224(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def make_dataset(
    df: pd.DataFrame,
    image_col: str,
    target_col: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    records = df[[image_col, target_col]].to_dict("records")

    def generator() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for row in records:
            image = load_rgb_224(str(row[image_col]), image_size)
            target = np.asarray([float(row[target_col])], dtype=np.float32)
            yield image, target

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(records), 2048), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def batch_count(num_rows: int, batch_size: int) -> int:
    if num_rows <= 0:
        raise ValueError("Cannot create steps for an empty dataset.")
    return int(math.ceil(num_rows / float(batch_size)))


def build_model(
    image_size: int,
    learning_rate: float,
    loss_name: str,
    huber_delta: float,
    backbone_weights: str,
    freeze_backbone: bool,
) -> Tuple[tf.keras.Model, Dict[str, object]]:
    weights_arg = None if backbone_weights == "none" else "imagenet"
    build_info: Dict[str, object] = {
        "requested_backbone_weights": backbone_weights,
        "backbone_weights_loaded": False,
        "backbone_weights_error": "",
        "backbone_trainable": not freeze_backbone,
        "include_preprocessing": True,
    }

    try:
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=weights_arg,
            include_preprocessing=True,
        )
        build_info["backbone_weights_loaded"] = weights_arg == "imagenet"
    except Exception as exc:
        if weights_arg != "imagenet":
            raise
        build_info["backbone_weights_error"] = repr(exc)
        backbone = tf.keras.applications.MobileNetV3Small(
            input_shape=(image_size, image_size, 3),
            include_top=False,
            weights=None,
            include_preprocessing=True,
        )

    backbone.trainable = not freeze_backbone
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = backbone(inputs, training=False if freeze_backbone else None)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="regression_dense")(x)
    x = tf.keras.layers.Dropout(0.2, name="regression_dropout")(x)
    outputs = tf.keras.layers.Dense(1, name="score")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobile_student_iqa_v1")

    if loss_name == "huber":
        loss = tf.keras.losses.Huber(delta=huber_delta, name="huber")
        build_info["loss"] = f"Huber(delta={huber_delta})"
    else:
        loss = tf.keras.losses.MeanAbsoluteError(name="mae")
        build_info["loss"] = "MAE"

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model, build_info


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    return pearson_corr(rankdata(y_true[mask]), rankdata(y_pred[mask]))


def regression_metrics(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    split: str,
    dataset: str,
) -> Dict[str, object]:
    y_true = df[target_col].to_numpy(dtype=np.float64)
    y_pred = df[pred_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {
            "split": split,
            "dataset": dataset,
            "n": len(df),
            "finite_n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "srcc": float("nan"),
            "plcc": float("nan"),
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "pred_mean": float("nan"),
            "target_min": float("nan"),
            "target_max": float("nan"),
            "target_mean": float("nan"),
        }
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    return {
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": int(finite.sum()),
        "mae": float(np.mean(np.abs(y_pred - y_true))),
        "rmse": float(np.sqrt(np.mean(np.square(y_pred - y_true)))),
        "srcc": spearman_corr(y_true, y_pred),
        "plcc": pearson_corr(y_true, y_pred),
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "pred_mean": float(np.mean(y_pred)),
        "target_min": float(np.min(y_true)),
        "target_max": float(np.max(y_true)),
        "target_mean": float(np.mean(y_true)),
    }


def safety_metrics(
    df: pd.DataFrame,
    reference_col: str,
    pred_col: str,
    split: str,
    dataset: str,
) -> Dict[str, object]:
    reference = df[reference_col].to_numpy(dtype=np.float64)
    pred = df[pred_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(reference) & np.isfinite(pred)
    reference = reference[finite]
    pred = pred[finite]
    fp_mask = (reference < 50.0) & (pred > 65.0)
    severe_fp_mask = (reference < 40.0) & (pred > 65.0)
    fn_mask = (reference > 75.0) & (pred < 45.0)
    return {
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": int(finite.sum()),
        "fp_count": int(fp_mask.sum()),
        "severe_fp_count": int(severe_fp_mask.sum()),
        "fn_count": int(fn_mask.sum()),
    }


def predict_split(
    model: tf.keras.Model,
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    ds = make_dataset(
        df,
        image_col=args.image_col,
        target_col=args.target_col,
        image_size=args.input_size,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    preds = model.predict(ds, steps=batch_count(len(df), args.batch_size), verbose=1).reshape(-1)
    out = df.copy()
    out["student_pred_0_100"] = preds.astype(np.float64)
    out["abs_error_guarded_target"] = np.abs(
        out["student_pred_0_100"].to_numpy(dtype=np.float64)
        - out[args.target_col].to_numpy(dtype=np.float64)
    )
    out["abs_error_quality_score"] = np.abs(
        out["student_pred_0_100"].to_numpy(dtype=np.float64)
        - out[args.reference_col].to_numpy(dtype=np.float64)
    )
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def final_decision(
    metrics_df: pd.DataFrame,
    safety_df: pd.DataFrame,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    test_overall = metrics_df[(metrics_df["split"] == "test") & (metrics_df["dataset"] == "overall")]
    if test_overall.empty:
        return "DO_NOT_CONTINUE_WITH_THIS_STUDENT", ["Missing test overall metrics."]
    test_row = test_overall.iloc[0]
    test_srcc = float(test_row["srcc"])
    test_mae = float(test_row["mae"])
    pred_min = float(test_row["pred_min"])
    pred_max = float(test_row["pred_max"])
    finite_ok = int(test_row["finite_n"]) == int(test_row["n"])
    severe_fp_total = int(safety_df["severe_fp_count"].sum()) if not safety_df.empty else 0

    test_dataset = metrics_df[(metrics_df["split"] == "test") & (metrics_df["dataset"] != "overall")]
    no_dataset_collapse = bool((test_dataset["srcc"].fillna(-1.0) >= 0.50).all()) if not test_dataset.empty else False
    reasonable_range = pred_min > -20.0 and pred_max < 120.0

    if not finite_ok:
        reasons.append("Non-finite predictions were found.")
    if test_srcc < 0.85:
        reasons.append(f"Test SRCC {test_srcc:.4f} is below 0.85.")
    if test_mae > 10.0:
        reasons.append(f"Test MAE {test_mae:.4f} is above 10.")
    if severe_fp_total != 0:
        reasons.append(f"Severe FP count is {severe_fp_total}, expected 0.")
    if not no_dataset_collapse:
        reasons.append("At least one test dataset SRCC is below the no-collapse guard of 0.50.")
    if not reasonable_range:
        reasons.append(f"Prediction range {pred_min:.4f}..{pred_max:.4f} is outside the reasonable guard.")

    if not reasons:
        return "READY_FOR_BASELINE_COMPARISON", ["All READY criteria passed."]
    if not finite_ok or not reasonable_range:
        return "DO_NOT_CONTINUE_WITH_THIS_STUDENT", reasons
    return "NEEDS_STUDENT_TUNING", reasons


def write_report(
    path: Path,
    args: argparse.Namespace,
    build_info: Dict[str, object],
    split_counts: pd.Series,
    load_errors: pd.DataFrame,
    metrics_df: pd.DataFrame,
    safety_df: pd.DataFrame,
    decision: str,
    decision_reasons: List[str],
    history: pd.DataFrame,
) -> None:
    overall_cols = ["split", "dataset", "n", "finite_n", "mae", "rmse", "srcc", "plcc", "pred_min", "pred_max", "pred_mean"]
    safety_cols = ["split", "dataset", "n", "finite_n", "fp_count", "severe_fp_count", "fn_count"]
    dataset_test = metrics_df[(metrics_df["split"] == "test") & (metrics_df["dataset"] != "overall")]
    overall = metrics_df[metrics_df["dataset"] == "overall"]

    lines = [
        "# Mobile Student IQA v1 Training Report",
        "",
        "## 1. Summary",
        f"- Final decision: `{decision}`",
        f"- Input manifest: `{args.manifest_csv}`",
        f"- Output directory: `{args.out_dir}`",
        f"- Best model: `{Path(args.out_dir) / 'best_model.keras'}`",
        f"- Final model: `{Path(args.out_dir) / 'final_model.keras'}`",
        "",
        "## 2. Dataset",
        f"- Split counts after image validation: `{split_counts.to_dict()}`",
        f"- Image load failures: `{len(load_errors)}`",
        f"- Target column: `{args.target_col}`",
        f"- Safety reference column: `{args.reference_col}`",
        "",
        "## 3. Model",
        "- Backbone: `tf.keras.applications.MobileNetV3Small`",
        f"- Input size: `{args.input_size}x{args.input_size}x3`",
        "- Head: `GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.2) -> Dense(1)`",
        f"- Include MobileNetV3 preprocessing in backbone: `{build_info['include_preprocessing']}`",
        f"- Requested backbone weights: `{build_info['requested_backbone_weights']}`",
        f"- ImageNet weights loaded: `{build_info['backbone_weights_loaded']}`",
        f"- Backbone trainable: `{build_info['backbone_trainable']}`",
        "",
        "## 4. Training Configuration",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Train steps per epoch: `{batch_count(int(split_counts.get('train', 0)), args.batch_size)}`",
        f"- Validation steps: `{batch_count(int(split_counts.get('val', 0)), args.batch_size)}`",
        f"- Test prediction steps: `{batch_count(int(split_counts.get('test', 0)), args.batch_size)}`",
        f"- Learning rate: `{args.learning_rate}`",
        f"- Loss: `{build_info['loss']}`",
        "- Callbacks: EarlyStopping on `val_mae`, ReduceLROnPlateau on `val_mae`, ModelCheckpoint on `val_mae`.",
        f"- Completed epochs: `{len(history)}`",
        "",
        "## 5. Overall Metrics",
        markdown_table(overall, overall_cols),
        "",
        "## 6. Dataset-wise Metrics",
        markdown_table(dataset_test, overall_cols),
        "",
        "## 7. Safety Metrics",
        markdown_table(safety_df, safety_cols),
        "",
        "## 8. Failure Cases",
        f"- Load failure CSV: `{Path(args.out_dir) / 'load_errors.csv'}`",
        f"- Load failure count: `{len(load_errors)}`",
        "",
        "## 9. Risks",
        "- This is a first baseline and is not exported to mobile.",
        "- The guarded target depends on MOS during label design and is not a Flutter runtime formula.",
        "- Dataset-wise FLIVE behavior must be reviewed before student training is expanded.",
        "",
        "## 10. Final Decision",
        f"- `{decision}`",
    ]
    lines.extend([f"- {reason}" for reason in decision_reasons])
    if build_info.get("backbone_weights_error"):
        lines.extend(["", "## Backbone Weight Fallback", f"- `{build_info['backbone_weights_error']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest_csv)
    require_columns(manifest, args.image_col, args.target_col, args.reference_col)
    manifest = manifest[manifest["student_split"].isin(SPLITS)].copy()
    manifest = apply_optional_caps(manifest, args)
    usable_df, load_errors = validate_image_rows(
        manifest,
        image_col=args.image_col,
        target_col=args.target_col,
        reference_col=args.reference_col,
    )
    load_errors.to_csv(out_dir / "load_errors.csv", index=False)

    split_counts = usable_df["student_split"].value_counts()
    missing_splits = [split for split in SPLITS if split_counts.get(split, 0) == 0]
    if missing_splits:
        raise ValueError(f"Usable manifest has empty splits after validation: {missing_splits}")

    train_df = usable_df[usable_df["student_split"] == "train"].reset_index(drop=True)
    val_df = usable_df[usable_df["student_split"] == "val"].reset_index(drop=True)
    test_df = usable_df[usable_df["student_split"] == "test"].reset_index(drop=True)
    train_steps = batch_count(len(train_df), args.batch_size)
    val_steps = batch_count(len(val_df), args.batch_size)

    train_ds = make_dataset(train_df, args.image_col, args.target_col, args.input_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(val_df, args.image_col, args.target_col, args.input_size, args.batch_size, False, args.seed)

    model, build_info = build_model(
        image_size=args.input_size,
        learning_rate=args.learning_rate,
        loss_name=args.loss,
        huber_delta=args.huber_delta,
        backbone_weights=args.backbone_weights,
        freeze_backbone=not args.unfreeze_backbone,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_model.keras"),
            monitor="val_mae",
            save_best_only=True,
            mode="min",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=5,
            restore_best_weights=True,
            mode="min",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            mode="min",
        ),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_history.csv")),
    ]

    history_obj = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1,
    )
    if not (out_dir / "training_history.csv").exists():
        pd.DataFrame(history_obj.history).to_csv(out_dir / "training_history.csv", index=False)

    model.save(out_dir / "final_model.keras")
    best_model_path = out_dir / "best_model.keras"
    eval_model = tf.keras.models.load_model(best_model_path, compile=False) if best_model_path.exists() else model

    pred_frames = {
        "train": predict_split(eval_model, train_df, args),
        "val": predict_split(eval_model, val_df, args),
        "test": predict_split(eval_model, test_df, args),
    }
    for split, pred_df in pred_frames.items():
        pred_df.to_csv(out_dir / f"predictions_{split}.csv", index=False)

    metrics_rows: List[Dict[str, object]] = []
    for split, pred_df in pred_frames.items():
        metrics_rows.append(regression_metrics(pred_df, args.target_col, "student_pred_0_100", split, "overall"))
        if split == "test":
            for dataset, group in pred_df.groupby("dataset", sort=True):
                metrics_rows.append(regression_metrics(group, args.target_col, "student_pred_0_100", split, str(dataset)))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "dataset_metrics.csv", index=False)

    safety_rows: List[Dict[str, object]] = []
    for split in ["val", "test"]:
        pred_df = pred_frames[split]
        safety_rows.append(safety_metrics(pred_df, args.reference_col, "student_pred_0_100", split, "overall"))
        for dataset, group in pred_df.groupby("dataset", sort=True):
            safety_rows.append(safety_metrics(group, args.reference_col, "student_pred_0_100", split, str(dataset)))
    safety_df = pd.DataFrame(safety_rows)
    safety_df.to_csv(out_dir / "safety_metrics.csv", index=False)

    history = pd.read_csv(out_dir / "training_history.csv")
    decision, decision_reasons = final_decision(metrics_df, safety_df)
    write_report(
        out_dir / "report.md",
        args=args,
        build_info=build_info,
        split_counts=split_counts,
        load_errors=load_errors,
        metrics_df=metrics_df,
        safety_df=safety_df,
        decision=decision,
        decision_reasons=decision_reasons,
        history=history,
    )
    print(f"Training complete. Final decision: {decision}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
