# 모바일 technical IQA v3 학생 모델을 50k 매니페스트로 학습하고 평가하는 스크립트
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


DEFAULT_MANIFEST = "outputs/student_training_manifest_50k_20260601/student_training_manifest.csv"
DEFAULT_OUT_DIR = "outputs/mobile_student_iqa_v3_20260601"
DEFAULT_V2_REPORT = "outputs/mobile_student_iqa_v2_20260601_fixed/report.md"
SPLITS = ["train", "val", "test"]
TARGET_NORM_COL = "target_norm"
PRED_COL = "pred_0_100"
REQUIRED_COLUMNS = [
    "image_path",
    "dataset",
    "student_split",
    "quality_score_0_100",
    "guarded_target_0_100",
]
V2_BASELINE = {
    "test_srcc": 0.666845,
    "test_mae": 7.234272,
    "test_severe_fp": 5.0,
    "flive_test_srcc": 0.361239,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mobile Student IQA v3 on the 50k guarded manifest.")
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--v2_report_md", default=DEFAULT_V2_REPORT)
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--head_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=20)
    parser.add_argument("--unfreeze_last_n", type=int, default=40)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber", "mae"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--debug_history", action="store_true")
    parser.add_argument("--seed", type=int, default=20260601)
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


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"manifest_csv is missing required columns: {missing}")


def apply_optional_caps(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    caps = {"train": args.max_train_samples, "val": args.max_val_samples, "test": args.max_test_samples}
    parts = []
    for split in SPLITS:
        split_df = df[df["student_split"] == split].copy()
        cap = caps[split]
        if cap is not None and len(split_df) > cap:
            split_df = split_df.sample(n=cap, random_state=args.seed)
        parts.append(split_df)
    return pd.concat(parts, ignore_index=True)


def validate_image_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    usable_rows = []
    error_rows = []
    numeric_cols = ["quality_score_0_100", "guarded_target_0_100"]
    for idx, row in df.iterrows():
        image_path = str(row["image_path"])
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
            error_rows.append({
                "row_index": idx,
                "dataset": row.get("dataset", ""),
                "student_split": row.get("student_split", ""),
                "image_path": image_path,
                "error": error,
            })
        else:
            usable_rows.append(row)
    return pd.DataFrame(usable_rows).reset_index(drop=True), pd.DataFrame(error_rows)


def add_normalized_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET_NORM_COL] = np.clip(out["guarded_target_0_100"].astype(float).to_numpy() / 100.0, 0.0, 1.0)
    return out


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def make_dataset(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    repeat: bool,
    seed: int,
) -> tf.data.Dataset:
    records = df[["image_path", TARGET_NORM_COL]].to_dict("records")

    def generator() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        for row in records:
            image = load_rgb_resized(str(row["image_path"]), image_size)
            target = np.asarray([float(row[TARGET_NORM_COL])], dtype=np.float32)
            yield image, target

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(records), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def batch_count(num_rows: int, batch_size: int) -> int:
    if num_rows <= 0:
        raise ValueError("Cannot create steps for an empty dataset.")
    return int(math.ceil(num_rows / float(batch_size)))


def backbone_factory(backbone_name: str) -> Any:
    if backbone_name == "efficientnetv2b0":
        return tf.keras.applications.EfficientNetV2B0
    if backbone_name == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0
    if backbone_name == "mobilenetv3small":
        return tf.keras.applications.MobileNetV3Small
    raise ValueError(f"Unsupported backbone: {backbone_name}")


def create_backbone(backbone_name: str, image_size: int, weights_arg: str | None) -> tuple[tf.keras.Model, bool]:
    app = backbone_factory(backbone_name)
    kwargs = {
        "input_shape": (image_size, image_size, 3),
        "include_top": False,
        "weights": weights_arg,
        "include_preprocessing": True,
    }
    try:
        return app(**kwargs), True
    except TypeError:
        kwargs.pop("include_preprocessing")
        return app(**kwargs), False


def build_model(args: argparse.Namespace) -> Tuple[tf.keras.Model, tf.keras.Model, Dict[str, Any]]:
    weights_arg = None if args.backbone_weights == "none" else "imagenet"
    build_info: Dict[str, Any] = {
        "backbone": args.backbone,
        "requested_backbone_weights": args.backbone_weights,
        "backbone_weights_loaded": False,
        "backbone_weights_error": "",
        "include_preprocessing": True,
        "preprocessing_note": "Input generator emits RGB float32 0..255. Keras application preprocessing is included when supported.",
        "output_activation": "sigmoid",
    }
    try:
        backbone, include_preprocessing = create_backbone(args.backbone, args.image_size, weights_arg)
        build_info["backbone_weights_loaded"] = weights_arg == "imagenet"
        build_info["include_preprocessing"] = include_preprocessing
    except Exception as exc:
        if weights_arg != "imagenet":
            raise
        build_info["backbone_weights_error"] = repr(exc)
        backbone, include_preprocessing = create_backbone(args.backbone, args.image_size, None)
        build_info["include_preprocessing"] = include_preprocessing
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3), name="image")
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="regression_dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="regression_dropout_03")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="regression_dense_64")(x)
    x = tf.keras.layers.Dropout(0.2, name="regression_dropout_02")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="score_norm")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobile_student_iqa_v3")
    compile_model(model, args.learning_rate_head, args.loss, args.huber_delta)
    return model, backbone, build_info


def compile_model(model: tf.keras.Model, learning_rate: float, loss_name: str, huber_delta: float) -> None:
    if loss_name == "huber":
        loss = tf.keras.losses.Huber(delta=huber_delta, name="huber_norm")
    else:
        loss = tf.keras.losses.MeanAbsoluteError(name="mae_norm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )


def configure_stage_a(backbone: tf.keras.Model) -> Dict[str, int]:
    backbone.trainable = False
    for layer in backbone.layers:
        layer.trainable = False
    return {"backbone_layers": len(backbone.layers), "backbone_trainable_layers": 0, "batch_norm_trainable_layers": 0}


def configure_stage_b(backbone: tf.keras.Model, unfreeze_last_n: int) -> Dict[str, int]:
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    selected_layers = backbone.layers[-max(unfreeze_last_n, 0):] if unfreeze_last_n > 0 else []
    for layer in selected_layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    return {
        "backbone_layers": len(backbone.layers),
        "requested_unfreeze_last_n": unfreeze_last_n,
        "backbone_trainable_layers": sum(int(layer.trainable) for layer in backbone.layers),
        "batch_norm_trainable_layers": sum(int(layer.trainable) for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)),
    }


def stage_callbacks(checkpoint: tf.keras.callbacks.Callback) -> List[tf.keras.callbacks.Callback]:
    return [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=6, restore_best_weights=True, mode="min"),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=3, min_lr=1e-7, mode="min"),
    ]


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


def regression_metrics(df: pd.DataFrame, target_col: str, split: str, dataset: str) -> Dict[str, Any]:
    y_true = df[target_col].to_numpy(dtype=np.float64)
    y_pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {
            "split": split, "dataset": dataset, "n": len(df), "finite_n": 0,
            "mae": float("nan"), "rmse": float("nan"), "srcc": float("nan"), "plcc": float("nan"),
            "pred_min": float("nan"), "pred_max": float("nan"), "pred_mean": float("nan"), "pred_std": float("nan"),
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
        "pred_std": float(np.std(y_pred)),
    }


def safety_metrics(df: pd.DataFrame, split: str, dataset: str) -> Dict[str, Any]:
    reference = df["quality_score_0_100"].to_numpy(dtype=np.float64)
    pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(reference) & np.isfinite(pred)
    reference = reference[finite]
    pred = pred[finite]
    return {
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": int(finite.sum()),
        "fp_count": int(((reference < 50.0) & (pred > 65.0)).sum()),
        "severe_fp_count": int(((reference < 40.0) & (pred > 65.0)).sum()),
        "fn_count": int(((reference > 75.0) & (pred < 45.0)).sum()),
    }


def predict_split(model: tf.keras.Model, df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ds = make_dataset(df, args.image_size, args.batch_size, shuffle=False, repeat=True, seed=args.seed)
    pred_norm = model.predict(ds, steps=batch_count(len(df), args.batch_size), verbose=1).reshape(-1)
    out = df.copy()
    out["pred_norm"] = np.clip(pred_norm.astype(np.float64), 0.0, 1.0)
    out[PRED_COL] = out["pred_norm"] * 100.0
    out["abs_error_to_target"] = np.abs(out[PRED_COL].to_numpy(dtype=np.float64) - out["guarded_target_0_100"].to_numpy(dtype=np.float64))
    out["abs_error_to_mos"] = np.abs(out[PRED_COL].to_numpy(dtype=np.float64) - out["quality_score_0_100"].to_numpy(dtype=np.float64))
    return out


def severe_fp_cases(pred_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split in ["val", "test"]:
        df = pred_frames[split]
        severe = df[(df["quality_score_0_100"] < 40.0) & (df[PRED_COL] > 65.0)]
        standard = df[(df["quality_score_0_100"] < 50.0) & (df[PRED_COL] > 65.0) & ~(df.index.isin(severe.index))]
        for fp_type, subset in [("severe_fp", severe), ("fp", standard)]:
            for _, row in subset.iterrows():
                rows.append({
                    "fp_type": fp_type,
                    "dataset": row.get("dataset", ""),
                    "split": split,
                    "image_path": row.get("image_path", ""),
                    "source_group": row.get("source_group", ""),
                    "quality_score_0_100": row.get("quality_score_0_100", ""),
                    "guarded_target_0_100": row.get("guarded_target_0_100", ""),
                    "pred_0_100": row.get(PRED_COL, ""),
                    "absolute_error_to_target": row.get("abs_error_to_target", ""),
                })
    return pd.DataFrame(rows).sort_values(["fp_type", "absolute_error_to_target"], ascending=[True, False]) if rows else pd.DataFrame(columns=[
        "fp_type", "dataset", "split", "image_path", "source_group", "quality_score_0_100", "guarded_target_0_100", "pred_0_100", "absolute_error_to_target"
    ])


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


def history_frame(history: tf.keras.callbacks.History, phase: str, start_epoch: int) -> pd.DataFrame:
    history_dict = {key: list(values) for key, values in history.history.items()}
    max_len = max((len(values) for values in history_dict.values()), default=0)
    normalized = {key: values + [float("nan")] * (max_len - len(values)) for key, values in history_dict.items()}
    df = pd.DataFrame(normalized)
    df.insert(0, "phase", phase)
    df.insert(1, "phase_epoch", np.arange(1, len(df) + 1))
    df.insert(2, "global_epoch", np.arange(start_epoch, start_epoch + len(df)))
    return df


def debug_history(history: tf.keras.callbacks.History, phase: str) -> None:
    print(f"[debug_history] phase={phase}")
    print(f"[debug_history] keys={list(history.history.keys())}")
    for key, values in history.history.items():
        print(f"[debug_history] {phase}.{key}={values}")


def load_v2_baseline(report_path: str) -> Dict[str, float]:
    baseline = dict(V2_BASELINE)
    if Path(report_path).is_file():
        baseline["report_found"] = 1.0
    else:
        baseline["report_found"] = 0.0
    return baseline


def final_decision(
    overall_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    safety_df: pd.DataFrame,
    load_errors: pd.DataFrame,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    test_row = overall_metrics[overall_metrics["split"] == "test"].iloc[0]
    test_srcc = float(test_row["srcc"])
    test_mae = float(test_row["mae"])
    finite_ok = int(test_row["finite_n"]) == int(test_row["n"])
    range_ok = 0.0 <= float(test_row["pred_min"]) <= 100.0 and 0.0 <= float(test_row["pred_max"]) <= 100.0
    severe_fp = int(safety_df[(safety_df["split"] == "test") & (safety_df["dataset"] == "overall")]["severe_fp_count"].sum())
    no_dataset_collapse = bool((dataset_metrics["srcc"].fillna(-1.0) >= 0.50).all())
    load_failure_rate = len(load_errors) / max(int(overall_metrics["n"].sum()), 1)
    if test_srcc < 0.85:
        reasons.append(f"Test SRCC {test_srcc:.4f} is below 0.85.")
    if test_mae > 10.0:
        reasons.append(f"Test MAE {test_mae:.4f} is above 10.")
    if severe_fp != 0:
        reasons.append(f"Test severe FP count is {severe_fp}, expected 0.")
    if not no_dataset_collapse:
        reasons.append("At least one test dataset SRCC is below 0.50.")
    if not finite_ok or not range_ok:
        reasons.append("Predictions are non-finite or outside 0..100.")
    if load_failure_rate > 0.001:
        reasons.append(f"Image load failure rate {load_failure_rate:.6f} is not negligible.")
    if not reasons:
        return "READY_FOR_BASELINE_COMPARISON", ["All READY criteria passed."]
    if (not finite_ok) or severe_fp > 50 or (not range_ok):
        return "DO_NOT_CONTINUE_WITH_THIS_STUDENT", reasons
    return "NEEDS_MORE_TUNING", reasons


def write_report(
    path: Path,
    args: argparse.Namespace,
    build_info: Dict[str, Any],
    stage_info: Dict[str, Dict[str, int]],
    split_counts: pd.Series,
    load_errors: pd.DataFrame,
    history: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    mos_metrics: pd.DataFrame,
    safety_df: pd.DataFrame,
    severe_cases: pd.DataFrame,
    v2: Dict[str, float],
    decision: str,
    decision_reasons: List[str],
) -> None:
    metric_cols = ["split", "dataset", "n", "finite_n", "mae", "rmse", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std"]
    safety_cols = ["split", "dataset", "n", "finite_n", "fp_count", "severe_fp_count", "fn_count"]
    test_overall = overall_metrics[overall_metrics["split"] == "test"].iloc[0]
    test_safety = safety_df[(safety_df["split"] == "test") & (safety_df["dataset"] == "overall")].iloc[0]
    flive_row = dataset_metrics[dataset_metrics["dataset"] == "flive"]
    flive_srcc = float(flive_row.iloc[0]["srcc"]) if not flive_row.empty else float("nan")
    comparison = pd.DataFrame([
        {
            "metric": "test_srcc",
            "v2": v2["test_srcc"],
            "v3": float(test_overall["srcc"]),
            "improvement": float(test_overall["srcc"]) - v2["test_srcc"],
        },
        {
            "metric": "test_mae",
            "v2": v2["test_mae"],
            "v3": float(test_overall["mae"]),
            "improvement": v2["test_mae"] - float(test_overall["mae"]),
        },
        {
            "metric": "test_severe_fp",
            "v2": v2["test_severe_fp"],
            "v3": float(test_safety["severe_fp_count"]),
            "improvement": v2["test_severe_fp"] - float(test_safety["severe_fp_count"]),
        },
        {
            "metric": "flive_test_srcc",
            "v2": v2["flive_test_srcc"],
            "v3": flive_srcc,
            "improvement": flive_srcc - v2["flive_test_srcc"],
        },
    ])
    lines = [
        "# Mobile Student IQA v3 Training Report",
        "",
        "## 1. Summary",
        f"- Final decision: `{decision}`",
        f"- Input manifest: `{args.manifest_csv}`",
        f"- Output directory: `{args.out_dir}`",
        "",
        "## 2. Dataset",
        f"- Split counts after image validation: `{split_counts.to_dict()}`",
        f"- Image load failures: `{len(load_errors)}`",
        "- Target: `target_norm = guarded_target_0_100 / 100.0`",
        "",
        "## 3. Model",
        f"- Backbone: `{build_info['backbone']}`",
        f"- Input size: `{args.image_size}x{args.image_size}x3`",
        "- Head: `GAP -> Dense(256, relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)`",
        f"- Include preprocessing in Keras application: `{build_info['include_preprocessing']}`",
        f"- Preprocessing behavior: {build_info['preprocessing_note']}",
        f"- ImageNet weights loaded: `{build_info['backbone_weights_loaded']}`",
        "",
        "## 4. Training Configuration",
        f"- Stage A head epochs: `{args.head_epochs}`",
        f"- Stage B fine-tune epochs: `{args.finetune_epochs}`",
        f"- Stage B requested unfreeze last N: `{args.unfreeze_last_n}`",
        f"- Stage B trainable backbone layers: `{stage_info['finetune']['backbone_trainable_layers']}`",
        f"- Stage B trainable BatchNorm layers: `{stage_info['finetune']['batch_norm_trainable_layers']}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Loss: `{args.loss}` with huber_delta `{args.huber_delta}` when applicable",
        "",
        "## 5. Training History Summary",
        f"- History rows: `{len(history)}`",
        f"- Best validation MAE norm: `{history['val_mae'].min() if 'val_mae' in history else 'n/a'}`",
        "",
        "## 6. Overall Metrics Against Guarded Target",
        markdown_table(overall_metrics, metric_cols),
        "",
        "## 7. Dataset-wise Metrics Against Guarded Target",
        markdown_table(dataset_metrics, metric_cols),
        "",
        "## 8. MOS Reference Metrics",
        markdown_table(mos_metrics, metric_cols),
        "",
        "## 9. Safety Metrics",
        markdown_table(safety_df, safety_cols),
        "",
        "## 10. Severe False Positive Cases",
        f"- Severe/standard FP case rows saved: `{len(severe_cases)}`",
        f"- CSV: `{Path(args.out_dir) / 'severe_false_positive_cases.csv'}`",
        "",
        "## 11. v2 Comparison",
        markdown_table(comparison, ["metric", "v2", "v3", "improvement"]),
        "",
        "## 12. Risks",
        "- This run is not a TFLite export and does not modify app runtime scoring.",
        "- The guarded target is a teacher-label design target, not a Flutter runtime formula.",
        "- If FLIVE SRCC remains low, inspect data balance, resolution, and target behavior before deployment work.",
        "",
        "## 13. Final Decision",
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
    require_columns(manifest)
    manifest = manifest[manifest["student_split"].isin(SPLITS)].copy()
    manifest = apply_optional_caps(manifest, args)
    usable_df, load_errors = validate_image_rows(manifest)
    usable_df = add_normalized_target(usable_df)
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

    train_ds = make_dataset(train_df, args.image_size, args.batch_size, shuffle=True, repeat=True, seed=args.seed)
    val_ds = make_dataset(val_df, args.image_size, args.batch_size, shuffle=False, repeat=True, seed=args.seed)

    model, backbone, build_info = build_model(args)
    stage_info = {"head": configure_stage_a(backbone)}
    checkpoint = tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best_model.keras"), monitor="val_mae", save_best_only=True, mode="min")

    compile_model(model, args.learning_rate_head, args.loss, args.huber_delta)
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        initial_epoch=0,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=stage_callbacks(checkpoint),
        verbose=1,
    )
    if args.debug_history:
        debug_history(head_history, "head")

    stage_info["finetune"] = configure_stage_b(backbone, args.unfreeze_last_n)
    compile_model(model, args.learning_rate_finetune, args.loss, args.huber_delta)
    finetune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.finetune_epochs,
        initial_epoch=0,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=stage_callbacks(checkpoint),
        verbose=1,
    )
    if args.debug_history:
        debug_history(finetune_history, "finetune")

    history = pd.concat([
        history_frame(head_history, "head", 1),
        history_frame(finetune_history, "finetune", len(head_history.history.get("loss", [])) + 1),
    ], ignore_index=True)
    history.to_csv(out_dir / "training_history.csv", index=False)

    model.save(out_dir / "final_model.keras")
    best_path = out_dir / "best_model.keras"
    eval_model = tf.keras.models.load_model(best_path, compile=False) if best_path.exists() else model

    pred_frames = {
        "train": predict_split(eval_model, train_df, args),
        "val": predict_split(eval_model, val_df, args),
        "test": predict_split(eval_model, test_df, args),
    }
    pred_cols = [
        "image_path", "dataset", "student_split", "source_group", "quality_score_0_100",
        "guarded_target_0_100", "pred_norm", "pred_0_100", "abs_error_to_target", "abs_error_to_mos",
    ]
    for split, pred_df in pred_frames.items():
        cols = [col for col in pred_cols if col in pred_df.columns]
        pred_df.to_csv(out_dir / f"predictions_{split}.csv", columns=cols, index=False)

    overall_rows = [regression_metrics(pred_frames[split], "guarded_target_0_100", split, "overall") for split in SPLITS]
    overall_metrics = pd.DataFrame(overall_rows)
    dataset_metrics = pd.DataFrame([
        regression_metrics(group, "guarded_target_0_100", "test", str(dataset))
        for dataset, group in pred_frames["test"].groupby("dataset", sort=True)
    ])
    mos_rows = [regression_metrics(pred_frames["test"], "quality_score_0_100", "test", "overall")]
    mos_rows.extend(
        regression_metrics(group, "quality_score_0_100", "test", str(dataset))
        for dataset, group in pred_frames["test"].groupby("dataset", sort=True)
    )
    mos_metrics = pd.DataFrame(mos_rows)
    safety_rows: List[Dict[str, Any]] = []
    for split in ["val", "test"]:
        safety_rows.append(safety_metrics(pred_frames[split], split, "overall"))
        safety_rows.extend(
            safety_metrics(group, split, str(dataset))
            for dataset, group in pred_frames[split].groupby("dataset", sort=True)
        )
    safety_df = pd.DataFrame(safety_rows)
    severe_cases = severe_fp_cases(pred_frames)

    overall_metrics.to_csv(out_dir / "overall_metrics.csv", index=False)
    dataset_metrics.to_csv(out_dir / "dataset_metrics.csv", index=False)
    mos_metrics.to_csv(out_dir / "mos_reference_metrics.csv", index=False)
    safety_df.to_csv(out_dir / "safety_metrics.csv", index=False)
    severe_cases.to_csv(out_dir / "severe_false_positive_cases.csv", index=False)

    decision, reasons = final_decision(overall_metrics, dataset_metrics, safety_df, load_errors)
    write_report(
        out_dir / "report.md",
        args,
        build_info,
        stage_info,
        split_counts,
        load_errors,
        history,
        overall_metrics,
        dataset_metrics,
        mos_metrics,
        safety_df,
        severe_cases,
        load_v2_baseline(args.v2_report_md),
        decision,
        reasons,
    )
    print(f"Training complete. Final decision: {decision}")
    print(f"Report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
