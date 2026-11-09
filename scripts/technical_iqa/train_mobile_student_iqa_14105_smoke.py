# 14105행 teacher-cache 매니페스트로 모바일 IQA 학생 모델 smoke 학습을 실행한다.
from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs" / "student_distillation_manifest_14105_20260603" / "student_distillation_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_14105_smoke_20260603"
PHASE7_REPORT = ROOT / "outputs" / "student_distillation_manifest_14105_20260603" / "report.md"
PHASE6_REPORT = ROOT / "outputs" / "known_dataset_full_label_dryrun_20260603" / "report.md"
COVERAGE_REPORT = ROOT / "outputs" / "known_dataset_full_label_coverage_audit_20260603" / "report.md"

SPLITS = ["train", "val", "test"]
DATASET_ORDER = ["flive", "koniq10k", "spaq"]
PRED_COL = "pred_0_100"
REQUIRED_COLUMNS = [
    "dataset",
    "split",
    "image_path",
    "source_group",
    "quality_score_0_100",
    "selected_teacher_target_0_100",
    "target_norm",
    "selected_teacher_source",
    "selected_policy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded smoke training on the 14105 teacher-cache student manifest.")
    parser.add_argument("--manifest_csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=1)
    parser.add_argument("--unfreeze_last_n", type=int, default=16)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max_train_samples", type=int, default=512)
    parser.add_argument("--max_val_samples", type=int, default=256)
    parser.add_argument("--max_test_samples", type=int, default=256)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--seed", type=int, default=20260603)
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
        raise ValueError(f"manifest missing columns: {missing}")


def apply_caps(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    caps = {"train": args.max_train_samples, "val": args.max_val_samples, "test": args.max_test_samples}
    parts = []
    for split in SPLITS:
        split_df = df[df["split"].astype(str).eq(split)].copy()
        cap = caps[split]
        if cap is not None and len(split_df) > cap:
            split_df = split_df.sample(n=cap, random_state=args.seed)
        parts.append(split_df)
    return pd.concat(parts, ignore_index=True)


def validate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = []
    failures = []
    for idx, row in df.iterrows():
        image_path = str(row.get("image_path", ""))
        error = ""
        if not image_path or image_path.lower() == "nan":
            error = "empty image_path"
        elif not Path(image_path).is_file():
            error = "missing image file"
        else:
            for col in ["quality_score_0_100", "selected_teacher_target_0_100", "target_norm"]:
                value = pd.to_numeric(row.get(col), errors="coerce")
                if not math.isfinite(float(value)):
                    error = f"non-finite {col}"
                    break
            if not error and not (0.0 <= float(row["target_norm"]) <= 1.0):
                error = "target_norm outside [0,1]"
            if not error and not (0.0 <= float(row["selected_teacher_target_0_100"]) <= 100.0):
                error = "target outside [0,100]"
        if not error:
            try:
                with Image.open(image_path) as img:
                    img.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                error = f"image load failed: {exc}"
        if error:
            failures.append(
                {
                    "row_index": idx,
                    "split": row.get("split", ""),
                    "dataset": row.get("dataset", ""),
                    "image_path": image_path,
                    "error": error,
                }
            )
        else:
            usable.append(row)
    return pd.DataFrame(usable).reset_index(drop=True), pd.DataFrame(failures)


def prepare_manifest(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.manifest_csv)
    require_columns(df)
    df["quality_score_0_100"] = pd.to_numeric(df["quality_score_0_100"], errors="coerce")
    df["selected_teacher_target_0_100"] = pd.to_numeric(df["selected_teacher_target_0_100"], errors="coerce")
    df["target_norm"] = pd.to_numeric(df["target_norm"], errors="coerce")
    df = apply_caps(df, args)
    usable, failures = validate_rows(df)
    return usable, failures


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def make_dataset(df: pd.DataFrame, image_size: int, batch_size: int, shuffle: bool, repeat: bool, seed: int) -> tf.data.Dataset:
    records = df[["image_path", "target_norm"]].to_dict("records")

    def generator() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for row in records:
            image = load_rgb_resized(str(row["image_path"]), image_size)
            target = np.asarray([float(row["target_norm"])], dtype=np.float32)
            yield image, target

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(records), 1024), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


def batch_count(num_rows: int, batch_size: int) -> int:
    if num_rows <= 0:
        raise ValueError("empty split")
    return int(math.ceil(num_rows / float(batch_size)))


def backbone_factory(backbone_name: str) -> Any:
    if backbone_name == "efficientnetv2b0":
        return tf.keras.applications.EfficientNetV2B0
    if backbone_name == "efficientnetb0":
        return tf.keras.applications.EfficientNetB0
    if backbone_name == "mobilenetv3small":
        return tf.keras.applications.MobileNetV3Small
    raise ValueError(f"unsupported backbone: {backbone_name}")


def create_backbone(backbone_name: str, image_size: int, weights_arg: str | None) -> tuple[tf.keras.Model, bool]:
    app = backbone_factory(backbone_name)
    kwargs = {"input_shape": (image_size, image_size, 3), "include_top": False, "weights": weights_arg, "include_preprocessing": True}
    try:
        return app(**kwargs), True
    except TypeError:
        kwargs.pop("include_preprocessing")
        return app(**kwargs), False


def compile_model(model: tf.keras.Model, learning_rate: float, loss_name: str, huber_delta: float) -> None:
    if loss_name == "huber":
        loss = tf.keras.losses.Huber(delta=huber_delta, name="huber_norm")
    elif loss_name == "mae":
        loss = tf.keras.losses.MeanAbsoluteError(name="mae_norm")
    else:
        loss = tf.keras.losses.MeanSquaredError(name="mse_norm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )


def build_model(args: argparse.Namespace) -> tuple[tf.keras.Model, tf.keras.Model, dict[str, Any]]:
    weights_arg = None if args.backbone_weights == "none" else "imagenet"
    info: dict[str, Any] = {
        "backbone": args.backbone,
        "requested_backbone_weights": args.backbone_weights,
        "backbone_weights_loaded": False,
        "backbone_weights_error": "",
        "include_preprocessing": True,
        "output_activation": "sigmoid",
    }
    try:
        backbone, include_preprocessing = create_backbone(args.backbone, args.image_size, weights_arg)
        info["backbone_weights_loaded"] = weights_arg == "imagenet"
        info["include_preprocessing"] = include_preprocessing
    except Exception as exc:  # noqa: BLE001
        if weights_arg is None:
            raise
        info["backbone_weights_error"] = repr(exc)
        backbone, include_preprocessing = create_backbone(args.backbone, args.image_size, None)
        info["include_preprocessing"] = include_preprocessing

    backbone.trainable = False
    inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3), name="image")
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="regression_dense_256")(x)
    x = tf.keras.layers.Dropout(0.3, name="regression_dropout_03")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="regression_dense_64")(x)
    x = tf.keras.layers.Dropout(0.2, name="regression_dropout_02")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="score_norm")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobile_student_iqa_14105_smoke")
    compile_model(model, args.learning_rate_head, args.loss, args.huber_delta)
    return model, backbone, info


def configure_finetune(backbone: tf.keras.Model, unfreeze_last_n: int) -> dict[str, int]:
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    selected = backbone.layers[-max(unfreeze_last_n, 0):] if unfreeze_last_n > 0 else []
    for layer in selected:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    return {
        "backbone_layers": len(backbone.layers),
        "requested_unfreeze_last_n": unfreeze_last_n,
        "backbone_trainable_layers": sum(int(layer.trainable) for layer in backbone.layers),
        "batch_norm_trainable_layers": sum(int(layer.trainable) for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return float("nan")
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return float("nan")
    return pearson_corr(rankdata(y_true[finite]), rankdata(y_pred[finite]))


def predict_split(model: tf.keras.Model, df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    ds = make_dataset(df, args.image_size, args.batch_size, shuffle=False, repeat=True, seed=args.seed)
    pred_norm = model.predict(ds, steps=batch_count(len(df), args.batch_size), verbose=0).reshape(-1)[: len(df)]
    out = df.copy()
    out["pred_norm"] = pred_norm.astype(np.float64)
    out[PRED_COL] = out["pred_norm"] * 100.0
    out["abs_error_to_target_0_100"] = np.abs(out[PRED_COL] - out["selected_teacher_target_0_100"])
    return out


def regression_metrics(df: pd.DataFrame, split: str, dataset: str) -> dict[str, Any]:
    y_true = df["selected_teacher_target_0_100"].to_numpy(dtype=np.float64)
    y_pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    finite_n = int(finite.sum())
    if finite_n == 0:
        return {"split": split, "dataset": dataset, "n": len(df), "finite_n": 0, "mae_0_100": float("nan"), "rmse_0_100": float("nan"), "srcc": float("nan"), "plcc": float("nan"), "pred_min": float("nan"), "pred_max": float("nan"), "pred_mean": float("nan"), "pred_std": float("nan"), "target_min": float("nan"), "target_max": float("nan"), "target_mean": float("nan"), "target_std": float("nan"), "nan_inf_count": int((~finite).sum())}
    yt = y_true[finite]
    yp = y_pred[finite]
    return {
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": finite_n,
        "mae_0_100": float(np.mean(np.abs(yp - yt))),
        "rmse_0_100": float(np.sqrt(np.mean(np.square(yp - yt)))),
        "srcc": spearman_corr(yt, yp),
        "plcc": pearson_corr(yt, yp),
        "pred_min": float(np.min(yp)),
        "pred_max": float(np.max(yp)),
        "pred_mean": float(np.mean(yp)),
        "pred_std": float(np.std(yp)),
        "target_min": float(np.min(yt)),
        "target_max": float(np.max(yt)),
        "target_mean": float(np.mean(yt)),
        "target_std": float(np.std(yt)),
        "nan_inf_count": int((~finite).sum()),
    }


def safety_counts(df: pd.DataFrame) -> dict[str, int]:
    mos = df["quality_score_0_100"].to_numpy(dtype=np.float64)
    pred = df[PRED_COL].to_numpy(dtype=np.float64)
    return {
        "fp_count": int(((mos < 40.0) & (pred > 60.0)).sum()),
        "severe_fp_count": int(((mos < 40.0) & (pred > 65.0)).sum()),
        "fn_count": int(((mos > 75.0) & (pred < 45.0)).sum()),
    }


def prediction_summary(pred_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, df in pred_frames.items():
        metrics = regression_metrics(df, split, "overall")
        metrics.update(safety_counts(df))
        rows.append(metrics)
        for dataset in sorted(df["dataset"].astype(str).unique()):
            sub = df[df["dataset"].astype(str).eq(dataset)]
            if len(sub) >= 2:
                metrics = regression_metrics(sub, split, dataset)
                metrics.update(safety_counts(sub))
                rows.append(metrics)
    return pd.DataFrame(rows)


def collapse_checks(summary: pd.DataFrame) -> dict[str, Any]:
    overall = summary[summary["dataset"].eq("overall")]
    min_std = float(overall["pred_std"].min()) if len(overall) else float("nan")
    pred_min = float(overall["pred_min"].min()) if len(overall) else float("nan")
    pred_max = float(overall["pred_max"].max()) if len(overall) else float("nan")
    nan_inf = int(overall["nan_inf_count"].sum()) if len(overall) else 0
    return {
        "pred_std_min_overall": min_std,
        "pred_min_overall": pred_min,
        "pred_max_overall": pred_max,
        "nan_inf_count": nan_inf,
        "pred_std_too_low": bool(math.isfinite(min_std) and min_std < 1.0),
        "pred_outside_0_100": bool((math.isfinite(pred_min) and pred_min < 0.0) or (math.isfinite(pred_max) and pred_max > 100.0)),
    }


def history_frame(history: tf.keras.callbacks.History, phase: str, start_epoch: int) -> pd.DataFrame:
    history_dict = {key: list(values) for key, values in history.history.items()}
    max_len = max((len(values) for values in history_dict.values()), default=0)
    normalized = {key: values + [float("nan")] * (max_len - len(values)) for key, values in history_dict.items()}
    df = pd.DataFrame(normalized)
    df.insert(0, "phase", phase)
    df.insert(1, "phase_epoch", np.arange(1, len(df) + 1))
    df.insert(2, "global_epoch", np.arange(start_epoch, start_epoch + len(df)))
    return df


def final_decision(
    gpu_count: int,
    training_ok: bool,
    sample_counts: dict[str, int],
    failures: pd.DataFrame,
    history: pd.DataFrame,
    summary: pd.DataFrame,
    collapse: dict[str, Any],
) -> str:
    if not training_ok:
        return "STUDENT_14105_SMOKE_FAILED"
    if sample_counts.get("train", 0) == 0 or sample_counts.get("val", 0) == 0 or sample_counts.get("test", 0) == 0:
        return "STUDENT_14105_SMOKE_NEEDS_DATA_FIX"
    if len(failures) > max(1, int(0.01 * sum(sample_counts.values()))):
        return "STUDENT_14105_SMOKE_NEEDS_DATA_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "STUDENT_14105_SMOKE_NEEDS_SCRIPT_FIX"
    if collapse.get("nan_inf_count", 0) != 0 or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        if gpu_count != 0:
            return "STUDENT_14105_SMOKE_NEEDS_SCRIPT_FIX"
    if gpu_count == 0:
        return "STUDENT_14105_SMOKE_NEEDS_TF_GPU_FIX"
    return "STUDENT_14105_SMOKE_READY_FOR_SHORT_TRAINING"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        vals = []
        for value in row.tolist():
            if isinstance(value, float):
                vals.append("" if math.isnan(value) else f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    sample_counts: dict[str, int],
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    prediction_summary_df: pd.DataFrame,
    failures: pd.DataFrame,
    collapse: dict[str, Any],
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    if history.empty:
        final_train_loss = float("nan")
        final_val_loss = float("nan")
    else:
        final_train_loss = float(history[history["phase"].eq("finetune")]["loss"].iloc[-1]) if len(history[history["phase"].eq("finetune")]) else float(history["loss"].iloc[-1])
        final_val_loss = float(history[history["phase"].eq("finetune")]["val_loss"].iloc[-1]) if len(history[history["phase"].eq("finetune")]) else float(history["val_loss"].iloc[-1])
    overall = metrics[metrics["dataset"].eq("overall")].copy()
    dataset_metrics = metrics[~metrics["dataset"].eq("overall")].copy()
    safety_cols = ["split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]
    lines = [
        "# Mobile Student IQA 14105 Smoke Training Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: bounded smoke only, not full training.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Train/val/test samples used: `{sample_counts}`.",
        f"- Final train loss: `{final_train_loss:.6f}`.",
        f"- Final val loss: `{final_val_loss:.6f}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.manifest_csv}`",
        f"- `{PHASE7_REPORT}`",
        f"- `{COVERAGE_REPORT}`",
        f"- `{PHASE6_REPORT}`",
        "- Existing trainers inspected: `train_mobile_student_iqa_v3.py`, `train_mobile_student_iqa_v4a.py`, `train_mobile_student_iqa_v4b.py`.",
        "",
        "## 3. Manifest and Scope",
        "- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.",
        "- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.",
        "- Target: `target_norm`; report metrics also use 0..100 scale.",
        f"- Image load failure rows: `{len(failures)}`.",
        "",
        "## 4. Environment / GPU Preflight",
        f"- TensorFlow: `{env_info.get('tf_version')}`.",
        f"- GPUs visible: `{env_info.get('gpus')}`.",
        f"- GPU count: `{env_info.get('gpu_count')}`.",
        f"- Execution mode: `{'GPU' if env_info.get('gpu_count') else 'CPU smoke capped <=128 train rows'}`.",
        "",
        "## 5. Smoke Training Configuration",
        f"- Backbone: `{build_info.get('backbone')}`.",
        f"- Image size: `{args.image_size}`.",
        f"- Batch size: `{args.batch_size}`.",
        f"- Head epochs: `{args.head_epochs}`.",
        f"- Finetune epochs: `{args.finetune_epochs}`.",
        f"- Loss: `{args.loss}`.",
        f"- ImageNet weights requested: `{args.backbone_weights}`.",
        f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
        f"- Backbone weights error: `{build_info.get('backbone_weights_error')}`.",
        f"- Include preprocessing: `{build_info.get('include_preprocessing')}`.",
        f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
        "",
        "## 6. Training History",
        markdown_table(history, list(history.columns)),
        "",
        "## 7. Prediction and Metric Summary",
        markdown_table(overall, ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std", "nan_inf_count"]),
        "",
        "## 8. Dataset-wise Metrics",
        markdown_table(dataset_metrics, ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std"]),
        "",
        "## 9. Failure Cases",
        f"- Failure CSV: `{path.parent / 'failure_cases.csv'}`.",
        f"- Failure rows: `{len(failures)}`.",
        "",
        "## 10. Collapse / Safety Checks",
        f"- Collapse check: `{collapse}`.",
        "",
        markdown_table(metrics, safety_cols),
        "",
        "## 11. Final Decision",
        decision,
        "",
        "## 12. Recommended Next Step",
    ]
    if decision == "STUDENT_14105_SMOKE_READY_FOR_SHORT_TRAINING":
        lines.append("- Proceed to a short controlled GPU training run; do not start full distillation yet.")
    elif decision == "STUDENT_14105_SMOKE_NEEDS_TF_GPU_FIX":
        lines.append("- Fix TensorFlow GPU visibility before any short controlled training run. CPU smoke evidence is recorded.")
    elif decision == "STUDENT_14105_SMOKE_NEEDS_SCRIPT_FIX":
        lines.append("- Fix smoke script/model pipeline before retry.")
    elif decision == "STUDENT_14105_SMOKE_NEEDS_DATA_FIX":
        lines.append("- Fix manifest/image data issues before retry.")
    else:
        lines.append("- Inspect failure cases/logs and rerun bounded smoke only.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    set_reproducibility(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    env_info = {"tf_version": tf.__version__, "gpus": [str(gpu) for gpu in gpus], "gpu_count": len(gpus)}

    sample_df, failures = prepare_manifest(args)
    if len(gpus) == 0 and args.max_train_samples and args.max_train_samples > 128:
        raise RuntimeError("GPU unavailable; CPU smoke requires --max_train_samples <= 128")
    sample_df.to_csv(out_dir / "sample_manifest_used.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)

    split_frames = {split: sample_df[sample_df["split"].astype(str).eq(split)].reset_index(drop=True) for split in SPLITS}
    sample_counts = {split: len(split_frames[split]) for split in SPLITS}

    training_ok = False
    history = pd.DataFrame()
    metrics = pd.DataFrame()
    prediction_summary_df = pd.DataFrame()
    collapse: dict[str, Any] = {"not_run": True}
    build_info: dict[str, Any] = {}
    finetune_info: dict[str, Any] = {}
    try:
        train_ds = make_dataset(split_frames["train"], args.image_size, args.batch_size, shuffle=True, repeat=True, seed=args.seed)
        val_ds = make_dataset(split_frames["val"], args.image_size, args.batch_size, shuffle=False, repeat=False, seed=args.seed)
        model, backbone, build_info = build_model(args)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=batch_count(sample_counts["train"], args.batch_size),
            validation_steps=batch_count(sample_counts["val"], args.batch_size),
            epochs=args.head_epochs,
            verbose=1,
        )
        finetune_info = configure_finetune(backbone, args.unfreeze_last_n)
        compile_model(model, args.learning_rate_finetune, args.loss, args.huber_delta)
        finetune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=batch_count(sample_counts["train"], args.batch_size),
            validation_steps=batch_count(sample_counts["val"], args.batch_size),
            epochs=args.finetune_epochs,
            verbose=1,
        )
        history = pd.concat(
            [history_frame(head_history, "head", 1), history_frame(finetune_history, "finetune", args.head_epochs + 1)],
            ignore_index=True,
        )
        pred_frames = {split: predict_split(model, split_frames[split], args) for split in SPLITS}
        prediction_summary_df = pd.concat(pred_frames.values(), ignore_index=True)
        metrics = prediction_summary(pred_frames)
        collapse = collapse_checks(metrics)
        if args.save_model:
            model.save(out_dir / "smoke_model.keras", include_optimizer=False)
        training_ok = True
    except Exception as exc:  # noqa: BLE001
        failures = pd.concat(
            [
                failures,
                pd.DataFrame([{"row_index": "", "split": "", "dataset": "", "image_path": "", "error": f"training failed: {exc!r}"}]),
            ],
            ignore_index=True,
        )
        failures.to_csv(out_dir / "failure_cases.csv", index=False)

    runtime_seconds = time.time() - start
    history.to_csv(out_dir / "history.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    prediction_summary_df.to_csv(out_dir / "prediction_summary.csv", index=False)

    decision = final_decision(env_info["gpu_count"], training_ok, sample_counts, failures, history, metrics, collapse)
    write_report(
        out_dir / "report.md",
        args,
        env_info,
        build_info,
        sample_counts,
        history,
        metrics,
        prediction_summary_df,
        failures,
        collapse,
        decision,
        runtime_seconds,
        finetune_info,
    )
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {decision}")
    print(f"Sample counts: {sample_counts}")


if __name__ == "__main__":
    main()
