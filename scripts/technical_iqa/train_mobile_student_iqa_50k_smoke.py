# 50k MDIQA distillation manifest로 모바일 IQA 학생 모델 smoke 학습을 실행한다.
from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs" / "mdiqa_distillation_50k_full_targets_20260608" / "final_50k_mdiqa_distillation_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_50k_smoke_20260608"
SOURCE_REPORT = ROOT / "outputs" / "mdiqa_distillation_50k_full_targets_20260608" / "report.md"

SPLITS = ["train", "val", "test"]
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
DATASET_ORDER = ["flive", "spaq", "koniq10k"]
PRED_COL = "pred_0_100"
TARGET_COL = "teacher_target_0_100"
TARGET_NORM_COL = "target_norm"
SELECTED_TARGET_COL = "selected_teacher_target_0_100"
FAILURE_COLUMNS = ["row_index", "split", "dataset", "image_path", "error"]
REQUIRED_COLUMNS = [
    "dataset",
    "image_path",
    "quality_score_0_100",
    "is_official_test",
    TARGET_COL,
    "target_source",
]
BASELINE_14K = {
    "student_official_srcc": 0.7436,
    "teacher_official_srcc": 0.8227,
    "gap_decision": "STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded smoke training on 50k MDIQA student distillation targets.")
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
    parser.add_argument("--dataset_weight_flive", type=float, default=1.0)
    parser.add_argument("--dataset_weight_koniq10k", type=float, default=1.0)
    parser.add_argument("--dataset_weight_spaq", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=4000)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--max_test_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260608)
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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def source_group_from_path(image_path: str) -> str:
    filename = Path(str(image_path)).name
    normalized = str(image_path).replace("\\", "/")
    if "/patches/" in normalized and "_patch_" in filename:
        return filename.split("_patch_", 1)[0]
    return str(image_path)


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.lower()
    out["image_path"] = out["image_path"].astype(str)
    out["source_group"] = out["image_path"].map(source_group_from_path)
    out["quality_score_0_100"] = pd.to_numeric(out["quality_score_0_100"], errors="coerce")
    out[SELECTED_TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out[TARGET_NORM_COL] = out[SELECTED_TARGET_COL] / 100.0
    try:
        deciles = pd.qcut(out[SELECTED_TARGET_COL], q=10, labels=False, duplicates="drop")
    except Exception:
        deciles = pd.cut(out[SELECTED_TARGET_COL].rank(method="first"), bins=10, labels=False, include_lowest=True)
    out["target_decile"] = deciles.astype("Int64").astype(str).map(lambda value: f"d{value}")
    return out


def split_groups(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    out = df.copy()
    assignments: dict[int, str] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_group, group_df in out.groupby("source_group", sort=False):
        key_counts = Counter((str(row.dataset), str(row.target_decile)) for row in group_df.itertuples())
        strat_key = sorted(key_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        grouped["|".join(strat_key)].append({"indices": list(group_df.index), "row_count": int(len(group_df)), "source_group": source_group})
    for strat_key in sorted(grouped):
        bucket = grouped[strat_key]
        rng.shuffle(bucket)
        total = sum(item["row_count"] for item in bucket)
        targets = {split: total * SPLIT_RATIOS[split] for split in SPLITS}
        counts = {split: 0 for split in SPLITS}
        for item in sorted(bucket, key=lambda value: (-value["row_count"], str(value["source_group"]))):
            chosen = max(SPLITS, key=lambda split: (targets[split] - counts[split], -counts[split]))
            for idx in item["indices"]:
                assignments[int(idx)] = chosen
            counts[chosen] += int(item["row_count"])
    out["student_split"] = pd.Series(assignments).reindex(out.index).values
    return out


def balanced_cap(split_df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(split_df) <= cap:
        return split_df.copy()
    rng = np.random.default_rng(seed)
    counts = split_df["dataset"].value_counts().to_dict()
    raw = {dataset: cap * count / len(split_df) for dataset, count in counts.items()}
    quotas = {dataset: int(math.floor(value)) for dataset, value in raw.items()}
    remainder = cap - sum(quotas.values())
    for dataset, _ in sorted(raw.items(), key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))[:remainder]:
        quotas[dataset] += 1
    parts = []
    for dataset in sorted(quotas):
        sub = split_df[split_df["dataset"].eq(dataset)]
        n = min(quotas[dataset], len(sub))
        indices = rng.choice(sub.index.to_numpy(), size=n, replace=False)
        parts.append(sub.loc[indices])
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def dataset_weight_map(args: argparse.Namespace) -> dict[str, float]:
    return {
        "flive": float(args.dataset_weight_flive),
        "koniq10k": float(args.dataset_weight_koniq10k),
        "spaq": float(args.dataset_weight_spaq),
    }


def add_dataset_sample_weights(frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    weights = dataset_weight_map(args)
    for frame in frames.values():
        frame["dataset_sample_weight"] = frame["dataset"].astype(str).str.lower().map(weights).fillna(1.0).astype(np.float32)


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
            for col in ["quality_score_0_100", SELECTED_TARGET_COL, TARGET_NORM_COL]:
                value = pd.to_numeric(row.get(col), errors="coerce")
                if not math.isfinite(float(value)):
                    error = f"non-finite {col}"
                    break
            if not error and not (0.0 <= float(row[TARGET_NORM_COL]) <= 1.0):
                error = "target_norm outside [0,1]"
            if not error and not (0.0 <= float(row[SELECTED_TARGET_COL]) <= 100.0):
                error = "target outside [0,100]"
        if not error:
            try:
                with Image.open(image_path) as img:
                    img.convert("RGB")
            except Exception as exc:  # noqa: BLE001
                error = f"image load failed: {exc}"
        if error:
            failures.append({"row_index": idx, "split": row.get("student_split", ""), "dataset": row.get("dataset", ""), "image_path": image_path, "error": error})
        else:
            usable.append(row)
    return pd.DataFrame(usable).reset_index(drop=True), pd.DataFrame(failures, columns=FAILURE_COLUMNS)


def prepare_manifest(args: argparse.Namespace, out_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(args.manifest_csv)
    require_columns(raw)
    full = add_target_columns(raw)
    official_test_rows = full[full["is_official_test"].map(as_bool)].copy()
    trainable = full[~full["is_official_test"].map(as_bool)].copy()
    split_full = split_groups(trainable, args.seed)
    capped = {
        "train": balanced_cap(split_full[split_full["student_split"].eq("train")], args.max_train_samples, args.seed + 1),
        "val": balanced_cap(split_full[split_full["student_split"].eq("val")], args.max_val_samples, args.seed + 2),
        "test": balanced_cap(split_full[split_full["student_split"].eq("test")], args.max_test_samples, args.seed + 3),
    }
    capped_all = pd.concat(capped.values(), ignore_index=True)
    usable, failures = validate_rows(capped_all)
    frames = {split: usable[usable["student_split"].eq(split)].reset_index(drop=True) for split in SPLITS}
    add_dataset_sample_weights(frames, args)
    frames["train"].to_csv(out_dir / "train_manifest.csv", index=False)
    frames["val"].to_csv(out_dir / "val_manifest.csv", index=False)
    frames["test"].to_csv(out_dir / "test_manifest.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)
    meta = {
        "input_rows": len(raw),
        "official_test_rows_excluded_before_split": len(official_test_rows),
        "duplicate_image_paths_input": int(raw["image_path"].duplicated().sum()),
        "full_split_counts": split_full["student_split"].value_counts().to_dict(),
        "capped_counts_before_validation": {split: len(df) for split, df in capped.items()},
        "usable_counts": {split: len(frames[split]) for split in SPLITS},
        "overlap_image_paths_across_splits": overlap_count(frames, "image_path"),
        "overlap_source_groups_across_splits": overlap_count(frames, "source_group"),
        "train_official_test_rows": int(frames["train"]["is_official_test"].map(as_bool).sum()) if len(frames["train"]) else 0,
    }
    return frames, failures, meta


def overlap_count(frames: dict[str, pd.DataFrame], col: str) -> int:
    seen: dict[str, str] = {}
    overlaps = 0
    for split in SPLITS:
        for value in frames[split][col].astype(str):
            if value in seen and seen[value] != split:
                overlaps += 1
            else:
                seen[value] = split
    return overlaps


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def make_dataset(df: pd.DataFrame, image_size: int, batch_size: int, shuffle: bool, repeat: bool, seed: int, sample_weight_col: str | None = None) -> tf.data.Dataset:
    columns = ["image_path", TARGET_NORM_COL]
    if sample_weight_col is not None:
        columns.append(sample_weight_col)
    records = df[columns].to_dict("records")

    def generator() -> Iterable[tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for row in records:
            image = load_rgb_resized(str(row["image_path"]), image_size)
            target = np.asarray([float(row[TARGET_NORM_COL])], dtype=np.float32)
            if sample_weight_col is None:
                yield image, target
            else:
                yield image, target, np.asarray(float(row[sample_weight_col]), dtype=np.float32)

    output_signature: tuple[tf.TensorSpec, ...] = (
            tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    if sample_weight_col is not None:
        output_signature = (*output_signature, tf.TensorSpec(shape=(), dtype=tf.float32))
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
    info = {"backbone": args.backbone, "requested_backbone_weights": args.backbone_weights, "backbone_weights_loaded": False, "backbone_weights_error": "", "include_preprocessing": True}
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
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mobile_student_iqa_50k_smoke")
    compile_model(model, args.learning_rate_head, args.loss, args.huber_delta)
    return model, backbone, info


def configure_finetune(backbone: tf.keras.Model, unfreeze_last_n: int) -> dict[str, int]:
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-max(unfreeze_last_n, 0):]:
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
    out["abs_error_to_target_0_100"] = np.abs(out[PRED_COL] - out[SELECTED_TARGET_COL])
    return out


def regression_metrics(df: pd.DataFrame, split: str, dataset: str) -> dict[str, Any]:
    y_true = df[SELECTED_TARGET_COL].to_numpy(dtype=np.float64)
    y_pred = df[PRED_COL].to_numpy(dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() == 0:
        return {"split": split, "dataset": dataset, "n": len(df), "finite_n": 0, "mae_0_100": float("nan"), "rmse_0_100": float("nan"), "srcc": float("nan"), "plcc": float("nan"), "pred_min": float("nan"), "pred_max": float("nan"), "pred_mean": float("nan"), "pred_std": float("nan"), "target_min": float("nan"), "target_max": float("nan"), "target_mean": float("nan"), "target_std": float("nan"), "nan_inf_count": int((~finite).sum())}
    yt = y_true[finite]
    yp = y_pred[finite]
    return {
        "split": split,
        "dataset": dataset,
        "n": len(df),
        "finite_n": int(finite.sum()),
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


def metric_tables(pred_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    dataset_rows = []
    for split, df in pred_frames.items():
        rows.append(regression_metrics(df, split, "overall"))
        for dataset in sorted(df["dataset"].astype(str).unique()):
            sub = df[df["dataset"].astype(str).eq(dataset)]
            if len(sub) >= 2:
                row = regression_metrics(sub, split, dataset)
                rows.append(row)
                dataset_rows.append(row)
    metrics = pd.DataFrame(rows)
    dataset_metrics = pd.DataFrame(dataset_rows)
    prediction_summary = metrics[metrics["dataset"].eq("overall")].copy()
    return metrics, dataset_metrics, prediction_summary


def collapse_checks(metrics: pd.DataFrame) -> dict[str, Any]:
    overall = metrics[metrics["dataset"].eq("overall")]
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
    if "phase" not in df.columns:
        df.insert(0, "phase", phase)
    if "phase_epoch" not in df.columns:
        df.insert(1, "phase_epoch", np.arange(1, len(df) + 1))
    if "global_epoch" not in df.columns:
        df.insert(2, "global_epoch", np.arange(start_epoch, start_epoch + len(df)))
    return df


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append("" if math.isnan(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def final_decision(training_ok: bool, env_info: dict[str, Any], split_meta: dict[str, Any], failures: pd.DataFrame, history: pd.DataFrame, collapse: dict[str, Any]) -> str:
    if not training_ok:
        return "STUDENT_50K_SMOKE_FAILED"
    if env_info["gpu_count"] == 0:
        return "STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX"
    if split_meta["overlap_image_paths_across_splits"] or split_meta["train_official_test_rows"]:
        return "STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX"
    if len(failures) > max(1, int(0.01 * sum(split_meta["usable_counts"].values()))):
        return "STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "STUDENT_50K_SMOKE_NEEDS_PIPELINE_FIX"
    if collapse.get("nan_inf_count", 0) or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        return "STUDENT_50K_SMOKE_NEEDS_TARGET_FIX"
    return "STUDENT_50K_SMOKE_READY_FOR_CONTROLLED_TRAINING"


def write_next_actions(out_dir: Path, decision: str, gpu_count: int) -> None:
    if decision == "STUDENT_50K_SMOKE_READY_FOR_CONTROLLED_TRAINING":
        rows = [{"priority": 1, "action": "Run controlled 50k training with unchanged split and explicit checkpoint output.", "status": "recommended"}]
    elif gpu_count == 0:
        rows = [{"priority": 1, "action": "Fix TensorFlow GPU visibility, then rerun the same bounded smoke with 4000/1000/1000 caps.", "status": "blocked"}]
    else:
        rows = [{"priority": 1, "action": "Inspect failure_cases.csv, history.csv, and prediction_summary.csv before any controlled training.", "status": "required"}]
    pd.DataFrame(rows).to_csv(out_dir / "next_actions.csv", index=False)


def write_report(
    path: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    split_meta: dict[str, Any],
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    dataset_metrics: pd.DataFrame,
    prediction_summary_df: pd.DataFrame,
    failures: pd.DataFrame,
    collapse: dict[str, Any],
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    final_train_loss = float(history["loss"].iloc[-1]) if not history.empty and "loss" in history else float("nan")
    final_val_loss = float(history["val_loss"].iloc[-1]) if not history.empty and "val_loss" in history else float("nan")
    metric_cols = ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std", "target_min", "target_max", "target_mean", "target_std", "nan_inf_count"]
    lines = [
        "# Mobile Student IQA 50k Smoke Training Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: bounded smoke training only; no full training, no TFLite export, no Flutter edits, no production `technical_score` edits.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Usable train/val/test counts: `{split_meta['usable_counts']}`.",
        f"- Final train loss: `{final_train_loss:.6f}`.",
        f"- Final val loss: `{final_val_loss:.6f}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.manifest_csv}`",
        f"- `{SOURCE_REPORT}`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_v4a.py`",
        "- `scripts/technical_iqa/train_mobile_student_iqa_v4b.py`",
        "- `scripts/technical_iqa/make_student_training_manifest_50k.py`",
        "",
        "## 3. Manifest and Split Construction",
        f"- Input rows: `{split_meta['input_rows']}`.",
        f"- Official Track B rows excluded before split: `{split_meta['official_test_rows_excluded_before_split']}`.",
        f"- Duplicate image paths in input: `{split_meta['duplicate_image_paths_input']}`.",
        f"- Full split counts before caps: `{split_meta['full_split_counts']}`.",
        f"- Capped counts before validation: `{split_meta['capped_counts_before_validation']}`.",
        f"- Image-path overlap across splits: `{split_meta['overlap_image_paths_across_splits']}`.",
        f"- Source-group overlap across splits: `{split_meta['overlap_source_groups_across_splits']}`.",
        f"- Official Track B rows in train: `{split_meta['train_official_test_rows']}`.",
        "",
        "## 4. Environment / GPU Preflight",
        f"- TensorFlow: `{env_info['tf_version']}`.",
        f"- GPUs visible to TensorFlow: `{env_info['gpus']}`.",
        f"- GPU count: `{env_info['gpu_count']}`.",
        f"- Execution mode: `{'GPU' if env_info['gpu_count'] else 'CPU fallback smoke recorded; not acceptable for READY unless explicitly approved'}`.",
        "",
        "## 5. Smoke Training Configuration",
        f"- Backbone: `{build_info.get('backbone')}`.",
        f"- Image size: `{args.image_size}`.",
        f"- Batch size: `{args.batch_size}`.",
        f"- Head epochs: `{args.head_epochs}`.",
        f"- Finetune epochs: `{args.finetune_epochs}`.",
        f"- Loss: `{args.loss}`.",
        f"- target_norm: `{TARGET_COL} / 100`.",
        f"- ImageNet weights requested: `{args.backbone_weights}`.",
        f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
        f"- Backbone weights error: `{build_info.get('backbone_weights_error')}`.",
        f"- Include preprocessing: `{build_info.get('include_preprocessing')}`.",
        f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
        f"- Dataset sample weights: `{dataset_weight_map(args)}`.",
        "- Dataset sample weights apply only to the training dataset.",
        "",
        "## 6. Training History",
        markdown_table(history, list(history.columns)),
        "",
        "## 7. Prediction and Metric Summary",
        markdown_table(prediction_summary_df, metric_cols),
        "",
        "## 8. Dataset-wise Metrics",
        markdown_table(dataset_metrics, metric_cols),
        "",
        "## 9. Failure Cases",
        f"- Failure CSV: `{path.parent / 'failure_cases.csv'}`.",
        f"- Failure rows: `{len(failures)}`.",
        "",
        "## 10. Collapse / Safety Checks",
        f"- Collapse check: `{collapse}`.",
        "",
        "## 11. Comparison Against 14k Student Baseline",
        f"- Previous 14k student official SRCC: `{BASELINE_14K['student_official_srcc']}`.",
        f"- MDIQA teacher official SRCC: `{BASELINE_14K['teacher_official_srcc']}`.",
        f"- Prior gap decision: `{BASELINE_14K['gap_decision']}`.",
        "- This smoke is not a final official benchmark; compare only for pipeline sanity.",
        "",
        "## 12. Final Decision",
        decision,
        "",
        "## 13. Recommended Next Step",
    ]
    if decision == "STUDENT_50K_SMOKE_READY_FOR_CONTROLLED_TRAINING":
        lines.append("- Start a controlled, still bounded 50k training run using the saved clean split.")
    elif env_info["gpu_count"] == 0:
        lines.append("- Fix TensorFlow GPU visibility and rerun the same smoke command before controlled training.")
    else:
        lines.append("- Inspect failures, target ranges, prediction spread, and dataloader logs before retry.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    set_reproducibility(args.seed)
    gpus = tf.config.list_physical_devices("GPU")
    env_info = {"tf_version": tf.__version__, "gpus": [str(gpu) for gpu in gpus], "gpu_count": len(gpus)}
    frames, failures, split_meta = prepare_manifest(args, out_dir)

    training_ok = False
    history = pd.DataFrame()
    metrics = pd.DataFrame()
    dataset_metrics = pd.DataFrame()
    prediction_summary_df = pd.DataFrame()
    collapse: dict[str, Any] = {"not_run": True}
    build_info: dict[str, Any] = {}
    finetune_info: dict[str, Any] = {}
    try:
        train_ds = make_dataset(frames["train"], args.image_size, args.batch_size, shuffle=True, repeat=True, seed=args.seed, sample_weight_col="dataset_sample_weight")
        val_ds = make_dataset(frames["val"], args.image_size, args.batch_size, shuffle=False, repeat=True, seed=args.seed)
        model, backbone, build_info = build_model(args)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=batch_count(len(frames["train"]), args.batch_size),
            validation_steps=batch_count(len(frames["val"]), args.batch_size),
            epochs=args.head_epochs,
            verbose=1,
        )
        finetune_info = configure_finetune(backbone, args.unfreeze_last_n)
        compile_model(model, args.learning_rate_finetune, args.loss, args.huber_delta)
        finetune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=batch_count(len(frames["train"]), args.batch_size),
            validation_steps=batch_count(len(frames["val"]), args.batch_size),
            epochs=args.finetune_epochs,
            verbose=1,
        )
        history = pd.concat([history_frame(head_history, "head", 1), history_frame(finetune_history, "finetune", args.head_epochs + 1)], ignore_index=True)
        pred_frames = {split: predict_split(model, frames[split], args) for split in SPLITS}
        prediction_cols = ["dataset", "student_split", "image_path", "quality_score_0_100", SELECTED_TARGET_COL, TARGET_NORM_COL, "pred_norm", PRED_COL, "abs_error_to_target_0_100", "target_source"]
        for split, pred_df in pred_frames.items():
            pred_df.to_csv(out_dir / f"predictions_{split}.csv", columns=[col for col in prediction_cols if col in pred_df.columns], index=False)
        metrics, dataset_metrics, prediction_summary_df = metric_tables(pred_frames)
        collapse = collapse_checks(metrics)
        training_ok = True
    except Exception as exc:  # noqa: BLE001
        failures = pd.concat([failures, pd.DataFrame([{"row_index": "", "split": "", "dataset": "", "image_path": "", "error": f"training failed: {exc!r}"}], columns=FAILURE_COLUMNS)], ignore_index=True)
        failures.to_csv(out_dir / "failure_cases.csv", index=False)

    runtime_seconds = time.time() - start
    history.to_csv(out_dir / "history.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    dataset_metrics.to_csv(out_dir / "datasetwise_metrics.csv", index=False)
    prediction_summary_df.to_csv(out_dir / "prediction_summary.csv", index=False)
    decision = final_decision(training_ok, env_info, split_meta, failures, history, collapse)
    write_next_actions(out_dir, decision, env_info["gpu_count"])
    write_report(out_dir / "report.md", args, env_info, build_info, split_meta, history, metrics, dataset_metrics, prediction_summary_df, failures, collapse, decision, runtime_seconds, finetune_info)
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {decision}")
    print(f"Counts: {split_meta['usable_counts']}")
    print(f"Runtime seconds: {runtime_seconds:.2f}")


if __name__ == "__main__":
    main()
