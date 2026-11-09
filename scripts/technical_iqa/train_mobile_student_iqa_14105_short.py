# 14105행 teacher-cache 매니페스트로 모바일 IQA 학생 모델 short controlled 학습을 실행한다.
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.technical_iqa import train_mobile_student_iqa_14105_smoke as smoke  # noqa: E402

DEFAULT_MANIFEST = ROOT / "outputs" / "student_distillation_manifest_14105_20260603" / "student_distillation_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_14105_short_train_20260604"
SMOKE_DIR = ROOT / "outputs" / "mobile_student_iqa_14105_smoke_gpu_imagenet_20260604"
PHASE7_REPORT = ROOT / "outputs" / "student_distillation_manifest_14105_20260603" / "report.md"
PHASE6_REPORT = ROOT / "outputs" / "known_dataset_full_label_dryrun_20260603" / "report.md"
COVERAGE_REPORT = ROOT / "outputs" / "known_dataset_full_label_coverage_audit_20260603" / "report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run short controlled GPU training on the 14105 teacher-cache manifest.")
    parser.add_argument("--manifest_csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--unfreeze_last_n", type=int, default=16)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max_train_samples", type=int, default=4000)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--max_test_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def load_smoke_metrics() -> pd.DataFrame:
    path = SMOKE_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def compare_against_smoke(metrics: pd.DataFrame, smoke_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if smoke_metrics.empty:
        return pd.DataFrame(columns=["split", "metric", "smoke", "short", "delta", "improved"])
    for split in ["train", "val", "test"]:
        short_row = metrics[(metrics["split"].eq(split)) & (metrics["dataset"].eq("overall"))]
        smoke_row = smoke_metrics[(smoke_metrics["split"].eq(split)) & (smoke_metrics["dataset"].eq("overall"))]
        if short_row.empty or smoke_row.empty:
            continue
        short = short_row.iloc[0]
        old = smoke_row.iloc[0]
        for metric, higher_is_better in [("mae_0_100", False), ("srcc", True), ("plcc", True), ("severe_fp_count", False), ("pred_std", True)]:
            old_val = float(old[metric])
            new_val = float(short[metric])
            delta = new_val - old_val
            improved = delta > 0 if higher_is_better else delta < 0
            rows.append({"split": split, "metric": metric, "smoke": old_val, "short": new_val, "delta": delta, "improved": improved})
    return pd.DataFrame(rows)


def final_decision(
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    training_ok: bool,
    sample_counts: dict[str, int],
    failures: pd.DataFrame,
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    collapse: dict[str, Any],
    comparison: pd.DataFrame,
) -> str:
    if not training_ok:
        return "STUDENT_14105_SHORT_FAILED"
    if int(env_info.get("gpu_count", 0)) == 0:
        return "STUDENT_14105_SHORT_NEEDS_TF_GPU_FIX"
    if not bool(build_info.get("backbone_weights_loaded")):
        return "STUDENT_14105_SHORT_NEEDS_HYPERPARAMETER_TUNING"
    if sample_counts.get("train", 0) == 0 or sample_counts.get("val", 0) == 0 or sample_counts.get("test", 0) == 0:
        return "STUDENT_14105_SHORT_NEEDS_DATA_FIX"
    if len(failures) > max(1, int(0.01 * sum(sample_counts.values()))):
        return "STUDENT_14105_SHORT_NEEDS_DATA_FIX"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "STUDENT_14105_SHORT_FAILED"
    if collapse.get("nan_inf_count", 0) != 0 or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        return "STUDENT_14105_SHORT_NEEDS_HYPERPARAMETER_TUNING"

    overall = metrics[metrics["dataset"].eq("overall")]
    val_row = overall[overall["split"].eq("val")].iloc[0]
    test_row = overall[overall["split"].eq("test")].iloc[0]
    val_srcc = float(val_row["srcc"])
    test_srcc = float(test_row["srcc"])
    severe_fp = int(val_row["severe_fp_count"]) + int(test_row["severe_fp_count"])
    compare_val_mae = comparison[(comparison["split"].eq("val")) & (comparison["metric"].eq("mae_0_100"))]
    compare_test_mae = comparison[(comparison["split"].eq("test")) & (comparison["metric"].eq("mae_0_100"))]
    mae_improved = (
        (not compare_val_mae.empty and bool(compare_val_mae.iloc[0]["improved"]))
        and (not compare_test_mae.empty and bool(compare_test_mae.iloc[0]["improved"]))
    )
    srcc_positive = val_srcc > 0.0 and test_srcc > 0.0
    if srcc_positive and mae_improved and severe_fp <= 10:
        return "STUDENT_14105_SHORT_READY_FOR_LONGER_TRAINING"
    return "STUDENT_14105_SHORT_NEEDS_HYPERPARAMETER_TUNING"


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


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    sample_counts: dict[str, int],
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    failures: pd.DataFrame,
    collapse: dict[str, Any],
    comparison: pd.DataFrame,
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    final_train_loss = float(history[history["phase"].eq("finetune")]["loss"].iloc[-1]) if len(history[history["phase"].eq("finetune")]) else float(history["loss"].iloc[-1])
    final_val_loss = float(history[history["phase"].eq("finetune")]["val_loss"].iloc[-1]) if len(history[history["phase"].eq("finetune")]) else float(history["val_loss"].iloc[-1])
    overall = metrics[metrics["dataset"].eq("overall")].copy()
    dataset_metrics = metrics[~metrics["dataset"].eq("overall")].copy()
    safety_cols = ["split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]
    lines = [
        "# Mobile Student IQA 14105 Short Controlled Training Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: short controlled training only, not full distillation.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Train/val/test samples used: `{sample_counts}`.",
        f"- Final train loss: `{final_train_loss:.6f}`.",
        f"- Final val loss: `{final_val_loss:.6f}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.manifest_csv}`",
        f"- `{SMOKE_DIR / 'report.md'}`",
        f"- `{PHASE7_REPORT}`",
        f"- `{COVERAGE_REPORT}`",
        f"- `{PHASE6_REPORT}`",
        "- Reused smoke trainer functions from `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`.",
        "",
        "## 3. Manifest and Scope",
        "- Scope: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.",
        "- Unknown/fallback excluded; c6_parity holdout only; Test_weight/product-gold validation only.",
        "- Target: `target_norm`; metrics use 0..100 scale.",
        f"- Image load failure rows: `{len(failures)}`.",
        "",
        "## 4. Environment / GPU Preflight",
        f"- TensorFlow: `{env_info.get('tf_version')}`.",
        f"- Built with CUDA: `{env_info.get('built_cuda')}`.",
        f"- GPUs visible: `{env_info.get('gpus')}`.",
        f"- GPU count: `{env_info.get('gpu_count')}`.",
        "",
        "## 5. Training Configuration",
        f"- Backbone: `{build_info.get('backbone')}`.",
        f"- Image size: `{args.image_size}`.",
        f"- Batch size: `{args.batch_size}`.",
        f"- Head epochs: `{args.head_epochs}`.",
        f"- Finetune epochs: `{args.finetune_epochs}`.",
        f"- Loss: `{args.loss}`.",
        f"- ImageNet weights requested: `{args.backbone_weights}`.",
        f"- ImageNet weights loaded: `{build_info.get('backbone_weights_loaded')}`.",
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
        markdown_table(dataset_metrics[(dataset_metrics["split"].isin(["val", "test"]))], ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std"]),
        "",
        "## 9. Comparison Against Smoke",
        markdown_table(comparison, ["split", "metric", "smoke", "short", "delta", "improved"]),
        "",
        "## 10. Failure Cases",
        f"- Failure CSV: `{out_dir / 'failure_cases.csv'}`.",
        f"- Failure rows: `{len(failures)}`.",
        "",
        "## 11. Collapse / Safety Checks",
        f"- Collapse check: `{collapse}`.",
        "",
        markdown_table(metrics, safety_cols),
        "",
        "## 12. Final Decision",
        decision,
        "",
        "## 13. Recommended Next Step",
    ]
    if decision == "STUDENT_14105_SHORT_READY_FOR_LONGER_TRAINING":
        lines.append("- Prepare a longer controlled GPU training run; do not start full distillation yet.")
    elif decision == "STUDENT_14105_SHORT_NEEDS_HYPERPARAMETER_TUNING":
        lines.append("- Tune short-run hyperparameters or sampling before longer training.")
    elif decision == "STUDENT_14105_SHORT_NEEDS_TF_GPU_FIX":
        lines.append("- Fix TensorFlow GPU visibility before retry.")
    elif decision == "STUDENT_14105_SHORT_NEEDS_DATA_FIX":
        lines.append("- Fix manifest/image data issues before retry.")
    else:
        lines.append("- Inspect failure cases/logs and rerun bounded short training only.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    smoke.set_reproducibility(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    env_info = {
        "tf_version": tf.__version__,
        "built_cuda": tf.test.is_built_with_cuda(),
        "gpus": [str(gpu) for gpu in gpus],
        "gpu_count": len(gpus),
    }
    if not gpus:
        raise RuntimeError("GPU unavailable; short controlled training requires GPU")

    sample_df, failures = smoke.prepare_manifest(args)
    sample_df.to_csv(out_dir / "sample_manifest_used.csv", index=False)
    failures.to_csv(out_dir / "failure_cases.csv", index=False)
    split_frames = {split: sample_df[sample_df["split"].astype(str).eq(split)].reset_index(drop=True) for split in smoke.SPLITS}
    sample_counts = {split: len(split_frames[split]) for split in smoke.SPLITS}

    training_ok = False
    history = pd.DataFrame()
    metrics = pd.DataFrame()
    prediction_summary_df = pd.DataFrame()
    collapse: dict[str, Any] = {"not_run": True}
    build_info: dict[str, Any] = {}
    finetune_info: dict[str, Any] = {}
    try:
        train_ds = smoke.make_dataset(split_frames["train"], args.image_size, args.batch_size, shuffle=True, repeat=True, seed=args.seed)
        val_ds = smoke.make_dataset(split_frames["val"], args.image_size, args.batch_size, shuffle=False, repeat=False, seed=args.seed)
        model, backbone, build_info = smoke.build_model(args)
        head_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(sample_counts["train"], args.batch_size),
            validation_steps=smoke.batch_count(sample_counts["val"], args.batch_size),
            epochs=args.head_epochs,
            verbose=1,
        )
        finetune_info = smoke.configure_finetune(backbone, args.unfreeze_last_n)
        smoke.compile_model(model, args.learning_rate_finetune, args.loss, args.huber_delta)
        finetune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(sample_counts["train"], args.batch_size),
            validation_steps=smoke.batch_count(sample_counts["val"], args.batch_size),
            epochs=args.finetune_epochs,
            verbose=1,
        )
        history = pd.concat(
            [smoke.history_frame(head_history, "head", 1), smoke.history_frame(finetune_history, "finetune", args.head_epochs + 1)],
            ignore_index=True,
        )
        pred_frames = {split: smoke.predict_split(model, split_frames[split], args) for split in smoke.SPLITS}
        prediction_summary_df = pd.concat(pred_frames.values(), ignore_index=True)
        metrics = smoke.prediction_summary(pred_frames)
        collapse = smoke.collapse_checks(metrics)
        model.save(out_dir / "short_model.keras", include_optimizer=False)
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
    comparison = compare_against_smoke(metrics, load_smoke_metrics())
    comparison.to_csv(out_dir / "comparison_against_smoke.csv", index=False)
    decision = final_decision(env_info, build_info, training_ok, sample_counts, failures, history, metrics, collapse, comparison)
    write_report(out_dir, args, env_info, build_info, sample_counts, history, metrics, failures, collapse, comparison, decision, runtime_seconds, finetune_info)
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {decision}")
    print(f"Sample counts: {sample_counts}")


if __name__ == "__main__":
    main()
