# 14105행 teacher-cache 매니페스트로 모바일 IQA 학생 모델 longer controlled 학습을 실행한다.
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
DEFAULT_OUT_DIR = ROOT / "outputs" / "mobile_student_iqa_14105_longer_controlled_20260604"
SHORT_DIR = ROOT / "outputs" / "mobile_student_iqa_14105_short_train_20260604"
PHASE7_REPORT = ROOT / "outputs" / "student_distillation_manifest_14105_20260603" / "report.md"
PHASE6_REPORT = ROOT / "outputs" / "known_dataset_full_label_dryrun_20260603" / "report.md"
COVERAGE_REPORT = ROOT / "outputs" / "known_dataset_full_label_coverage_audit_20260603" / "report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run longer controlled GPU training on the 14105 teacher-cache manifest.")
    parser.add_argument("--manifest_csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--backbone", choices=["efficientnetv2b0", "efficientnetb0", "mobilenetv3small"], default="efficientnetv2b0")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--head_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--unfreeze_last_n", type=int, default=24)
    parser.add_argument("--learning_rate_head", type=float, default=3e-4)
    parser.add_argument("--learning_rate_finetune", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["huber", "mae", "mse"], default="huber")
    parser.add_argument("--huber_delta", type=float, default=0.05)
    parser.add_argument("--backbone_weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--max_train_samples", type=int, default=11283)
    parser.add_argument("--max_val_samples", type=int, default=1409)
    parser.add_argument("--max_test_samples", type=int, default=1413)
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values: list[str] = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append("" if math.isnan(float(value)) else f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_short_metrics() -> pd.DataFrame:
    path = SHORT_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def compare_against_short(metrics: pd.DataFrame, short_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = ["split", "dataset", "metric", "short", "longer", "delta", "improved"]
    if metrics.empty or short_metrics.empty:
        return pd.DataFrame(columns=columns)
    metric_specs = [
        ("mae_0_100", False),
        ("srcc", True),
        ("plcc", True),
        ("severe_fp_count", False),
        ("pred_std", True),
    ]
    datasets = ["overall", "flive", "koniq10k", "spaq"]
    for split in ["train", "val", "test"]:
        for dataset in datasets:
            new_row = metrics[(metrics["split"].eq(split)) & (metrics["dataset"].eq(dataset))]
            old_row = short_metrics[(short_metrics["split"].eq(split)) & (short_metrics["dataset"].eq(dataset))]
            if new_row.empty or old_row.empty:
                continue
            new = new_row.iloc[0]
            old = old_row.iloc[0]
            for metric, higher_is_better in metric_specs:
                old_val = float(old[metric])
                new_val = float(new[metric])
                delta = new_val - old_val
                improved = delta > 0 if higher_is_better else delta < 0
                rows.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "metric": metric,
                        "short": old_val,
                        "longer": new_val,
                        "delta": delta,
                        "improved": improved,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def best_checkpoint_summary(history: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["monitor", "best_epoch", "best_value", "checkpoint"])
    rows = []
    for monitor in ["val_loss", "val_mae"]:
        idx = history[monitor].astype(float).idxmin()
        rows.append(
            {
                "monitor": monitor,
                "best_epoch": int(history.loc[idx, "global_epoch"]),
                "best_value": float(history.loc[idx, monitor]),
                "checkpoint": str(out_dir / "best_by_val_mae_model.keras") if monitor == "val_mae" else "not_saved_separately",
            }
        )
    return pd.DataFrame(rows)


def flive_analysis(metrics: pd.DataFrame, comparison: pd.DataFrame) -> list[str]:
    lines = []
    for split in ["val", "test"]:
        row = metrics[(metrics["split"].eq(split)) & (metrics["dataset"].eq("flive"))]
        cmp_row = comparison[
            (comparison["split"].eq(split))
            & (comparison["dataset"].eq("flive"))
            & (comparison["metric"].eq("srcc"))
        ]
        if row.empty:
            continue
        srcc = float(row.iloc[0]["srcc"])
        mae = float(row.iloc[0]["mae_0_100"])
        if cmp_row.empty:
            lines.append(f"- {split} FLIVE: SRCC `{srcc:.4f}`, MAE `{mae:.4f}`; no short baseline row.")
        else:
            delta = float(cmp_row.iloc[0]["delta"])
            lines.append(f"- {split} FLIVE: SRCC `{srcc:.4f}` (`{delta:+.4f}` vs short), MAE `{mae:.4f}`.")
    if not lines:
        lines.append("- FLIVE metrics unavailable.")
    return lines


def pick_final_decision(
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
        return "STUDENT_14105_LONGER_FAILED"
    if int(env_info.get("gpu_count", 0)) == 0 or not bool(build_info.get("backbone_weights_loaded")):
        return "STUDENT_14105_LONGER_FAILED"
    if sample_counts.get("train", 0) == 0 or sample_counts.get("val", 0) == 0 or sample_counts.get("test", 0) == 0:
        return "STUDENT_14105_LONGER_NEEDS_DATA_OR_LABEL_REVIEW"
    if len(failures) > max(1, int(0.005 * sum(sample_counts.values()))):
        return "STUDENT_14105_LONGER_NEEDS_DATA_OR_LABEL_REVIEW"
    if history.empty or not np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all():
        return "STUDENT_14105_LONGER_FAILED"
    if collapse.get("nan_inf_count", 0) != 0 or collapse.get("pred_outside_0_100") or collapse.get("pred_std_too_low"):
        return "STUDENT_14105_LONGER_NEEDS_HYPERPARAMETER_TUNING"

    overall = metrics[metrics["dataset"].eq("overall")]
    if overall.empty:
        return "STUDENT_14105_LONGER_FAILED"
    train = overall[overall["split"].eq("train")].iloc[0]
    val = overall[overall["split"].eq("val")].iloc[0]
    test = overall[overall["split"].eq("test")].iloc[0]
    val_mae = float(val["mae_0_100"])
    test_mae = float(test["mae_0_100"])
    train_mae = float(train["mae_0_100"])
    val_srcc = float(val["srcc"])
    test_srcc = float(test["srcc"])
    severe_fp_total = int(val["severe_fp_count"]) + int(test["severe_fp_count"])
    train_val_gap = val_mae - train_mae

    if train_val_gap > 4.0 and (val_srcc < 0.60 or test_srcc < 0.60):
        return "STUDENT_14105_LONGER_OVERFITTING"

    def improved(split: str, dataset: str, metric: str, allow_equal_margin: float = 0.0) -> bool:
        row = comparison[
            (comparison["split"].eq(split))
            & (comparison["dataset"].eq(dataset))
            & (comparison["metric"].eq(metric))
        ]
        if row.empty:
            return False
        delta = float(row.iloc[0]["delta"])
        if metric in {"mae_0_100", "severe_fp_count"}:
            return delta <= allow_equal_margin
        return delta >= -allow_equal_margin

    overall_stable = (
        improved("val", "overall", "mae_0_100", 0.25)
        and improved("test", "overall", "mae_0_100", 0.25)
        and improved("val", "overall", "srcc", 0.02)
        and improved("test", "overall", "srcc", 0.02)
    )
    flive_test_srcc_row = metrics[(metrics["split"].eq("test")) & (metrics["dataset"].eq("flive"))]
    flive_test_srcc = float(flive_test_srcc_row.iloc[0]["srcc"]) if not flive_test_srcc_row.empty else float("nan")
    flive_ok = math.isfinite(flive_test_srcc) and (flive_test_srcc >= 0.35 or improved("test", "flive", "srcc", 0.0))

    if overall_stable and val_srcc >= 0.60 and test_srcc >= 0.60 and severe_fp_total <= 2 and train_val_gap <= 3.0 and flive_ok:
        return "STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING"
    return "STUDENT_14105_LONGER_NEEDS_HYPERPARAMETER_TUNING"


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    build_info: dict[str, Any],
    sample_counts: dict[str, int],
    history: pd.DataFrame,
    best_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    failures: pd.DataFrame,
    collapse: dict[str, Any],
    comparison: pd.DataFrame,
    decision: str,
    runtime_seconds: float,
    finetune_info: dict[str, Any],
) -> None:
    final_train_loss = float(history["loss"].iloc[-1]) if not history.empty else float("nan")
    final_val_loss = float(history["val_loss"].iloc[-1]) if not history.empty else float("nan")
    best_val_loss = float(history["val_loss"].min()) if not history.empty else float("nan")
    overall = metrics[metrics["dataset"].eq("overall")].copy()
    dataset_metrics = metrics[~metrics["dataset"].eq("overall")].copy()
    safety_cols = ["split", "dataset", "n", "fp_count", "severe_fp_count", "fn_count"]
    comparison_focus = comparison[
        (comparison["dataset"].isin(["overall", "flive", "koniq10k", "spaq"]))
        & (comparison["split"].isin(["val", "test"]))
        & (comparison["metric"].isin(["mae_0_100", "srcc", "plcc", "severe_fp_count", "pred_std"]))
    ].copy()
    lines = [
        "# Mobile Student IQA 14105 Longer Controlled Training Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: longer controlled training only, not final full distillation.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Train/val/test samples used: `{sample_counts}`.",
        f"- Final train loss: `{final_train_loss:.6f}`.",
        f"- Final val loss: `{final_val_loss:.6f}`.",
        f"- Best val loss: `{best_val_loss:.6f}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.manifest_csv}`",
        f"- `{SHORT_DIR / 'report.md'}`",
        f"- `{SHORT_DIR / 'history.csv'}`",
        f"- `{SHORT_DIR / 'metrics.csv'}`",
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
        f"- Requested unfreeze_last_n: `{args.unfreeze_last_n}`.",
        f"- Finetune trainable backbone layers: `{finetune_info.get('backbone_trainable_layers')}`.",
        "",
        "## 6. Training History",
        markdown_table(history, list(history.columns)),
        "",
        "## 7. Best Checkpoint Summary",
        markdown_table(best_summary, list(best_summary.columns)),
        "- `best_by_val_srcc_model.keras` not implemented; SRCC is computed after final prediction pass to keep training bounded.",
        "",
        "## 8. Prediction and Metric Summary",
        markdown_table(
            overall,
            [
                "split",
                "dataset",
                "n",
                "finite_n",
                "mae_0_100",
                "rmse_0_100",
                "srcc",
                "plcc",
                "pred_min",
                "pred_max",
                "pred_mean",
                "pred_std",
                "target_min",
                "target_max",
                "target_mean",
                "target_std",
                "nan_inf_count",
            ],
        ),
        "",
        "## 9. Dataset-wise Metrics",
        markdown_table(
            dataset_metrics[dataset_metrics["split"].isin(["val", "test"])],
            ["split", "dataset", "n", "finite_n", "mae_0_100", "rmse_0_100", "srcc", "plcc", "pred_min", "pred_max", "pred_mean", "pred_std"],
        ),
        "",
        "## 10. Comparison Against Short Controlled Training",
        markdown_table(comparison_focus, ["split", "dataset", "metric", "short", "longer", "delta", "improved"]),
        "",
        "## 11. FLIVE Weakness Analysis",
        *flive_analysis(metrics, comparison),
        "",
        "## 12. Failure Cases",
        f"- Failure CSV: `{out_dir / 'failure_cases.csv'}`.",
        f"- Failure rows: `{len(failures)}`.",
        "",
        "## 13. Collapse / Safety Checks",
        f"- Collapse check: `{collapse}`.",
        "",
        markdown_table(metrics, safety_cols),
        "",
        "## 14. Final Decision",
        decision,
        "",
        "## 15. Recommended Next Step",
    ]
    if decision == "STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING":
        lines.append("- Prepare a full controlled training plan with explicit stop criteria; do not run final full distillation yet.")
    elif decision == "STUDENT_14105_LONGER_OVERFITTING":
        lines.append("- Reduce finetune depth/epochs or add regularization before another controlled run.")
    elif decision == "STUDENT_14105_LONGER_NEEDS_DATA_OR_LABEL_REVIEW":
        lines.append("- Review manifest/image/label rows before further training.")
    elif decision == "STUDENT_14105_LONGER_NEEDS_HYPERPARAMETER_TUNING":
        lines.append("- Tune sampling or fine-tune schedule, with special attention to FLIVE and severe FP rows.")
    else:
        lines.append("- Inspect failure cases/logs and rerun only bounded controlled training after fix.")
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
        raise RuntimeError("GPU unavailable; longer controlled training requires GPU")

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
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best_by_val_mae_model.keras"),
            monitor="val_mae",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        )
        finetune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            steps_per_epoch=smoke.batch_count(sample_counts["train"], args.batch_size),
            validation_steps=smoke.batch_count(sample_counts["val"], args.batch_size),
            epochs=args.finetune_epochs,
            callbacks=[checkpoint_cb],
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
        model.save(out_dir / "final_model.keras", include_optimizer=False)
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
    comparison = compare_against_short(metrics, load_short_metrics())
    comparison.to_csv(out_dir / "comparison_against_short.csv", index=False)
    best_summary = best_checkpoint_summary(history, out_dir)
    best_summary.to_csv(out_dir / "best_checkpoint_summary.csv", index=False)
    decision = pick_final_decision(env_info, build_info, training_ok, sample_counts, failures, history, metrics, collapse, comparison)
    write_report(out_dir, args, env_info, build_info, sample_counts, history, best_summary, metrics, failures, collapse, comparison, decision, runtime_seconds, finetune_info)
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {decision}")
    print(f"Sample counts: {sample_counts}")


if __name__ == "__main__":
    main()
