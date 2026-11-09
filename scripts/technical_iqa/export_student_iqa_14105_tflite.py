# 14105행 모바일 IQA 학생 모델을 TFLite로 내보내고 Keras parity를 점검한다.
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "outputs/mobile_student_iqa_14105_full_controlled_20260604/best_by_val_mae_model.keras"
DEFAULT_MANIFEST = ROOT / "outputs/student_distillation_manifest_14105_20260603/student_distillation_manifest.csv"
DEFAULT_OUT_DIR = ROOT / "outputs/mobile_student_iqa_14105_tflite_export_20260604"
PHASE11_REPORT = ROOT / "outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/report.md"
PHASE11_METRICS = ROOT / "outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/checkpoint_metrics.csv"
PHASE11_DATASET_METRICS = ROOT / "outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/datasetwise_checkpoint_metrics.csv"

DATASET_ORDER = ["flive", "koniq10k", "spaq"]
TFLITE_FP32_NAME = "student_iqa_14105_fp32.tflite"
TFLITE_FP16_NAME = "student_iqa_14105_fp16.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export 14105-row mobile student IQA checkpoint to TFLite and audit parity.")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--manifest_csv", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample_count", type=int, default=300)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


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


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    x = a[finite].astype(np.float64)
    y = b[finite].astype(np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 2:
        return float("nan")
    return pearson_corr(rankdata(a[finite]), rankdata(b[finite]))


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


def load_rgb_resized(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)


def sample_manifest(manifest_csv: Path, split: str, sample_count: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)
    if "split" in df.columns:
        df = df[df["split"].astype(str).eq(split)].copy()
    if df.empty:
        raise ValueError(f"no rows for split={split}")
    per_dataset = max(1, sample_count // max(1, len(DATASET_ORDER)))
    parts = []
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"].astype(str).eq(dataset)].copy()
        if sub.empty:
            continue
        take = min(per_dataset, len(sub))
        parts.append(sub.sample(n=take, random_state=seed + len(parts)))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(out) < sample_count:
        remaining = df.drop(index=out.index, errors="ignore")
        if len(remaining):
            take = min(sample_count - len(out), len(remaining))
            out = pd.concat([out, remaining.sample(n=take, random_state=seed + 99)], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def export_tflite(model: tf.keras.Model, out_path: Path, fp16: bool) -> dict[str, Any]:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if fp16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    t0 = time.perf_counter()
    content = converter.convert()
    out_path.write_bytes(content)
    return {"path": str(out_path), "bytes": len(content), "mb": len(content) / (1024.0 * 1024.0), "export_seconds": time.perf_counter() - t0}


def interpreter_info(model_path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    ops = interpreter._get_ops_details()  # noqa: SLF001
    op_names = [str(op.get("op_name", "")) for op in ops]
    flex_ops = [name for name in op_names if name.startswith("Flex") or "Flex" in name]
    return {
        "input_shape": input_details[0].get("shape").tolist(),
        "input_dtype": str(input_details[0].get("dtype")),
        "output_shape": output_details[0].get("shape").tolist(),
        "output_dtype": str(output_details[0].get("dtype")),
        "op_count": len(op_names),
        "unique_ops": sorted(set(op_names)),
        "flex_ops": sorted(set(flex_ops)),
    }


def analyzer_text(model_paths: list[Path]) -> str:
    chunks = []
    analyzer = getattr(getattr(tf.lite, "experimental", object()), "Analyzer", None)
    if analyzer is None:
        return "tf.lite.experimental.Analyzer unavailable in this TensorFlow build.\n"
    for model_path in model_paths:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                analyzer.analyze(model_path=str(model_path), gpu_compatibility=True)
            except TypeError:
                analyzer.analyze(model_path=str(model_path))
        chunks.append(f"===== {model_path.name} =====\n{buf.getvalue()}")
    return "\n".join(chunks)


def keras_predict(model: tf.keras.Model, images: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    pred = model.predict(images, batch_size=batch_size, verbose=0).reshape(-1).astype(np.float64) * 100.0
    return pred, time.perf_counter() - t0


def tflite_predict(model_path: Path, images: np.ndarray) -> tuple[np.ndarray, float]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_index = int(input_detail["index"])
    output_index = int(output_detail["index"])
    shape = list(input_detail["shape"])
    if shape[0] != 1:
        interpreter.resize_tensor_input(input_index, [1, images.shape[1], images.shape[2], images.shape[3]])
        interpreter.allocate_tensors()
    preds = []
    t0 = time.perf_counter()
    for image in images:
        interpreter.set_tensor(input_index, image[None, ...].astype(input_detail["dtype"]))
        interpreter.invoke()
        preds.append(float(interpreter.get_tensor(output_index).reshape(-1)[0]) * 100.0)
    return np.asarray(preds, dtype=np.float64), time.perf_counter() - t0


def parity_metrics(df: pd.DataFrame, model_name: str) -> dict[str, Any]:
    keras_pred = df["keras_pred_0_100"].to_numpy(dtype=np.float64)
    tflite_pred = df[f"{model_name}_pred_0_100"].to_numpy(dtype=np.float64)
    diff = tflite_pred - keras_pred
    return {
        "model": model_name,
        "n": len(df),
        "mae_to_keras_0_100": float(np.mean(np.abs(diff))),
        "max_abs_diff_0_100": float(np.max(np.abs(diff))),
        "mean_diff_0_100": float(np.mean(diff)),
        "srcc_vs_keras": spearman_corr(keras_pred, tflite_pred),
        "plcc_vs_keras": pearson_corr(keras_pred, tflite_pred),
        "tflite_pred_min": float(np.min(tflite_pred)),
        "tflite_pred_max": float(np.max(tflite_pred)),
        "tflite_pred_mean": float(np.mean(tflite_pred)),
        "tflite_pred_std": float(np.std(tflite_pred)),
    }


def dataset_parity_metrics(df: pd.DataFrame, model_name: str) -> list[dict[str, Any]]:
    rows = [parity_metrics(df, model_name) | {"dataset": "overall"}]
    for dataset in DATASET_ORDER:
        sub = df[df["dataset"].astype(str).eq(dataset)]
        if len(sub):
            rows.append(parity_metrics(sub, model_name) | {"dataset": dataset})
    return rows


def write_hybrid_design(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Hybrid Runtime Design Note",
                "",
                "## Scope",
                "- Phase 12 is export/parity/profiling preparation only.",
                "- Student TFLite is a log-only mobile profiling candidate.",
                "- This is not a production replacement for `technical_score`.",
                "",
                "## Evidence",
                "- Phase 11 best checkpoint is strong overall and on KonIQ/SPAQ.",
                "- FLIVE remains weak: test MAE/SRCC/PLCC 8.9626/0.5176/0.5168.",
                "- Severe FP was 0, but FLIVE ranking/absolute alignment is not production-safe alone.",
                "",
                "## Candidate Runtime Plan",
                "- Keep current production/FLIVE robustness signal active.",
                "- Add student TFLite only behind telemetry/log-only switch.",
                "- Log student score, production technical score, image metadata, model version, runtime ms, and disagreement bands.",
                "- Review disagreement cases before any hybrid weighting.",
                "",
                "## Non-goals",
                "- Do not alter Flutter production scoring in Phase 12.",
                "- Do not replace `technical_score` with student model.",
                "- Do not ship final model assets from this phase.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    env_info: dict[str, Any],
    export_rows: list[dict[str, Any]],
    structure_rows: list[dict[str, Any]],
    parity_df: pd.DataFrame,
    size_df: pd.DataFrame,
    decision: str,
    runtime_seconds: float,
) -> None:
    parity_overall = parity_df[parity_df["dataset"].eq("overall")].copy()
    lines = [
        "# Mobile Student IQA 14105 TFLite Export and Hybrid Design Report",
        "",
        "## 1. Executive Summary",
        f"- Final decision: `{decision}`.",
        "- Run type: TFLite export/parity/profiling preparation only.",
        "- Mobile status: log-only candidate, not production replacement.",
        f"- Runtime seconds: `{runtime_seconds:.2f}`.",
        f"- Parity sample rows: `{int(parity_overall['n'].max()) if len(parity_overall) else 0}`.",
        "",
        "## 2. Sources Inspected",
        f"- `{args.model_path}`",
        f"- `{args.manifest_csv}`",
        f"- `{PHASE11_REPORT}`",
        f"- `{PHASE11_METRICS}`",
        f"- `{PHASE11_DATASET_METRICS}`",
        "- `scripts/technical_iqa/test_tflite_parity.py` (old hard-coded parity script).",
        "- `outputs/final_topiq_lite_mixed112_export_20260517/export_script.py` (prior builtin-only converter pattern).",
        "- `scripts/technical_iqa/evaluate_student_iqa_14105_checkpoints.py` (metric/preprocess reference).",
        "",
        "## 3. Export Configuration",
        f"- TensorFlow: `{env_info.get('tf_version')}`.",
        f"- Built with CUDA: `{env_info.get('built_cuda')}`.",
        f"- GPUs visible: `{env_info.get('gpus')}`.",
        "- Converter ops: `TFLITE_BUILTINS` only.",
        "- FP16 conversion: `Optimize.DEFAULT` plus `supported_types=[tf.float16]`.",
        "",
        markdown_table(pd.DataFrame(export_rows), ["model", "path", "mb", "export_seconds"]),
        "",
        "## 4. TFLite Model Structure",
        markdown_table(pd.DataFrame(structure_rows), ["model", "input_shape", "input_dtype", "output_shape", "output_dtype", "op_count", "has_flex_ops", "unique_ops"]),
        "",
        "## 5. Keras vs TFLite Parity",
        markdown_table(parity_df, ["model", "dataset", "n", "mae_to_keras_0_100", "max_abs_diff_0_100", "mean_diff_0_100", "srcc_vs_keras", "plcc_vs_keras", "tflite_pred_min", "tflite_pred_max", "tflite_pred_mean", "tflite_pred_std"]),
        "",
        "## 6. Model Size / Runtime Notes",
        markdown_table(size_df, ["model", "bytes", "mb", "runtime_seconds", "avg_runtime_ms_per_image"]),
        "",
        "## 7. FLIVE Weakness and Hybrid Plan",
        "- Student alone is not recommended as production replacement.",
        "- Production/FLIVE robustness signal should remain active.",
        "- Candidate deployment is log-only mobile profiling.",
        "- Hybrid design note: `hybrid_runtime_design.md`.",
        "- Do not alter Flutter production score yet.",
        "",
        "## 8. Final Decision",
        decision,
        "",
        "## 9. Recommended Next Step",
        "- Run mobile-device profiling/log-only parity against production technical score and FLIVE-heavy disagreement cases.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    model_path = Path(args.model_path)
    manifest_csv = Path(args.manifest_csv)
    fp32_path = out_dir / TFLITE_FP32_NAME
    fp16_path = out_dir / TFLITE_FP16_NAME

    env_info = {
        "tf_version": tf.__version__,
        "built_cuda": bool(tf.test.is_built_with_cuda()),
        "gpus": [str(g) for g in tf.config.list_physical_devices("GPU")],
    }

    model = tf.keras.models.load_model(model_path, compile=False)
    image_size = int(model.input_shape[1])
    sample_df = sample_manifest(manifest_csv, args.split, args.sample_count, args.seed)
    sample_df.to_csv(out_dir / "sample_manifest_used.csv", index=False)

    images = np.stack([load_rgb_resized(str(path), image_size) for path in sample_df["image_path"].tolist()], axis=0)
    keras_pred, keras_seconds = keras_predict(model, images, args.batch_size)

    export_rows = [
        {"model": "fp32", **export_tflite(model, fp32_path, fp16=False)},
        {"model": "fp16", **export_tflite(model, fp16_path, fp16=True)},
    ]

    structures = {"fp32": interpreter_info(fp32_path), "fp16": interpreter_info(fp16_path)}
    structure_rows = []
    for name, info in structures.items():
        structure_rows.append(
            {
                "model": name,
                "input_shape": info["input_shape"],
                "input_dtype": info["input_dtype"],
                "output_shape": info["output_shape"],
                "output_dtype": info["output_dtype"],
                "op_count": info["op_count"],
                "unique_ops": ",".join(info["unique_ops"]),
                "has_flex_ops": bool(info["flex_ops"]),
            }
        )

    (out_dir / "tflite_model_analyzer.txt").write_text(analyzer_text([fp32_path, fp16_path]), encoding="utf-8")

    fp32_pred, fp32_seconds = tflite_predict(fp32_path, images)
    fp16_pred, fp16_seconds = tflite_predict(fp16_path, images)

    pred_df = sample_df.copy()
    pred_df["keras_pred_0_100"] = keras_pred
    pred_df["fp32_pred_0_100"] = fp32_pred
    pred_df["fp16_pred_0_100"] = fp16_pred
    pred_df["fp32_abs_diff_to_keras_0_100"] = np.abs(fp32_pred - keras_pred)
    pred_df["fp16_abs_diff_to_keras_0_100"] = np.abs(fp16_pred - keras_pred)
    pred_df.to_csv(out_dir / "tflite_parity_predictions.csv", index=False)

    parity_rows = []
    parity_rows.extend(dataset_parity_metrics(pred_df, "fp32"))
    parity_rows.extend(dataset_parity_metrics(pred_df, "fp16"))
    parity_df = pd.DataFrame(parity_rows)
    parity_df.to_csv(out_dir / "tflite_parity_metrics.csv", index=False)

    size_rows = []
    for row in export_rows:
        model_name = row["model"]
        runtime = fp32_seconds if model_name == "fp32" else fp16_seconds
        size_rows.append(
            {
                "model": model_name,
                "bytes": row["bytes"],
                "mb": row["mb"],
                "runtime_seconds": runtime,
                "avg_runtime_ms_per_image": (runtime / max(1, len(sample_df))) * 1000.0,
            }
        )
    size_rows.append(
        {
            "model": "keras_reference",
            "bytes": model_path.stat().st_size,
            "mb": model_path.stat().st_size / (1024.0 * 1024.0),
            "runtime_seconds": keras_seconds,
            "avg_runtime_ms_per_image": (keras_seconds / max(1, len(sample_df))) * 1000.0,
        }
    )
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(out_dir / "model_size_summary.csv", index=False)

    critical_flex = any(row["has_flex_ops"] for row in structure_rows)
    parity_ok = bool(
        (parity_df[parity_df["dataset"].eq("overall")]["mae_to_keras_0_100"] <= 0.05).all()
        and (parity_df[parity_df["dataset"].eq("overall")]["max_abs_diff_0_100"] <= 0.25).all()
    )
    if critical_flex:
        decision = "STUDENT_TFLITE_NEEDS_OP_COMPATIBILITY_FIX"
    elif not parity_ok:
        decision = "STUDENT_TFLITE_NEEDS_PARITY_FIX"
    else:
        decision = "STUDENT_TFLITE_READY_FOR_MOBILE_PROFILING_LOG_ONLY"

    write_hybrid_design(out_dir / "hybrid_runtime_design.md")
    pd.DataFrame(
        [
            {"priority": 1, "action": "Profile FP16 TFLite on-device as log-only candidate; do not change production technical_score."},
            {"priority": 2, "action": "Collect production-vs-student disagreement logs with FLIVE-heavy real mobile samples."},
            {"priority": 3, "action": "Design hybrid gate/weight only after log-only mobile evidence review."},
        ]
    ).to_csv(out_dir / "next_actions.csv", index=False)
    write_report(out_dir, args, env_info, export_rows, structure_rows, parity_df, size_df, decision, time.perf_counter() - t0)
    print(f"Report: {out_dir / 'report.md'}")
    print(f"Decision: {decision}")
    print(f"FP32 MB: {size_rows[0]['mb']:.2f}")
    print(f"FP16 MB: {size_rows[1]['mb']:.2f}")


if __name__ == "__main__":
    main()
