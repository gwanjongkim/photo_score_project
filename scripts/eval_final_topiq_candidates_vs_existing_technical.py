# Final benchmark script: TOPIQ-lite candidates vs existing technical models
"""
Evaluate TOPIQ-lite candidates against existing production technical models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

class TFLiteModelWrapper:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        self.name = Path(model_path).stem

    def predict(self, image_tensor: tf.Tensor) -> np.ndarray:
        # image_tensor expected to be (H, W, 3) or (1, H, W, 3)
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], image_tensor.numpy())
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

def preprocess_existing(image_path: str, size: int = 224) -> tf.Tensor:
    """Existing models use resize (stretch) and / 255.0 normalization."""
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (size, size))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_topiq(image_path: str, size: int = 384) -> tf.Tensor:
    """TOPIQ models use resize_with_pad and NO normalization (0..255)."""
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def evaluate_model_on_dataset(
    model: TFLiteModelWrapper,
    dataset_name: str,
    csv_path: str,
    model_type: str, # 'existing' or 'topiq'
    max_samples: int | None = None
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    csv_dir = Path(csv_path).parent
    
    # Pre-calculate absolute paths
    abs_paths = []
    for p in df['image_path'].values:
        if not os.path.isabs(p):
            abs_paths.append(str((csv_dir / p).resolve()))
        else:
            abs_paths.append(p)
    df['abs_path'] = abs_paths

    print(f"Evaluating {model.name} ({model_type}) on {dataset_name} ({len(df)} samples)...")
    
    y_true = []
    y_pred = []
    
    start_time = time.time()
    for i, row in df.iterrows():
        img_path = row['abs_path']
        try:
            if model_type == 'topiq':
                input_tensor = preprocess_topiq(img_path)
                # TOPIQ output is 0-1, so * 100 for score_100
                raw_out = model.predict(input_tensor)
                score = float(raw_out[0][0]) * 100.0
            else:
                input_tensor = preprocess_existing(img_path)
                # Existing models output is already 0-100
                raw_out = model.predict(input_tensor)
                score = float(raw_out[0][0])
            
            # Target MOS
            mos = float(row['mos'])
            # Ensure y_true is on 100 scale if it's 0-1
            if mos <= 1.01 and df['mos'].max() <= 1.01:
                mos *= 100.0

            y_true.append(mos)
            y_pred.append(score)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
            
    elapsed = time.time() - start_time
    avg_ms = (elapsed / len(y_true)) * 1000 if len(y_true) > 0 else 0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) < 2:
        return []
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    srcc = spearmanr(y_true, y_pred).correlation
    plcc = pearsonr(y_true, y_pred).statistic
    bias = np.mean(y_pred - y_true)
    
    results = {
        "dataset": dataset_name,
        "model": model.name,
        "type": model_type,
        "n": len(y_true),
        "mae": float(mae),
        "rmse": float(rmse),
        "srcc": float(srcc),
        "plcc": float(plcc),
        "bias": float(bias),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
        "std_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0,
        "avg_ms": avg_ms
    }
    
    # Store predictions for analysis
    predictions = pd.DataFrame({
        "image_path": df['image_path'].iloc[:len(y_pred)],
        "y_true": y_true,
        "y_pred": y_pred
    })
    
    return [results, predictions]

def main():
    _configure_tensorflow()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/eval_final_topiq_candidates_vs_existing_technical_20260520")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = [
        ("exports/tflite/koniq_mobile.tflite", "existing"),
        ("exports/tflite/flive_image_mobile.tflite", "existing"),
        ("outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite", "topiq"),
        ("outputs/final_topiq_lite_ranking_lam01_export_20260517/topiq_lite_ranking_lam01_gap05_fp16.tflite", "topiq"),
    ]
    
    datasets = [
        ("flive", "data/processed/flive/image_test.csv"),
        ("koniq", "data/processed/koniq10k/splits/test.csv"),
        ("spaq", "data/processed/spaq/test.csv"),
    ]
    
    all_metrics = []
    
    # Track predictions to create an "existing_avg" baseline
    existing_preds = {} # (dataset, model_name) -> df
    
    for model_path, model_type in models_to_test:
        if not os.path.exists(model_path):
            print(f"SKIP: {model_path} not found")
            continue
            
        model = TFLiteModelWrapper(model_path)
        
        for ds_name, ds_csv in datasets:
            res_tuple = evaluate_model_on_dataset(model, ds_name, ds_csv, model_type, max_samples=args.max_samples)
            if not res_tuple:
                continue
            
            metrics, preds = res_tuple
            all_metrics.append(metrics)
            
            if model_type == "existing":
                existing_preds[(ds_name, model.name)] = preds
            
            # Save individual predictions
            preds.to_csv(out_dir / f"predictions_{model.name}_{ds_name}.csv", index=False)
            
        del model
        tf.keras.backend.clear_session()

    # Create Simple Average Baseline for existing models
    for ds_name, _ in datasets:
        koniq_key = (ds_name, "koniq_mobile")
        flive_key = (ds_name, "flive_image_mobile")
        
        if koniq_key in existing_preds and flive_key in existing_preds:
            kp = existing_preds[koniq_key]
            fp = existing_preds[flive_key]
            
            # Merge on image_path to be safe
            merged = pd.merge(kp, fp, on="image_path", suffixes=("_k", "_f"))
            merged["y_pred"] = (merged["y_pred_k"] + merged["y_pred_f"]) / 2.0
            merged["y_true"] = merged["y_true_k"]
            
            y_true = merged["y_true"].values
            y_pred = merged["y_pred"].values
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            srcc = spearmanr(y_true, y_pred).correlation
            plcc = pearsonr(y_true, y_pred).statistic
            bias = np.mean(y_pred - y_true)
            
            avg_metrics = {
                "dataset": ds_name,
                "model": "existing_avg_technical",
                "type": "existing_combo",
                "n": len(y_true),
                "mae": float(mae),
                "rmse": float(rmse),
                "srcc": float(srcc),
                "plcc": float(plcc),
                "bias": float(bias),
                "pred_mean": float(np.mean(y_pred)),
                "pred_std": float(np.std(y_pred)),
                "target_mean": float(np.mean(y_true)),
                "target_std": float(np.std(y_true)),
                "std_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0,
                "avg_ms": 0 # Not measured for combo
            }
            all_metrics.append(avg_metrics)
            merged.to_csv(out_dir / f"predictions_existing_avg_technical_{ds_name}.csv", index=False)

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(out_dir / "summary.csv", index=False)
    
    # Generate report.md
    report = "# Final Benchmark: TOPIQ-lite Candidates vs Existing Technical Models\n\n"
    
    report += "## 2. Scaling Audit\n"
    report += "| Model | Input | Preprocess | Raw output | score_100 conversion | Evidence |\n"
    report += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    report += "| koniq_mobile | 224x224 | resize (stretch), /255.0 | 0..100 MOS | score = raw | metadata interpretation |\n"
    report += "| flive_image_mobile | 224x224 | resize (stretch), /255.0 | 0..100 MOS | score = raw | metadata interpretation |\n"
    report += "| TOPIQ mixed112 | 384x384 | resize_with_pad, NO /255.0 | 0..1 normalized | score = raw * 100 | context notes, trainer logs |\n"
    report += "| TOPIQ ranking_lam01 | 384x384 | resize_with_pad, NO /255.0 | 0..1 normalized | score = raw * 100 | context notes, trainer logs |\n"
    report += "\n"

    report += "## 3. Benchmark Metrics\n\n"
    report += "| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |\n"
    report += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    # Sort for readable table: Dataset then Model
    df_metrics = df_metrics.sort_values(["dataset", "model"])
    
    for _, row in df_metrics.iterrows():
        # Highlight TOPIQ candidates
        model_name = f"**{row['model']}**" if row['type'] == 'topiq' else row['model']
        report += f"| {row['dataset']} | {model_name} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['srcc']:.3f} | {row['plcc']:.3f} | {row['bias']:.2f} | {row['std_ratio']:.2f} |\n"
    
    (out_dir / "report.md").write_text(report)
    print(f"\nEvaluation complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
