"""
Evaluate TOPIQ-RankCalib v2 against TOPIQ-lite baselines.
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

from src.models.topiq_lite import build_topiq_lite
from src.models.topiq_rankcalib import build_topiq_rankcalib_v2

def _configure_tensorflow() -> None:
    tf.keras.mixed_precision.set_global_policy("float32")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def load_topiq_lite_keras(model_path: str) -> tf.keras.Model:
    model = build_topiq_lite(input_shape=(384, 384, 3))
    temp_model = tf.keras.models.load_model(model_path, compile=False)
    model.set_weights(temp_model.get_weights())
    return model

def load_rankcalib_v2_keras(model_path: str) -> tf.keras.Model:
    # Use the builder to ensure architecture matches, then load weights
    model = build_topiq_rankcalib_v2(input_shape=(384, 384, 3))
    temp_model = tf.keras.models.load_model(model_path, compile=False)
    model.set_weights(temp_model.get_weights())
    return model

def preprocess_image(image_path: str, size: int = 384) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def evaluate_model(
    model: tf.keras.Model,
    model_name: str,
    dataset_name: str,
    csv_path: str,
    is_rankcalib: bool,
    max_samples: int | None = None
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    csv_dir = Path(csv_path).parent
    
    heads = ["unified", "flive", "koniq", "spaq"] if is_rankcalib else ["default"]
    
    y_true = []
    y_preds = {h: [] for h in heads}
    
    print(f"Evaluating {model_name} on {dataset_name} ({len(df)} samples)...")
    
    for _, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = str((csv_dir / img_path).resolve())
        
        try:
            input_tensor = preprocess_image(img_path)
            raw_preds = model.predict(tf.expand_dims(input_tensor, 0), verbose=0)
            
            if is_rankcalib:
                # raw_preds is a dict or list depending on how it was saved
                # In our build_topiq_rankcalib_v2, outputs is a dict.
                for h in heads:
                    y_preds[h].append(float(raw_preds[h][0][0]) * 100.0)
            else:
                y_preds["default"].append(float(raw_preds[0][0]) * 100.0)
            
            mos = float(row['mos'])
            if mos <= 1.0 and df['mos'].max() <= 1.0:
                mos *= 100.0
            y_true.append(mos)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
            
    y_true = np.array(y_true)
    results = []
    
    for h in heads:
        y_pred = np.array(y_preds[h])
        if len(y_true) < 2:
            continue
            
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        srcc = spearmanr(y_true, y_pred).correlation
        plcc = pearsonr(y_true, y_pred).statistic
        bias = np.mean(y_pred - y_true)
        
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "head": h,
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
        })
        
    return results

def main():
    _configure_tensorflow()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/eval_topiq_rankcalib_v2_stage1_regression_e15_20260517")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = [
        ("mixed_112_frozen", "outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517/final_model.keras", False),
        ("ranking_lam01_gap05_e10", "outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517/final_model.keras", False),
        ("rankcalib_v2_stage1", "outputs/probe_topiq_rankcalib_v2_stage1_regression_e15_20260517/final_model.keras", True),
        ("rankcalib_v2_stage2", "outputs/probe_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517/final_model.keras", True),
    ]
    
    datasets = [
        ("flive", "data/processed/flive/image_test.csv"),
        ("koniq", "data/processed/koniq10k/splits/test.csv"),
        ("spaq", "data/processed/spaq/test.csv"),
    ]
    
    all_results = []
    
    for model_name, model_path, is_rankcalib in models_to_test:
        if not os.path.exists(model_path):
            print(f"SKIP: {model_path} not found")
            continue
            
        print(f"\n>>> Loading model: {model_name} from {model_path}")
        if is_rankcalib:
            model = load_rankcalib_v2_keras(model_path)
        else:
            model = load_topiq_lite_keras(model_path)
            
        for ds_name, ds_csv in datasets:
            res_list = evaluate_model(model, model_name, ds_name, ds_csv, is_rankcalib, max_samples=args.max_samples)
            all_results.extend(res_list)
            
        del model
        tf.keras.backend.clear_session()

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "summary.csv", index=False)
    
    # Generate report.md
    report = "# TOPIQ-RankCalib v2 Stage 1 Regression Probe Full Evaluation\n\n"
    report += "## 3. Metrics Summary\n\n"
    report += "| Dataset | Model | Head | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |\n"
    report += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for _, row in df.iterrows():
        report += f"| {row['dataset']} | {row['model']} | {row['head']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['srcc']:.3f} | {row['plcc']:.3f} | {row['bias']:.2f} | {row['std_ratio']:.2f} |\n"
    
    (out_dir / "report.md").write_text(report)
    print(f"\nEvaluation complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
