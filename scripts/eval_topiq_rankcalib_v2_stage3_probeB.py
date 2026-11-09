"""
Evaluate TOPIQ-RankCalib v2 Stage 3 Probe B against baselines.
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
    if model_path.endswith(".h5"):
        model.load_weights(model_path)
    else:
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(temp_model.get_weights())
    return model

def load_rankcalib_v2_keras(model_path: str) -> tf.keras.Model:
    model = build_topiq_rankcalib_v2(input_shape=(384, 384, 3))
    if model_path.endswith(".h5"):
        model.load_weights(model_path)
    else:
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(temp_model.get_weights())
    return model

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
    
    # Pre-calculate absolute paths
    abs_paths = []
    for p in df['image_path'].values:
        if not os.path.isabs(p):
            abs_paths.append(str((csv_dir / p).resolve()))
        else:
            abs_paths.append(p)
    df['abs_path'] = abs_paths

    heads = ["unified", "flive", "koniq", "spaq"] if is_rankcalib else ["default"]
    
    print(f"Evaluating {model_name} on {dataset_name} ({len(df)} samples)...")
    
    def load_and_preprocess(path, mos):
        image_bytes = tf.io.read_file(path)
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_pad(image, 384, 384)
        return image, mos

    # Create dataset for batching
    ds = tf.data.Dataset.from_tensor_slices((df['abs_path'].values, df['mos'].values))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)
    
    all_preds = []
    for batch_images, _ in ds:
        preds = model.predict(batch_images, verbose=0)
        all_preds.append(preds)
            
    # Reshape predictions
    results = []
    for i, head in enumerate(heads):
        y_pred = []
        for batch_preds in all_preds:
            if isinstance(batch_preds, dict):
                y_pred.extend(batch_preds[head].flatten())
            elif isinstance(batch_preds, list):
                y_pred.extend(batch_preds[i].flatten())
            else:
                y_pred.extend(batch_preds.flatten())
        y_pred = np.array(y_pred)
        
        # Use normalized_mos if available
        if 'normalized_mos' in df.columns:
            y_true = df['normalized_mos'].values
        else:
            y_true = df['mos'].values / 100.0
        
        # Scaling to 0-100 for metrics
        y_pred_100 = y_pred * 100
        y_true_100 = y_true * 100
        
        mae = np.mean(np.abs(y_pred_100 - y_true_100))
        rmse = np.sqrt(np.mean((y_pred_100 - y_true_100)**2))
        srcc, _ = spearmanr(y_pred, y_true)
        plcc, _ = pearsonr(y_pred, y_true)
        bias = np.mean(y_pred_100 - y_true_100)
        
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "head": head,
            "mae": float(mae),
            "rmse": float(rmse),
            "srcc": float(srcc),
            "plcc": float(plcc),
            "bias": float(bias),
            "pred_mean": float(np.mean(y_pred_100)),
            "pred_std": float(np.std(y_pred_100)),
            "target_mean": float(np.mean(y_true_100)),
            "target_std": float(np.std(y_true_100)),
            "std_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0,
        })
        
    return results

def main():
    _configure_tensorflow()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/eval_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = [
        ("mixed_112_frozen", "outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517/best.weights.h5", False),
        ("ranking_lam01_gap05", "outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517/best.weights.h5", False),
        ("rankcalib_v2_stage2", "outputs/probe_topiq_rankcalib_v2_stage2_flive_rank_lam005_e10_20260517/best.weights.h5", True),
        ("rankcalib_v2_stage3_probeA", "outputs/probe_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517/best.weights.h5", True),
        ("rankcalib_v2_stage3_probeB", "outputs/probe_topiq_rankcalib_v2_stage3_probeB_dist05_rank0005_e10_20260517/best.weights.h5", True),
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
    report = "# TOPIQ-RankCalib v2 Stage 3 Probe B Distillation Evaluation\n\n"
    
    report += "## 3. Full Test Metrics Summary\n\n"
    report += "| Dataset | Model | Head | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |\n"
    report += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for _, row in df.iterrows():
        # Highlight unified head
        head_str = f"**{row['head']}**" if row['head'] == "unified" else row['head']
        model_str = f"**{row['model']}**" if row['model'] == "rankcalib_v2_stage3_probeB" else row['model']
        report += f"| {row['dataset']} | {model_str} | {head_str} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['srcc']:.3f} | {row['plcc']:.3f} | {row['bias']:.2f} | {row['std_ratio']:.2f} |\n"
    
    (out_dir / "report.md").write_text(report)
    print(f"\nEvaluation complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
