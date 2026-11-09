"""
Evaluate TOPIQ-lite ranking probe against baseline and reference models.
Specialized for lam01 experiment.
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
from PIL import Image

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.topiq_lite import build_topiq_lite

def _set_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)

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

    def predict(self, image_tensor: tf.Tensor) -> np.ndarray:
        # image_tensor expected to be (H, W, 3) or (1, H, W, 3)
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], image_tensor.numpy())
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

def load_topiq_keras(model_path: str) -> tf.keras.Model:
    # Build then load weights to be safe with custom objects/structure
    model = build_topiq_lite(input_shape=(384, 384, 3))
    if model_path.endswith('.weights.h5'):
        model.load_weights(model_path)
    else:
        # Load from .keras file
        temp_model = tf.keras.models.load_model(model_path, compile=False)
        model.set_weights(temp_model.get_weights())
    return model

def preprocess_mobile(image_path: str, size: int = 224) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, (size, size))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_topiq(image_path: str, size: int = 384) -> tf.Tensor:
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def evaluate_model_on_dataset(
    model: Any,
    dataset_name: str,
    csv_path: str,
    is_topiq: bool,
    max_samples: int | None = None
) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    csv_dir = Path(csv_path).parent
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating on {dataset_name} ({len(df)} samples)...")
    
    start_time = time.time()
    for _, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.isabs(img_path):
            img_path = str((csv_dir / img_path).resolve())
        
        try:
            if is_topiq:
                input_tensor = preprocess_topiq(img_path)
                score = model.predict(tf.expand_dims(input_tensor, 0), verbose=0)[0][0] * 100.0
            else:
                input_tensor = preprocess_mobile(img_path)
                score = model.predict(input_tensor)[0][0] * 100.0
            
            # Normalize MOS to 100 scale
            mos = float(row['mos'])
            if mos <= 1.0 and dataset_name in ["flive", "koniq", "spaq"]:
                if df['mos'].max() <= 1.0:
                    mos *= 100.0

            y_true.append(mos)
            y_pred.append(float(score))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    elapsed = time.time() - start_time
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) < 2:
        return {"error": "Too few samples"}
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    srcc = spearmanr(y_true, y_pred).correlation
    plcc = pearsonr(y_true, y_pred).statistic
    bias = np.mean(y_pred - y_true)
    
    return {
        "dataset": dataset_name,
        "n": len(y_true),
        "mae": float(mae),
        "rmse": float(rmse),
        "srcc": float(srcc),
        "plcc": float(plcc),
        "bias": float(bias),
        "y_true_mean": float(np.mean(y_true)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_pred_std": float(np.std(y_pred)),
        "std_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0,
        "ms_per_img": (elapsed / len(y_true)) * 1000
    }

def main():
    _configure_tensorflow()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/eval_topiq_lite_ranking_probe_lam01_gap05_e10_20260517")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = [
        ("mobile_flive", "exports/tflite/flive_image_mobile.tflite", False),
        ("mobile_koniq", "exports/tflite/koniq_mobile.tflite", False),
        ("mixed_112_frozen", "outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517/final_model.keras", True),
        ("ranking_lam05_e12", "outputs/ranking_topiq_lite_mixed112_flive_pairs_lam05_e12_20260517/final_model.keras", True),
        ("ranking_lam02_gap10_e10", "outputs/ranking_topiq_lite_mixed112_flive_pairs_lam02_gap10_e10_20260517/final_model.keras", True),
        ("ranking_lam01_gap05_e10", "outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517/final_model.keras", True),
    ]
    
    datasets = [
        ("flive", "data/processed/flive/image_test.csv"),
        ("koniq", "data/processed/koniq10k/splits/test.csv"),
        ("spaq", "data/processed/spaq/test.csv"),
    ]
    
    all_results = []
    
    for model_name, model_path, is_topiq in models_to_test:
        print(f"\n>>> Loading model: {model_name} from {model_path}")
        if not os.path.exists(model_path):
            print(f"SKIP: {model_path} not found")
            continue

        if model_path.endswith(".tflite"):
            model = TFLiteModelWrapper(model_path)
        else:
            model = load_topiq_keras(model_path)
            
        for ds_name, ds_csv in datasets:
            res = evaluate_model_on_dataset(model, ds_name, ds_csv, is_topiq, max_samples=args.max_samples)
            res["model"] = model_name
            all_results.append(res)
            
        # Clear Keras session to save memory if it's a Keras model
        if is_topiq:
            del model
            tf.keras.backend.clear_session()

    df = pd.DataFrame(all_results)
    # Reorder columns
    cols = ["dataset", "model", "n", "mae", "rmse", "srcc", "plcc", "bias", "std_ratio", "y_true_mean", "y_true_std", "y_pred_mean", "y_pred_std", "ms_per_img"]
    df = df[cols]
    df.to_csv(out_dir / "summary.csv", index=False)
    
    # Generate report.md
    report = "# TOPIQ-lite Ranking Probe Full Evaluation (lam01_gap05)\n\n"
    report += "## 3. Metrics Summary\n\n"
    report += "| Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |\n"
    report += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    
    for _, row in df.iterrows():
        report += f"| {row['dataset']} | {row['model']} | {row['mae']:.2f} | {row['rmse']:.2f} | {row['srcc']:.3f} | {row['plcc']:.3f} | {row['bias']:.2f} | {row['std_ratio']:.2f} |\n"
    
    (out_dir / "report.md").write_text(report)
    print(f"\nEvaluation complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
