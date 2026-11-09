# Evaluation script for TechIQA-Guard v1 Stage 1
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _configure_tensorflow() -> None:
    # Use CPU for evaluation if needed to avoid OOM, but candidate is small.
    # We will use GPU if available.
    tf.keras.mixed_precision.set_global_policy("float32")
    # Enable unsafe deserialization for Lambda layers (Keras 3)
    try:
        import keras
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
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
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'], image_tensor.numpy())
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

class KerasModelWrapper:
    def __init__(self, model_path: str, name: str):
        # safe_mode=False is required for Lambda layers with python lambdas in Keras 3
        custom_objects = {'tf': tf}
        try:
            self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom_objects)
        except Exception:
            # Fallback for older versions or different parameter support
            self.model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
        
        # Injection hack for Lambda layers referring to 'tf'
        for layer in self.model.layers:
            if 'Lambda' in str(type(layer)):
                try:
                    layer.function.__globals__['tf'] = tf
                except Exception:
                    pass
        self.name = name
        # model.input_shape might be a list or single shape
        if isinstance(self.model.input_shape, list):
             self.input_shape = self.model.input_shape[0][1:3]
        else:
             self.input_shape = self.model.input_shape[1:3]

    def predict(self, image_tensor: tf.Tensor) -> np.ndarray:
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)
        return self.model(image_tensor, training=False).numpy()

def preprocess_existing(image_path: str, size: int = 224) -> tf.Tensor:
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
    model,
    dataset_name: str,
    csv_path: str,
    model_type: str,
    max_samples: int | None = None
) -> list:
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)
    
    y_true = []
    y_pred = []
    
    start_time = time.time()
    for i, row in df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
        try:
            if model_type in ['topiq', 'techiqa_guard']:
                input_tensor = preprocess_topiq(img_path)
                raw_out = model.predict(input_tensor)
                score = float(raw_out[0][0]) * 100.0
            else:
                input_tensor = preprocess_existing(img_path)
                raw_out = model.predict(input_tensor)
                score = float(raw_out[0][0])
            
            # Use mos_100 as the target
            y_true.append(float(row['mos_100']))
            y_pred.append(score)
        except Exception as e:
            if i < 3: # Print first few errors for debugging
                print(f"Error processing {img_path}: {e}")
            continue
            
    elapsed = time.time() - start_time
    avg_ms = (elapsed / len(y_true)) * 1000 if len(y_true) > 0 else 0
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(y_true) < 2: return []
    
    metrics = {
        "dataset": dataset_name,
        "model": model.name,
        "type": model_type,
        "n": len(y_true),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred)**2))),
        "srcc": float(spearmanr(y_true, y_pred).correlation),
        "plcc": float(pearsonr(y_true, y_pred).statistic),
        "bias": float(np.mean(y_pred - y_true)),
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred)),
        "target_mean": float(np.mean(y_true)),
        "target_std": float(np.std(y_true)),
        "std_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0,
        "avg_ms": avg_ms
    }
    
    predictions = pd.DataFrame({
        "image_path": df['image_path'].iloc[:len(y_pred)],
        "y_true": y_true,
        "y_pred": y_pred
    })
    
    return [metrics, predictions]

def main():
    _configure_tensorflow()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dir", default="outputs/techiqa_guard_v1_direct_stage1_frozen_20260520/")
    parser.add_argument("--out_dir", default="outputs/eval_techiqa_guard_v1_stage1_frozen_20260520/")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading candidate model...")
    candidate_model = KerasModelWrapper(os.path.join(args.candidate_dir, "final_model.keras"), "techiqa_guard_v1_s1_frozen")
    
    models_to_test = [
        ("exports/tflite/koniq_mobile.tflite", "existing"),
        ("exports/tflite/flive_image_mobile.tflite", "existing"),
        ("outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite", "topiq"),
        ("outputs/final_topiq_lite_ranking_lam01_export_20260517/topiq_lite_ranking_lam01_gap05_fp16.tflite", "topiq"),
    ]
    
    datasets = [
        ("flive", "data/processed/techiqa_guard/test_flive.csv"),
        ("koniq", "data/processed/techiqa_guard/test_koniq.csv"),
        ("spaq", "data/processed/techiqa_guard/test_spaq.csv"),
        ("smoke", "data/processed/techiqa_guard/smoke_v1.csv"),
        ("hard_fp", "data/processed/techiqa_guard/hard_false_positive.csv"),
    ]
    
    all_metrics = []
    
    # 1. Candidate Evaluation
    for ds_name, ds_csv in datasets:
        print(f"Evaluating candidate on {ds_name}...")
        res = evaluate_model_on_dataset(candidate_model, ds_name, ds_csv, "techiqa_guard", max_samples=args.max_samples)
        if res:
            all_metrics.append(res[0])
            res[1].to_csv(out_dir / f"predictions_{candidate_model.name}_{ds_name}.csv", index=False)

    # 1b. Special False Positive Check for Candidate
    print("Running special false positive check for candidate...")
    fp_images = [
        "test_vila/20230201_181300.jpg",
        "test_vila/1675342165226-13.jpg",
        "test_vila/1675342165226-3.jpg"
    ]
    fp_results = []
    for img_path in fp_images:
        if not os.path.exists(img_path):
            print(f"FP Image not found: {img_path}")
            continue
        try:
            it = preprocess_topiq(img_path)
            score = float(candidate_model.predict(it)[0][0]) * 100.0
            fp_results.append({"image": os.path.basename(img_path), "model": candidate_model.name, "score": score})
        except Exception as e:
            print(f"Error candidate FP {img_path}: {e}")

    # 2. Baselines Evaluation
    for model_path, model_type in models_to_test:
        if not os.path.exists(model_path):
            print(f"SKIP: {model_path} not found")
            continue
        print(f"Loading baseline {model_path}...")
        model = TFLiteModelWrapper(model_path)
        for ds_name, ds_csv in datasets:
            print(f"Evaluating {model.name} on {ds_name}...")
            res = evaluate_model_on_dataset(model, ds_name, ds_csv, model_type, max_samples=args.max_samples)
            if res:
                all_metrics.append(res[0])
                res[1].to_csv(out_dir / f"predictions_{model.name}_{ds_name}.csv", index=False)
        
        # FP Check for baseline
        for img_path in fp_images:
            if not os.path.exists(img_path): continue
            try:
                if model_type == 'topiq':
                    it = preprocess_topiq(img_path)
                    s = float(model.predict(it)[0][0]) * 100.0
                else:
                    it = preprocess_existing(img_path)
                    s = float(model.predict(it)[0][0])
                fp_results.append({"image": os.path.basename(img_path), "model": model.name, "score": s})
            except Exception as e:
                print(f"Error baseline FP {img_path}: {e}")

        del model
        tf.keras.backend.clear_session()

    pd.DataFrame(fp_results).to_csv(out_dir / "false_positive_check.csv", index=False)
    
    # Existing Average Technical (koniq_mobile + flive_image_mobile) / 2
    for ds_name, _ in datasets:
        k_pred_file = out_dir / f"predictions_koniq_mobile_{ds_name}.csv"
        f_pred_file = out_dir / f"predictions_flive_image_mobile_{ds_name}.csv"
        if k_pred_file.exists() and f_pred_file.exists():
            kd = pd.read_csv(k_pred_file)
            fd = pd.read_csv(f_pred_file)
            merged = pd.merge(kd, fd, on="image_path", suffixes=("_k", "_f"))
            merged["y_pred"] = (merged["y_pred_k"] + merged["y_pred_f"]) / 2.0
            merged["y_true"] = merged["y_true_k"]
            
            y_true = merged["y_true"].values
            y_pred = merged["y_pred"].values
            
            all_metrics.append({
                "dataset": ds_name,
                "model": "existing_avg_technical",
                "type": "existing_combo",
                "n": len(y_true),
                "mae": float(np.mean(np.abs(y_true - y_pred))),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred)**2))),
                "srcc": float(spearmanr(y_true, y_pred).correlation),
                "plcc": float(pearsonr(y_true, y_pred).statistic),
                "bias": float(np.mean(y_pred - y_true)),
                "pred_mean": float(np.mean(y_pred)),
                "pred_std": float(np.std(y_pred)),
                "target_mean": float(np.mean(y_true)),
                "target_std": float(np.std(y_true)),
                "std_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else 0,
                "avg_ms": 0
            })
            merged.to_csv(out_dir / f"predictions_existing_avg_technical_{ds_name}.csv", index=False)

    pd.DataFrame(all_metrics).to_csv(out_dir / "summary.csv", index=False)
    print(f"Evaluation complete. Results in {out_dir}")

if __name__ == "__main__":
    main()
