# A-LAMP(AVA) 및 RGNet(AADB) 모델의 구버전과 신버전을 논문 벤치마크 지표로 비교 평가한다.
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Preprocessing Constants
ALAMP_GLOBAL_SIZE = 384
ALAMP_PATCH_SIZE = 224
ALAMP_PATCH_COUNT = 5
RGNET_SIZE = 256

def load_image_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32)

def extract_patches(image_path: Path, boxes_abs: list[list[float]], patch_size: int) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        patches = []
        for box in boxes_abs:
            crop = img.crop((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
            crop = crop.resize((patch_size, patch_size), Image.BILINEAR)
            patches.append(np.asarray(crop, dtype=np.float32))
        return np.asarray(patches, dtype=np.float32)

def make_interpreter(path: Path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter

def tflite_predict(interpreter, inputs: dict[str, np.ndarray]) -> float:
    input_details = interpreter.get_input_details()
    for detail in input_details:
        name = detail['name']
        # Handle different naming conventions
        key = None
        if 'patches' in name:
            key = 'patches'
        elif 'full_image' in name or 'global_view' in name:
            key = 'full_image'
        elif 'image' in name or 'keras_tensor' in name:
            key = 'image'
        
        if key and key in inputs:
            interpreter.set_tensor(detail['index'], inputs[key][None, ...])
    
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    return float(interpreter.get_tensor(output_details[0]['index']).flatten()[0])

def evaluate_alamp(old_path, new_path, patch_jsonl, output_dir, max_samples=None):
    print(f"Evaluating A-LAMP on {patch_jsonl}...")
    old_interpreter = make_interpreter(old_path)
    new_interpreter = make_interpreter(new_path)
    
    records = []
    with open(patch_jsonl, 'r') as f:
        for i, line in enumerate(f):
            records.append(json.loads(line))
            if max_samples and i + 1 >= max_samples:
                break
    
    results = []
    y_true = []
    y_old = []
    y_new = []
    
    for record in records:
        img_path = REPO_ROOT / record['image_path']
        if not img_path.exists():
            continue
            
        full_img = load_image_rgb(img_path, ALAMP_GLOBAL_SIZE)
        patches = extract_patches(img_path, record['boxes_abs_xyxy'][:ALAMP_PATCH_COUNT], ALAMP_PATCH_SIZE)
        
        # Old ALAMP expects [0, 1]
        old_inputs = {
            'full_image': full_img / 255.0,
            'patches': patches / 255.0
        }
        # New ALAMP expects [0, 255]
        new_inputs = {
            'full_image': full_img,
            'patches': patches
        }
        
        pred_old = tflite_predict(old_interpreter, old_inputs)
        pred_new = tflite_predict(new_interpreter, new_inputs)
        
        label = 1 if record['mean_score'] > 5.0 else 0
        y_true.append(label)
        y_old.append(pred_old)
        y_new.append(pred_new)
        
        results.append({
            'image_path': record['image_path'],
            'mean_score': record['mean_score'],
            'label': label,
            'old_pred': pred_old,
            'new_pred': pred_new
        })
        
    y_true = np.array(y_true)
    y_old = np.array(y_old)
    y_new = np.array(y_new)
    
    metrics = {
        'old': calculate_binary_metrics(y_true, y_old),
        'new': calculate_binary_metrics(y_true, y_new)
    }
    
    # Save predictions
    save_predictions(results, output_dir / "alamp_predictions.csv")
    return metrics

def calculate_binary_metrics(y_true, y_pred, threshold=0.5):
    p = y_pred >= threshold
    tp = np.sum((p == 1) & (y_true == 1))
    tn = np.sum((p == 0) & (y_true == 0))
    fp = np.sum((p == 1) & (y_true == 0))
    fn = np.sum((p == 0) & (y_true == 1))
    
    acc = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred)),
        'pred_min': float(np.min(y_pred)),
        'pred_max': float(np.max(y_pred))
    }

def evaluate_rgnet(old_path, new_path, test_csv, output_dir, max_samples=None):
    print(f"Evaluating RGNet on {test_csv}...")
    old_interpreter = make_interpreter(old_path)
    new_interpreter = make_interpreter(new_path)
    
    df = pd.read_csv(test_csv)
    if max_samples:
        df = df.head(max_samples)
        
    results = []
    y_true = []
    y_old = []
    y_new = []
    
    for _, row in df.iterrows():
        img_path = REPO_ROOT / row['image_path']
        if not img_path.exists():
            continue
            
        img = load_image_rgb(img_path, RGNET_SIZE)
        
        # Both expect [0, 1]
        inputs = {'image': img / 255.0}
        
        pred_old = tflite_predict(old_interpreter, inputs)
        pred_new = tflite_predict(new_interpreter, inputs)
        
        y_true.append(row['score'])
        y_old.append(pred_old)
        y_new.append(pred_new)
        
        results.append({
            'image_path': row['image_path'],
            'true_score': row['score'],
            'old_pred': pred_old,
            'new_pred': pred_new
        })
        
    y_true = np.array(y_true)
    y_old = np.array(y_old)
    y_new = np.array(y_new)
    
    metrics = {
        'old': calculate_regression_metrics(y_true, y_old),
        'new': calculate_regression_metrics(y_true, y_new)
    }
    
    # Save predictions
    save_predictions(results, output_dir / "rgnet_predictions.csv")
    return metrics

def calculate_regression_metrics(y_true, y_pred):
    from scipy.stats import spearmanr, pearsonr
    srcc = spearmanr(y_true, y_pred)[0]
    plcc = pearsonr(y_true, y_pred)[0]
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    return {
        'srcc': float(srcc),
        'plcc': float(plcc),
        'mae': float(mae),
        'rmse': float(rmse),
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred)),
        'pred_min': float(np.min(y_pred)),
        'pred_max': float(np.max(y_pred))
    }

def save_predictions(results, path):
    if not results:
        return
    keys = results[0].keys()
    with open(path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

import pandas as pd # Needed for RGNet CSV read

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_alamp", default="models/aesthetic/alamp_aadb_gpu.tflite")
    parser.add_argument("--new_alamp", default="outputs/mobile_alamp_v2_full_ava_20260519_tflite/mobile_alamp_v2_fp16.tflite")
    parser.add_argument("--old_rgnet", default="models/aesthetic/rgnet_aadb_gpu.tflite")
    parser.add_argument("--new_rgnet", default="models/aesthetic/rgnet_paper_aadb_fp16.tflite")
    parser.add_argument("--ava_patch_jsonl", default="outputs/alamp_v4_full_ava_20260517/subsets/test_patch_boxes_full_v4.jsonl")
    parser.add_argument("--aadb_test_csv", default="data/processed/aadb/test.csv")
    parser.add_argument("--output_dir", default="outputs/aesthetic_paper_benchmark_old_vs_new_20260523")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args)
    }
    
    # Evaluate A-LAMP
    alamp_metrics = evaluate_alamp(
        REPO_ROOT / args.old_alamp,
        REPO_ROOT / args.new_alamp,
        REPO_ROOT / args.ava_patch_jsonl,
        output_dir,
        max_samples=args.max_samples
    )
    summary['alamp'] = alamp_metrics
    
    # Evaluate RGNet
    rgnet_metrics = evaluate_rgnet(
        REPO_ROOT / args.old_rgnet,
        REPO_ROOT / args.new_rgnet,
        REPO_ROOT / args.aadb_test_csv,
        output_dir,
        max_samples=args.max_samples
    )
    summary['rgnet'] = rgnet_metrics
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Audit completed. Summary saved to {output_dir / 'summary.json'}")

if __name__ == "__main__":
    main()
