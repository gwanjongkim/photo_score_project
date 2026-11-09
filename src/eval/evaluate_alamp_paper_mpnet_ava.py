"""A-LAMP-paper-MPNet evaluation script."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from src.models.alamp_paper_mpnet import get_alamp_paper_mpnet_custom_objects
from src.train.train_alamp_paper_mpnet_ava import _read_yaml, _resolve, _make_dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--test_jsonl")
    parser.add_argument("--val_jsonl")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--label_rule", default="paper_strict")
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--selector_name")
    args = parser.parse_args()
    
    config = _read_yaml(args.config)
    
    test_jsonl = _resolve(args.test_jsonl, config, "dataset", "test_jsonl", "")
    val_jsonl = _resolve(args.val_jsonl, config, "dataset", "val_jsonl", "")
    model_path = _resolve(args.model_path, config, "evaluation", "model_path", "")
    output_dir = Path(_resolve(args.output_dir, config, "evaluation", "output_dir", "outputs/alamp_paper_mpnet_selector_compare_20260512/eval"))
    if args.selector_name:
        output_dir = output_dir / args.selector_name
        
    batch_size = int(_resolve(args.batch_size, config, "evaluation", "batch_size", 4))
    max_test = _resolve(args.max_test_samples, config, "evaluation", "max_test_samples", None)
    max_val = _resolve(args.max_val_samples, config, "evaluation", "max_val_samples", None)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects=get_alamp_paper_mpnet_custom_objects()
    )
    
    def evaluate_split(jsonl_path, name, max_samples):
        ds = _make_dataset(jsonl_path, batch_size=batch_size, max_samples=max_samples, training=False)
        
        y_true = []
        y_pred_probs = []
        
        start_time = time.time()
        for batch_images, batch_labels in ds:
            probs = model(batch_images, training=False).numpy()
            y_true.extend(batch_labels.numpy().flatten())
            y_pred_probs.extend(probs.flatten())
        elapsed = time.time() - start_time
        
        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        y_pred = (y_pred_probs > 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        
        # Binary Cross Entropy
        bce = tf.keras.losses.binary_crossentropy(y_true.reshape(-1, 1), y_pred_probs.reshape(-1, 1)).numpy().mean()
        
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_probs)),
            "average_precision": float(average_precision_score(y_true, y_pred_probs)),
            "bce_loss": float(bce),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "seconds_per_image": float(elapsed / len(y_true)) if len(y_true) > 0 else 0.0,
            "total_samples": len(y_true)
        }
        
        # Save predictions
        df = pd.DataFrame({"true": y_true, "pred": y_pred_probs})
        df.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        
        return metrics

    test_metrics = evaluate_split(test_jsonl, "test", max_test)
    val_metrics = evaluate_split(val_jsonl, "val", max_val)
    
    summary = {
        "selector_name": args.selector_name,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics
    }
    
    with open(output_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
