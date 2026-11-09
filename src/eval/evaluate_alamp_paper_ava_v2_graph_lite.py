"""A-LAMP-paper-AVA-v2_graph_lite evaluation script."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

from src.models.alamp_paper_ava import get_alamp_paper_ava_custom_objects
from src.train.train_alamp_paper_ava_v2_graph_lite import _read_yaml, _resolve, _load_frame, _make_dataset, get_v2_custom_objects

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model_path")
    parser.add_argument("--test_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--test_graph_jsonl")
    parser.add_argument("--val_graph_jsonl")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--label_rule")
    args = parser.parse_args()
    config = _read_yaml(args.config)

    test_csv = _resolve(args.test_csv, config, "dataset", "test_csv", "")
    val_csv = _resolve(args.val_csv, config, "dataset", "val_csv", "")
    test_graph = _resolve(args.test_graph_jsonl, config, "dataset", "test_graph_jsonl", "")
    val_graph = _resolve(args.val_graph_jsonl, config, "dataset", "val_graph_jsonl", "")
    image_dir = _resolve(None, config, "dataset", "image_dir", "data/raw/ava/images")
    image_col = _resolve(None, config, "dataset", "image_col", "image_path")
    score_col = _resolve(None, config, "dataset", "score_col", "mean_score")
    label_threshold = 5.0
    label_rule = str(_resolve(args.label_rule, config, "dataset", "label_rule", "paper_strict"))
    
    patch_size = int(_resolve(None, config, "model", "patch_size", 224))
    global_size = int(_resolve(None, config, "model", "global_size", 384))
    patch_count = int(_resolve(None, config, "model", "patch_count", 5))
    max_objects = int(_resolve(None, config, "model", "max_objects", 4))
    batch_size = int(_resolve(args.batch_size, config, "training", "batch_size", 4))
    model_path = _resolve(args.model_path, config, "experiment", "model_path", "")
    out_dir = Path(_resolve(args.output_dir, config, "experiment", "output_dir", "outputs/alamp_paper_ava_v2_graph_lite_20260511/eval"))

    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_frame, test_arrays = _load_frame(test_csv, test_graph, image_col, score_col, image_dir, label_threshold, label_rule, None, max_objects)
    val_frame, val_arrays = _load_frame(val_csv, val_graph, image_col, score_col, image_dir, label_threshold, label_rule, None, max_objects)

    test_ds = _make_dataset(test_frame, test_arrays, patch_size, global_size, patch_count, batch_size, False, False)
    val_ds = _make_dataset(val_frame, val_arrays, patch_size, global_size, patch_count, batch_size, False, False)

    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=get_v2_custom_objects())
    
    def evaluate_split(ds, frame, arrays, name):
        start = time.time()
        preds = model.predict(ds, verbose=1).flatten()
        elapsed = time.time() - start
        
        y_true = frame["_label"].values
        y_pred = (preds > 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        
        bce = tf.keras.losses.binary_crossentropy(y_true, preds).numpy()
        
        no_obj = int(np.sum(np.sum(arrays[0], axis=1) == 0))
        
        df = pd.DataFrame({"image_path": frame[image_col], "true_label": y_true, "prediction": preds})
        df.to_csv(out_dir / f"{name}_predictions.csv", index=False)
        
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, preds)),
            "average_precision": float(average_precision_score(y_true, preds)),
            "bce_loss": float(bce),
            "confusion_matrix": f"tn={tn}, fp={fp}, fn={fn}, tp={tp}",
            "seconds_per_image": elapsed / max(1, len(y_true)),
            "no_object_count": no_obj,
            "total_count": len(y_true)
        }

    test_metrics = evaluate_split(test_ds, test_frame, test_arrays, "test")
    val_metrics = evaluate_split(val_ds, val_frame, val_arrays, "val")
    
    summary = {
        "test": test_metrics,
        "val": val_metrics
    }
    
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
