import os
import sys
import csv
import json
import argparse
import traceback
import random
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# Production-related imports
# We use .venv_tf for this script
import tensorflow as tf
from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image, add_bundle_arguments
from src.infer.select_best_shots import DEFAULT_SELECTOR_WEIGHTS, score_component

# Paths
MDIQA_PREDS = ROOT / "outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv"
OUTPUT_DIR = ROOT / "outputs/paired_production_mdiqa_significance_expanded_20260603"
PAIRED_MANIFEST = OUTPUT_DIR / "paired_eval_manifest_expanded.csv"
PAIRED_PREDICTIONS = OUTPUT_DIR / "paired_predictions_expanded.csv"
PRODUCTION_FAILURES = OUTPUT_DIR / "production_failures_expanded.csv"
SUMMARY_CSV = OUTPUT_DIR / "production_prediction_summary_expanded.csv"

def run_predictions(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not MDIQA_PREDS.exists():
        print(f"Error: {MDIQA_PREDS} not found.")
        return

    full_df = pd.read_csv(MDIQA_PREDS)
    
    # Filter and Sample
    dfs = []
    datasets = args.datasets.split(",")
    for ds in datasets:
        ds_df = full_df[full_df["dataset"] == ds].copy()
        if ds == "koniq10k" and args.koniq_limit:
            # Deterministic shuffle
            ds_df = ds_df.sample(n=min(len(ds_df), args.koniq_limit), random_state=args.seed)
        elif ds == "spaq" and not args.spaq_all:
             ds_df = ds_df.sample(n=min(len(ds_df), 100), random_state=args.seed) # default 100
        
        dfs.append(ds_df)
    
    df = pd.concat(dfs).copy()
    if args.smoke:
        print("Smoke mode: taking 10 images per dataset.")
        df = pd.concat([d.head(10) for d in dfs])

    # Drop duplicates by image_path to avoid re-evaluating same image multiple times
    # (MDIQA benchmark had 10 runs of same KonIQ images)
    df = df.drop_duplicates("image_path").copy()
    
    print(f"Total unique images to evaluate: {len(df)}")
    df.to_csv(PAIRED_MANIFEST, index=False)

    # Load bundle config
    with open(ROOT / "configs/stage5_reference.json", "r") as f:
        config = json.load(f)
    
    # Setup args for QualityBundle
    parser_bundle = argparse.ArgumentParser()
    add_bundle_arguments(parser_bundle)
    bundle_args = config["bundle_args"]
    args_list = []
    for k, v in bundle_args.items():
        if v is not None:
            # Ensure paths are absolute if relative to project root
            val = str(v)
            if val.startswith("checkpoints/") or val.startswith("configs/"):
                val = str(ROOT / val)
            args_list.extend([f"--{k}", val])
    bundle_namespace = parser_bundle.parse_args(args_list)

    print("Loading Production Bundle Models...")
    models = load_bundle_models(bundle_namespace)
    
    # Resume support
    processed_paths = set()
    predictions = []
    if args.resume and PAIRED_PREDICTIONS.exists():
        try:
            existing_df = pd.read_csv(PAIRED_PREDICTIONS)
            processed_paths = set(existing_df["image_path"].tolist())
            predictions = existing_df.to_dict(orient="records")
            print(f"Found existing predictions. Resuming from {len(processed_paths)} images.")
        except Exception as e:
            print(f"Warning: Could not load existing predictions for resume: {e}")

    failures = []
    total = len(df)
    
    start_time = time.time()
    
    for i, (idx, row) in enumerate(df.iterrows()):
        img_path = row["image_path"]
        if img_path in processed_paths:
            continue

        if (i+1) % 10 == 0:
            elapsed = time.time() - start_time
            avg = elapsed / (i+1) if i > 0 else 0
            eta = avg * (total - i - 1) / 60
            print(f"[{i+1}/{total}] Predicting... ETA: {eta:.1f}m")
        
        try:
            res = predict_bundle_for_image(Path(img_path), args=bundle_namespace, models=models)
            
            record = {}
            for key in ["koniq_score", "flive_image_score", "flive_patch_mean", "flive_patch_min", "musiq_score"]:
                if key in res:
                    record[key] = res[key]
            
            weights = DEFAULT_SELECTOR_WEIGHTS["technical"]
            tech_score, _, _ = score_component(record, weights, "technical")
            
            pred_row = {
                "dataset": row["dataset"],
                "image_path": row["image_path"],
                "quality_score_0_100": row["quality_score_0_100"],
                "mdiqa_variant": row.get("mdiqa_variant", "mdiqa"),
                "mdiqa_pred_0_100": row["pred_0_100"],
                "production_pred_0_100": tech_score * 100.0 if tech_score is not None else None,
                "koniq_mobile": record.get("koniq_score"),
                "flive_image_mobile": record.get("flive_image_score"),
                "flive_patch_mean": record.get("flive_patch_mean"),
                "flive_patch_min": record.get("flive_patch_min"),
                "musiq_aadb": record.get("musiq_score"),
                "production_success": True,
                "production_error": ""
            }
            predictions.append(pred_row)
            
            # Periodic save
            if len(predictions) % 50 == 0:
                pd.DataFrame(predictions).to_csv(PAIRED_PREDICTIONS, index=False)

        except Exception as e:
            failures.append({
                "dataset": row["dataset"],
                "image_path": row["image_path"],
                "error": str(e)
            })
            predictions.append({
                "dataset": row["dataset"],
                "image_path": row["image_path"],
                "quality_score_0_100": row["quality_score_0_100"],
                "mdiqa_pred_0_100": row["pred_0_100"],
                "production_pred_0_100": None,
                "production_success": False,
                "production_error": str(e)
            })

    # Final Save
    pd.DataFrame(predictions).to_csv(PAIRED_PREDICTIONS, index=False)
    print(f"Saved predictions to {PAIRED_PREDICTIONS}")
    
    if failures:
        pd.DataFrame(failures).to_csv(PRODUCTION_FAILURES, index=False)

    # Summary
    success_df = pd.DataFrame([p for p in predictions if p["production_success"]])
    if len(success_df) > 0:
        summary = success_df.groupby("dataset").agg({
            "production_pred_0_100": ["count", "min", "max", "mean"],
            "quality_score_0_100": ["mean"]
        })
        summary.to_csv(SUMMARY_CSV)

import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="koniq10k,spaq")
    parser.add_argument("--koniq_limit", type=int, default=1000)
    parser.add_argument("--spaq_all", action="store_true")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_predictions(args)
