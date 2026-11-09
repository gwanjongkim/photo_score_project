import time
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import tensorflow as tf
from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image, add_bundle_arguments
from src.infer.select_best_shots import DEFAULT_SELECTOR_WEIGHTS, score_component

def diagnostic():
    # Load config
    import json
    with open("configs/stage5_reference.json", "r") as f:
        config = json.load(f)
    
    import argparse
    parser = argparse.ArgumentParser()
    add_bundle_arguments(parser)
    bundle_args = config["bundle_args"]
    args_list = []
    for k, v in bundle_args.items():
        if v is not None:
            args_list.extend([f"--{k}", str(v)])
    args = parser.parse_args(args_list)

    print("--- START DIAGNOSTIC ---")
    start_load = time.perf_counter()
    models = load_bundle_models(args)
    print(f"Model Load Time: {time.perf_counter() - start_load:.2f}s")

    manifest = pd.read_csv("outputs/paired_production_mdiqa_significance_20260603/paired_eval_manifest.csv").head(10)
    
    timings = []
    for i, row in manifest.iterrows():
        img_path = Path(row["image_path"])
        
        t0 = time.perf_counter()
        # Internal loading in predict_bundle_for_image
        res = predict_bundle_for_image(img_path, args=args, models=models)
        t_total = time.perf_counter() - t0
        
        timings.append(t_total)
        print(f"Image {i+1}: {t_total:.4f}s")

    print(f"Average Inference Time per Image: {np.mean(timings):.4f}s")
    print(f"Estimated time for 10,000 images: {np.mean(timings) * 10000 / 60:.2f} minutes")
    print("--- END DIAGNOSTIC ---")

if __name__ == "__main__":
    diagnostic()
