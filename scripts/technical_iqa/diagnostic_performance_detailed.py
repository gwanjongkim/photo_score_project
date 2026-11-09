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
from src.infer.predict_quality_bundle import load_bundle_models, predict_bundle_for_image, add_bundle_arguments, preprocess_image_pil, make_patch_batch, build_musiq_inputs, parse_scale_sizes

def detailed_diagnostic():
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

    print("--- START DETAILED DIAGNOSTIC ---")
    models = load_bundle_models(args)
    
    img_path = Path("data/raw/koniq10k/images/5517232825.jpg")
    if not img_path.exists():
        # Fallback to any image
        img_path = next(Path("data/raw").rglob("*.jpg"))

    print(f"Testing on: {img_path}")
    
    # 1. Image Loading
    t0 = time.perf_counter()
    img = Image.open(img_path).convert("RGB")
    t_load = time.perf_counter() - t0
    print(f"PIL Load: {t_load:.4f}s")

    # 2. Global Preprocess
    t0 = time.perf_counter()
    img_arr = preprocess_image_pil(img, size=args.image_size)
    t_prep = time.perf_counter() - t0
    print(f"Global Preprocess: {t_prep:.4f}s")

    # 3. Model AADB
    t0 = time.perf_counter()
    p_aadb = models["aadb"].predict(img_arr[None, ...], verbose=0)
    t_aadb = time.perf_counter() - t0
    print(f"AADB Predict: {t_aadb:.4f}s")

    # 4. Model KonIQ
    t0 = time.perf_counter()
    p_koniq = models["koniq"].predict(img_arr[None, ...], verbose=0)
    t_koniq = time.perf_counter() - t0
    print(f"KonIQ Predict: {t_koniq:.4f}s")

    # 5. Model FLIVE Image
    t0 = time.perf_counter()
    p_flive = models["flive_image"].predict(img_arr[None, ...], verbose=0)
    t_flive = time.perf_counter() - t0
    print(f"FLIVE Image Predict: {t_flive:.4f}s")

    # 6. FLIVE Patches
    t0 = time.perf_counter()
    patch_batch = make_patch_batch(img, size=args.image_size)
    t_patch_prep = time.perf_counter() - t0
    print(f"Patch Prep (5): {t_patch_prep:.4f}s")
    
    t0 = time.perf_counter()
    p_patches = models["flive_patch"].predict(patch_batch, verbose=0)
    t_patch_pred = time.perf_counter() - t0
    print(f"FLIVE Patch Predict (5): {t_patch_pred:.4f}s")

    # 7. MUSIQ
    t0 = time.perf_counter()
    scale_sizes = parse_scale_sizes(args.musiq_scale_sizes)
    musiq_inputs = build_musiq_inputs(img, args.musiq_patch_size, scale_sizes, args.musiq_patches_per_scale)
    t_musiq_prep = time.perf_counter() - t0
    print(f"MUSIQ Prep: {t_musiq_prep:.4f}s")
    
    t0 = time.perf_counter()
    p_musiq = models["musiq"].predict(musiq_inputs, verbose=0)
    t_musiq_pred = time.perf_counter() - t0
    print(f"MUSIQ Predict: {t_musiq_pred:.4f}s")

    print("--- END DETAILED DIAGNOSTIC ---")

if __name__ == "__main__":
    detailed_diagnostic()
