import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.topiq_lite import build_topiq_lite

def get_tflite_ops(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    op_details = interpreter._get_ops_details()
    ops = sorted(list(set([op['op_name'] for op in op_details])))
    return ops

def preprocess_topiq(image_path, size=384):
    image_bytes = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/ranking_topiq_lite_mixed112_flive_pairs_lam01_gap05_e10_20260517/final_model.keras")
    parser.add_argument("--out_dir", default="outputs/final_topiq_lite_ranking_lam01_export_20260517")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Keras Model
    print(f"Loading Keras model from {args.model_path}...")
    try:
        keras_model = tf.keras.models.load_model(args.model_path, compile=False)
    except Exception as e:
        print(f"Failed to load .keras model: {e}. Rebuilding and loading weights.")
        keras_model = build_topiq_lite(input_shape=(384, 384, 3))
        weights_path = os.path.join(os.path.dirname(args.model_path), "best.weights.h5")
        keras_model.load_weights(weights_path)

    # 2. Export FP32
    print("Exporting FP32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp32_content = converter.convert()
    fp32_path = out_dir / "topiq_lite_ranking_lam01_gap05_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32_content)

    # 3. Export FP16
    print("Exporting FP16 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_fp16_content = converter.convert()
    fp16_path = out_dir / "topiq_lite_ranking_lam01_gap05_fp16.tflite"
    fp16_path.write_bytes(tflite_fp16_content)

    # 4. Analyze Ops
    ops_fp32 = get_tflite_ops(str(fp32_path))
    ops_fp16 = get_tflite_ops(str(fp16_path))
    with open(out_dir / "tflite_ops.txt", "w") as f:
        f.write("FP32 Ops:\n" + "\n".join(ops_fp32) + "\n\n")
        f.write("FP16 Ops:\n" + "\n".join(ops_fp16) + "\n")

    # 5. Parity Check
    datasets = {
        "spaq": "data/processed/spaq/test.csv",
        "koniq": "data/processed/koniq10k/splits/test.csv",
        "flive": "data/processed/flive/image_test.csv"
    }

    interpreters = {
        "fp32": tf.lite.Interpreter(model_path=str(fp32_path)),
        "fp16": tf.lite.Interpreter(model_path=str(fp16_path))
    }
    for interp in interpreters.values():
        interp.allocate_tensors()

    all_parity_results = []

    for ds_name, ds_csv in datasets.items():
        print(f"Checking parity on {ds_name}...")
        df = pd.read_csv(ds_csv).head(8)
        csv_dir = Path(ds_csv).parent
        
        ds_results = []
        for _, row in df.iterrows():
            img_path = row['image_path']
            if not os.path.isabs(img_path):
                img_path = str((csv_dir / img_path).resolve())
            
            input_tensor = preprocess_topiq(img_path)
            input_batch = tf.expand_dims(input_tensor, 0)

            # Keras
            keras_pred = keras_model.predict(input_batch, verbose=0)[0][0] * 100.0

            # TFLite FP32
            interp = interpreters["fp32"]
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()
            interp.set_tensor(input_details[0]['index'], input_batch.numpy())
            interp.invoke()
            tflite_fp32_pred = interp.get_tensor(output_details[0]['index'])[0][0] * 100.0

            # TFLite FP16
            interp = interpreters["fp16"]
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()
            interp.set_tensor(input_details[0]['index'], input_batch.numpy())
            interp.invoke()
            tflite_fp16_pred = interp.get_tensor(output_details[0]['index'])[0][0] * 100.0

            res = {
                "dataset": ds_name,
                "image": os.path.basename(img_path),
                "keras": float(keras_pred),
                "tflite_fp32": float(tflite_fp32_pred),
                "tflite_fp16": float(tflite_fp16_pred),
                "diff_fp32": float(abs(keras_pred - tflite_fp32_pred)),
                "diff_fp16": float(abs(keras_pred - tflite_fp16_pred))
            }
            ds_results.append(res)
            all_parity_results.append(res)
        
        pd.DataFrame(ds_results).to_csv(out_dir / f"parity_{ds_name}.csv", index=False)

    full_parity_df = pd.DataFrame(all_parity_results)
    summary_data = []
    for ds_name in datasets.keys():
        ds_df = full_parity_df[full_parity_df["dataset"] == ds_name]
        summary_data.append({
            "dataset": ds_name,
            "fp32_max_diff": ds_df["diff_fp32"].max(),
            "fp32_mean_diff": ds_df["diff_fp32"].mean(),
            "fp16_max_diff": ds_df["diff_fp16"].max(),
            "fp16_mean_diff": ds_df["diff_fp16"].mean(),
        })
    pd.DataFrame(summary_data).to_csv(out_dir / "summary.csv", index=False)

    print(f"Export and Parity complete. Files in {out_dir}")

if __name__ == "__main__":
    main()
