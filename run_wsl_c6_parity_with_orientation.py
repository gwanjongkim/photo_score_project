import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from PIL import Image, ImageOps

class TFLiteModelWrapper:
    def __init__(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.name = Path(model_path).stem

    def predict(self, input_tensor: tf.Tensor) -> np.ndarray:
        if input_tensor.shape.rank == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor.numpy())
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

def load_image_to_tensor(image_path: str) -> tf.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img) # ADDED
    img_np = np.array(img)
    return tf.convert_to_tensor(img_np)

def preprocess_koniq(image: tf.Tensor, size: int = 224) -> tf.Tensor:
    image = tf.image.resize(image, (size, size))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_topiq(image: tf.Tensor, size: int = 384) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def compute_c6(koniq: float, topiq: float) -> float:
    return min(0.7 * topiq + 0.3 * koniq, koniq + 8.0)

def main():
    android_csv = "data/debug/c6_parity_20260527/android_c6_scores.csv"
    image_dir = Path("test_vila")
    out_dir = Path("outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/wsl_with_orientation")
    out_dir.mkdir(parents=True, exist_ok=True)

    koniq_model_path = "exports/tflite/koniq_mobile.tflite"
    topiq_model_path = "outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite"

    koniq_model = TFLiteModelWrapper(koniq_model_path)
    topiq_model = TFLiteModelWrapper(topiq_model_path)

    df_android = pd.read_csv(android_csv)
    results = []
    
    for _, row in df_android.iterrows():
        filename = row['filename']
        img_path = str(image_dir / filename)
        try:
            image_tensor = load_image_to_tensor(img_path)
            k_input = preprocess_koniq(image_tensor)
            k_raw = koniq_model.predict(k_input)
            k_score = float(k_raw[0][0])
            t_input = preprocess_topiq(image_tensor)
            t_raw = topiq_model.predict(t_input)
            t_score = float(t_raw[0][0]) * 100.0
            c6_score = compute_c6(k_score, t_score)
            results.append({
                "filename": filename,
                "wsl_koniq_score_100": k_score,
                "wsl_topiq_score_100": t_score,
                "wsl_c6_score": c6_score
            })
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    df_wsl = pd.DataFrame(results)
    df_compare = pd.merge(df_android, df_wsl, on="filename")
    df_compare["diff_c6"] = df_compare["wsl_c6_score"] - df_compare["candidate_c6_score"]
    df_compare["abs_diff_c6"] = df_compare["diff_c6"].abs()
    
    mae = df_compare["abs_diff_c6"].mean()
    max_ae = df_compare["abs_diff_c6"].max()
    srcc = spearmanr(df_compare["wsl_c6_score"], df_compare["candidate_c6_score"]).correlation
    
    print(f"MAE: {mae:.4f}")
    print(f"Max AE: {max_ae:.4f}")
    print(f"SRCC: {srcc:.4f}")
    
    df_compare.to_csv(out_dir / "parity_with_orientation.csv", index=False)

if __name__ == "__main__":
    main()
