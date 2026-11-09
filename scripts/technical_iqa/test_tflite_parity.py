import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from scipy.stats import spearmanr
from pathlib import Path

def test_tflite():
    interpreter = tf.lite.Interpreter(model_path="exports/tflite/koniq_mobile.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    df = pd.read_csv("outputs/paired_production_mdiqa_significance_20260603/paired_predictions.csv")
    vdf = df[df["dataset"] == "koniq10k"].head(100).copy()
    
    tflite_preds = []
    for i, row in vdf.iterrows():
        img = Image.open(row["image_path"]).convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        
        interpreter.set_tensor(input_details[0]['index'], arr[None, ...])
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])[0][0]
        tflite_preds.append(pred)
    
    vdf["tflite_pred"] = tflite_preds
    srcc_tflite = spearmanr(vdf["tflite_pred"], vdf["quality_score_0_100"])[0]
    srcc_keras = spearmanr(vdf["koniq_mobile"], vdf["quality_score_0_100"])[0]
    
    print(f"TFLite SRCC: {srcc_tflite:.4f}")
    print(f"Keras SRCC: {srcc_keras:.4f}")
    print(f"Mean Abs Diff: {np.mean(np.abs(vdf['tflite_pred'] - vdf['koniq_mobile'])):.4e}")

if __name__ == "__main__":
    test_tflite()
