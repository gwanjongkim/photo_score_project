import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pandas as pd

def preprocess_koniq(image: tf.Tensor, size: int = 224) -> tf.Tensor:
    image = tf.image.resize(image, (size, size))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def preprocess_topiq(image: tf.Tensor, size: int = 384) -> tf.Tensor:
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, size, size)
    return image

def save_preview(tensor, path, scale=1.0):
    # tensor is float32, usually 0..1 (KonIQ) or 0..255 (TOPIQ)
    arr = tensor.numpy()
    if scale != 1.0:
        arr = arr * scale
    
    # Clip and convert to uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

df_meta = pd.read_csv('outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/top_mismatch_image_metadata.csv')
top_5 = df_meta.head(5)

out_dir = 'outputs/c6_android_wsl_parity_mismatch_diagnosis_20260528/preprocess_debug/'
os.makedirs(out_dir, exist_ok=True)

stats = []

for idx, row in top_5.iterrows():
    filename = row['filename']
    filepath = os.path.join('test_vila', filename)
    
    # Original Preview (Raw PIL load)
    img = Image.open(filepath).convert("RGB")
    img.save(os.path.join(out_dir, f"{filename}_original.jpg"))
    
    # Original with Transpose (Hypothesized Android behavior)
    img_t = ImageOps.exif_transpose(img)
    img_t.save(os.path.join(out_dir, f"{filename}_transposed.jpg"))
    
    image_tensor = tf.convert_to_tensor(np.array(img))
    
    # KonIQ WSL Preprocessed
    k_pre = preprocess_koniq(image_tensor)
    save_preview(k_pre, os.path.join(out_dir, f"{filename}_koniq_wsl.jpg"), scale=255.0)
    
    # TOPIQ WSL Preprocessed
    t_pre = preprocess_topiq(image_tensor)
    save_preview(t_pre, os.path.join(out_dir, f"{filename}_topiq_wsl.jpg"))
    
    # Stats
    for name, tensor in [("koniq", k_pre), ("topiq", t_pre)]:
        stats.append({
            "filename": filename,
            "model": name,
            "min": float(tf.reduce_min(tensor)),
            "max": float(tf.reduce_max(tensor)),
            "mean": float(tf.reduce_mean(tensor)),
            "std": float(tf.math.reduce_std(tensor)),
            "shape": str(tensor.shape.as_list())
        })

pd.DataFrame(stats).to_csv(os.path.join(out_dir, "preprocess_stats.csv"), index=False)

