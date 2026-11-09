import os
import csv
import io
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps
import numpy as np

failed_image_path = "data/raw/paq2piq/voc_emotic_ava/AVA__325065.jpg"
output_dir = "outputs/musiq_kaggle_failure_diagnosis_20260601"
os.makedirs(output_dir, exist_ok=True)

variants = {
    "koniq-10k": "https://www.kaggle.com/models/google/musiq/TensorFlow2/koniq-10k/1",
    "spaq": "https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1",
    "paq2piq": "https://www.kaggle.com/models/google/musiq/TensorFlow2/paq2piq/1"
}

# Force CPU
tf.config.set_visible_devices([], 'GPU')

# Load models
models = {}
for name, handle in variants.items():
    print(f"Loading {name}...")
    models[name] = hub.load(handle)

def get_image_bytes_original(path):
    with open(path, "rb") as f:
        return f.read()

def get_image_bytes_reencode(path):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

def get_image_bytes_resize(path, long_side):
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        if w > h:
            new_w = long_side
            new_h = int(h * (long_side / w))
        else:
            new_h = long_side
            new_w = int(w * (long_side / h))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

fallbacks = {
    "1_original": lambda p: get_image_bytes_original(p),
    "2_reencode_rgb": lambda p: get_image_bytes_reencode(p),
    "3_resize_1024": lambda p: get_image_bytes_resize(p, 1024),
    "4_resize_768": lambda p: get_image_bytes_resize(p, 768),
}

results = []

for fb_name, fb_func in fallbacks.items():
    print(f"Testing fallback: {fb_name}")
    try:
        img_bytes = fb_func(failed_image_path)
        for v_name, model in models.items():
            result = {
                "fallback": fb_name,
                "variant": v_name,
                "success": False,
                "score": None,
                "error": None
            }
            try:
                signature = model.signatures["serving_default"]
                input_name = list(signature.structured_input_signature[1].keys())[0]
                output = signature(**{input_name: img_bytes})
                val = list(output.values())[0].numpy()
                if isinstance(val, (list, np.ndarray)):
                    val = float(val.flatten()[0])
                result["score"] = val
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)
            results.append(result)
    except Exception as e:
        print(f"Error preparing fallback {fb_name}: {e}")

with open(os.path.join(output_dir, "fallback_test_results.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fallback", "variant", "success", "score", "error"])
    writer.writeheader()
    writer.writerows(results)

print(f"Fallback results saved to {output_dir}/fallback_test_results.csv")
