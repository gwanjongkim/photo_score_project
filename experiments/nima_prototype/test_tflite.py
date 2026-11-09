import os
import numpy as np
import tensorflow as tf
from PIL import Image

TFLITE_PATH = os.path.expanduser("~/nima_project/out/nima_aesthetic_fp16_flex.tflite")
IMG_PATH = os.path.expanduser("~/ava/images/953417.jpg")  # 아무 이미지 1장

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    x = np.array(img).astype(np.float32)
    x = (x / 127.5) - 1.0
    x = np.expand_dims(x, axis=0)   # [1,224,224,3]
    return x

def mean_score(p):
    scores = np.arange(1, 11, dtype=np.float32)
    return float((p[0] * scores).sum())

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

x = preprocess(IMG_PATH)
interpreter.set_tensor(inp["index"], x.astype(inp["dtype"]))
interpreter.invoke()
p = interpreter.get_tensor(out["index"])

print("output shape:", p.shape)
print("distribution:", p)
print("sum:", p.sum())
print("mean score:", mean_score(p))
