import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision

# --------------------
# Mixed precision
# --------------------
mixed_precision.set_global_policy("mixed_float16")

# --------------------
# GPU memory growth
# --------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# XLA는 현재 비활성화 (속도/안정성 문제 방지)
tf.config.optimizer.set_jit(False)

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# --------------------
# Paths
# --------------------
BASE = os.path.expanduser("~/nima_project")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "out")
CKPT_DIR = os.path.join(OUT_DIR, "ckpt_technical")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

IMG_DIR = os.path.expanduser("~/tid2013/distorted_images")
CSV_PATH = os.path.join(DATA_DIR, "labels_technical.csv")

# --------------------
# Load labels
# --------------------
df = pd.read_csv(CSV_PATH)
score_cols = [f"c{i}" for i in range(1, 11)]

def make_prob(row_counts):
    c = np.array(row_counts, dtype=np.float32)
    s = float(c.sum())
    return (c / s) if s > 0 else (np.ones((10,), np.float32) / 10.0)

probs = np.stack(
    [make_prob(df.loc[i, score_cols].values) for i in range(len(df))],
    axis=0
)

filepaths = np.array(
    [os.path.join(IMG_DIR, fn) for fn in df["filename"].values],
    dtype=object
)

assert len(filepaths) > 0, "No filepaths found."
assert os.path.exists(filepaths[0]), f"Missing first image: {filepaths[0]}"

# --------------------
# Split train / val
# --------------------
rng = np.random.RandomState(42)
idx = np.arange(len(filepaths))
rng.shuffle(idx)

split = int(len(idx) * 0.9)
train_idx, val_idx = idx[:split], idx[split:]

train_files = filepaths[train_idx].tolist()
train_probs = probs[train_idx]
val_files = filepaths[val_idx].tolist()
val_probs = probs[val_idx]

print("train:", len(train_files), "val:", len(val_files))

# --------------------
# tf.data
# --------------------
def load_train(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_bmp(img)  # TID2013은 BMP
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img, y

def load_val(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_bmp(img)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img, y

BATCH = 64

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_files, train_probs))
    .shuffle(1000)
    .map(load_train, num_parallel_calls=8)
    .ignore_errors().repeat() 
    .batch(BATCH)
    .prefetch(2)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_files, val_probs))
    .map(load_val, num_parallel_calls=8)
    .ignore_errors().repeat() 
    .batch(BATCH)
    .prefetch(2)
)

# --------------------
# Model
# --------------------
def build_nima_mobilenetv2(dropout=0.75):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout)(x)

    # 출력은 float32로 고정
    x = tf.keras.layers.Dense(10, dtype="float32")(x)
    out = tf.keras.layers.Softmax(name="score_dist", dtype="float32")(x)

    model = tf.keras.Model(base.input, out)
    return model, base

def emd_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    cdf_true = tf.cumsum(y_true, axis=-1)
    cdf_pred = tf.cumsum(y_pred, axis=-1)
    return tf.reduce_mean(tf.square(cdf_true - cdf_pred))

model, base = build_nima_mobilenetv2()

# --------------------
# Resume from checkpoint if exists
# --------------------
latest_ckpt = os.path.join(CKPT_DIR, "latest.weights.h5")
resume = os.path.exists(latest_ckpt)

if resume:
    print(">> Resuming from:", latest_ckpt)
    model.load_weights(latest_ckpt)
else:
    print(">> No checkpoint found. Starting fresh.")

# --------------------
# Callbacks
# --------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=latest_ckpt,
        save_weights_only=True,
        save_best_only=False
    ),
    tf.keras.callbacks.BackupAndRestore(
        backup_dir=os.path.join(CKPT_DIR, "backup")
    ),
    tf.keras.callbacks.TerminateOnNaN(),
]

# --------------------
# Training settings
# --------------------
STEPS_PER_EPOCH = 80
VAL_STEPS = 8

# --------------------
# Stage 1: head only
# --------------------
if not resume:
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=emd_loss
    )

    print(">> Stage1 fit")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        callbacks=callbacks
    )

# --------------------
# Stage 2: finetune
# --------------------
base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=emd_loss
)

print(">> Stage2 fit")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

# --------------------
# Save SavedModel (Keras 3 방식)
# --------------------
saved_path = os.path.join(OUT_DIR, "saved_nima_technical")
model.export(saved_path)
print("SavedModel:", saved_path)

# --------------------
# Convert to TFLite FP16 (Flex)
# --------------------
converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

tflite_path = os.path.join(OUT_DIR, "nima_technical_fp16_flex.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Saved TFLite:", tflite_path)

# --------------------
# Quick validation
# --------------------
def run_tflite(path, x):
    itp = tf.lite.Interpreter(model_path=path)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    itp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    itp.invoke()
    return itp.get_tensor(out["index"])

def mean_score(p):
    scores = np.arange(1, 11, dtype=np.float32)
    return float((p[0] * scores).sum())

for xb, yb in val_ds.take(1):
    x = xb[:1].numpy()
    break

p_keras = model(x, training=False).numpy()
p_tfl = run_tflite(tflite_path, x)

print("keras mean:", mean_score(p_keras), "sum:", float(p_keras.sum()))
print("tfl   mean:", mean_score(p_tfl), "sum:", float(p_tfl.sum()))
print("max abs diff:", float(np.max(np.abs(p_keras - p_tfl))))