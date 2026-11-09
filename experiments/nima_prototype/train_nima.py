'PY'
import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# --------------------
# GPU 안정화: VRAM을 한 번에 다 잡아먹지 않게
# --------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# XLA는 메모리 스파이크/초기 컴파일 지연이 있을 수 있어 끔(필요하면 True로 바꿔도 됨)
tf.config.optimizer.set_jit(False)

# --------------------
# Paths
# --------------------
BASE = os.path.expanduser("~/nima_project")
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR  = os.path.join(BASE, "out")
CKPT_DIR = os.path.join(OUT_DIR, "ckpt")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

IMG_DIR  = os.path.expanduser("~/ava/images")
CSV_PATH = os.path.join(DATA_DIR, "labels.csv")

# --------------------
# Load labels
# --------------------
df = pd.read_csv(CSV_PATH)
score_cols = [f"c{i}" for i in range(1, 11)]

def make_prob(row_counts):
    c = np.array(row_counts, dtype=np.float32)
    s = float(c.sum())
    return (c / s) if s > 0 else (np.ones((10,), np.float32) / 10.0)

probs = np.stack([make_prob(df.loc[i, score_cols].values) for i in range(len(df))], axis=0)
filepaths = np.array([os.path.join(IMG_DIR, fn) for fn in df["filename"].values], dtype=object)

assert os.path.exists(filepaths[0]), f"Missing: {filepaths[0]}"

# --------------------
# Split train/val
# --------------------
rng = np.random.RandomState(42)
idx = np.arange(len(filepaths))
rng.shuffle(idx)

split = int(len(idx) * 0.9)
train_idx, val_idx = idx[:split], idx[split:]

train_files = filepaths[train_idx].tolist()
train_probs = probs[train_idx]
val_files   = filepaths[val_idx].tolist()
val_probs   = probs[val_idx]

print("train:", len(train_files), "val:", len(val_files))

# --------------------
# tf.data (깨진 JPEG는 자동 스킵)
# --------------------
def load_train(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)          # 깨진 파일이면 여기서 에러
    img = tf.image.resize(img, (256, 256))
    img = tf.image.random_crop(img, (224, 224, 3))
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0                            # MobileNetV2: [-1,1]
    return img, y

def load_val(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256, 256))
    img = tf.image.central_crop(img, 224/256)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img, y

# ★ Killed(OOM) 방지용: 배치부터 낮춰 시작 (필요하면 8/16으로 더 낮추기)
BATCH = 32

train_ds = (tf.data.Dataset.from_tensor_slices((train_files, train_probs))
            .shuffle(2000)
            .map(load_train, num_parallel_calls=8)
            .ignore_errors()                 # 깨진 JPEG 스킵
            .batch(BATCH)
            .prefetch(2))

val_ds = (tf.data.Dataset.from_tensor_slices((val_files, val_probs))
          .map(load_val, num_parallel_calls=2)
          .ignore_errors()
          .batch(BATCH)
          .prefetch(2))

# --------------------
# Model: NIMA MobileNetV2
# --------------------
def build_nima_mobilenetv2(dropout=0.75):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(10, dtype="float32")(x)   # 중요
    out = tf.keras.layers.Softmax(name="score_dist", dtype="float32")(x)  # 중요
    return tf.keras.Model(base.input, out), base


def emd_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    cdf_true = tf.cumsum(y_true, axis=-1)
    cdf_pred = tf.cumsum(y_pred, axis=-1)
    return tf.reduce_mean(tf.square(cdf_true - cdf_pred))

model, base = build_nima_mobilenetv2()

# --------------------
# Resume from latest checkpoint if exists
# --------------------
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "latest.weights.h5")))
if ckpts:
    print(">> Resuming weights from:", ckpts[-1])
    model.load_weights(ckpts[-1])
else:
    print(">> No checkpoint found. Starting fresh.")

# --------------------
# Callbacks: checkpoint + auto backup
# --------------------
callbacks = [
    # 가장 최근 weights를 계속 덮어쓰기(재개용)
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CKPT_DIR, "latest.weights.h5"),
        save_weights_only=True,
        save_best_only=False
    ),
    # 가능한 범위에서 중간 상태 복구(강제 종료에도 도움이 됨)
    tf.keras.callbacks.BackupAndRestore(
        backup_dir=os.path.join(CKPT_DIR, "backup")
    ),
    tf.keras.callbacks.TerminateOnNaN(),
]

# --------------------
# Training
# --------------------
# 1) head only (짧게)
base.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=emd_loss)

# ★ "빨리 저장까지 성공"을 위해 steps_per_epoch를 걸어둠 (원하면 늘리기)
STEPS_PER_EPOCH = 800
VAL_STEPS = 200

print(">> Stage1 fit")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    callbacks=callbacks
)

# 2) finetune (짧게)
base.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss=emd_loss)

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
# Save SavedModel
# --------------------
saved_path = os.path.join(OUT_DIR, "saved_nima_aesthetic")
model.export(saved_path)
print("SavedModel:", saved_path)

# --------------------
# Convert to TFLite FP16 (with Select TF Ops fallback)
# --------------------
converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# 핵심: 기본 TFLite op + Select TF Ops 둘 다 허용
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

# 필요하면 resource variable 허용
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

tflite_path = os.path.join(OUT_DIR, "nima_aesthetic_fp16_flex.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("Saved TFLite:", tflite_path)
# --------------------
# Quick validation: keras vs tflite
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
p_tfl   = run_tflite(tflite_path, x)

print("keras mean:", mean_score(p_keras), "sum:", float(p_keras.sum()))
print("tfl   mean:", mean_score(p_tfl),   "sum:", float(p_tfl.sum()))
print("max abs diff:", float(np.max(np.abs(p_keras - p_tfl))))
PY