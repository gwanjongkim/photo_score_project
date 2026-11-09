'PY'
import os, time
import numpy as np
import pandas as pd
import tensorflow as tf

IMG_DIR = os.path.expanduser("~/ava/images")
CSV_PATH = os.path.expanduser("~/nima_project/data/labels.csv")

df = pd.read_csv(CSV_PATH)
score_cols = [f"c{i}" for i in range(1,11)]

def make_prob(row_counts):
    c = np.array(row_counts, dtype=np.float32)
    s = c.sum()
    return (c/s) if s>0 else np.ones((10,),np.float32)/10.0

probs = np.stack([make_prob(df.loc[i, score_cols].values) for i in range(len(df))], axis=0)
filepaths = np.array([os.path.join(IMG_DIR, fn) for fn in df["filename"].values], dtype=object)

rng = np.random.RandomState(42)
idx = np.arange(len(filepaths))
rng.shuffle(idx)
train_idx = idx[:int(len(idx)*0.9)]

train_files = filepaths[train_idx].tolist()
train_probs = probs[train_idx]

def load_train(path, y):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (256,256))
    img = tf.image.random_crop(img, (224,224,3))
    img = tf.image.random_flip_left_right(img)
    img = tf.cast(img, tf.float32)
    img = (img/127.5) - 1.0
    return img, y

BATCH = 32
train_ds = (tf.data.Dataset.from_tensor_slices((train_files, train_probs))
            .shuffle(2000)
            .map(load_train, num_parallel_calls=8)
            .ignore_errors()
            .batch(BATCH)
            .prefetch(2))

t0 = time.time()
for i, (xb, yb) in enumerate(train_ds.take(20), start=1):
    if i % 5 == 0:
        dt = time.time() - t0
        print(f"{i} batches loaded in {dt:.2f}s  |  avg {dt/i:.2f}s/batch  |  xb={xb.shape}")
PY
