import tensorflow as tf
import numpy as np
from pathlib import Path

def find_latest_saved_model(root: Path) -> Path:
    import glob
    pattern = str(root / "outputs" / "icaa_tf_native_savedmodel_*" / "saved_model")
    paths = glob.glob(pattern)
    if not paths:
        raise RuntimeError(f"No SavedModel found with pattern: {pattern}")
    return Path(sorted(paths)[-1])

root = Path(__file__).resolve().parents[0]
sm_path = find_latest_saved_model(root)
print(f"Loading SavedModel from {sm_path}")
model = tf.saved_model.load(str(sm_path))
infer = model.signatures["serving_default"]

zero_in = np.zeros((1, 224, 224, 3), dtype=np.float32)
one_in = np.ones((1, 224, 224, 3), dtype=np.float32)

out_zero = infer(image=tf.constant(zero_in))["scores"].numpy()
out_one = infer(image=tf.constant(one_in))["scores"].numpy()

print("SavedModel Zero Output:", out_zero)
print("SavedModel One Output:", out_one)

if np.isnan(out_zero).any() or np.isnan(out_one).any():
    print("ERROR: SavedModel produced NaNs!")
else:
    print("SavedModel outputs are finite.")
