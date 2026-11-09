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

# In SavedModel, variables are often nested.
# Let's use model.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if it was v1, 
# but for v2, we can just look at model.model (the Keras model if it was saved as one)
# or just walk through the object tree.

def walk_and_check(obj, path=""):
    if hasattr(obj, "numpy"):
        val = obj.numpy()
        if np.isnan(val).any():
            print(f"ERROR: {path} contains NaNs!")
        elif np.isinf(val).any():
            print(f"ERROR: {path} contains Infs!")
        else:
            pass # print(f"OK: {path} [{val.min()}, {val.max()}]")
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            walk_and_check(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            walk_and_check(v, f"{path}[{i}]")

# model.model should be the TFICAA17KDAT instance if saved correctly.
if hasattr(model, "model"):
    print("Walking through model variables...")
    for v in model.model.variables:
        val = v.numpy()
        if np.isnan(val).any():
            print(f"Variable {v.name} has NaNs!")
        if np.isinf(val).any():
            print(f"Variable {v.name} has Infs!")
else:
    print("Walking through root variables...")
    walk_and_check(model)

print("Inspection complete.")
