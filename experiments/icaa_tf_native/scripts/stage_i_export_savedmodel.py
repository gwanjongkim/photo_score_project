# ICAA17K DAT TensorFlow 모델을 SavedModel로 내보내고 동등성을 검증하는 스크립트
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import csv
import io
import json
import os
import sys
import traceback
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
OUTPUT_PREFIX = "icaa_tf_native_savedmodel"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
REAL_IMAGE_COUNT = 16
PREFERRED_MAX_ABS_DIFF = 1e-6
ACCEPTABLE_MAX_ABS_DIFF = 1e-5

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def import_tf_dat(root: Path):
    sys.path.insert(0, str(root / "experiments" / "icaa_tf_native"))
    from tf_models.tf_dat_model import TFICAA17KDAT
    return TFICAA17KDAT

def load_checkpoint(torch, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu")

def is_tensor_like(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "numel") and hasattr(value, "dtype")

def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, OrderedDict) and all(is_tensor_like(v) for v in checkpoint_obj.values()):
        return checkpoint_obj, "checkpoint object is an OrderedDict state_dict"
    if isinstance(checkpoint_obj, Mapping):
        for key in ["state_dict", "model", "model_state_dict"]:
            value = checkpoint_obj.get(key)
            if isinstance(value, Mapping) and all(is_tensor_like(v) for v in value.values()):
                return OrderedDict(value.items()), f"checkpoint object contains a {key!r} state_dict"
        if all(is_tensor_like(v) for v in checkpoint_obj.values()):
            return OrderedDict(checkpoint_obj.items()), "checkpoint object is a mapping of tensor state entries"
    raise RuntimeError(f"Checkpoint is not a recognized tensor state_dict: {type(checkpoint_obj)!r}")

def preprocess_real_image(path: Path):
    import numpy as np
    from PIL import Image
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std

def load_real_image_batch(root: Path, limit: int):
    import numpy as np
    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    if not csv_path.is_file() or not image_root.is_dir():
        return None, []
    arrays = []
    paths = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row.get("ID")
            if not image_id:
                continue
            image_path = image_root / image_id
            if not image_path.is_file():
                continue
            arrays.append(preprocess_real_image(image_path))
            paths.append(str(image_path))
            if len(arrays) >= limit:
                break
    if not arrays:
        return None, paths
    return np.stack(arrays, axis=0).astype(np.float32), paths

def compare_outputs(name: str, tf_in_memory, tf_saved_model) -> dict:
    import numpy as np
    diff = np.abs(tf_in_memory.astype(np.float32) - tf_saved_model.astype(np.float32))
    mos_diff = diff[:, 0]
    color_diff = diff[:, 1]
    full_max = float(diff.max()) if diff.size else 0.0
    full_mean = float(diff.mean()) if diff.size else 0.0
    color_max = float(color_diff.max()) if color_diff.size else 0.0
    color_mean = float(color_diff.mean()) if color_diff.size else 0.0
    mos_max = float(mos_diff.max()) if mos_diff.size else 0.0
    mos_mean = float(mos_diff.mean()) if mos_diff.size else 0.0
    preferred = full_max <= PREFERRED_MAX_ABS_DIFF
    acceptable = full_max <= ACCEPTABLE_MAX_ABS_DIFF
    status = "pass_preferred" if preferred else "pass_acceptable" if acceptable else "fail"
    return {
        "name": name,
        "status": status,
        "pass_preferred": bool(preferred),
        "pass_acceptable": bool(acceptable),
        "full_max_abs_diff": full_max,
        "full_mean_abs_diff": full_mean,
        "mos_max_abs_diff": mos_max,
        "mos_mean_abs_diff": mos_mean,
        "color_max_abs_diff": color_max,
        "color_mean_abs_diff": color_mean,
        "output_shape": list(tf_in_memory.shape),
    }

def run():
    import numpy as np
    import tensorflow as tf
    import torch

    root = project_root()
    checkpoint_path = root / CHECKPOINT_REL_PATH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"{OUTPUT_PREFIX}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    saved_model_path = output_dir / "saved_model"

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(msg)

    try:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        log(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_obj = load_checkpoint(torch, checkpoint_path)
        state_dict, _ = extract_state_dict(checkpoint_obj)

        log("Instantiating TFICAA17KDAT model")
        TFICAA17KDAT = import_tf_dat(root)
        model_tf = TFICAA17KDAT(epsilon=1e-5)
        model_tf.build_for_input_shape((1, 224, 224, 3))
        model_tf.load_from_pytorch_state_dict(state_dict)

        log("Exporting to SavedModel")
        class ServingModule(tf.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            @tf.function(input_signature=[tf.TensorSpec([None, 224, 224, 3], tf.float32, name="image")])
            def __call__(self, image):
                # Using call_with_debug or just call? call is sufficient for export.
                return {"scores": self.model(image, training=False)}

        module = ServingModule(model_tf)
        tf.saved_model.save(module, str(saved_model_path), signatures={"serving_default": module.__call__})
        log(f"SavedModel exported to {saved_model_path}")

        log("Reloading SavedModel for parity verification")
        reloaded = tf.saved_model.load(str(saved_model_path))
        infer = reloaded.signatures["serving_default"]

        log("Running parity tests")
        results = []
        predictions = []

        # Test A: Random Input
        random_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
        mem_out = model_tf(random_input, training=False).numpy()
        sm_out = infer(tf.constant(random_input))["scores"].numpy()
        res_a = compare_outputs("random_normalized_input", mem_out, sm_out)
        results.append(res_a)
        log(f"Test A (Random): {res_a['status']}, max_diff={res_a['full_max_abs_diff']:.9g}")

        # Test B: Real Images
        real_input, real_paths = load_real_image_batch(root, REAL_IMAGE_COUNT)
        if real_input is not None:
            mem_out_real = model_tf(real_input, training=False).numpy()
            sm_out_real = infer(tf.constant(real_input))["scores"].numpy()
            res_b = compare_outputs(f"{len(real_paths)}_real_images", mem_out_real, sm_out_real)
            results.append(res_b)
            log(f"Test B (Real Images): {res_b['status']}, max_diff={res_b['full_max_abs_diff']:.9g}")

            for i, path in enumerate(real_paths):
                predictions.append({
                    "case": "real_images",
                    "path": path,
                    "mem_mos": float(mem_out_real[i, 0]),
                    "sm_mos": float(sm_out_real[i, 0]),
                    "mem_color": float(mem_out_real[i, 1]),
                    "sm_color": float(sm_out_real[i, 1]),
                })
        else:
            log("Warning: Real images could not be loaded for Test B")

        overall_pass = all(r["pass_acceptable"] for r in results)
        
        report = {
            "status": "ok" if overall_pass else "parity_failed",
            "overall_pass": overall_pass,
            "command_used": COMMAND_USED,
            "checkpoint_path": str(checkpoint_path),
            "saved_model_path": str(saved_model_path),
            "output_dir": str(output_dir),
            "seed": SEED,
            "thresholds": {
                "preferred": PREFERRED_MAX_ABS_DIFF,
                "acceptable": ACCEPTABLE_MAX_ABS_DIFF,
            },
            "serving_signature": {
                "input": "image [None, 224, 224, 3] tf.float32",
                "output": "scores [None, 2] tf.float32",
            },
            "results": results,
            "real_image_paths": real_paths if real_input is not None else [],
            "log_lines": log_lines,
        }

        return report, predictions

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "failed",
            "errors": [str(exc)] + traceback.format_exc().splitlines(),
            "log_lines": log_lines,
        }, []

def main():
    report, predictions = run()
    output_dir = Path(report.get("output_dir", "."))
    
    with open(output_dir / "stage_i_savedmodel_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    with open(output_dir / "stage_i_savedmodel_log.txt", "w") as f:
        f.write("\n".join(report.get("log_lines", [])))
    
    if predictions:
        with open(output_dir / "stage_i_savedmodel_predictions.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "path", "mem_mos", "sm_mos", "mem_color", "sm_color"])
            writer.writeheader()
            writer.writerows(predictions)
    
    if report["status"] == "ok":
        print("Stage I completed successfully.")
        return 0
    else:
        print(f"Stage I failed with status: {report['status']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
