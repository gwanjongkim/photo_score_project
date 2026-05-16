# ICAA17K DAT SavedModel을 FP32 TFLite로 변환하고 검증하는 스크립트
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv
import json
import os
import sys
import traceback
import warnings
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

sys.dont_write_bytecode = True

OUTPUT_PREFIX = "icaa_tf_native_tflite_fp32"
COMMAND_USED = "python experiments/icaa_tf_native/scripts/stage_j_convert_tflite_fp32.py"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
SEED = 123
REAL_IMAGE_COUNT = 16

# FP32 TFLite Thresholds
PREFERRED_MAX_ABS_DIFF = 1e-4
PREFERRED_MEAN_ABS_DIFF = 1e-5
ACCEPTABLE_MAX_ABS_DIFF = 1e-3
ACCEPTABLE_MEAN_ABS_DIFF = 1e-4

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def find_latest_saved_model(root: Path) -> Path:
    import glob
    # Look for the latest timestamped output directory
    pattern = str(root / "outputs" / "icaa_tf_native_savedmodel_*" / "saved_model")
    paths = glob.glob(pattern)
    if not paths:
        # Fallback to specific path mentioned in prompt
        fallback = root / "outputs/icaa_tf_native_savedmodel_20260516_042006/saved_model"
        if fallback.exists():
            return fallback
        raise RuntimeError(f"No SavedModel found with pattern: {pattern}")
    # Sort by name (timestamp is in name)
    return Path(sorted(paths)[-1])

def preprocess_real_image(path: Path):
    from PIL import Image
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (array - mean) / std

def load_real_image_batch(root: Path, limit: int):
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
            if not image_id: continue
            image_path = image_root / image_id
            if not image_path.is_file(): continue
            arrays.append(preprocess_real_image(image_path))
            paths.append(str(image_path))
            if len(arrays) >= limit: break
    if not arrays: return None, paths
    return np.stack(arrays, axis=0).astype(np.float32), paths

def compare_outputs(name: str, sm_out, tflite_out) -> dict:
    diff = np.abs(sm_out.astype(np.float32) - tflite_out.astype(np.float32))
    full_max = float(diff.max()) if diff.size else 0.0
    full_mean = float(diff.mean()) if diff.size else 0.0
    
    mos_max = float(np.abs(sm_out[:, 0] - tflite_out[:, 0]).max()) if sm_out.shape[1] > 0 else 0.0
    color_max = float(np.abs(sm_out[:, 1] - tflite_out[:, 1]).max()) if sm_out.shape[1] > 1 else 0.0
    
    preferred = (full_max <= PREFERRED_MAX_ABS_DIFF and full_mean <= PREFERRED_MEAN_ABS_DIFF)
    acceptable = (full_max <= ACCEPTABLE_MAX_ABS_DIFF and full_mean <= ACCEPTABLE_MEAN_ABS_DIFF)
    status = "pass_preferred" if preferred else "pass_acceptable" if acceptable else "fail"
    
    return {
        "name": name,
        "status": status,
        "pass_preferred": bool(preferred),
        "pass_acceptable": bool(acceptable),
        "full_max_abs_diff": full_max,
        "full_mean_abs_diff": full_mean,
        "mos_max_abs_diff": mos_max,
        "color_max_abs_diff": color_max,
        "output_shape": list(sm_out.shape),
    }

def run_tflite_inference(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    batch_size = input_data.shape[0]
    outputs = []
    for i in range(batch_size):
        interpreter.set_tensor(input_details[0]['index'], input_data[i:i+1])
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        outputs.append(out)
    return np.concatenate(outputs, axis=0)

def run():
    root = project_root()
    saved_model_path = find_latest_saved_model(root)
    log(f"Loading SavedModel from {saved_model_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"{OUTPUT_PREFIX}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    
    try:
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        log("Loading SavedModel")
        sm_model = tf.saved_model.load(str(saved_model_path))
        infer = sm_model.signatures["serving_default"]

        log("Converting SavedModel to TFLite (Fixed shape [1, 224, 224, 3])")
        # Load the SavedModel function and wrap it with a fixed-shape signature
        @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="image")])
        def model_func(image):
            return infer(image)
            
        concrete_func = model_func.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], sm_model)
        
        builtin_success = False
        select_tf_ops_required = False
        tflite_model = None

        try:
            tflite_model = converter.convert()
            builtin_success = True
            log("TFLite conversion with builtin ops succeeded.")
        except Exception as e:
            log(f"Builtin ops conversion failed: {e}")
            log("Retrying with SELECT_TF_OPS...")
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            select_tf_ops_required = True
            log("TFLite conversion with SELECT_TF_OPS succeeded.")

        tflite_filename = "icaa_dat_tf_native_fp32.tflite"
        if select_tf_ops_required:
            tflite_filename = "icaa_dat_tf_native_fp32_select_tf_ops.tflite"
            
        tflite_path = output_dir / tflite_filename
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        log(f"TFLite model saved to {tflite_path}")
        tflite_size = tflite_path.stat().st_size

        log("Loading TFLite Interpreter")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        log(f"Input details: {input_details}")
        log(f"Output details: {output_details}")
        
        log("Running Input Sensitivity Probe")
        probe_results = {}
        zero_in = np.zeros((1, 224, 224, 3), dtype=np.float32)
        one_in = np.ones((1, 224, 224, 3), dtype=np.float32)
        rand_in = np.random.randn(1, 224, 224, 3).astype(np.float32)
        
        zero_out = run_tflite_inference(interpreter, zero_in)
        one_out = run_tflite_inference(interpreter, one_in)
        rand_out = run_tflite_inference(interpreter, rand_in)
        
        diff_zero_one = float(np.abs(zero_out - one_out).max())
        diff_zero_rand = float(np.abs(zero_out - rand_out).max())
        
        log(f"Zero vs One max diff: {diff_zero_one:.9g}")
        log(f"Zero vs Random max diff: {diff_zero_rand:.9g}")
        
        input_sensitive = not (diff_zero_one < 1e-6 and diff_zero_rand < 1e-6)
        if input_sensitive:
            log("TFLite model passed input sensitivity probe.")
        else:
            log("ERROR: TFLite model failed input sensitivity probe (outputs are constant).")

        log("Running SavedModel vs TFLite Parity Verification")
        results = []
        predictions = []

        # Test A: Random Input
        sm_rand_out = infer(image=tf.constant(rand_in))["scores"].numpy()
        res_a = compare_outputs("random_normalized_input", sm_rand_out, rand_out)
        results.append(res_a)
        log(f"Test A (Random): {res_a['status']}, max_diff={res_a['full_max_abs_diff']:.9g}")

        # Test B: Real Images
        real_input_nhwc, real_paths = load_real_image_batch(root, REAL_IMAGE_COUNT)
        if real_input_nhwc is not None:
            sm_real_out = infer(image=tf.constant(real_input_nhwc))["scores"].numpy()
            tflite_real_out = run_tflite_inference(interpreter, real_input_nhwc)
            res_b = compare_outputs(f"{len(real_paths)}_real_images", sm_real_out, tflite_real_out)
            results.append(res_b)
            log(f"Test B (Real Images): {res_b['status']}, max_diff={res_b['full_max_abs_diff']:.9g}")
            for i, path in enumerate(real_paths):
                predictions.append({
                    "case": "real_images", "path": path,
                    "sm_mos": float(sm_real_out[i, 0]), "tflite_mos": float(tflite_real_out[i, 0]),
                    "sm_color": float(sm_real_out[i, 1]), "tflite_color": float(tflite_real_out[i, 1]),
                })
        
        overall_pass = input_sensitive and all(r["pass_acceptable"] for r in results)
        
        report = {
            "status": "ok" if overall_pass else "failed",
            "overall_pass": overall_pass,
            "saved_model_path": str(saved_model_path),
            "tflite_path": str(tflite_path),
            "output_dir": str(output_dir),
            "tflite_size_bytes": tflite_size,
            "builtin_success": builtin_success,
            "select_tf_ops_required": select_tf_ops_required,
            "input_sensitive": input_sensitive,
            "input_details": [str(d) for d in input_details],
            "output_details": [str(d) for d in output_details],
            "results": results,
            "log_lines": log_lines,
        }
        return report, predictions, {"zero_vs_one": diff_zero_one, "zero_vs_random": diff_zero_rand}

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "failed",
            "errors": [str(exc)] + traceback.format_exc().splitlines(),
            "log_lines": log_lines,
        }, [], {}

def main():
    report, predictions, probe_results = run()
    output_dir = Path(report.get("output_dir", "."))
    
    with open(output_dir / "stage_j_tflite_fp32_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(output_dir / "stage_j_tflite_fp32_log.txt", "w") as f:
        f.write("\n".join(report.get("log_lines", [])))
    if probe_results:
        with open(output_dir / "tflite_input_sensitivity_report.json", "w") as f:
            json.dump(probe_results, f, indent=2)
    if predictions:
        with open(output_dir / "stage_j_tflite_fp32_predictions.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case", "path", "sm_mos", "tflite_mos", "sm_color", "tflite_color"])
            writer.writeheader()
            writer.writerows(predictions)
    
    if report["status"] == "ok":
        print("Stage J completed successfully.")
        return 0
    else:
        print(f"Stage J failed with status: {report['status']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
