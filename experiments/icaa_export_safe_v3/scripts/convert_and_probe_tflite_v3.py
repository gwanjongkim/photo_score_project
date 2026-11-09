# ICAA17K final-pooling-safe v3 ONNX 모델을 TFLite로 변환하고 입력 민감도를 점검하는 스크립트
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
sys.dont_write_bytecode = True

CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
INPUT_SENSITIVE_TOL = 1e-6


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def latest_v3_onnx(root: Path) -> Path:
    candidates = sorted(
        (root / "outputs").glob("icaa_export_safe_v3_onnx_*/icaa_dat_export_safe_v3.onnx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("No v3 ONNX model found under outputs/icaa_export_safe_v3_onnx_*/.")
    return candidates[0]


def read_first_real_image(csv_path: Path, image_root: Path):
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = row["ID"]
            image_path = image_root / image_id
            if image_path.is_file():
                return image_id, image_path
    raise RuntimeError("No real image was found from the ICAA17K CSV.")


def preprocess_real_image(image_path: Path):
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(image_path) as image:
        nchw = transform(image.convert("RGB")).unsqueeze(0).numpy().astype("float32")
    nhwc = np.transpose(nchw, (0, 2, 3, 1))
    return nchw, nhwc


def locate_float32_tflite(conversion_dir: Path) -> Path:
    candidates = sorted(conversion_dir.glob("*_float32.tflite"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError(f"No *_float32.tflite file found in {conversion_dir}")
    return candidates[0]


def details_for_json(details):
    result = []
    for item in details:
        result.append(
            {
                "name": item.get("name"),
                "index": int(item["index"]),
                "shape": item["shape"].tolist(),
                "shape_signature": item["shape_signature"].tolist(),
                "dtype": str(item["dtype"]),
            }
        )
    return result


def run_probe(tflite_path: Path, real_nchw, real_nhwc):
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(f"Expected one input and one output, got {len(input_details)} inputs and {len(output_details)} outputs.")

    input_detail = input_details[0]
    output_detail = output_details[0]
    input_shape = tuple(input_detail["shape"].tolist())
    if input_shape == tuple(real_nhwc.shape):
        real_input = real_nhwc
    elif input_shape == tuple(real_nchw.shape):
        real_input = real_nchw
    else:
        raise RuntimeError(f"TFLite input shape {input_shape} does not match NCHW {tuple(real_nchw.shape)} or NHWC {tuple(real_nhwc.shape)}.")

    rng = np.random.default_rng(123)
    probe_inputs = {
        "zero": np.zeros(input_shape, dtype=np.float32),
        "one": np.ones(input_shape, dtype=np.float32),
        "random": rng.normal(0.0, 1.0, input_shape).astype(np.float32),
        "real_image": real_input.astype(np.float32, copy=False),
    }
    outputs = {}
    for name, array in probe_inputs.items():
        interpreter.set_tensor(int(input_detail["index"]), array)
        interpreter.invoke()
        outputs[name] = interpreter.get_tensor(int(output_detail["index"])).tolist()

    zero = np.asarray(outputs["zero"], dtype=np.float32)
    diffs = {}
    max_nonzero_input_diff = 0.0
    for name, output in outputs.items():
        diff = np.abs(np.asarray(output, dtype=np.float32) - zero)
        max_abs_diff = float(diff.max())
        mean_abs_diff = float(diff.mean())
        diffs[name] = {"max_abs_diff_vs_zero": max_abs_diff, "mean_abs_diff_vs_zero": mean_abs_diff}
        if name != "zero":
            max_nonzero_input_diff = max(max_nonzero_input_diff, max_abs_diff)

    return {
        "input_details": details_for_json(input_details),
        "output_details": details_for_json(output_details),
        "outputs": outputs,
        "diffs_vs_zero": diffs,
        "max_nonzero_input_diff_vs_zero": max_nonzero_input_diff,
        "input_sensitive": max_nonzero_input_diff > INPUT_SENSITIVE_TOL,
        "input_sensitive_threshold": INPUT_SENSITIVE_TOL,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Convert v3 ONNX with onnx2tf and probe FP32 TFLite input sensitivity.")
    parser.add_argument("--onnx_model", default=None)
    parser.add_argument("--onnx2tf", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_safe_v3_tflite_probe_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    conversion_dir = output_dir / "onnx2tf_no_flex"
    report_path = output_dir / "tflite_probe_report.json"
    log_path = output_dir / "tflite_probe_log.txt"
    stdout_path = output_dir / "onnx2tf_stdout.txt"
    stderr_path = output_dir / "onnx2tf_stderr.txt"
    log_lines = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    onnx_path = resolve_path(root, args.onnx_model) or latest_v3_onnx(root)
    onnx2tf_bin = resolve_path(root, args.onnx2tf) if args.onnx2tf else None
    onnx2tf_cmd = str(onnx2tf_bin) if onnx2tf_bin is not None else shutil.which("onnx2tf")
    if onnx2tf_cmd is None:
        raise RuntimeError("onnx2tf executable was not found on PATH. Use --onnx2tf to provide a path.")

    command = [
        onnx2tf_cmd,
        "-i",
        str(onnx_path),
        "-o",
        str(conversion_dir),
        "-tb",
        "flatbuffer_direct",
        "-n",
    ]
    report = {
        "status": "failed",
        "output_dir": str(output_dir),
        "onnx_model": str(onnx_path),
        "conversion_dir": str(conversion_dir),
        "command": command,
        "input_sensitive_threshold": INPUT_SENSITIVE_TOL,
    }

    try:
        if not onnx_path.is_file():
            raise RuntimeError(f"ONNX model not found: {onnx_path}")
        image_id, image_path = read_first_real_image(root / CSV_REL_PATH, root / IMAGE_ROOT_REL_PATH)
        real_nchw, real_nhwc = preprocess_real_image(image_path)
        report["real_image_id"] = image_id
        report["real_image_path"] = str(image_path)
        report["real_nchw_shape"] = list(real_nchw.shape)
        report["real_nhwc_shape"] = list(real_nhwc.shape)

        log(f"ONNX model: {onnx_path}")
        log(f"onnx2tf command: {' '.join(command)}")
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-photo-score")
        process = subprocess.run(command, cwd=str(root), env=env, text=True, capture_output=True, check=False)
        stdout_path.write_text(process.stdout, encoding="utf-8")
        stderr_path.write_text(process.stderr, encoding="utf-8")
        report["conversion_returncode"] = process.returncode
        report["onnx2tf_stdout_txt"] = str(stdout_path)
        report["onnx2tf_stderr_txt"] = str(stderr_path)
        log(f"onnx2tf return code: {process.returncode}")
        if process.returncode != 0:
            raise RuntimeError("onnx2tf conversion failed.")

        tflite_path = locate_float32_tflite(conversion_dir)
        float16_sidecars = sorted(str(path) for path in conversion_dir.glob("*_float16.tflite"))
        report["tflite_float32_path"] = str(tflite_path)
        report["float16_sidecars_created_by_converter"] = float16_sidecars
        if float16_sidecars:
            log("onnx2tf produced float16 sidecar(s); this script does not probe or use them.")
        log(f"float32 TFLite model: {tflite_path}")

        probe = run_probe(tflite_path, real_nchw, real_nhwc)
        report["tflite_probe"] = probe
        report["status"] = "passed" if probe["input_sensitive"] else "failed"
        log(f"TFLite input details: {probe['input_details']}")
        log(f"TFLite output details: {probe['output_details']}")
        log(f"zero output: {probe['outputs']['zero']}")
        log(f"one output: {probe['outputs']['one']}")
        log(f"random output: {probe['outputs']['random']}")
        log(f"real image output: {probe['outputs']['real_image']}")
        log(f"max non-zero-input diff vs zero: {probe['max_nonzero_input_diff_vs_zero']:.10f}")
        log(f"TFLite input-sensitive: {probe['input_sensitive']}")
        if not probe["input_sensitive"]:
            raise RuntimeError("Generated FP32 TFLite output is effectively constant.")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(f"error: {exc}")
    finally:
        report["tflite_probe_report_json"] = str(report_path)
        report["tflite_probe_log_txt"] = str(log_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"tflite_probe_report.json: {report_path}")
        log(f"tflite_probe_log.txt: {log_path}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
