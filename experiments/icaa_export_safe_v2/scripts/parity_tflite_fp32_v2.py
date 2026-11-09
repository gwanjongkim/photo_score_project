# ICAA17K export-safe v2 PyTorch 모델과 FP32 TFLite 모델의 실제 이미지 동등성을 검증하는 스크립트
from pathlib import Path
from datetime import datetime
import argparse
import csv
import importlib
import importlib.util
import json
import sys
import traceback

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
DEFAULT_TFLITE_REL_PATH = "outputs/icaa_tflite_conversion_v2_no_flex_20260514_220549/icaa_dat_export_safe_v2_float32.tflite"
MAX_ABS_TOL = 1e-4
MEAN_ABS_TOL = 1e-5

MODEL_KWARGS = {
    "img_size": 224,
    "patch_size": 4,
    "num_classes": 1,
    "expansion": 4,
    "dim_stem": 128,
    "dims": [128, 256, 512, 1024],
    "depths": [2, 2, 18, 2],
    "stage_spec": [
        ["L", "S"],
        ["L", "S"],
        ["L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D", "L", "D"],
        ["L", "D"],
    ],
    "heads": [4, 8, 16, 32],
    "window_sizes": [7, 7, 7, 7],
    "groups": [-1, -1, 4, 8],
    "use_pes": [False, False, True, True],
    "dwc_pes": [False, False, False, False],
    "strides": [-1, -1, 1, 1],
    "sr_ratios": [-1, -1, -1, -1],
    "offset_range_factor": [-1, -1, 2, 2],
    "no_offs": [False, False, False, False],
    "fixed_pes": [False, False, False, False],
    "use_dwc_mlps": [False, False, False, False],
    "use_conv_patches": False,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.5,
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def import_dat_from_package_dir(package_name: str, package_dir: Path):
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for {package_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(f"{package_name}.dat").DAT


def import_dependencies(root: Path):
    try:
        import tensorflow as tf
        import torch
        from PIL import Image
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("Failed to import TensorFlow, torch, PIL, or torchvision dependencies.") from exc

    v2_dir = root / "experiments" / "icaa_export_safe_v2" / "export_safe_models"
    v2_dat = import_dat_from_package_dir("icaa_export_safe_v2_models_tflite_fp32", v2_dir)
    return tf, torch, Image, transforms, v2_dat


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return root / path


def load_checkpoint(torch, path: Path):
    if not path.is_file():
        raise RuntimeError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_samples(csv_path: Path, image_root: Path, limit: int):
    samples = []
    missing_ids = []
    total_rows = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_rows += 1
            image_id = row["ID"]
            image_path = image_root / image_id
            if not image_path.is_file():
                missing_ids.append(image_id)
                continue
            samples.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "target_mos": float(row["MOS"]) / 10.0,
                    "target_color": float(row["color"]) / 10.0,
                }
            )
            if len(samples) >= limit:
                break
    return samples, total_rows, missing_ids


def write_predictions(path: Path, rows) -> None:
    fieldnames = [
        "image_id",
        "target_mos",
        "target_color",
        "pytorch_mos",
        "pytorch_color",
        "tflite_mos",
        "tflite_color",
        "mos_abs_diff",
        "color_abs_diff",
        "full_row_max_abs_diff",
        "full_row_mean_abs_diff",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def run_tflite(interpreter, input_index: int, output_index: int, image_nchw):
    image_nhwc = image_nchw.detach().cpu().permute(0, 2, 3, 1).numpy().astype("float32", copy=False)
    interpreter.set_tensor(input_index, image_nhwc)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare export-safe ICAA DAT v2 PyTorch and FP32 TFLite outputs.")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tflite_model", default=DEFAULT_TFLITE_REL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise RuntimeError("--limit must be >= 1.")

    root = project_root()
    tf, torch, Image, transforms, DAT = import_dependencies(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_safe_v2_tflite_fp32_parity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "tflite_fp32_parity_report.json"
    predictions_path = output_dir / "tflite_fp32_parity_predictions.csv"
    log_path = output_dir / "tflite_fp32_parity_log.txt"
    log_lines = []
    prediction_rows = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    tflite_path = resolve_path(root, args.tflite_model)
    report = {
        "status": "failed",
        "output_dir": str(output_dir),
        "limit": args.limit,
        "device": str(device),
        "thresholds": {"max_abs_diff": MAX_ABS_TOL, "mean_abs_diff": MEAN_ABS_TOL},
        "model_config": MODEL_KWARGS,
        "tflite_model": str(tflite_path),
    }

    try:
        ckpt_path = root / CHECKPOINT_REL_PATH
        checkpoint = load_checkpoint(torch, ckpt_path)
        model = DAT(**MODEL_KWARGS)
        load_result = model.load_state_dict(checkpoint, strict=True)
        model.to(device).eval()

        if not tflite_path.is_file():
            raise RuntimeError(f"TFLite model not found: {tflite_path}")
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            raise RuntimeError(f"Expected one input and one output, got {len(input_details)} inputs and {len(output_details)} outputs.")

        input_detail = input_details[0]
        output_detail = output_details[0]
        input_shape = tuple(input_detail["shape"].tolist())
        output_shape = tuple(output_detail["shape"].tolist())
        input_dtype = str(input_detail["dtype"])
        output_dtype = str(output_detail["dtype"])
        log(f"checkpoint path: {ckpt_path}")
        log(f"checkpoint key count: {len(checkpoint)}")
        log(f"strict load result: {load_result}")
        log(f"TFLite model path: {tflite_path}")
        log(f"TFLite input details: {details_for_json(input_details)}")
        log(f"TFLite output details: {details_for_json(output_details)}")
        if input_shape != (1, 224, 224, 3):
            raise RuntimeError(f"Expected TFLite input shape (1, 224, 224, 3), got {input_shape}.")
        if output_shape != (1, 2):
            raise RuntimeError(f"Expected TFLite output shape (1, 2), got {output_shape}.")
        if "float32" not in input_dtype or "float32" not in output_dtype:
            raise RuntimeError(f"Expected FP32 TFLite tensors, got input {input_dtype}, output {output_dtype}.")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        samples, total_rows, missing_ids = read_samples(root / CSV_REL_PATH, root / IMAGE_ROOT_REL_PATH, args.limit)
        if not samples:
            raise RuntimeError("No evaluable images were found.")
        log(f"csv rows read: {total_rows}")
        log(f"evaluable images: {len(samples)}")
        log(f"missing image rows skipped: {len(missing_ids)}")

        all_diffs = []
        mos_diffs = []
        color_diffs = []
        with torch.no_grad():
            for sample in samples:
                with Image.open(sample["image_path"]) as image:
                    image_tensor = transform(image.convert("RGB")).unsqueeze(0)
                if tuple(image_tensor.shape) != (1, 3, 224, 224):
                    raise RuntimeError(f"Expected PyTorch input shape (1, 3, 224, 224), got {tuple(image_tensor.shape)}.")

                pytorch_scores = model(image_tensor.to(device))[0].detach().cpu()
                tflite_scores = torch.from_numpy(
                    run_tflite(interpreter, int(input_detail["index"]), int(output_detail["index"]), image_tensor)
                )
                if tuple(pytorch_scores.shape) != tuple(tflite_scores.shape):
                    raise RuntimeError(
                        f"Shape mismatch: PyTorch={tuple(pytorch_scores.shape)}, TFLite={tuple(tflite_scores.shape)}"
                    )

                diff = (pytorch_scores - tflite_scores).abs()
                all_diffs.append(diff.reshape(-1))
                mos_diffs.append(diff[:, 0])
                color_diffs.append(diff[:, 1])
                row_diff = diff[0]
                prediction_rows.append(
                    {
                        "image_id": sample["image_id"],
                        "target_mos": sample["target_mos"],
                        "target_color": sample["target_color"],
                        "pytorch_mos": pytorch_scores[0, 0].item(),
                        "pytorch_color": pytorch_scores[0, 1].item(),
                        "tflite_mos": tflite_scores[0, 0].item(),
                        "tflite_color": tflite_scores[0, 1].item(),
                        "mos_abs_diff": row_diff[0].item(),
                        "color_abs_diff": row_diff[1].item(),
                        "full_row_max_abs_diff": row_diff.max().item(),
                        "full_row_mean_abs_diff": row_diff.mean().item(),
                    }
                )

        full_diff = torch.cat(all_diffs)
        mos_diff = torch.cat(mos_diffs)
        color_diff = torch.cat(color_diffs)
        metrics = {
            "full_tensor_max_abs_diff": full_diff.max().item(),
            "full_tensor_mean_abs_diff": full_diff.mean().item(),
            "mos_max_abs_diff": mos_diff.max().item(),
            "mos_mean_abs_diff": mos_diff.mean().item(),
            "color_max_abs_diff": color_diff.max().item(),
            "color_mean_abs_diff": color_diff.mean().item(),
        }
        passed = metrics["full_tensor_max_abs_diff"] <= MAX_ABS_TOL and metrics["full_tensor_mean_abs_diff"] <= MEAN_ABS_TOL
        report.update(
            {
                "status": "passed" if passed else "failed",
                "checkpoint_path": str(ckpt_path),
                "strict_load_result": str(load_result),
                "tflite_input_details": details_for_json(input_details),
                "tflite_output_details": details_for_json(output_details),
                "csv_rows_read": total_rows,
                "evaluable_images": len(samples),
                "missing_image_rows_skipped": len(missing_ids),
                "first_missing_image_ids": missing_ids[:10],
                "parity_metrics": metrics,
                "pass_condition": "full_tensor_max_abs_diff <= 1e-4 and full_tensor_mean_abs_diff <= 1e-5",
            }
        )
        log(f"full tensor max_abs_diff: {metrics['full_tensor_max_abs_diff']:.10f}")
        log(f"full tensor mean_abs_diff: {metrics['full_tensor_mean_abs_diff']:.10f}")
        log(f"MOS max_abs_diff: {metrics['mos_max_abs_diff']:.10f}")
        log(f"MOS mean_abs_diff: {metrics['mos_mean_abs_diff']:.10f}")
        log(f"color max_abs_diff: {metrics['color_max_abs_diff']:.10f}")
        log(f"color mean_abs_diff: {metrics['color_mean_abs_diff']:.10f}")
        log(f"TFLite FP32 parity pass: {passed}")
        if not passed:
            raise RuntimeError("TFLite FP32 parity failed thresholds.")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(f"error: {exc}")
    finally:
        write_predictions(predictions_path, prediction_rows)
        report["tflite_fp32_parity_report_json"] = str(report_path)
        report["tflite_fp32_parity_predictions_csv"] = str(predictions_path)
        report["tflite_fp32_parity_log_txt"] = str(log_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"tflite_fp32_parity_report.json: {report_path}")
        log(f"tflite_fp32_parity_predictions.csv: {predictions_path}")
        log(f"tflite_fp32_parity_log.txt: {log_path}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
