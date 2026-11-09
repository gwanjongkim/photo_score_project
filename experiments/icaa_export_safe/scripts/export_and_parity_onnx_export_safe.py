# ICAA17K export-safe DAT 모델의 ONNX 내보내기와 ONNX Runtime 동등성 검증 스크립트
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import sys
import traceback
import warnings

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
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


def import_dependencies(root: Path):
    export_safe_root = root / "experiments" / "icaa_export_safe"
    sys.path.insert(0, str(export_safe_root))

    try:
        import onnx
        import onnxruntime as ort
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from export_safe_models.dat import DAT
    except Exception as exc:
        raise RuntimeError("Failed to import ONNX/ORT, torch/PIL/torchvision, or export-safe DAT.") from exc

    return onnx, ort, torch, nn, Image, DataLoader, Dataset, transforms, DAT


def load_checkpoint(torch, checkpoint_path: Path):
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


def check_state_dict_compatibility(model, checkpoint):
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint.keys())
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    shape_mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        if shape_of(model_state[key]) != shape_of(checkpoint[key]):
            shape_mismatches.append(
                {
                    "key": key,
                    "model_shape": shape_of(model_state[key]),
                    "checkpoint_shape": shape_of(checkpoint[key]),
                }
            )
    return missing, unexpected, shape_mismatches


def make_wrapper_class(nn):
    class DATScoresOnly(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            scores, positions, references = self.model(image)
            return scores

    return DATScoresOnly


def read_samples(csv_path: Path, image_root: Path, limit: int):
    samples = []
    missing_ids = []
    total_rows = 0
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"ID", "MOS", "color"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise RuntimeError(f"CSV is missing required fields {sorted(required_fields)}: {csv_path}")

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


def make_dataset_class(Dataset, Image):
    class ICAA17KDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            sample = self.samples[index]
            with Image.open(sample["image_path"]) as image:
                image_tensor = self.transform(image.convert("RGB"))
            target = image_tensor.new_tensor([sample["target_mos"], sample["target_color"]])
            return image_tensor, target, sample["image_id"]

    return ICAA17KDataset


def warning_messages(recorded_warnings):
    counts = {}
    for item in recorded_warnings:
        message = f"{item.category.__name__}: {item.message}"
        counts[message] = counts.get(message, 0) + 1
    return [f"{message} (count={count})" for message, count in counts.items()]


def export_onnx(torch, wrapper, dummy_input, onnx_path: Path, opset: int):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            input_names=["image"],
            output_names=["scores"],
            dynamic_axes=None,
            opset_version=opset,
            do_constant_folding=True,
        )
    return warning_messages(caught)


def write_predictions(path: Path, rows) -> None:
    fieldnames = [
        "image_id",
        "target_mos",
        "target_color",
        "pytorch_mos",
        "pytorch_color",
        "onnx_mos",
        "onnx_color",
        "mos_abs_diff",
        "color_abs_diff",
        "full_row_max_abs_diff",
        "full_row_mean_abs_diff",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Export export-safe ICAA17K DAT to ONNX and check ORT parity.")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise RuntimeError("--limit must be >= 1.")
    if args.batch_size != 1:
        raise RuntimeError("Static mobile export parity requires --batch_size 1.")
    if args.num_workers < 0:
        raise RuntimeError("--num_workers must be >= 0.")

    root = project_root()
    onnx, ort, torch, nn, Image, DataLoader, Dataset, transforms, DAT = import_dependencies(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_safe_onnx_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = output_dir / "icaa_dat_export_safe.onnx"
    report_path = output_dir / "onnx_export_report.json"
    predictions_path = output_dir / "onnx_parity_predictions.csv"
    log_path = output_dir / "onnx_export_log.txt"
    log_lines = []
    prediction_rows = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    checkpoint_path = root / CHECKPOINT_REL_PATH
    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    report = {
        "status": "failed",
        "output_dir": str(output_dir),
        "onnx_path": str(onnx_path),
        "checkpoint_path": str(checkpoint_path),
        "csv_path": str(csv_path),
        "image_root": str(image_root),
        "device": str(device),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "static_input_shape": [1, 3, 224, 224],
        "thresholds": {"max_abs_diff": MAX_ABS_TOL, "mean_abs_diff": MEAN_ABS_TOL},
        "model_source": "experiments/icaa_export_safe/export_safe_models",
        "model_config": MODEL_KWARGS,
        "export_attempts": [],
    }

    try:
        log(f"project root: {root}")
        log(f"export-safe model package: {root / 'experiments' / 'icaa_export_safe' / 'export_safe_models'}")
        log(f"checkpoint path: {checkpoint_path}")
        log(f"csv path: {csv_path}")
        log(f"image root: {image_root}")
        log(f"output dir: {output_dir}")
        log(f"device: {device}")
        log(f"limit: {args.limit}")
        log(f"batch_size: {args.batch_size}")

        checkpoint = load_checkpoint(torch, checkpoint_path)
        model = DAT(**MODEL_KWARGS)
        missing, unexpected, shape_mismatches = check_state_dict_compatibility(model, checkpoint)
        log(f"checkpoint key count: {len(checkpoint)}")
        log(f"patch_proj.0.weight shape: {shape_of(checkpoint.get('patch_proj.0.weight'))}")
        log(f"missing keys: {missing}")
        log(f"unexpected keys: {unexpected}")
        if shape_mismatches:
            raise RuntimeError(f"Checkpoint shape mismatches found: {shape_mismatches}")
        if missing or unexpected:
            raise RuntimeError("Missing or unexpected checkpoint keys found; refusing to load.")

        model.load_state_dict(checkpoint, strict=True)
        model.to(device)
        model.eval()
        wrapper_class = make_wrapper_class(nn)
        wrapper = wrapper_class(model).to(device)
        wrapper.eval()
        log("export-safe checkpoint strict load: ok")

        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        export_success = False
        opset_to_try = [args.opset]
        if args.opset == 17:
            opset_to_try.append(18)
        
        for opset in opset_to_try:
            try:
                log(f"ONNX export attempt opset {opset}: start")
                warnings_seen = export_onnx(torch, wrapper, dummy_input, onnx_path, opset)
                report["export_attempts"].append(
                    {
                        "opset": opset,
                        "succeeded": True,
                        "warnings": warnings_seen,
                    }
                )
                log(f"ONNX export attempt opset {opset}: succeeded")
                if warnings_seen:
                    log("ONNX export warnings:")
                    for message in warnings_seen:
                        log(f"  {message}")
                report["opset_version"] = opset
                export_success = True
                break
            except Exception as exc:
                error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                report["export_attempts"].append(
                    {
                        "opset": opset,
                        "succeeded": False,
                        "error": error,
                    }
                )
                log(f"ONNX export attempt opset {opset}: failed")
                log(f"  error: {exc}")

        if not export_success:
            raise RuntimeError(f"ONNX export failed for opset versions: {opset_to_try}")

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        log("onnx.checker.check_model: ok")
        report["onnx_checker"] = "passed"

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        providers = session.get_providers()
        report["onnxruntime"] = {
            "providers": providers,
            "input_name": input_name,
            "output_name": output_name,
        }
        log(f"ONNX Runtime providers: {providers}")
        log(f"ONNX input: {input_name}")
        log(f"ONNX output: {output_name}")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        samples, total_rows, missing_ids = read_samples(csv_path, image_root, args.limit)
        if not samples:
            raise RuntimeError("No evaluable images were found.")
        dataset_class = make_dataset_class(Dataset, Image)
        loader = DataLoader(
            dataset_class(samples, transform),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        log(f"csv rows read: {total_rows}")
        log(f"evaluable images: {len(samples)}")
        log(f"missing image rows skipped: {len(missing_ids)}")
        if missing_ids:
            log(f"first missing image IDs: {missing_ids[:10]}")

        all_diffs = []
        mos_diffs = []
        color_diffs = []
        with torch.no_grad():
            for images, targets, image_ids in loader:
                images = images.to(device)
                targets = targets.to(device=device, dtype=torch.float32)
                pytorch_output = wrapper(images)
                onnx_output_array = session.run([output_name], {input_name: images.detach().cpu().numpy()})[0]
                onnx_output = torch.from_numpy(onnx_output_array)
                pytorch_cpu = pytorch_output.detach().cpu()
                targets_cpu = targets.detach().cpu()
                if tuple(pytorch_cpu.shape) != tuple(onnx_output.shape):
                    raise RuntimeError(
                        f"Output shape mismatch: PyTorch={tuple(pytorch_cpu.shape)}, ONNX={tuple(onnx_output.shape)}"
                    )

                diff = (pytorch_cpu - onnx_output).abs()
                all_diffs.append(diff.reshape(-1))
                mos_diffs.append(diff[:, 0])
                color_diffs.append(diff[:, 1])

                row_diff = diff[0]
                prediction_rows.append(
                    {
                        "image_id": image_ids[0],
                        "target_mos": targets_cpu[0, 0].item(),
                        "target_color": targets_cpu[0, 1].item(),
                        "pytorch_mos": pytorch_cpu[0, 0].item(),
                        "pytorch_color": pytorch_cpu[0, 1].item(),
                        "onnx_mos": float(onnx_output[0, 0].item()),
                        "onnx_color": float(onnx_output[0, 1].item()),
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
                "csv_rows_read": total_rows,
                "evaluable_images": len(samples),
                "missing_image_rows_skipped": len(missing_ids),
                "first_missing_image_ids": missing_ids[:10],
                "checkpoint": {
                    "key_count": len(checkpoint),
                    "patch_proj_0_weight_shape": shape_of(checkpoint.get("patch_proj.0.weight")),
                    "missing_keys": missing,
                    "unexpected_keys": unexpected,
                    "shape_mismatches": shape_mismatches,
                    "strict_load": True,
                },
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
        log(f"parity pass: {passed}")
        if not passed:
            raise RuntimeError("ONNX parity thresholds failed.")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log("final status: failed")
        log(f"error: {exc}")
    finally:
        write_predictions(predictions_path, prediction_rows)
        report["onnx_export_report_json"] = str(report_path)
        report["onnx_parity_predictions_csv"] = str(predictions_path)
        report["onnx_export_log_txt"] = str(log_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"onnx model: {onnx_path}")
        log(f"onnx_export_report.json: {report_path}")
        log(f"onnx_parity_predictions.csv: {predictions_path}")
        log(f"onnx_export_log.txt: {log_path}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
