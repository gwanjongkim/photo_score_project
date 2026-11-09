# ICAA17K GridSample-free v2 모델의 ONNX 내보내기와 ORT 동등성 검증 스크립트
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import csv
import importlib
import importlib.util
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
        import onnx
        import onnxruntime as ort
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("Failed to import ONNX/ORT, torch/PIL, or torchvision dependencies.") from exc

    v2_dir = root / "experiments" / "icaa_export_safe_v2" / "export_safe_models"
    v2_dat = import_dat_from_package_dir("icaa_export_safe_v2_models_onnx", v2_dir)
    return onnx, ort, torch, nn, Image, DataLoader, Dataset, transforms, v2_dat


def load_checkpoint(torch, path: Path):
    if not path.is_file():
        raise RuntimeError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def export_onnx(torch, wrapper, dummy_input, path: Path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(path),
            input_names=["image"],
            output_names=["scores"],
            dynamic_axes=None,
            opset_version=17,
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
    parser = argparse.ArgumentParser(description="Export GridSample-free ICAA DAT v2 to ONNX and verify ORT parity.")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise RuntimeError("--limit must be >= 1.")
    if args.batch_size != 1:
        raise RuntimeError("Static single-image ONNX export requires --batch_size 1.")

    root = project_root()
    onnx, ort, torch, nn, Image, DataLoader, Dataset, transforms, DAT = import_dependencies(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_safe_v2_onnx_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    onnx_path = output_dir / "icaa_dat_export_safe_v2.onnx"
    op_hist_path = output_dir / "onnx_op_histogram.json"
    report_path = output_dir / "onnx_parity_report.json"
    predictions_path = output_dir / "onnx_parity_predictions.csv"
    log_path = output_dir / "onnx_export_log.txt"
    log_lines = []
    prediction_rows = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    report = {
        "status": "failed",
        "output_dir": str(output_dir),
        "onnx_path": str(onnx_path),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": str(device),
        "thresholds": {"max_abs_diff": MAX_ABS_TOL, "mean_abs_diff": MEAN_ABS_TOL},
        "model_config": MODEL_KWARGS,
    }

    try:
        checkpoint_path = root / CHECKPOINT_REL_PATH
        checkpoint = load_checkpoint(torch, checkpoint_path)
        model = DAT(**MODEL_KWARGS)
        load_result = model.load_state_dict(checkpoint, strict=True)
        model.to(device).eval()
        wrapper_class = make_wrapper_class(nn)
        wrapper = wrapper_class(model).to(device).eval()
        log(f"checkpoint path: {checkpoint_path}")
        log(f"checkpoint key count: {len(checkpoint)}")
        log(f"strict load result: {load_result}")

        dummy_input = torch.randn(1, 3, 224, 224, device=device)
        export_warnings = export_onnx(torch, wrapper, dummy_input, onnx_path)
        log(f"ONNX export opset 17: ok")
        if export_warnings:
            log("ONNX export warnings:")
            for warning in export_warnings:
                log(f"  {warning}")

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        op_histogram = dict(sorted(Counter(node.op_type for node in onnx_model.graph.node).items()))
        grid_sample_count = op_histogram.get("GridSample", 0)
        op_hist_path.write_text(json.dumps(op_histogram, indent=2), encoding="utf-8")
        log("onnx.checker.check_model: ok")
        log(f"GridSample op count: {grid_sample_count}")
        if grid_sample_count != 0:
            raise RuntimeError(f"Expected GridSample count 0, got {grid_sample_count}.")

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        log(f"ONNX Runtime providers: {session.get_providers()}")
        log(f"ONNX input: {input_name}")
        log(f"ONNX output: {output_name}")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        csv_path = root / CSV_REL_PATH
        image_root = root / IMAGE_ROOT_REL_PATH
        samples, total_rows, missing_ids = read_samples(csv_path, image_root, args.limit)
        if not samples:
            raise RuntimeError("No evaluable images were found.")
        dataset_class = make_dataset_class(Dataset, Image)
        loader = DataLoader(dataset_class(samples, transform), batch_size=1, shuffle=False, num_workers=0)
        log(f"csv rows read: {total_rows}")
        log(f"evaluable images: {len(samples)}")
        log(f"missing image rows skipped: {len(missing_ids)}")

        all_diffs = []
        mos_diffs = []
        color_diffs = []
        with torch.no_grad():
            for images, targets, image_ids in loader:
                images = images.to(device)
                targets = targets.to(device=device, dtype=torch.float32)
                pytorch_scores = wrapper(images).detach().cpu()
                onnx_scores = torch.from_numpy(session.run([output_name], {input_name: images.detach().cpu().numpy()})[0])
                if tuple(pytorch_scores.shape) != tuple(onnx_scores.shape):
                    raise RuntimeError(f"Shape mismatch: PyTorch={tuple(pytorch_scores.shape)}, ONNX={tuple(onnx_scores.shape)}")
                diff = (pytorch_scores - onnx_scores).abs()
                targets_cpu = targets.detach().cpu()
                all_diffs.append(diff.reshape(-1))
                mos_diffs.append(diff[:, 0])
                color_diffs.append(diff[:, 1])
                row_diff = diff[0]
                prediction_rows.append(
                    {
                        "image_id": image_ids[0],
                        "target_mos": targets_cpu[0, 0].item(),
                        "target_color": targets_cpu[0, 1].item(),
                        "pytorch_mos": pytorch_scores[0, 0].item(),
                        "pytorch_color": pytorch_scores[0, 1].item(),
                        "onnx_mos": onnx_scores[0, 0].item(),
                        "onnx_color": onnx_scores[0, 1].item(),
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
                "checkpoint_path": str(checkpoint_path),
                "opset_version": 17,
                "export_warnings": export_warnings,
                "onnx_checker": "passed",
                "onnx_op_histogram_json": str(op_hist_path),
                "onnx_op_histogram": op_histogram,
                "grid_sample_count": grid_sample_count,
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
        log(f"ONNX parity pass: {passed}")
        if not passed:
            raise RuntimeError("ONNX Runtime parity failed thresholds.")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(f"error: {exc}")
    finally:
        write_predictions(predictions_path, prediction_rows)
        report["onnx_parity_report_json"] = str(report_path)
        report["onnx_parity_predictions_csv"] = str(predictions_path)
        report["onnx_export_log_txt"] = str(log_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"onnx model: {onnx_path}")
        log(f"onnx_op_histogram.json: {op_hist_path}")
        log(f"onnx_parity_report.json: {report_path}")
        log(f"onnx_parity_predictions.csv: {predictions_path}")
        log(f"onnx_export_log.txt: {log_path}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
