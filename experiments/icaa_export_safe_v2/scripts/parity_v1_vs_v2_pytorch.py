# ICAA17K export-safe v1과 GridSample-free v2 PyTorch 출력 동등성을 검증하는 스크립트
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
PREFERRED_MAX_ABS_TOL = 1e-5
ACCEPTABLE_MAX_ABS_TOL = 1e-4

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
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("Failed to import torch/PIL/torchvision dependencies.") from exc

    v1_dir = root / "experiments" / "icaa_export_safe" / "export_safe_models"
    v2_dir = root / "experiments" / "icaa_export_safe_v2" / "export_safe_models"
    v1_dat = import_dat_from_package_dir("icaa_export_safe_v1_models", v1_dir)
    v2_dat = import_dat_from_package_dir("icaa_export_safe_v2_models", v2_dir)
    return torch, Image, DataLoader, Dataset, transforms, v1_dat, v2_dat


def checkpoint_path(root: Path) -> Path:
    return root / CHECKPOINT_REL_PATH


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


def write_predictions(path: Path, rows) -> None:
    fieldnames = [
        "image_id",
        "target_mos",
        "target_color",
        "v1_mos",
        "v1_color",
        "v2_mos",
        "v2_color",
        "mos_abs_diff",
        "color_abs_diff",
        "full_row_max_abs_diff",
        "full_row_mean_abs_diff",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shape_of(value):
    return tuple(value.shape) if hasattr(value, "shape") else None


def parse_args():
    parser = argparse.ArgumentParser(description="Compare export-safe ICAA DAT v1 and GridSample-free v2 in PyTorch.")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise RuntimeError("--limit must be >= 1.")
    if args.batch_size < 1:
        raise RuntimeError("--batch_size must be >= 1.")

    root = project_root()
    torch, Image, DataLoader, Dataset, transforms, V1DAT, V2DAT = import_dependencies(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_safe_v2_pytorch_parity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "parity_report.json"
    predictions_path = output_dir / "parity_predictions.csv"
    log_path = output_dir / "parity_log.txt"
    log_lines = []
    prediction_rows = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    report = {
        "status": "failed",
        "output_dir": str(output_dir),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "device": str(device),
        "thresholds": {
            "preferred_max_abs_diff": PREFERRED_MAX_ABS_TOL,
            "acceptable_max_abs_diff": ACCEPTABLE_MAX_ABS_TOL,
        },
        "model_config": MODEL_KWARGS,
    }

    try:
        ckpt_path = checkpoint_path(root)
        checkpoint = load_checkpoint(torch, ckpt_path)
        v1_model = V1DAT(**MODEL_KWARGS)
        v2_model = V2DAT(**MODEL_KWARGS)
        v1_result = v1_model.load_state_dict(checkpoint, strict=True)
        v2_result = v2_model.load_state_dict(checkpoint, strict=True)
        v1_model.to(device).eval()
        v2_model.to(device).eval()

        csv_path = root / CSV_REL_PATH
        image_root = root / IMAGE_ROOT_REL_PATH
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
        loader = DataLoader(dataset_class(samples, transform), batch_size=args.batch_size, shuffle=False, num_workers=0)

        log(f"checkpoint path: {ckpt_path}")
        log(f"checkpoint key count: {len(checkpoint)}")
        log(f"patch_proj.0.weight shape: {shape_of(checkpoint.get('patch_proj.0.weight'))}")
        log(f"v1 strict load result: {v1_result}")
        log(f"v2 strict load result: {v2_result}")
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
                v1_scores = v1_model(images)[0]
                v2_scores = v2_model(images)[0]
                if tuple(v1_scores.shape) != tuple(v2_scores.shape):
                    raise RuntimeError(f"Shape mismatch: v1={tuple(v1_scores.shape)}, v2={tuple(v2_scores.shape)}")

                v1_cpu = v1_scores.detach().cpu()
                v2_cpu = v2_scores.detach().cpu()
                targets_cpu = targets.detach().cpu()
                diff = (v1_cpu - v2_cpu).abs()
                all_diffs.append(diff.reshape(-1))
                mos_diffs.append(diff[:, 0])
                color_diffs.append(diff[:, 1])

                for idx, image_id in enumerate(image_ids):
                    row_diff = diff[idx]
                    prediction_rows.append(
                        {
                            "image_id": image_id,
                            "target_mos": targets_cpu[idx, 0].item(),
                            "target_color": targets_cpu[idx, 1].item(),
                            "v1_mos": v1_cpu[idx, 0].item(),
                            "v1_color": v1_cpu[idx, 1].item(),
                            "v2_mos": v2_cpu[idx, 0].item(),
                            "v2_color": v2_cpu[idx, 1].item(),
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
        preferred_pass = metrics["full_tensor_max_abs_diff"] <= PREFERRED_MAX_ABS_TOL
        acceptable_pass = metrics["full_tensor_max_abs_diff"] <= ACCEPTABLE_MAX_ABS_TOL
        report.update(
            {
                "status": "passed" if acceptable_pass else "failed",
                "pass_level": "preferred" if preferred_pass else "acceptable" if acceptable_pass else "failed",
                "checkpoint_path": str(ckpt_path),
                "csv_rows_read": total_rows,
                "evaluable_images": len(samples),
                "missing_image_rows_skipped": len(missing_ids),
                "first_missing_image_ids": missing_ids[:10],
                "metrics": metrics,
            }
        )
        log(f"full tensor max_abs_diff: {metrics['full_tensor_max_abs_diff']:.10f}")
        log(f"full tensor mean_abs_diff: {metrics['full_tensor_mean_abs_diff']:.10f}")
        log(f"MOS max_abs_diff: {metrics['mos_max_abs_diff']:.10f}")
        log(f"MOS mean_abs_diff: {metrics['mos_mean_abs_diff']:.10f}")
        log(f"color max_abs_diff: {metrics['color_max_abs_diff']:.10f}")
        log(f"color mean_abs_diff: {metrics['color_mean_abs_diff']:.10f}")
        log(f"pass level: {report['pass_level']}")
        if not acceptable_pass:
            raise RuntimeError("v1-v2 PyTorch parity failed acceptable threshold.")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(f"error: {exc}")
    finally:
        write_predictions(predictions_path, prediction_rows)
        report["parity_report_json"] = str(report_path)
        report["parity_predictions_csv"] = str(predictions_path)
        report["parity_log_txt"] = str(log_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"parity_report.json: {report_path}")
        log(f"parity_predictions.csv: {predictions_path}")
        log(f"parity_log.txt: {log_path}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
