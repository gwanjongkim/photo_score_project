# ICAA17K 공식 DAT와 TorchScript 추적 모델의 실제 이미지 출력 동등성을 확인하는 스크립트
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import sys
import traceback

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
DEFAULT_TRACED_REL_PATH = "outputs/icaa_export_torchscript_20260514_150436/icaa_dat_traced.pt"
MAX_ABS_TOL = 1e-4
MEAN_ABS_TOL = 1e-5

OFFICIAL_DAT_CONFIG = {
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
    return Path(__file__).resolve().parents[1]


def add_official_repo_to_path(root: Path) -> Path:
    official_repo = root / "external" / "icaa_official_repo" / "ICAA17K_code"
    if not official_repo.is_dir():
        raise RuntimeError(f"Official ICAA17K repo path not found: {official_repo}")
    sys.path.insert(0, str(official_repo))
    return official_repo


def import_dependencies():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("Failed to import torch, PIL, DataLoader, or torchvision transforms.") from exc

    try:
        from models.dat import DAT
    except Exception as exc:
        raise RuntimeError(
            "Failed to import DAT from external/icaa_official_repo/ICAA17K_code/models/dat.py. "
            "Check that official repo dependencies such as timm and einops are installed."
        ) from exc

    return torch, nn, Image, DataLoader, Dataset, transforms, DAT


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
    class DATOutputWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            output, positions, references = self.model(x)
            return output

    return DATOutputWrapper


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
            try:
                target_mos = float(row["MOS"]) / 10.0
                target_color = float(row["color"]) / 10.0
            except ValueError as exc:
                raise RuntimeError(f"Failed to parse MOS/color labels for image ID {image_id}.") from exc

            samples.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "target_mos": target_mos,
                    "target_color": target_color,
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
        "pytorch_mos",
        "pytorch_color",
        "torchscript_mos",
        "torchscript_color",
        "mos_abs_diff",
        "color_abs_diff",
        "full_row_max_abs_diff",
        "full_row_mean_abs_diff",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_traced_path(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise RuntimeError(f"Traced TorchScript model not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Real-image parity check for official ICAA17K DAT and traced TorchScript.")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--traced_model", default=DEFAULT_TRACED_REL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise RuntimeError("--limit must be >= 1.")
    if args.batch_size < 1:
        raise RuntimeError("--batch_size must be >= 1.")
    if args.num_workers < 0:
        raise RuntimeError("--num_workers must be >= 0.")

    root = project_root()
    official_repo = add_official_repo_to_path(root)
    torch, nn, Image, DataLoader, Dataset, transforms, DAT = import_dependencies()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_torchscript_parity_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "parity_report.json"
    predictions_path = output_dir / "parity_predictions.csv"
    log_path = output_dir / "parity_log.txt"
    log_lines = []
    prediction_rows = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    traced_path = resolve_traced_path(root, args.traced_model)
    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    checkpoint_path = root / CHECKPOINT_REL_PATH

    report = {
        "status": "failed",
        "output_dir": str(output_dir),
        "official_repo": str(official_repo),
        "checkpoint_path": str(checkpoint_path),
        "traced_model_path": str(traced_path),
        "csv_path": str(csv_path),
        "image_root": str(image_root),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
        "thresholds": {"max_abs_diff": MAX_ABS_TOL, "mean_abs_diff": MEAN_ABS_TOL},
        "model_config": OFFICIAL_DAT_CONFIG,
    }

    try:
        checkpoint = load_checkpoint(torch, checkpoint_path)
        original_model = DAT(**OFFICIAL_DAT_CONFIG)
        missing, unexpected, shape_mismatches = check_state_dict_compatibility(original_model, checkpoint)
        log(f"official repo path: {official_repo}")
        log(f"checkpoint path: {checkpoint_path}")
        log(f"traced model path: {traced_path}")
        log(f"output dir: {output_dir}")
        log(f"device: {device}")
        log(f"checkpoint key count: {len(checkpoint)}")
        log(f"patch_proj.0.weight shape: {shape_of(checkpoint.get('patch_proj.0.weight'))}")
        log(f"missing keys: {missing}")
        log(f"unexpected keys: {unexpected}")
        if shape_mismatches:
            raise RuntimeError(f"Checkpoint shape mismatches found: {shape_mismatches}")
        if missing or unexpected:
            raise RuntimeError("Missing or unexpected checkpoint keys found; refusing to load.")
        original_model.load_state_dict(checkpoint, strict=True)
        log("original PyTorch checkpoint strict load: ok")

        wrapper_class = make_wrapper_class(nn)
        original_wrapper = wrapper_class(original_model).to(device)
        original_wrapper.eval()
        traced_model = torch.jit.load(str(traced_path), map_location=device)
        traced_model.eval()
        log("traced TorchScript load: ok")

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
        dataset = dataset_class(samples, transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
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
                pytorch_output = original_wrapper(images)
                torchscript_output = traced_model(images)
                if tuple(pytorch_output.shape) != tuple(torchscript_output.shape):
                    raise RuntimeError(
                        f"Output shape mismatch: PyTorch={tuple(pytorch_output.shape)}, "
                        f"TorchScript={tuple(torchscript_output.shape)}"
                    )
                if pytorch_output.ndim != 2 or pytorch_output.shape[1] != 2:
                    raise RuntimeError(f"Expected output shape (batch, 2), got {tuple(pytorch_output.shape)}.")

                diff = (pytorch_output - torchscript_output).abs().detach().cpu()
                pytorch_cpu = pytorch_output.detach().cpu()
                torchscript_cpu = torchscript_output.detach().cpu()
                targets_cpu = targets.detach().cpu()
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
                            "pytorch_mos": pytorch_cpu[idx, 0].item(),
                            "pytorch_color": pytorch_cpu[idx, 1].item(),
                            "torchscript_mos": torchscript_cpu[idx, 0].item(),
                            "torchscript_color": torchscript_cpu[idx, 1].item(),
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
                "metrics": metrics,
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
            raise RuntimeError("TorchScript parity thresholds failed.")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        log(f"parity pass: False")
        log(f"error: {exc}")
    finally:
        write_predictions(predictions_path, prediction_rows)
        report["predictions_csv"] = str(predictions_path)
        report["parity_report_json"] = str(report_path)
        report["parity_log_txt"] = str(log_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"parity_predictions.csv: {predictions_path}")
        log(f"parity_report.json: {report_path}")
        log(f"parity_log.txt: {log_path}")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
