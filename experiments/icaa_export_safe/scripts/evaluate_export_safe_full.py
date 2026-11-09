# ICAA17K export-safe DAT 모델을 전체 테스트셋에서 평가하는 스크립트
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import sys

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"

OFFICIAL_REFERENCE = {
    "accuracy_at_0_5": 0.963513,
    "srcc": 0.880682,
    "plcc": 0.898182,
}

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
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
        from export_safe_models.dat import DAT
    except Exception as exc:
        raise RuntimeError("Failed to import torch/PIL/torchvision or export-safe DAT.") from exc

    return torch, Image, DataLoader, Dataset, transforms, DAT


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


def read_samples(csv_path: Path, image_root: Path, limit: int | None):
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
            if limit is not None and len(samples) >= limit:
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


def rankdata(torch, values):
    values = values.double()
    order = values.argsort()
    sorted_values = values[order]
    ranks = values.new_empty(values.numel())
    i = 0
    while i < values.numel():
        j = i + 1
        while j < values.numel() and sorted_values[j].item() == sorted_values[i].item():
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def pearson_corr(values, targets) -> float:
    values = values.double()
    targets = targets.double()
    values = values - values.mean()
    targets = targets - targets.mean()
    denom = values.norm() * targets.norm()
    if denom.item() == 0:
        return float("nan")
    return ((values * targets).sum() / denom).item()


def spearman_corr(torch, values, targets) -> float:
    return pearson_corr(rankdata(torch, values), rankdata(torch, targets))


def compute_metrics(torch, predictions, targets, include_accuracy: bool):
    diff = predictions - targets
    metrics = {
        "srcc": spearman_corr(torch, predictions, targets),
        "plcc": pearson_corr(predictions, targets),
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff * diff).mean()).item(),
    }
    if include_accuracy:
        metrics["accuracy_at_0_5"] = ((predictions >= 0.5) == (targets >= 0.5)).float().mean().item()
    return metrics


def write_predictions(path: Path, rows) -> None:
    fieldnames = [
        "image_id",
        "pred_mos",
        "pred_color",
        "target_mos",
        "target_color",
        "color_pred_binary",
        "color_target_binary",
        "mos_abs_error",
        "color_abs_error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate export-safe ICAA17K DAT on the ICAA17K test set.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke limit; omit for full test set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise RuntimeError("--batch_size must be >= 1.")
    if args.num_workers < 0:
        raise RuntimeError("--num_workers must be >= 0.")
    if args.limit is not None and args.limit < 1:
        raise RuntimeError("--limit must be >= 1 when provided.")

    root = project_root()
    torch, Image, DataLoader, Dataset, transforms, DAT = import_dependencies(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_export_safe_full_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    log_path = output_dir / "eval_log.txt"
    log_lines = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    checkpoint_path = root / CHECKPOINT_REL_PATH
    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    if not csv_path.is_file():
        raise RuntimeError(f"ICAA17K test CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise RuntimeError(f"ICAA17K image root not found: {image_root}")

    log(f"project root: {root}")
    log(f"export-safe model package: {root / 'experiments' / 'icaa_export_safe' / 'export_safe_models'}")
    log(f"checkpoint path: {checkpoint_path}")
    log(f"csv path: {csv_path}")
    log(f"image root: {image_root}")
    log(f"output dir: {output_dir}")
    log(f"device: {device}")
    log(f"batch_size: {args.batch_size}")
    log(f"num_workers: {args.num_workers}")
    log(f"limit: {args.limit}")

    checkpoint = load_checkpoint(torch, checkpoint_path)
    if not hasattr(checkpoint, "keys"):
        raise RuntimeError(f"Expected checkpoint to be state_dict-like, got {type(checkpoint).__name__}.")

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
    log("export-safe checkpoint strict load: ok")

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
    log(f"csv rows read: {total_rows}")
    log(f"evaluable images: {len(samples)}")
    log(f"missing image rows skipped: {len(missing_ids)}")
    if missing_ids:
        log(f"first missing image IDs: {missing_ids[:10]}")

    dataset_class = make_dataset_class(Dataset, Image)
    loader = DataLoader(
        dataset_class(samples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    prediction_rows = []
    pred_mos_parts = []
    pred_color_parts = []
    target_mos_parts = []
    target_color_parts = []

    with torch.no_grad():
        for images, targets, image_ids in loader:
            images = images.to(device)
            targets = targets.to(device=device, dtype=torch.float32)
            outputs = model(images)
            if not isinstance(outputs, tuple) or len(outputs) != 3:
                raise RuntimeError(f"Expected DAT forward to return a 3-item tuple, got {type(outputs).__name__}.")

            x = outputs[0]
            if x.ndim != 2 or x.shape[1] != 2:
                raise RuntimeError(f"Expected output shape (batch, 2), got {tuple(x.shape)}.")

            pred_mos = x[:, 0].detach().cpu()
            pred_color = x[:, 1].detach().cpu()
            target_mos = targets[:, 0].detach().cpu()
            target_color = targets[:, 1].detach().cpu()
            pred_mos_parts.append(pred_mos)
            pred_color_parts.append(pred_color)
            target_mos_parts.append(target_mos)
            target_color_parts.append(target_color)

            for idx, image_id in enumerate(image_ids):
                pred_mos_value = pred_mos[idx].item()
                pred_color_value = pred_color[idx].item()
                target_mos_value = target_mos[idx].item()
                target_color_value = target_color[idx].item()
                prediction_rows.append(
                    {
                        "image_id": image_id,
                        "pred_mos": pred_mos_value,
                        "pred_color": pred_color_value,
                        "target_mos": target_mos_value,
                        "target_color": target_color_value,
                        "color_pred_binary": int(pred_color_value >= 0.5),
                        "color_target_binary": int(target_color_value >= 0.5),
                        "mos_abs_error": abs(pred_mos_value - target_mos_value),
                        "color_abs_error": abs(pred_color_value - target_color_value),
                    }
                )

    pred_mos = torch.cat(pred_mos_parts)
    pred_color = torch.cat(pred_color_parts)
    target_mos = torch.cat(target_mos_parts)
    target_color = torch.cat(target_color_parts)
    color_metrics = compute_metrics(torch, pred_color, target_color, include_accuracy=True)
    mos_metrics = compute_metrics(torch, pred_mos, target_mos, include_accuracy=False)
    color_reference_delta = {
        "accuracy_at_0_5": color_metrics["accuracy_at_0_5"] - OFFICIAL_REFERENCE["accuracy_at_0_5"],
        "srcc": color_metrics["srcc"] - OFFICIAL_REFERENCE["srcc"],
        "plcc": color_metrics["plcc"] - OFFICIAL_REFERENCE["plcc"],
    }

    metrics = {
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "csv_path": str(csv_path),
        "image_root": str(image_root),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "csv_rows_read": total_rows,
        "evaluable_images": len(samples),
        "missing_image_rows_skipped": len(missing_ids),
        "first_missing_image_ids": missing_ids[:10],
        "model_source": "experiments/icaa_export_safe/export_safe_models",
        "model_config": MODEL_KWARGS,
        "checkpoint": {
            "key_count": len(checkpoint),
            "patch_proj_0_weight_shape": shape_of(checkpoint.get("patch_proj.0.weight")),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "shape_mismatches": shape_mismatches,
            "strict_load": True,
        },
        "color_metrics": color_metrics,
        "mos_reference_metrics": mos_metrics,
        "official_pytorch_reference_color_metrics": OFFICIAL_REFERENCE,
        "color_reference_delta": color_reference_delta,
    }

    write_predictions(predictions_path, prediction_rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"predictions.csv: {predictions_path}")
    log(f"metrics.json: {metrics_path}")
    log(f"eval_log.txt: {log_path}")
    log("color metrics:")
    log(f"  Accuracy@0.5: {color_metrics['accuracy_at_0_5']:.6f} vs official PyTorch {OFFICIAL_REFERENCE['accuracy_at_0_5']:.6f}")
    log(f"  SRCC: {color_metrics['srcc']:.6f} vs official PyTorch {OFFICIAL_REFERENCE['srcc']:.6f}")
    log(f"  PLCC: {color_metrics['plcc']:.6f} vs official PyTorch {OFFICIAL_REFERENCE['plcc']:.6f}")
    log(f"  MAE: {color_metrics['mae']:.6f}")
    log(f"  RMSE: {color_metrics['rmse']:.6f}")
    log("MOS reference metrics:")
    log(f"  SRCC: {mos_metrics['srcc']:.6f}")
    log(f"  PLCC: {mos_metrics['plcc']:.6f}")
    log(f"  MAE: {mos_metrics['mae']:.6f}")
    log(f"  RMSE: {mos_metrics['rmse']:.6f}")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
