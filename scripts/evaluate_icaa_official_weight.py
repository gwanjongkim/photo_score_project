# ICAA17K 공식 DAT 체크포인트를 테스트셋에서 평가하는 스크립트
from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import math
import sys

sys.dont_write_bytecode = True

CHECKPOINT_REL_PATH = "weights/icaa_official/e_30_ICAA17K_multi_tacc0.9622_srcc0.8811_tlcc0.8981.pth"
IMAGE_ROOT_REL_PATH = "data/raw/icaa17k"
CSV_REL_PATH = "external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv"

FILENAME_TARGETS = {
    "color_accuracy": 0.9622,
    "color_srcc": 0.8811,
    "color_plcc": 0.8981,
}

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

    return torch, Image, DataLoader, Dataset, transforms, DAT


def resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


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


def regression_metrics(torch, predictions, targets, include_accuracy: bool = False):
    diff = predictions - targets
    result = {
        "srcc": spearman_corr(torch, predictions, targets),
        "plcc": pearson_corr(predictions, targets),
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt((diff * diff).mean()).item(),
    }
    if include_accuracy:
        result["accuracy_at_0_5"] = ((predictions >= 0.5) == (targets >= 0.5)).float().mean().item()
    return result


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


def write_predictions_csv(path: Path, rows) -> None:
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
    parser = argparse.ArgumentParser(description="Evaluate official ICAA17K DAT checkpoint on the ICAA17K test CSV.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
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
    official_repo = add_official_repo_to_path(root)
    torch, Image, DataLoader, Dataset, transforms, DAT = import_dependencies()
    device = resolve_device(torch, args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "outputs" / f"icaa_official_eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_lines = []

    def log(message: str) -> None:
        print(message)
        log_lines.append(message)

    csv_path = root / CSV_REL_PATH
    image_root = root / IMAGE_ROOT_REL_PATH
    checkpoint_path = root / CHECKPOINT_REL_PATH
    if not csv_path.is_file():
        raise RuntimeError(f"ICAA17K test CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise RuntimeError(f"ICAA17K image root not found: {image_root}")

    log(f"project root: {root}")
    log(f"official repo path: {official_repo}")
    log(f"csv path: {csv_path}")
    log(f"image root: {image_root}")
    log(f"checkpoint path: {checkpoint_path}")
    log(f"output dir: {output_dir}")
    log(f"device: {device}")
    log(f"batch_size: {args.batch_size}")
    log(f"num_workers: {args.num_workers}")
    log(f"limit: {args.limit}")

    checkpoint = load_checkpoint(torch, checkpoint_path)
    if not hasattr(checkpoint, "keys"):
        raise RuntimeError(f"Expected checkpoint to be a state_dict-like object, got {type(checkpoint).__name__}.")
    log(f"checkpoint key count: {len(checkpoint)}")
    log(f"patch_proj.0.weight shape: {shape_of(checkpoint.get('patch_proj.0.weight'))}")

    model = DAT(**OFFICIAL_DAT_CONFIG)
    missing, unexpected, shape_mismatches = check_state_dict_compatibility(model, checkpoint)
    log(f"missing keys: {missing}")
    log(f"unexpected keys: {unexpected}")
    if shape_mismatches:
        log(f"shape mismatches: {shape_mismatches}")
        raise RuntimeError("Checkpoint shape mismatches found; refusing to load.")
    if missing or unexpected:
        raise RuntimeError("Missing or unexpected checkpoint keys found; refusing to load.")

    load_result = model.load_state_dict(checkpoint, strict=True)
    log(f"strict load result: {load_result}")
    log("checkpoint loaded successfully: yes")

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
    dataset = dataset_class(samples, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = model.to(device)
    model.eval()
    prediction_rows = []
    all_pred_mos = []
    all_pred_color = []
    all_target_mos = []
    all_target_color = []

    try:
        with torch.no_grad():
            for images, targets, image_ids in loader:
                images = images.to(device)
                targets = targets.to(device=device, dtype=torch.float32)
                outputs = model(images)
                if not isinstance(outputs, tuple) or len(outputs) != 3:
                    raise RuntimeError(f"Expected DAT forward to return a 3-item tuple, got {type(outputs).__name__}.")
                x, positions, references = outputs
                if x.ndim != 2 or x.shape[1] != 2:
                    raise RuntimeError(f"Expected x shape (batch, 2), got {tuple(x.shape)}.")

                pred_mos = x[:, 0].detach().cpu()
                pred_color = x[:, 1].detach().cpu()
                target_mos = targets[:, 0].detach().cpu()
                target_color = targets[:, 1].detach().cpu()

                all_pred_mos.append(pred_mos)
                all_pred_color.append(pred_color)
                all_target_mos.append(target_mos)
                all_target_color.append(target_color)

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
    except RuntimeError as exc:
        if device.type == "cuda" and "out of memory" in str(exc).lower():
            raise RuntimeError("CUDA out of memory during evaluation. Rerun with --batch_size 4 or --batch_size 2.") from exc
        raise

    pred_mos_tensor = torch.cat(all_pred_mos)
    pred_color_tensor = torch.cat(all_pred_color)
    target_mos_tensor = torch.cat(all_target_mos)
    target_color_tensor = torch.cat(all_target_color)

    color_metrics = regression_metrics(torch, pred_color_tensor, target_color_tensor, include_accuracy=True)
    mos_metrics = regression_metrics(torch, pred_mos_tensor, target_mos_tensor, include_accuracy=False)

    metrics = {
        "output_dir": str(output_dir),
        "csv_path": str(csv_path),
        "image_root": str(image_root),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "csv_rows_read": total_rows,
        "evaluable_images": len(samples),
        "missing_image_rows_skipped": len(missing_ids),
        "first_missing_image_ids": missing_ids[:10],
        "model_config": OFFICIAL_DAT_CONFIG,
        "checkpoint": {
            "key_count": len(checkpoint),
            "patch_proj_0_weight_shape": shape_of(checkpoint.get("patch_proj.0.weight")),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "strict_load": True,
        },
        "color_metrics": color_metrics,
        "mos_metrics": mos_metrics,
        "filename_targets": FILENAME_TARGETS,
        "filename_target_delta": {
            "color_accuracy": color_metrics["accuracy_at_0_5"] - FILENAME_TARGETS["color_accuracy"],
            "color_srcc": color_metrics["srcc"] - FILENAME_TARGETS["color_srcc"],
            "color_plcc": color_metrics["plcc"] - FILENAME_TARGETS["color_plcc"],
        },
    }

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    log_path = output_dir / "eval_log.txt"
    write_predictions_csv(predictions_path, prediction_rows)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log(f"predictions.csv: {predictions_path}")
    log(f"metrics.json: {metrics_path}")
    log(f"eval_log.txt: {log_path}")
    log("final color summary:")
    log(f"  measured Accuracy@0.5: {color_metrics['accuracy_at_0_5']:.6f} vs filename tacc0.9622")
    log(f"  measured SRCC: {color_metrics['srcc']:.6f} vs filename srcc0.8811")
    log(f"  measured PLCC: {color_metrics['plcc']:.6f} vs filename tlcc0.8981")
    log(f"  measured MAE: {color_metrics['mae']:.6f}")
    log(f"  measured RMSE: {color_metrics['rmse']:.6f}")
    log("MOS reference summary:")
    log(f"  SRCC: {mos_metrics['srcc']:.6f}")
    log(f"  PLCC: {mos_metrics['plcc']:.6f}")
    log(f"  MAE: {mos_metrics['mae']:.6f}")
    log(f"  RMSE: {mos_metrics['rmse']:.6f}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
