# ICAA17K 공식 DAT 모델을 실제 이미지 배치로 확인하는 스모크 테스트
from pathlib import Path
import argparse
import csv
import sys

sys.dont_write_bytecode = True

DAT_SMOKE_KWARGS = {
    "num_classes": 1,
    # Smoke-only DAT config: keeps the official 1024-wide head input, with one block per stage.
    "dim_stem": 128,
    "dims": [128, 256, 512, 1024],
    "depths": [1, 1, 1, 1],
    "stage_spec": [["L"], ["L"], ["D"], ["D"]],
    "heads": [4, 8, 16, 32],
    "groups": [-1, -1, 4, 8],
    "use_pes": [False, False, True, True],
    "strides": [-1, -1, 1, 1],
    "sr_ratios": [-1, -1, -1, -1],
    "offset_range_factor": [-1, -1, 2, 2],
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
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("Failed to import torch, PIL, or torchvision transforms.") from exc

    try:
        from models.dat import DAT
    except Exception as exc:
        raise RuntimeError(
            "Failed to import DAT from external/icaa_official_repo/ICAA17K_code/models/dat.py. "
            "Check that official repo dependencies such as timm and einops are installed."
        ) from exc

    return torch, Image, transforms, DAT


def rankdata(values):
    order = values.argsort()
    sorted_values = values[order]
    ranks = values.new_empty(values.numel(), dtype=values.dtype)
    i = 0
    while i < values.numel():
        j = i + 1
        while j < values.numel() and sorted_values[j].item() == sorted_values[i].item():
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def pearson_corr(a, b) -> float:
    a = a.double()
    b = b.double()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float("nan")
    return ((a * b).sum() / denom).item()


def spearman_corr(a, b) -> float:
    return pearson_corr(rankdata(a.double()), rankdata(b.double()))


def load_rows(csv_path: Path, image_root: Path, count: int, transform, Image):
    images = []
    targets = []
    image_ids = []
    skipped = 0

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"ID", "MOS", "color"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise RuntimeError(f"CSV is missing required fields {sorted(required_fields)}: {csv_path}")

        for row in reader:
            image_id = row["ID"]
            image_path = image_root / image_id
            if not image_path.is_file():
                skipped += 1
                continue

            try:
                mos = float(row["MOS"]) / 10.0
                color = float(row["color"]) / 10.0
            except ValueError as exc:
                raise RuntimeError(f"Failed to parse MOS/color labels for image ID {image_id}.") from exc

            try:
                with Image.open(image_path) as image:
                    images.append(transform(image.convert("RGB")))
            except Exception:
                skipped += 1
                continue

            targets.append([mos, color])
            image_ids.append(image_id)
            if len(images) == count:
                break

    if len(images) != count:
        raise RuntimeError(f"Loaded {len(images)} valid images, expected {count}. Skipped rows: {skipped}.")

    return image_ids, images, targets, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test official ICAA17K DAT on local real images.")
    parser.add_argument("--num-images", type=int, default=16, help="Number of valid rows/images to load.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    official_repo = add_official_repo_to_path(root)
    torch, Image, transforms, DAT = import_dependencies()
    print("DAT import: ok")

    csv_path = official_repo / "dataset" / "ICAA17K" / "1test.csv"
    image_root = root / "data" / "raw" / "icaa17k"
    if not csv_path.is_file():
        raise RuntimeError(f"Official CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise RuntimeError(f"ICAA17K image root not found: {image_root}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_ids, images, targets, skipped = load_rows(csv_path, image_root, args.num_images, transform, Image)
    print("label parsing: ok")
    print(f"loaded images: {len(images)}")
    print(f"skipped rows before loaded batch: {skipped}")
    print(f"first 3 image IDs: {image_ids[:3]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = torch.stack(images).to(device)
    target = torch.tensor(targets, dtype=torch.float32, device=device)
    print(f"device: {device}")
    print(f"input batch shape: {tuple(batch.shape)}")
    print(f"target shape: {tuple(target.shape)}")

    try:
        model = DAT(**DAT_SMOKE_KWARGS).to(device)
        model.eval()
    except Exception as exc:
        raise RuntimeError("DAT model construction failed with num_classes=1 and smoke config.") from exc
    print("DAT construction: ok (num_classes=1)")
    print("pretrained weights: not loaded")

    with torch.no_grad():
        try:
            outputs = model(batch)
        except Exception as exc:
            raise RuntimeError(f"DAT forward pass failed for input batch shape {tuple(batch.shape)}.") from exc

        if not isinstance(outputs, tuple) or len(outputs) != 3:
            raise RuntimeError(f"Expected DAT forward to return a 3-item tuple, got {type(outputs).__name__}.")
        x, positions, references = outputs
        if tuple(x.shape) != (len(images), 2):
            raise RuntimeError(f"Expected x.shape == ({len(images)}, 2), got {tuple(x.shape)}.")

        loss = torch.nn.MSELoss()(x, target)
        pred_color = x[:, 1].detach().cpu()
        label_color = target[:, 1].detach().cpu()
        srcc = spearman_corr(pred_color, label_color)
        plcc = pearson_corr(pred_color, label_color)
        accuracy = ((pred_color >= 0.5) == (label_color >= 0.5)).float().mean().item()

    print(f"output x shape: {tuple(x.shape)}")
    print(f"loss: {loss.item():.6f}")
    print("metrics note: meaningless smoke metrics because the model is randomly initialized.")
    print(f"color SRCC: {srcc:.6f}")
    print(f"color PLCC: {plcc:.6f}")
    print(f"color binary Accuracy@0.5: {accuracy:.6f}")
    print("first 5 color predictions and labels:")
    for idx, (pred, label) in enumerate(zip(pred_color[:5], label_color[:5]), start=1):
        print(f"  {idx}: pred={pred.item():.6f}, label={label.item():.6f}")

    print("ICAA17K DAT real-image smoke test succeeded.")


if __name__ == "__main__":
    main()
