# AVA 이미지에서 YOLO 객체 감지 결과를 JSONL로 추출하는 도구
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO


DEFAULT_MODEL_CANDIDATES = ["yolo11n.pt", "yolov8n.pt"]


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configs.") from exc

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return data


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Extract Ultralytics object detections for a bounded AVA subset."
    )
    arg_parser.add_argument("--config")
    arg_parser.add_argument("--split", choices=["train", "val", "test"])
    arg_parser.add_argument("--csv")
    arg_parser.add_argument("--image_dir")
    arg_parser.add_argument("--output_jsonl")
    arg_parser.add_argument("--max_images", type=int)
    arg_parser.add_argument("--model")
    arg_parser.add_argument("--confidence", type=float)
    arg_parser.add_argument("--max_objects", type=int)
    arg_parser.add_argument("--device")
    arg_parser.add_argument("--batch_size", type=int)
    return arg_parser


def first_set(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def split_config(config: dict[str, Any], split: str | None) -> dict[str, Any]:
    if not split:
        return {}
    splits = config.get("splits", {})
    if not isinstance(splits, dict):
        return {}
    value = splits.get(split, {})
    return value if isinstance(value, dict) else {}


def resolve_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    split = first_set(args.split, config.get("split"))
    split_cfg = split_config(config, split)

    csv_path = first_set(args.csv, split_cfg.get("csv"))
    image_dir = first_set(args.image_dir, config.get("image_dir"), split_cfg.get("image_dir"))
    output_jsonl = first_set(
        args.output_jsonl,
        split_cfg.get("output_jsonl"),
        split_cfg.get("detection_jsonl"),
    )

    missing = [
        name
        for name, value in [
            ("split", split),
            ("csv", csv_path),
            ("image_dir", image_dir),
            ("output_jsonl", output_jsonl),
        ]
        if value in (None, "")
    ]
    if missing:
        raise ValueError(f"Missing required setting(s): {', '.join(missing)}")

    max_images = int(first_set(args.max_images, config.get("max_images"), default=1024))
    max_objects = int(first_set(args.max_objects, config.get("max_objects"), default=4))
    batch_size = int(first_set(args.batch_size, config.get("batch_size"), default=16))
    confidence = float(first_set(args.confidence, config.get("confidence"), default=0.25))
    device = first_set(args.device, config.get("device"))
    if device in (None, "", "auto"):
        device = default_device()

    candidates = config.get("model_candidates", DEFAULT_MODEL_CANDIDATES)
    if not isinstance(candidates, list) or not candidates:
        candidates = DEFAULT_MODEL_CANDIDATES

    model = first_set(args.model, config.get("model"))
    return {
        "split": str(split),
        "csv": Path(str(csv_path)),
        "image_dir": Path(str(image_dir)),
        "output_jsonl": Path(str(output_jsonl)),
        "max_images": max_images,
        "max_objects": max_objects,
        "batch_size": batch_size,
        "confidence": confidence,
        "device": str(device),
        "model": str(model) if model else None,
        "model_candidates": [str(candidate) for candidate in candidates],
        "image_col": str(config.get("image_col", "image_path")),
    }


def default_device() -> str:
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model(model_name: str | None, candidates: list[str]) -> tuple[YOLO, str]:
    names = [model_name] if model_name else candidates
    errors: list[str] = []
    for name in names:
        try:
            return YOLO(name), name
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Unable to load any YOLO model candidate.\n{joined}")


def read_limited_rows(csv_path: Path, image_col: str, max_images: int) -> tuple[list[dict[str, Any]], str]:
    if max_images <= 0:
        raise ValueError("--max_images must be positive")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")

        selected_col = choose_image_column(reader.fieldnames, image_col)
        for row_index, row in enumerate(reader):
            if len(rows) >= max_images:
                break
            rows.append({"row_index": row_index, "row": row})
    return rows, selected_col


def choose_image_column(fieldnames: list[str], preferred: str) -> str:
    if preferred in fieldnames:
        return preferred
    for candidate in ["image_path", "image", "filename", "file_name", "path"]:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"Could not find image column. Preferred '{preferred}', available {fieldnames}"
    )


def resolve_image_path(raw_value: str, image_dir: Path) -> Path | None:
    raw_path = Path(raw_value)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        candidates.append(image_dir / raw_path)
        candidates.append(image_dir / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def validate_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        image.verify()
    return width, height


def chunked(records: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [records[index : index + size] for index in range(0, len(records), size)]


def extract_objects(result: Any, width: int, height: int, max_objects: int, names: dict[Any, str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = boxes.xyxy.detach().cpu().tolist()
    conf_values = boxes.conf.detach().cpu().tolist()
    class_values = boxes.cls.detach().cpu().tolist()

    objects: list[dict[str, Any]] = []
    for xyxy, confidence, class_value in zip(xyxy_values, conf_values, class_values):
        class_id = int(class_value)
        x1, y1, x2, y2 = clamp_box(xyxy, width, height)
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            continue

        norm_xyxy = [x1 / width, y1 / height, x2 / width, y2 / height]
        center_norm = [((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height]
        class_name = names.get(class_id, names.get(str(class_id), str(class_id)))
        objects.append(
            {
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": round(float(confidence), 6),
                "bbox_xyxy": [round_float(value) for value in [x1, y1, x2, y2]],
                "bbox_xyxy_normalized": [round_float(value) for value in norm_xyxy],
                "center_normalized": [round_float(value) for value in center_norm],
                "area_ratio": round_float((box_width * box_height) / (width * height)),
            }
        )

    objects.sort(key=lambda item: item["confidence"], reverse=True)
    return objects[:max_objects]


def clamp_box(xyxy: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    return x1, y1, x2, y2


def round_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), 6)


def write_detection_summary(summary_path: Path, split: str, split_summary: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)

    existing.pop("blocked_reason", None)
    existing.setdefault("selected_detector", "ultralytics")
    existing["status"] = "available"
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    existing["confidence"] = split_summary["confidence"]
    existing["max_objects"] = split_summary["max_objects"]
    existing["model"] = split_summary["model"]
    existing["model_file"] = split_summary["model_file"]
    existing.setdefault("splits", {})
    existing["splits"][split] = split_summary

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parser().parse_args()
    config = load_config(args.config)
    settings = resolve_settings(args, config)

    image_dir = settings["image_dir"]
    if not image_dir.is_absolute():
        image_dir = (Path.cwd() / image_dir).resolve()

    output_jsonl = settings["output_jsonl"]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    raw_rows, image_col = read_limited_rows(
        settings["csv"],
        settings["image_col"],
        settings["max_images"],
    )

    skipped_examples: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    for entry in raw_rows:
        row_index = entry["row_index"]
        row = entry["row"]
        image_path = str(row.get(image_col, "")).strip()
        resolved_path = resolve_image_path(image_path, image_dir)
        if resolved_path is None:
            skipped_examples.append(
                {"row_index": row_index, "image_path": image_path, "reason": "missing"}
            )
            continue

        try:
            width, height = validate_image(resolved_path)
        except Exception as exc:
            skipped_examples.append(
                {
                    "row_index": row_index,
                    "image_path": image_path,
                    "resolved_image_path": str(resolved_path),
                    "reason": f"undecodable: {exc}",
                }
            )
            continue

        valid_records.append(
            {
                "row_index": row_index,
                "image_path": image_path,
                "resolved_image_path": str(resolved_path),
                "width": width,
                "height": height,
            }
        )

    model, model_name = load_model(settings["model"], settings["model_candidates"])
    names = getattr(model, "names", {}) or {}
    model_file = str(
        getattr(model, "ckpt_path", None)
        or getattr(model, "model_name", None)
        or model_name
    )

    class_counts: Counter[str] = Counter()
    no_object_count = 0
    object_counts: list[int] = []
    records_written = 0

    with output_jsonl.open("w", encoding="utf-8") as output_handle:
        for batch in chunked(valid_records, settings["batch_size"]):
            paths = [record["resolved_image_path"] for record in batch]
            results = model.predict(
                paths,
                conf=settings["confidence"],
                device=settings["device"],
                batch=settings["batch_size"],
                verbose=False,
            )

            for record, result in zip(batch, results):
                width = int(record["width"])
                height = int(record["height"])
                objects = extract_objects(result, width, height, settings["max_objects"], names)
                if not objects:
                    no_object_count += 1
                object_counts.append(len(objects))
                class_counts.update(obj["class_name"] for obj in objects)
                output_record = {
                    "row_index": record["row_index"],
                    "split": settings["split"],
                    "image_path": record["image_path"],
                    "resolved_image_path": record["resolved_image_path"],
                    "width": width,
                    "height": height,
                    "objects": objects,
                }
                output_handle.write(json.dumps(output_record, sort_keys=True) + "\n")
                records_written += 1

    processed = len(valid_records)
    split_summary = {
        "split": settings["split"],
        "csv": str(settings["csv"]),
        "image_col": image_col,
        "image_dir": str(image_dir),
        "output_jsonl": str(output_jsonl),
        "requested": len(raw_rows),
        "processed": processed,
        "records_written": records_written,
        "skipped": len(skipped_examples),
        "skipped_examples": skipped_examples[:10],
        "no_object": no_object_count,
        "avg_objects_per_image": round_float(sum(object_counts) / processed) if processed else 0.0,
        "confidence": settings["confidence"],
        "max_objects": settings["max_objects"],
        "batch_size": settings["batch_size"],
        "device": settings["device"],
        "model": model_name,
        "model_file": model_file,
        "top_classes": [
            {"class_name": class_name, "count": count}
            for class_name, count in class_counts.most_common(20)
        ],
    }
    write_detection_summary(output_jsonl.parent / "detection_summary.json", settings["split"], split_summary)
    print(json.dumps(split_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
