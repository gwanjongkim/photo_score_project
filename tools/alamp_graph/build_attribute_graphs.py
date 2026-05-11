# 객체 감지 JSONL을 고정 크기 객체/글로벌 속성 그래프로 변환하는 도구
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description="Build fixed-size object/global attribute graph JSONL from detection JSONL."
    )
    arg_parser.add_argument("--input_jsonl", required=True)
    arg_parser.add_argument("--output_jsonl", required=True)
    arg_parser.add_argument("--max_objects", type=int, default=4)
    return arg_parser


def round_float(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), 6)


def zero_matrix(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def zero_edge_tensor(max_objects: int) -> list[list[list[float]]]:
    return [[[0.0, 0.0, 0.0] for _ in range(max_objects)] for _ in range(max_objects)]


def iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def angle_sin(dx: float, dy: float) -> float:
    distance = math.hypot(dx, dy)
    if distance <= 0.0:
        return 0.0
    return dy / distance


def normalized_local_distance(dx: float, dy: float) -> float:
    return math.hypot(dx, dy) / math.sqrt(2.0)


def normalized_center_distance(dx: float, dy: float) -> float:
    return math.hypot(dx, dy) / math.sqrt(0.5)


def build_graph(record: dict[str, Any], max_objects: int) -> dict[str, Any]:
    objects = sorted(
        record.get("objects", []),
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )[:max_objects]

    object_mask = [0 for _ in range(max_objects)]
    boxes = zero_matrix(max_objects, 4)
    centers = zero_matrix(max_objects, 2)
    areas = [0.0 for _ in range(max_objects)]
    class_ids = [0 for _ in range(max_objects)]
    confidences = [0.0 for _ in range(max_objects)]

    for index, obj in enumerate(objects):
        object_mask[index] = 1
        boxes[index] = [round_float(value) for value in obj.get("bbox_xyxy_normalized", [0, 0, 0, 0])]
        centers[index] = [round_float(value) for value in obj.get("center_normalized", [0, 0])]
        areas[index] = round_float(float(obj.get("area_ratio", 0.0)))
        class_ids[index] = int(obj.get("class_id", 0))
        confidences[index] = round_float(float(obj.get("confidence", 0.0)))

    local_edges = zero_edge_tensor(max_objects)
    global_edges = zero_matrix(max_objects, 3)
    for i in range(max_objects):
        if not object_mask[i]:
            continue
        for j in range(max_objects):
            if not object_mask[j]:
                continue
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            local_edges[i][j] = [
                round_float(normalized_local_distance(dx, dy)),
                round_float(angle_sin(dx, dy)),
                round_float(iou_xyxy(boxes[i], boxes[j])),
            ]

        center_dx = 0.5 - centers[i][0]
        center_dy = 0.5 - centers[i][1]
        global_edges[i] = [
            round_float(normalized_center_distance(center_dx, center_dy)),
            round_float(angle_sin(center_dx, center_dy)),
            areas[i],
        ]

    graph_record = {
        "row_index": record.get("row_index"),
        "split": record.get("split"),
        "image_path": record.get("image_path"),
        "width": int(record.get("width", 0)),
        "height": int(record.get("height", 0)),
        "num_objects": int(sum(object_mask)),
        "object_mask": object_mask,
        "boxes_norm_xyxy": boxes,
        "centers_norm_xy": centers,
        "area_ratio": areas,
        "class_ids": class_ids,
        "confidences": confidences,
        "local_edges": local_edges,
        "global_edges": global_edges,
    }
    validate_graph_record(graph_record, max_objects)
    return graph_record


def validate_graph_record(record: dict[str, Any], max_objects: int) -> None:
    if len(record["object_mask"]) != max_objects:
        raise ValueError("object_mask has invalid length")
    if len(record["boxes_norm_xyxy"]) != max_objects:
        raise ValueError("boxes_norm_xyxy has invalid outer length")
    if any(len(row) != 4 for row in record["boxes_norm_xyxy"]):
        raise ValueError("boxes_norm_xyxy has invalid inner length")
    if len(record["local_edges"]) != max_objects:
        raise ValueError("local_edges has invalid outer length")
    if any(len(row) != max_objects for row in record["local_edges"]):
        raise ValueError("local_edges has invalid middle length")
    if any(len(edge) != 3 for row in record["local_edges"] for edge in row):
        raise ValueError("local_edges has invalid feature length")
    if len(record["global_edges"]) != max_objects:
        raise ValueError("global_edges has invalid outer length")
    if any(len(row) != 3 for row in record["global_edges"]):
        raise ValueError("global_edges has invalid feature length")


def write_graph_summary(summary_path: Path, split: str, split_summary: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)

    existing.pop("blocked_reason", None)
    existing["status"] = "available"
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    existing["max_objects"] = split_summary["max_objects"]
    existing["field_shapes"] = split_summary["field_shapes"]
    existing.setdefault("splits", {})
    existing["splits"][split] = split_summary

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parser().parse_args()
    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    max_objects = int(args.max_objects)
    if max_objects <= 0:
        raise ValueError("--max_objects must be positive")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Detection JSONL not found: {input_jsonl}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    total_objects = 0
    split = output_jsonl.stem.split("_")[0]

    with input_jsonl.open("r", encoding="utf-8") as input_handle, output_jsonl.open(
        "w", encoding="utf-8"
    ) as output_handle:
        for line in input_handle:
            if not line.strip():
                continue
            detection_record = json.loads(line)
            graph_record = build_graph(detection_record, max_objects)
            split = str(graph_record.get("split") or split)
            records += 1
            total_objects += int(graph_record["num_objects"])
            output_handle.write(json.dumps(graph_record, sort_keys=True) + "\n")

    split_summary = {
        "split": split,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "records": records,
        "max_objects": max_objects,
        "avg_objects_per_image": round_float(total_objects / records) if records else 0.0,
        "field_shapes": {
            "object_mask": [max_objects],
            "boxes_norm_xyxy": [max_objects, 4],
            "centers_norm_xy": [max_objects, 2],
            "area_ratio": [max_objects],
            "class_ids": [max_objects],
            "confidences": [max_objects],
            "local_edges": [max_objects, max_objects, 3],
            "global_edges": [max_objects, 3],
        },
    }
    write_graph_summary(output_jsonl.parent / "graph_summary.json", split, split_summary)
    print(json.dumps(split_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
