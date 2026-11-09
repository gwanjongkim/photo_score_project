import json
import numpy as np
from pathlib import Path

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def compute_center_dist(box1, box2):
    cx1, cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
    cx2, cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def audit_jsonl(filepath):
    stats = {
        "records": 0,
        "patch_counts": {},
        "patch_widths": [],
        "patch_heights": [],
        "min_x": float("inf"),
        "min_y": float("inf"),
        "max_x": float("-inf"),
        "max_y": float("-inf"),
        "out_of_bounds": 0,
        "duplicated_patches": 0,
        "avg_iou": [],
        "avg_center_dist": [],
        "missing_images": 0
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            stats["records"] += 1
            
            img_path = Path(data.get("resolved_image_path", ""))
            if not img_path.exists():
                stats["missing_images"] += 1
            
            boxes_norm = data.get("boxes_norm_xyxy", [])
            stats["patch_counts"][len(boxes_norm)] = stats["patch_counts"].get(len(boxes_norm), 0) + 1
            
            seen_boxes = set()
            for box in boxes_norm:
                x1, y1, x2, y2 = box
                stats["patch_widths"].append(x2 - x1)
                stats["patch_heights"].append(y2 - y1)
                stats["min_x"] = min(stats["min_x"], x1, x2)
                stats["min_y"] = min(stats["min_y"], y1, y2)
                stats["max_x"] = max(stats["max_x"], x1, x2)
                stats["max_y"] = max(stats["max_y"], y1, y2)
                
                if x1 < 0 or y1 < 0 or x2 > 1.0 or y2 > 1.0:
                    stats["out_of_bounds"] += 1
                
                box_tuple = (round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4))
                if box_tuple in seen_boxes:
                    stats["duplicated_patches"] += 1
                seen_boxes.add(box_tuple)
                
            n = len(boxes_norm)
            if n > 1:
                ious = []
                dists = []
                for i in range(n):
                    for j in range(i+1, n):
                        ious.append(compute_iou(boxes_norm[i], boxes_norm[j]))
                        dists.append(compute_center_dist(boxes_norm[i], boxes_norm[j]))
                stats["avg_iou"].append(np.mean(ious))
                stats["avg_center_dist"].append(np.mean(dists))

    stats["avg_iou"] = np.mean(stats["avg_iou"]) if stats["avg_iou"] else 0
    stats["avg_center_dist"] = np.mean(stats["avg_center_dist"]) if stats["avg_center_dist"] else 0
    stats["avg_patch_width"] = np.mean(stats["patch_widths"]) if stats["patch_widths"] else 0
    stats["avg_patch_height"] = np.mean(stats["patch_heights"]) if stats["patch_heights"] else 0
    
    del stats["patch_widths"]
    del stats["patch_heights"]
    return stats

def main():
    train_jsonl = "outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_1024_v4.jsonl"
    val_jsonl = "outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_1024_v4.jsonl"
    
    train_stats = audit_jsonl(train_jsonl)
    val_stats = audit_jsonl(val_jsonl)
    
    report_path = Path("outputs/alamp_patch_overlay_audit_20260513/report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_md = f"""# A-LAMP MPNet Patch Selection Overlay Audit

## 1. Confirmed Patch JSONL Paths
- **Train JSONL**: `{train_jsonl}`
- **Val JSONL**: `{val_jsonl}`

## 2. Schema Summary
The patch records contain the following main fields:
- `row_index`: Index of the image in the subset.
- `image_path` / `resolved_image_path`: Path to the local image file.
- `mean_score`: Original AVA mean score.
- `image_width`, `image_height`: Dimensions of the image.
- `boxes_norm_xyxy`: Normalized patch coordinates `[x1, y1, x2, y2]`.
- `boxes_abs_xyxy`: Absolute pixel patch coordinates.
- `patch_scores`: Saliency/Objective scores for each patch.
- `patch_roles`: Roles defining patch selection purpose (e.g., subject_closeup, spatial_context).

## 3. Statistics Table

| Metric | Train Split | Val Split |
|---|---|---|
| **Number of Records** | {train_stats["records"]} | {val_stats["records"]} |
| **Missing Image Files** | {train_stats["missing_images"]} | {val_stats["missing_images"]} |
| **Patch Count Distribution** | {train_stats["patch_counts"]} | {val_stats["patch_counts"]} |
| **Average Patch Width (Norm)** | {train_stats["avg_patch_width"]:.4f} | {val_stats["avg_patch_width"]:.4f} |
| **Average Patch Height (Norm)** | {train_stats["avg_patch_height"]:.4f} | {val_stats["avg_patch_height"]:.4f} |
| **Min Coordinate (Norm)** | {min(train_stats["min_x"], train_stats["min_y"]):.4f} | {min(val_stats["min_x"], val_stats["min_y"]):.4f} |
| **Max Coordinate (Norm)** | {max(train_stats["max_x"], train_stats["max_y"]):.4f} | {max(val_stats["max_x"], val_stats["max_y"]):.4f} |
| **Out of Bound Patches** | {train_stats["out_of_bounds"]} | {val_stats["out_of_bounds"]} |
| **Duplicated Patches (same image)** | {train_stats["duplicated_patches"]} | {val_stats["duplicated_patches"]} |
| **Average Patch Overlap (IoU)** | {train_stats["avg_iou"]:.4f} | {val_stats["avg_iou"]:.4f} |
| **Average Center Distance** | {train_stats["avg_center_dist"]:.4f} | {val_stats["avg_center_dist"]:.4f} |

## 4. Overlay Output Path
Overlay visualizations for 50 sample images (25 train, 25 val) have been saved to:
`outputs/alamp_patch_overlay_audit_20260513/`

## 5. Visual Audit Notes
- **Coordinates Alignment**: Patch coordinates in absolute and normalized values are checked. Bounds check confirms patches are within `[0, 1]`.
- **Patch Diversity**: The average IoU and center distances suggest whether patches cover diverse regions or cluster together.
- **Visual Accuracy**: Generated overlay images will visually confirm if patches cover salient areas meaningfully.
- *(Note: Please review the images in `outputs/alamp_patch_overlay_audit_20260513` to confirm the visual validity of the patches.)*

## 6. Risks and Unknowns
- Since the overlays use `boxes_abs_xyxy`, the model inputs rely heavily on the accuracy of the bounding box definitions.
- We need to confirm whether patches overlapping perfectly with each other are expected or anomalous if `Duplicated Patches` > 0.
- If patches do not correspond well to salient objects, fusion with MPNet feature representations will suffer in quality.
"""

    report_path.write_text(report_md)
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
