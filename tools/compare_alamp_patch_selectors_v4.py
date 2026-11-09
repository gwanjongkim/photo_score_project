"""Compare A-LAMP patch selector v1 vs v2 vs v3 vs v4."""
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / max(union_area, 1e-6)

def main():
    v1_path = Path("outputs/alamp_paper_mpnet_design_audit_20260511/subsets/train_patch_boxes_1024.jsonl")
    v2_path = Path("outputs/alamp_paper_mpnet_patch_selector_v2_20260511/subsets/train_patch_boxes_1024_v2.jsonl")
    v3_path = Path("outputs/alamp_paper_mpnet_patch_selector_v3_20260512/subsets/train_patch_boxes_1024_v3.jsonl")
    v4_path = Path("outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_1024_v4.jsonl")
    yolo_path = Path("outputs/alamp_object_graph_subset_20260511/detections_conf010/train_objects_1024.jsonl")
    
    paths = {"V1": v1_path, "V2": v2_path, "V3": v3_path, "V4": v4_path}
    
    yolo_data = {}
    if yolo_path.exists():
        with open(yolo_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                name = Path(d["image_path"]).name
                objs = [obj["bbox_xyxy_normalized"] for obj in d.get("objects", [])]
                yolo_data[name] = objs

    def analyze_v(path):
        metrics = {"scores": [], "ious": [], "obj_covered": [], "main_obj_covered": [], "total": 0}
        with open(path, 'r') as f:
            for line in f:
                d = json.loads(line)
                if d.get("skipped"): continue
                metrics["total"] += 1
                metrics["scores"].append(np.mean(d["patch_scores"]))
                
                boxes = d["boxes_norm_xyxy"]
                ious = []
                for i in range(len(boxes)):
                    for j in range(i+1, len(boxes)):
                        ious.append(compute_iou(boxes[i], boxes[j]))
                metrics["ious"].append(np.mean(ious) if ious else 0.0)
                
                # Object coverage
                name = Path(d["image_path"]).name
                objs = yolo_data.get(name, [])
                if objs:
                    covered = 0
                    for obj in objs:
                        if any(compute_iou(p, obj) > 0.3 for p in boxes):
                            covered += 1
                    metrics["obj_covered"].append(covered / len(objs))
                    # Main object (usually the first in YOLO sorted by conf/area)
                    if any(compute_iou(p, objs[0]) > 0.3 for p in boxes):
                        metrics["main_obj_covered"].append(1.0)
                    else:
                        metrics["main_obj_covered"].append(0.0)
                    
        return metrics

    all_results = {}
    for label, path in paths.items():
        if path.exists():
            logging.info(f"Analyzing {label}...")
            all_results[label] = analyze_v(path)

    report = "# Patch Selector V1 vs V2 vs V3 vs V4 Comparison\n\n## Metrics Summary\n"
    report += "| Metric | V1 | V2 | V3 | V4 | Change (V3->V4) |\n"
    report += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    m3 = all_results.get("V3")
    m4 = all_results.get("V4")
    
    def get_val(m, key):
        return np.mean(m[key]) if m and key in m and m[key] else 0.0

    m1, m2 = all_results.get("V1"), all_results.get("V2")
    
    for key, label in [("scores", "Mean Patch Score"), ("ious", "Mean Pairwise IoU"), 
                       ("obj_covered", "Object Coverage"), ("main_obj_covered", "Main Object Coverage")]:
        v1, v2, v3, v4 = get_val(m1, key), get_val(m2, key), get_val(m3, key), get_val(m4, key)
        change = f"{(v4/v3-1)*100:+.1f}%" if v3 > 0 else "N/A"
        if "Coverage" in label:
            report += f"| **{label}** | {v1:.1%} | {v2:.1%} | {v3:.1%} | {v4:.1%} | {v4-v3:+.1%} |\n"
        else:
            report += f"| **{label}** | {v1:.4f} | {v2:.4f} | {v3:.4f} | {v4:.4f} | {change} |\n"

    report += """
## Analysis
- **V4 Strategy**: Directly targets salient components identified by U²-Net, allocating roles for close-ups and context.
- **Subject Coverage**: V4 is expected to show significant improvement in Main Object Coverage as it explicitly places patches on the largest salient contours.
- **Diversity**: V4 maintains spatial diversity by filling remaining slots with context-aware patches while avoiding subject-overlap.
"""
    print(report)
    out_dir = Path("outputs/alamp_paper_mpnet_patch_selector_v4_20260512")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patch_selector_v4_comparison_report.md").write_text(report)
    
    summary = {
        label: {
            "score": float(get_val(res, "scores")),
            "iou": float(get_val(res, "ious")),
            "obj": float(get_val(res, "obj_covered")),
            "main_obj": float(get_val(res, "main_obj_covered"))
        } for label, res in all_results.items()
    }
    with open(out_dir / "patch_selector_v4_comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
