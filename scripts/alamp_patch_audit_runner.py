# A-LAMP 패치 데이터셋의 통계적 정합성과 오류 상관관계를 분석한다.
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import math
from tqdm import tqdm

def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    union_area = area1 + area2 - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area

def audit_patches(jsonl_path, split_name, sample_size=500):
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    
    np.random.seed(42)
    if len(records) > sample_size:
        indices = np.random.choice(len(records), sample_size, replace=False)
        sample_records = [records[i] for i in indices]
    else:
        sample_records = records
        
    stats = []
    for record in tqdm(sample_records, desc=f"Auditing {split_name}"):
        img_path = Path(record['image_path'])
        if not img_path.exists():
            continue
            
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception:
            continue
            
        boxes = record['patch_boxes']
        img_area = width * height
        
        record_stats = {
            'image_id': record['image_id'],
            'split': split_name,
            'width': width,
            'height': height,
            'boxes': boxes,
            'mean_score': record.get('mean_score')
        }
        
        patch_areas = []
        ious = []
        centers = []
        out_of_bounds = 0
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            patch_areas.append(area / img_area)
            centers.append(((x1 + x2) / 2 / width, (y1 + y2) / 2 / height))
            
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                out_of_bounds += 1
            
            for j in range(i + 1, len(boxes)):
                ious.append(calculate_iou(box, boxes[j]))
        
        record_stats['avg_patch_area_ratio'] = np.mean(patch_areas)
        record_stats['max_iou'] = np.max(ious) if ious else 0
        record_stats['avg_iou'] = np.mean(ious) if ious else 0
        record_stats['out_of_bounds_count'] = out_of_bounds
        record_stats['near_duplicates'] = sum(1 for iou in ious if iou > 0.9)
        
        # Center distribution (normalized)
        record_stats['center_dist_from_mid'] = np.mean([math.sqrt((cx-0.5)**2 + (cy-0.5)**2) for cx, cy in centers])
        
        stats.append(record_stats)
        
    return pd.DataFrame(stats)

def main():
    base_path = Path("outputs/alamp_external_patch_full_conversion_20260524")
    train_df = audit_patches(base_path / "alamp_external_patches_train.jsonl", "train", 500)
    val_df = audit_patches(base_path / "alamp_external_patches_val.jsonl", "val", 200)
    test_df = audit_patches(base_path / "alamp_external_patches_test.jsonl", "test", 200)
    
    full_stats_df = pd.concat([train_df, val_df, test_df])
    
    # Error Correlation
    pred_path = Path("outputs/eval_alamp_multipatch_teacher_full_ava_20260524/predictions.csv")
    if pred_path.exists():
        preds = pd.read_csv(pred_path)
        preds['image_id'] = preds['image_id'].astype(str)
        full_stats_df['image_id'] = full_stats_df['image_id'].astype(str)
        
        merged = pd.merge(full_stats_df, preds[['image_id', 'y_true', 'y_score', 'y_pred']], on='image_id', how='left')
        
        # Identify FP/FN
        merged['is_fp'] = (merged['y_true'] == 0) & (merged['y_pred'] == 1)
        merged['is_fn'] = (merged['y_true'] == 1) & (merged['y_pred'] == 0)
        merged['is_correct'] = (merged['y_true'] == merged['y_pred'])
        
        print("\n=== Patch Stats by Error Type ===")
        error_stats = merged.groupby(['is_fp', 'is_fn', 'is_correct'])[['avg_patch_area_ratio', 'max_iou', 'avg_iou', 'center_dist_from_mid', 'near_duplicates']].mean()
        print(error_stats)
        
        # Save sample for report
        full_stats_df.to_csv("outputs/alamp_patch_fidelity_audit_20260525/patch_stats_sample.csv", index=False)
        
        summary = {
            "overall_avg_patch_area_ratio": float(full_stats_df['avg_patch_area_ratio'].mean()),
            "overall_avg_iou": float(full_stats_df['avg_iou'].mean()),
            "overall_max_iou": float(full_stats_df['max_iou'].mean()),
            "total_out_of_bounds": int(full_stats_df['out_of_bounds_count'].sum()),
            "total_near_duplicates": int(full_stats_df['near_duplicates'].sum()),
            "avg_center_dist_from_mid": float(full_stats_df['center_dist_from_mid'].mean()),
            "error_correlation": error_stats.to_dict('index')
        }
        
        # Convert keys to string for JSON
        summary['error_correlation'] = {str(k): v for k, v in summary['error_correlation'].items()}
        
        with open("outputs/alamp_patch_fidelity_audit_20260525/patch_stats_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == "__main__":
    Path("outputs/alamp_patch_fidelity_audit_20260525").mkdir(parents=True, exist_ok=True)
    main()
