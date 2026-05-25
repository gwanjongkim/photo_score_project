# A-LAMP Object Graph Subset Report

## 1. Environment and Context
- CWD: `/home/omen_pc1/photo_score_project`
- The previous report indicated detection and graph generation were blocked due to missing YOLO weights. This issue has been resolved.
- Note: This is still an object/global attribute graph approximation, not an official A-LAMP reproduction.

## 2. Detection Results
- Model: `models/detectors/yolo11n.pt`
- Confidence threshold: `0.10` (selected to reduce the no_object rate).
- Max objects per image: `4`
- Batch size: `16`
- 1024-image subset detection succeeded for train, val, and test splits.
- Skipped images: `0` across all splits.

### Detection Statistics
- **Train (1024 images)**
  - Average objects per image: `1.969727`
  - Images with no objects: `163` (~15.9%)
- **Val (1024 images)**
  - Average objects per image: `1.943359`
  - Images with no objects: `166` (~16.2%)
- **Test (1024 images)**
  - Average objects per image: `1.958984`
  - Images with no objects: `161` (~15.7%)

*Limitation*: The no_object rate remains around 16%, which should be considered a limitation when analyzing model behavior on these images.

## 3. Graph Generation Results
Train, val, and test splits each successfully produced 1024 graph records. 

Graph fields are fixed shape and ready for `A-LAMP-paper-AVA-v2_graph_lite`. 
Shape validation succeeded for all splits (train shape ok, val shape ok, test shape ok).

### Generated Field Shapes
- `object_mask`: `[4]`
- `boxes_norm_xyxy`: `[4, 4]`
- `centers_norm_xy`: `[4, 2]`
- `area_ratio`: `[4]`
- `class_ids`: `[4]`
- `confidences`: `[4]`
- `local_edges`: `[4, 4, 3]`
- `global_edges`: `[4, 3]`

## 4. Generated Artifacts
**Detection Outputs:**
- `outputs/alamp_object_graph_subset_20260511/detections_conf010/train_objects_1024.jsonl`
- `outputs/alamp_object_graph_subset_20260511/detections_conf010/val_objects_1024.jsonl`
- `outputs/alamp_object_graph_subset_20260511/detections_conf010/test_objects_1024.jsonl`

**Graph Outputs:**
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/train_graphs_1024.jsonl`
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/val_graphs_1024.jsonl`
- `outputs/alamp_object_graph_subset_20260511/graphs_conf010/test_graphs_1024.jsonl`