# A-LAMP MPNet Patch Selection Overlay Audit

## 1. Confirmed Patch JSONL Paths
- **Train JSONL**: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/train_patch_boxes_1024_v4.jsonl`
- **Val JSONL**: `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/subsets/val_patch_boxes_1024_v4.jsonl`

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
| **Number of Records** | 1024 | 1024 |
| **Missing Image Files** | 0 | 0 |
| **Patch Count Distribution** | {5: 1024} | {5: 1024} |
| **Average Patch Width (Norm)** | 0.2534 | 0.2551 |
| **Average Patch Height (Norm)** | 0.2867 | 0.2835 |
| **Min Coordinate (Norm)** | 0.0000 | 0.0000 |
| **Max Coordinate (Norm)** | 0.9988 | 0.9988 |
| **Out of Bound Patches** | 0 | 0 |
| **Duplicated Patches (same image)** | 0 | 0 |
| **Average Patch Overlap (IoU)** | 0.0589 | 0.0612 |
| **Average Center Distance** | 0.4240 | 0.4201 |

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
