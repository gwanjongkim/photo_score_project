# A-LAMP-paper-AVA-v2_graph_lite Report

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 2026-05-11
- Note: This execution was isolated to avoid interfering with any practical codebase, and was executed on CPU for stability regarding cuDNN versions.

## 2. Scope
Implement and evaluate `A-LAMP-paper-AVA-v2_graph_lite` as a paper-oriented approximation. This introduces an explicit graph-lite branch to consume the newly generated 1024-image object/global attribute graph subset artifacts from YOLOv11 (confidence 0.10).

## 3. Graph Artifact Inputs
- Object bounding boxes (norm xyxy)
- Object centers (norm xy)
- Area ratios
- Class IDs (embedded into 16 dims)
- Confidences
- Local edges (neighbor distances/orientations)
- Global edges (scene layout relations)

## 4. Model Architecture
- Added `GraphLiteBranch` custom layer: It processes up to 4 objects per image.
- Node features are aggregated via `boxes`, `centers`, `area`, `confidence`, and `class embedding`.
- Local edges are pooled over valid neighbors.
- Combined node features are mapped via a `64->64` dense network and globally averaged (ignoring padded masks).
- Concatenated with standard multi-patch VGG16 features and global VGG16 branch.

## 5. Smoke Results
- Forward smoke check: Passed. (Shape `[2,1]`, outputs inside `[0,1]`)
- Tiny train: Passed on 64 sample subset.

## 6. Subset Training Results
- Configured for a 1024-sample train split and 1024-sample validation split.
- `epochs`: 1 (limited due to CPU inference time).
- `no_object_count`: 163 images out of 1024 in train had zero objects (~16%).
- `best_val_loss`: 0.7068

## 7. Subset Evaluation Results
Evaluated on the exact 1024-sample test subset:
- **accuracy**: 0.7109
- **precision**: 0.7157
- **recall**: 0.9863
- **specificity**: 0.0272
- **F1**: 0.8295
- **ROC-AUC**: 0.5826
- **average precision**: 0.7637
- **BCE loss**: 0.6976
- **confusion matrix**: tn=8, fp=286, fn=10, tp=720

## 8. Comparison With v0_a and v0_b_fixed
| Metric | v0_a | v0_b_fixed | v2_graph_lite |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.6678 | 0.6645 | **0.5826** |
| **Specificity** | 0.1224 | 0.1394 | **0.0272** |
| **Accuracy** | 0.7236 | 0.7187 | **0.7109** |
| **AP** | 0.8199 | 0.8180 | **0.7637** |
| **F1** | 0.8328 | 0.8283 | **0.8295** |

*Note: RGNet-paper-AVA reference ROC-AUC is > 0.74, making all A-LAMP variants severely underperform.*

## 9. Failure Mode Analysis
The `v2_graph_lite` model performed worse than the `v0` models. 
1. **Low Semantic Value for Aesthetics**: YOLOv11n objects (e.g., detecting a "person", "dog", "chair") do not inherently map directly to compositional quality or aesthetics without deeper relation analysis.
2. **Missing Objects**: ~16% of the images have no objects at all, essentially acting as noise to the graph branch.
3. **Training Time Limitation**: The subset ran for only 1 epoch to bypass local GPU cuDNN issues, meaning the graph branch weights were severely undertrained compared to standard 10-20 epoch loops. However, the initial trajectory of ROC-AUC collapsing below 0.60 indicates a fundamental architecture struggle to fuse the signals effectively.

## 10. Official Reproduction Boundary
This model remains an "object/global attribute graph approximation". It utilizes a heuristic YOLO detector rather than the exact object/global attribute pipeline used by the original paper authors.

## 11. Next Step Recommendation
**Decision**: C. `v2_graph_lite` does not help; pause A-LAMP graph work.

The inclusion of the object graph actually hurt specificity and ROC-AUC. Without true semantic object-attribute mappings from an aesthetic dataset (like AVA attributes) instead of generic COCO bounding boxes, the graph branch merely acts as a regularizer driving the model towards predicting the majority class (positive aesthetic). 

We should pause the `A-LAMP` explicit graph track and focus entirely on the `RGNet-paper-AVA` variants which natively demonstrated >0.74 ROC-AUC through image-centric region compositions.