# A-LAMP-paper-MPNet Design Audit

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 2026-05-11
- Note: This is a read-only design audit. No practical A-LAMP models, Flutter app, or forWeights components are modified.

## 2. Why v2_graph_lite Is Paused
The `v2_graph_lite` experiment attempted to use YOLO-generated object graphs. However, it severely underperformed the `v0` baselines:
- v2_graph_lite ROC-AUC: 0.5826 (vs v0_b_fixed: 0.6645)
- v2_graph_lite Specificity: 0.0272 (vs v0_b_fixed: 0.1394)
Generic COCO bounding boxes lack the explicit aesthetic semantic context (like lighting, blur, compositional balance) required to improve aesthetic scoring. Furthermore, ~16% of images lacked objects entirely. Therefore, explicit layout graphs are paused.

## 3. Paper Evidence For Multi-Patch Subnet
From local audit reports (`outputs/alamp_paper_faithfulness_audit_20260511/` and `outputs/alamp_paper_ava_v1_design_audit_20260511/`):
- **Adaptive Patch Selection**: Driven by a saliency-map and a pattern diversity constraint.
- **Overlap Constraint**: Enforced spatial distance/overlap constraint during selection.
- **Patch count/size**: 5 patches, 224x224 each.
- **Backbone**: Shared VGG16.
- **Aggregation**: Orderless aggregation over the patches.
- **Label Rule**: AVA binary classification (good vs bad).
*Specific details like the exact edge/chrominance distribution algorithms were not found locally and will require approximation.*

## 4. Current Implementation Gap
The current `v0_a` implements a shared VGG16 backbone but falls short in patch selection:
- It uses a dynamic, on-the-fly TensorFlow dataset heuristic (Sobel edges + luminance/color variance).
- It applies simple IoU NMS (0.35 threshold) instead of a true pattern diversity + saliency joint optimization.
- The patch boxes are not inspectable or saved offline, making reproducibility and debugging difficult.

## 5. Target A-LAMP-paper-MPNet Definition
The `A-LAMP-paper-MPNet` track will isolate the Multi-Patch Subnet (New-MP-Net):
- **Task**: AVA binary classification (`mean_score > 5.0`).
- **Architecture**: Shared VGG16 processing 5 patches (224x224), followed by orderless statistical aggregation over the patch axis, and a sigmoid classification head.
- **Exclusions**: No layout-aware branch, no global view, no object detector, no Flutter integration.

## 6. Patch Selection Design
The patch selector must be moved out of the TensorFlow `tf.data` pipeline into a dedicated offline script (`src/datasets/alamp_paper_patch_selector.py`).
**Design**:
- **Candidate Generation**: Sample predefined scales and grid centers across the native image.
- **Saliency Approximation**: Since exact paper implementations are missing locally, we will use OpenCV's Fine Grained Saliency/Spectral Residual (if available) or a NumPy/OpenCV translation of our edge+variance heuristic as a non-speculative approximation.
- **Diversity & Overlap**: Select the top patch, then iteratively select the next patches that maximize saliency while penalizing high IoU (overlap constraint) or color histogram similarity (pattern diversity constraint).
- **Output**: Save the 5 patch bounding boxes (normalized `xyxy`) to a JSONL file per split.

## 7. Model Design
**File**: `src/models/alamp_paper_mpnet.py`
- Input: `patches` tensor `[B, 5, 224, 224, 3]`.
- Reshape to `[B*5, 224, 224, 3]`, pass through VGG16 (ImageNet weights, trainable=False initially).
- Apply GlobalAveragePooling2D.
- Reshape to `[B, 5, Feature_Dim]`.
- Aggregate using a combination of Mean/Max pooling across the `5` axis, followed by dropout and dense layers.

## 8. Validation Plan
Full AVA training is expensive and premature. We will follow a strict stage-gate process:
1. **Patch selector smoke**: Test `src/datasets/alamp_paper_patch_selector.py` on 10 images.
2. **Subset Generation**: Generate patch box JSONL files for a 1024-image train/val/test subset.
3. **Visual/JSON Validation**: Inspect the generated JSONL to ensure coordinates are valid.
4. **MPNet forward smoke**: Compile model, verify shapes.
5. **Tiny train**: Train on 64 samples to verify loss calculation.
6. **Subset train/eval**: Train and evaluate on the 1024 subsets.
7. **Comparison**: Compare metrics against `v0_a`.
8. **Full Expansion**: Only expand to full AVA (4096 or 250k) if subset metrics show promise.

## 9. What Not To Claim
- Do not claim this is an "official reproduction" or "same as the paper". It is an "A-LAMP-paper-oriented approximation" using saliency approximations.

## 10. Next Implementation Prompt
*See `next_implementation_prompt.md`*
