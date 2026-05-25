# A-LAMP Dual-Branch GCN Teacher Preflight

## 1. Summary
This preflight audit confirms that the core components for building a Dual-Branch GCN Teacher (Multi-Patch + Layout GCN) are present in the repository. While a full AVA-wide graph dataset is currently missing, the scripts to generate it are available and validated on 4096-sample subsets. The project has an established pattern for using YOLO-based object graphs and Graph Convolutional Networks (GCNs), making the "YOLO-GCN" approach the most viable path forward.

## 2. Existing Graph/Object/Layout Data
- **Subset Data**: Graph JSONLs exist for 1024 and 4096 AVA samples under `outputs/alamp_object_graph_subset_20260511/`.
- **Full Data**: No full AVA-wide (254k) graph JSONL was found. 
- **Generator Scripts**: 
  - `tools/alamp_graph/extract_ava_objects.py`: Extracts YOLO detections.
  - `tools/alamp_graph/build_attribute_graphs.py`: Converts detections to object graphs with local/global edges.
- **Saliency Metadata**: `outputs/alamp_v4_full_ava_20260517/saliency_maps/` contains full AVA saliency maps (PNG) and metadata (JSONL), which can be used for salient object extraction if YOLO is not preferred.

## 3. Reusable GCN Code
- **`src/models/alamp_paper_fusion.py`**: Contains `GraphConvolutionLayer` and `MaskedMeanPoolingLayer`.
- **Compatibility**: These are standard TF2/Keras layers, fully compatible with the current project's training pipeline.
- **Math**: The logic for converting `local_edges` (distance) to a scalar `adjacency` matrix is already implemented in `src/train/train_alamp_paper_mpnet_graph_fusion.py`.

## 4. GraphLiteBranch Feasibility
- **Location**: `src/models/alamp_paper_ava_v2_graph_lite.py`.
- **Inputs**: It uses a flat vector of object features (boxes, centers, areas, class_ids, confidences).
- **Assessment**: While `GraphLiteBranch` is easy to implement, it lacks inter-node message passing. It should be used as a "Lite" baseline, but the teacher should upgrade to the GCN variant for maximum strength.

## 5. External SOD Feasibility
- **Location**: `external/A-Lamp_external_audit/A-Lamp/layout_aware_subnet/SOD_class.py`.
- **Status**: TensorFlow 2 compatible. Requires `.h5` weights.
- **Runtime Risk**: Processing 254k images is estimated to take ~35 hours on a single GPU.
- **Recommendation**: Prioritize YOLO-based detection as it is already integrated into the `alamp_graph` tools and `ultralytics` is easier to maintain.

## 6. Minimal Viable Layout Branch Options
- **Option A (Baseline)**: Pick IoU from existing patch boxes as layout features (Too simple).
- **Option B (Recommended)**: Use YOLO-based object boxes and a 2-layer GCN (Option C/D from previous audit).
- **Option C (Paper-Strict)**: Use external SOD to extract salient objects (Higher complexity/runtime).

## 7. Recommended First Implementation
**"A-LAMP Dual-Branch GCN Teacher (YOLO-based)"**
- **Architecture**: VGG16 Multi-Patch (frozen or fine-tuned) + 2-layer GCN Layout Branch.
- **Data**: Generate a 4096-sample graph JSONL (if not already fully validated) then scale to full AVA.
- **Fusion**: Concatenate Multi-Patch aggregation (Mean+Max) and GCN Masked Mean Pooling.

## 8. Required Code Changes
- **New Model**: `src/models/alamp_dual_branch_teacher.py` combining Multi-Patch and GCN branches.
- **New Dataset**: Extend `src/datasets/alamp_external_patch_dataset.py` to optionally load graph features (boxes, adjacency, mask).
- **New Train Script**: `src/train/train_alamp_dual_branch_teacher.py` supporting dictionary-based inputs.

## 9. Runtime / GPU Risk
- **Full Graph Generation**: High runtime (30-40 hours for 254k images). Should be run as a background batch job.
- **VRAM**: Dual-branch VGG16 + GCN will be heavy. Recommend `batch_size=4` or `8` and potential mixed precision.

## 10. Final Recommendation
Proceed with building the **Dual-Branch GCN Teacher**. The code infrastructure is 90% ready; the primary bottleneck is the **Full AVA Graph Generation**. Start by implementing the model and training script, and verify performance on the existing 4096-sample subsets before launching the full-scale graph extraction.
