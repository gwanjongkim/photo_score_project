# A-LAMP External Integration Feasibility Audit

## 1. Summary
This audit evaluates the feasibility of integrating external A-LAMP code (from `external/A-Lamp_external_audit/`) into the current project to build a high-performance teacher model. The external code provides a mature implementation of **Adaptive Patch Selection** and a baseline **Multi-Patch Subnet**. External code is useful for building a Multi-Patch teacher baseline, but it is not a complete A-LAMP reproduction because the Layout-Aware subnet is missing/incomplete.

## 2. Paper Architecture Requirements
| Component | Paper Specification (CVPR 2017) | Status |
| :--- | :--- | :--- |
| **Backbone** | Heavier CNN (e.g., VGG16) | Available in `alamp_paper_ava.py` |
| **Multi-Patch Subnet** | Shared-weight CNN over multiple patches | Available in `train_complete.py` (external) |
| **Adaptive Patching** | Saliency-based subset optimization | Available in `patch_selection.py` (external) |
| **Layout-Aware Subnet**| Models spatial attributes of patches/regions | **Gap** (Fragments only) |
| **Aggregation** | Orderless Mean + Max pooling | Available in `train_complete.py` (external) |
| **Resolution** | Arbitrary-size support | Partial (via padding/resizing logic) |

## 3. External Code Inventory
- `multi_patch_subnet/adaptive_patch_selection/patch_selection.py`: Implements saliency-based optimization for 5 patches. Uses "Graph-Based Manifold Ranking".
- `multi_patch_subnet/orderless_aggregation/train_complete.py`: Keras/VGG16 implementation of the Multi-Patch subnet with Mean+Max aggregation.
- `layout_aware_subnet/functions/SOD_class.py`: Salient Object Detection (SOD) wrapper using VGG16.
- `layout_aware_subnet/output-split/`: Precomputed AVA patch pickle files exist, but usability is unverified because sample pickle loading failed with a pandas BlockManager compatibility error.

## 4. Current Mobile A-LAMP v2 Inventory
- `src/models/mobile_alamp_v2.py`: Lightweight MobileNetV3 approximation. Uses 1D attention for layout.
- `src/models/alamp_paper_ava_v2_graph_lite.py`: Contains `GraphLiteBranch`, which uses object detections to model spatial layout (possible project-specific substitute candidate, but not confirmed to match the paper’s Layout-Aware subnet).

## 5. Compatibility with AVA Manifests
- The external `patch_selection.py` outputs pickle files. These must be converted to JSONL or directly integrated into the `tf.data` pipeline.
- Current AVA manifests in `data/processed/ava/` use image IDs and paths that are compatible with the external filenames.

## 6. Adaptive Patch Selection Feasibility
- **High Feasibility.** The `patch_selection.py` script is self-contained. It can be used to generate patches for a "Heavy" teacher.
- **Precomputed Data:** Usability is currently blocked by pickle compatibility issues. If the pickle files in `output-split` can be successfully parsed, we have immediate access to adaptive patches for 250k images.

## 7. Layout-Aware Subnet Status
- **Missing/Incomplete.** The file `layout_aware_subnet.py` is empty. The provided notebook and functions focus on saliency and object proposal utilities rather than a trainable layout subnet. No complete trainable Layout-Aware subnet was confirmed in the external artifacts.
- **Action:** Must be implemented by combining holistic image features from a heavy backbone (VGG16) with spatial modeling logic.

## 8. Aggregation Layer Status
- **Implemented.** The external `train_complete.py` uses `Average` and `Maximum` layers followed by `Concatenate`. This matches the "orderless statistical aggregation" mentioned in the paper.

## 9. Reusable Components
- **Logic:** Adaptive patch selection optimization loop.
- **Data:** Precomputed AVA patches (subject to parsing verification).
- **Architecture:** `Multi_Patch_Model` structure from `train_complete.py`.

## 10. Required Rewrites
- **Data Pipeline:** Bridge the gap between pickle/notebook data loading and the project's `TrainingGenerator` or `tf.data`.
- **Layout Subnet:** A faithful implementation of spatial relationship modeling between regions/objects.
- **Integration:** Assembly of Multi-Patch and Layout-Aware subnets into a single `alamp_faithful_teacher` model.

## 11. Expected Runtime / GPU Memory
- **Model Size:** Estimated ~150M parameters (2x VGG16 + Dense aggregation layers).
- **GPU Memory:** High (~12GB-16GB) for training with batch size >= 8 due to 5 patches + 1 holistic image per sample.
- **Training Time:** Significant. Full AVA training (250k samples) on a single high-end GPU may take 2-4 days.

## 12. Recommended Next Implementation
**Experiment T2: Adaptive Patch Teacher Baseline**
First run a pre-implementation validation:
1. Verify whether precomputed pickle files can be parsed or converted (e.g., using an older pandas environment or a conversion script).
2. Verify filename/image-id alignment with our current AVA manifests.
3. Run a 1,000-image adaptive patch manifest conversion test to verify coordinate sanity.
4. Only then implement a VGG16-based Multi-Patch teacher baseline to evaluate the impact of adaptive patching.

## 13. Risks
- **IO Bottleneck:** On-the-fly adaptive patching is slow. Pre-extraction to JSONL is mandatory for efficiency.
- **Paper SOTA:** Even with faithful subnets, reaching 82.5% Accuracy requires precise hyperparameter tuning (SGD, LR decay) as seen in the external training code.

## 14. Before Implementation Checklist
- [ ] Verify pickle compatibility (resolve pandas BlockManager error).
- [ ] Verify patch coordinates match current AVA image paths and orientations.
- [ ] Verify whether an old pandas version or a specific conversion route is required.
- [ ] Double-check if the Layout-Aware subnet is truly missing from all subdirectories.
- [ ] Verify whether `GraphLiteBranch` can be considered paper-equivalent or is only a substitute.
- [ ] Verify expected GPU memory requirements for VGG16 with 5 patches at 224x224.
