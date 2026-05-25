# A-LAMP Layout-Aware Teacher Strengthening Plan

## 1. Summary
The current A-LAMP teacher baseline is a Multi-Patch-only model (Accuracy ≈ 0.763), which lacks the "Layout-Aware Subnet" that is central to the A-LAMP paper's performance (Accuracy ≈ 0.825). To bridge this ~6% accuracy gap, we must integrate a relational layout branch—ideally a Graph Convolutional Network (GCN)—that captures spatial relationships between salient objects. This plan proposes three strengthening paths, with a recommendation to upgrade the current Multi-Patch teacher to a full Dual-Branch GCN architecture leveraging existing project components.

## 2. Current Teacher Strength
- **Model**: VGG16 shared-patch Multi-Patch teacher with orderless mean+max aggregation.
- **Accuracy**: 0.7633 (on full AVA test set).
- **F1 Score**: 0.8482.
- **Positive Ratio**: 0.71 (high recall bias: 0.93).
- **Limitation**: Uses external adaptive patches but ignores global composition and object-level layout relationships.

## 3. Gap to Paper Target
- **Current Accuracy**: 0.763.
- **A-LAMP Paper Target**: 0.825.
- **Absolute Gap**: +6.17%.
- **Diagnosis**: The current model is "blind" to layout. The paper claims that adaptive multi-patching and layout-aware GCN are synergistic. We have the patches (via external JSONLs) but not the GCN.

## 4. External Layout-Aware Code Status
- **Location**: `external/A-Lamp_external_audit/A-Lamp/layout_aware_subnet/`
- **Findings**: 
    - `SOD_class.py`: Implementation of Salient Object Detection (SOD).
    - `salient_object_detection.ipynb`: Demonstration of saliency map generation.
- **Status**: The code for feature extraction (SOD) exists but is not integrated into a trainable TensorFlow pipeline within the project. It currently serves as a reference for offline feature generation.

## 5. GraphLiteBranch Feasibility
- **File**: `src/models/alamp_paper_ava_v2_graph_lite.py`
- **Assessment**:
    - `GraphLiteBranch` is a simplified MLP-based pooling of object geometric features (boxes, centers, areas).
    - It **lacks message passing** (GCN), which is critical for modeling "relational aesthetics" (e.g., rule of thirds, balance).
    - It is a good "lite" substitute for mobile inference but insufficient for a high-performance teacher.
- **Recommendation**: Use `GraphLiteBranch` as a baseline for layout, but upgrade to a proper GCN for the teacher.

## 6. Teacher Strengthening Options

| Option | Description | Performance Gain | Difficulty | Risk | Reaches 0.80+? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Lightweight Layout-Aware** | Add a global branch and a simple MLP layout branch (like GraphLite) to the current Multi-Patch teacher. | +1-2% Acc | Low | Low (incremental) | Unlikely (≈0.78) |
| **B. Dual-Branch GCN (Paper-Near)** | Implement a full Layout-Aware GCN branch using `GraphConvolutionLayer` (from `alamp_paper_fusion.py`) on salient objects. | +3-5% Acc | High | High (Data prep) | **Yes (0.80-0.81)** |
| **C. Ensemble Teacher** | Combine current Multi-Patch (0.763), Mobile A-LAMP v2 (0.705), and RGNet (GCN-based). | +2-3% Acc | Medium | Medium (Complex distill) | Possible (≈0.79) |

## 7. Recommended Next Experiment
**Upgrade the Teacher to a "Dual-Branch GCN" model.**
1. **Backbone**: Retain VGG16 (shared) for patches.
2. **Layout Branch**: Implement a GCN branch using the `GraphConvolutionLayer` found in `src/models/alamp_paper_fusion.py`.
3. **Inputs**: Use the existing `graph_jsonl` (object boxes/categories) or generate new salient object features using the external SOD code.
4. **Fusion**: Concatenate Multi-Patch features, Global features, and GCN-based Layout features before the final classification head.

## 8. Success Criteria
- **Weak Success**: Accuracy > 0.7633 (Improved over current baseline).
- **Strong Teacher**: Accuracy >= 0.80 (Crosses the psychological barrier for "strong" aesthetics).
- **Paper-Near**: Accuracy >= 0.815 (Within 1% of paper results).
- **Paper Target**: Accuracy ≈ 0.825 (Full parity).

## 9. Distillation Readiness
- **Status**: **NOT READY.**
- **Reasoning**: Distilling from a 0.763 teacher will likely result in a 0.70-0.72 student (Mobile A-LAMP v2 is already at 0.70). We need a teacher at 0.80+ to significantly improve the mobile student's quality.

## 10. Risks
- **Overclaiming Reproduction**: Unless we use the exact same SOD and GCN parameters as the paper, we should continue to label results as "A-LAMP-oriented approximation."
- **Data Bottleneck**: Generating salient object features for the full AVA dataset (250k images) is computationally expensive. We should start with the existing 4096-sample subsets for prototyping.
- **GPU Memory**: A dual-branch VGG16 model with 5+ patches and a GCN will require ~16GB+ VRAM for training if not optimized.
