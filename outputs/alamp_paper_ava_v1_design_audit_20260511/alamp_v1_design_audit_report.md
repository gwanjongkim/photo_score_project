# A-LAMP-paper-AVA-v1 Design Audit

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 2026-05-11
- Python & TF: Python 3.12 / TensorFlow 2.20.0
- Scope: Read-only audit focusing on designing `A-LAMP-paper-AVA-v1`. No modifications to practical A-LAMP models, Flutter app, or forWeights.

## 2. Current v0 Inventory
The `A-LAMP-paper-AVA-v0` track provides two models inside `src/models/alamp_paper_ava.py`:
- **`v0_a`**: An approximation of the Multi-Patch Subnet. It processes 5 patches extracted from native resolution images through a shared VGG16 backbone, applying an attention mechanism (`WeightedPatchPooling`) to merge features before a classifier.
- **`v0_b_fixed`**: Incorporates the Multi-Patch Subnet (without dynamic per-patch attention due to XLA limitations), a Global Subnet (using resize-with-pad), and a simplistic Layout-Aware Subnet (currently just dense layers over bounding box coordinates).
- Both use a `Vgg16UnitPreprocess` block and depend on `src/datasets/native_size_dataset.py` to extract patches.

**XLA Failure Fix**: The original `v0_b` used `TimeDistributed` layers and dynamic sequence attention across the patch axis. This lowered into `pfor`/`while` loops that clashed with XLA memory allocations. `v0_b_fixed` circumvented this by using explicit fixed shape flattening (`MergePatchBatch` and `RestorePatchBatch`), applying global average pooling across the patches without learned dynamic attention.

**Gaps from Paper**: 
- The paper dictates adaptive saliency-based patch extraction; v0 uses a basic deterministic approximation (edge/variance).
- The paper dictates explicit Layout Cue Augmentation using attribute graphs (object relational graphs); v0 uses none.

## 3. v0 Results and Failure Mode
Despite structural completeness, `v0_b_fixed` showed poor classification performance on AVA (ROC-AUC ~0.665).
- **Weak Negative Discrimination**: Specificity is extremely low (~0.14 for `v0_b_fixed`, ~0.12 for `v0_a`). It produces high numbers of False Positives.
- **Why?**: The simple layout branch and basic edge/color variance patch extraction fail to capture the subtle compositional flaws that make an image aesthetically poor. RGNet (using region graphs) handles this much better.

## 4. Patch Selection Audit
Located in `src/datasets/native_size_dataset.py`.
- **How patches are selected**: The script computes a synthetic saliency map directly from the image array combining Sobel edge strength, local luminance variance, and local color variance using simple heuristics (average pooling). It then searches over grid centers and predefined scales (0.22, 0.35, 0.5) to pick bounding boxes with the highest sum of synthetic saliency, applying a non-maximum suppression (NMS) based on an IoU threshold of 0.35 to enforce spatial diversity.
- **Are they adaptive?**: Yes, dynamically generated per image.
- **Do they use real saliency/objectness?**: No, it is a pure image processing heuristic, lacking semantic context.
- **Saved/Inspectable?**: The bounding boxes and saliency scores are generated on-the-fly inside the `tf.data.Dataset` pipeline. They are not saved to disk or inspectable natively without custom debug runs.

## 5. Saliency Proposal Feasibility
Based on searches (`rg -n "saliency|objectness..."`), there is **no external object detector (YOLO/COCO)** or learned saliency model integrated directly into the `photo_score_project` data pipeline.
- **A. Image Processing Saliency (Current)**: Easy, low cost, but semantically weak.
- **B. Object Detector Proposal**: Not feasible immediately without heavy dependencies and offline extraction.
- **C. Learned Saliency Map**: Not feasible; requires a separate heavy model (like a pre-trained segmentation net) just for preprocessing.
- **D. External/Offline Saliency Maps**: Feasible but high setup cost (requires running 250k AVA images through a separate detector script and saving coordinates).

## 6. Object/Global Attribute Graph Feasibility
The A-LAMP paper utilizes an Object/Global Attribute Graph to encode spatial relations and layout cues.
- **Feasibility**: **Not currently feasible.**
- Our local datasets (AVA/AADB CSVs) contain no bounding box annotations, no object labels, and no scene attributes.
- Constructing an attribute graph would be entirely speculative and computationally wasted without true object-level data. The best approximation we can achieve is feeding the coordinates of the chosen heuristic patches into the layout branch.

## 7. v1 Candidate Designs

**Candidate v1_patch**
- **Idea**: Retain the current `v0_b_fixed` model architecture but vastly improve the heuristic patch selection in `native_size_dataset.py` by incorporating a more robust OpenCV-based saliency algorithm or advanced heuristic combinations, tuning the scales and diversity thresholds.
- **Risk**: Low. Purely data pipeline changes.
- **Cost**: Low.
- **Expected Benefit**: Minimal. The bottleneck is likely semantic, not just algorithmic edge detection.

**Candidate v1_layout**
- **Idea**: Focus on the `v0_b_fixed` model architecture itself. Instead of a speculative attribute graph, enhance the global and layout branches. Explicitly feed spatial configuration metrics (patch bounding box coordinates, relative areas, distances from center) into a more sophisticated Layout-Aware Subnet (e.g., using multi-head attention over the layout coordinates) to implicitly learn layout rules without explicit object labels.
- **Risk**: Medium (model architecture tuning).
- **Cost**: Low (modifying `alamp_paper_ava.py`).
- **Expected Benefit**: Medium. Explicit attention over layout cues might improve the low specificity.

**Candidate v1_graph_lite**
- **Idea**: Speculative approximation of the paper's graph. Use the coordinates of the 5 heuristic patches as "nodes" and define edges based on spatial distances, running a simple Graph Convolution on top of them.
- **Risk**: High (speculative, unproven on heuristic patches).
- **Cost**: Medium.
- **Expected Benefit**: Unknown, possibly negative if heuristic patches are noisy.

## 8. Recommended Next Target
**Target**: `Candidate v1_layout`

## 9. Why This Target Is Chosen
The primary failure mode of `v0` is weak negative discrimination (low specificity). The current patch extraction, while heuristic, provides 5 diverse spatial views. The weakness lies in how the model understands the *arrangement* of those views. Since we lack true object detectors to build the paper's exact attribute graph (eliminating `v1_graph_lite`), the most logical step is to improve the Layout-Aware Subnet to heavily attend to the geometric properties (coordinates, relative scale) of the extracted patches. This avoids speculative external dependencies while directly targeting the model's structural awareness.

## 10. What Not To Claim
- Do not claim this is an "official reproduction" or "same as the paper". The paper relies on true object attributes and learned saliency.
- This remains an "A-LAMP-paper-oriented approximation".

## 11. Next Codex Prompt
```text
Implement the `v1_layout` candidate for the A-LAMP-paper-AVA track. Create a new model script `src/models/alamp_paper_ava_v1.py` building upon `v0_b_fixed`. Replace the simple dense layout branch with a self-attention mechanism over the layout features (patch coordinates/areas). Create a new training script `src/train/train_alamp_paper_ava_v1.py` and evaluation script. Add a new configuration `configs/paper_benchmarks/alamp_paper_ava_v1_classification.yaml`. Ensure changes are strictly isolated from practical A-LAMP models, Flutter, and RGNet. Run `py_compile`, a forward smoke test, a tiny smoke train, and finally a mid-run on the `v1_layout` design to compare its specificity against `v0_b_fixed`. Output all summaries and reports to `outputs/alamp_paper_ava_v1_layout_20260511/`.
```