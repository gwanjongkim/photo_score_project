Implement A-LAMP-paper-MPNet Patch Selector V3 using a salient object detection model.

**Goal**: Improve subject coverage and diversity using high-quality saliency maps.

**Core Requirements**:
1. **Model Selection**: Use a pretrained **U²-Net** model (Apache 2.0 license) to generate saliency maps.
2. **Saliency Map Generation**:
   - Create a standalone script/module to run U²-Net on the 1024 subset.
   - Save the saliency maps as 1-channel grayscale PNGs or compressed NPZ.
3. **Selector V3 Implementation**:
   - Update `src/datasets/alamp_paper_patch_selector.py` to support `--selector_version v3`.
   - Score candidate patches based on the **saliency mass** (average pixel value) in the U²-Net mask.
   - Combine with **V2 pattern diversity** (Lab color/edge histograms) and **spatial constraints** in the weighted objective.
4. **Isolated Output**:
   - Save to `outputs/alamp_paper_mpnet_patch_selector_v3_YYYYMMDD/`.
5. **Quality Comparison**:
   - Automatically compare V1, V2, and V3.
   - Metrics: Object coverage (YOLO IoU > 0.3), Saliency lift, Pairwise IoU.
6. **Visualization**:
   - Generate 50 overlay images for manual inspection.

**Constraints**:
- Do not train MPNet yet.
- Do not modify Flutter or forWeights.
- Keep V1/V2 artifacts untouched.
- Clearly document the saliency model source and license.
