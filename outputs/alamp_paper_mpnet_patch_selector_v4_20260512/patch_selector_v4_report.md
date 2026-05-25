# A-LAMP-paper-MPNet Patch Selector v4 Report

## 1. Context
Patch Selector V3 (U²-Net grid scoring) failed the manual quality gate, capturing meaningful subjects in only ~20/50 images. V4 was implemented to directly leverage the geometry of salient components identified by U²-Net.

## 2. Improvements in v4
- **Component-Based Placement**: Instead of scoring a predefined grid, V4 extracts connected components (contours) from the U²-Net saliency mask and centers patches on them.
- **Role-Aware Selection**: Explicitly allocates patches to specific roles:
  - `subject_closeup`: Tight fit around the primary salient component.
  - `subject_secondary`: Centered on the next most salient component (if available).
  - `subject_context`: A larger view expanding around the primary subject.
  - `diverse_context`: Background/Context patch selected via pattern diversity (Lab color/edge histograms).
  - `spatial_context`: Region spatially distant from the main subject.
- **Robust Diversity**: Remaining slots are filled using the V3 weighted objective to ensure no overlaps and broad coverage.

## 3. Results (1024 train images)
- **Object Coverage**: **57.3%** (V3: 8.4%). This is a **+48.9% absolute increase** in capturing YOLO-detected subjects.
- **Main Object Coverage**: **64.3%** (V3: 8.0%). This proves that V4 is highly effective at centering on the primary salient element.
- **Mean Pairwise IoU**: 0.0589 (V3: 0.0258). Overlap increased slightly as expected (closeup + context roles), but remains low.
- **Mean Patch Score**: 0.6168 (V3: 0.5742).

## 4. Manual Quality Gate
**IMPORTANT**: MPNet training is allowed **ONLY** if manual inspection of the 50 overlays in `outputs/alamp_paper_mpnet_patch_selector_v4_20260512/patch_visualizations/train/` finds about **40/50 acceptable images**. 
Acceptable means:
- At least one patch is clearly centered on the main subject.
- Patches are diverse (not all clustered on one spot).
- At least one patch represents the background/context.

## 5. Conclusion
V4 represents a fundamental shift from "blind sampling" to "component-aware extraction". The quantitative jump in object coverage suggests that V4 is the most promising candidate for reproducing the A-LAMP Multi-Patch Subnet.
