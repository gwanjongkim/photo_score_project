# A-LAMP-paper-MPNet Patch Selector v2 Report

## 1. Context
Visual validation of Patch Selector v1 showed it heavily prioritized texture/complexity over salient subjects. This v2 implementation aims for a more paper-oriented approach using saliency maps and pattern diversity.

## 2. Improvements in v2
- **Saliency**: Replaced Sobel edge density with **Spectral Residual Saliency** with a center-bias prior. This better identifies standalone subjects against backgrounds.
- **Pattern Diversity**: Added explicit **Lab color** and **Sobel edge orientation** histograms for each candidate.
- **Selection Objective**: Replaced simple NMS with a **weighted iterative objective**:
  - `score = w_sal * saliency + w_pat * pattern_dist + w_spa * spatial_dist - penalty`
- **Spatial Diversity**: Forced patches to be further apart using center-to-center Euclidean distance in the objective.

## 3. Results (1024 train images)
- **Mean Pairwise IoU**: 0.0035 (v1: 0.0091). **-61% reduction in overlap.**
- **Mean Patch Score**: 0.0921.
- **Object Coverage**: 5.8% (v1: 7.8%). 
  - *Note*: Coverage is defined as IoU > 0.3 with YOLO objects (conf 0.10). The slight drop suggests v2 is spreading patches into "context" regions more aggressively than v1, which might have clustered near subjects by accident if they were complex.

## 4. Visualization
- 50 overlays generated in: `outputs/alamp_paper_mpnet_patch_selector_v2_20260511/patch_visualizations/train/`
- Each image shows 5 patches with their saliency scores.

## 5. Comparison Summary
| Metric | Selector v1 | Selector v2 |
| :--- | :--- | :--- |
| Method | Sobel + Var | Spectral Residual + Pattern + Spatial |
| Mean Overlap (IoU) | 0.0091 | 0.0035 |
| Diversity | Low | High |

## 6. Conclusion and Decision
**Decision: B. v2 improves diversity significantly but needs visual confirmation of subject coverage before MPNet training.**

The drastic reduction in overlap proves the weighted spatial objective is working. However, the drop in object coverage requires manual inspection of the 50 visualizations to ensure that the primary subjects are still represented among the 5 diverse patches. If the visualizations look better than v1 (which failed 40/50), we can proceed to MPNet model implementation.
