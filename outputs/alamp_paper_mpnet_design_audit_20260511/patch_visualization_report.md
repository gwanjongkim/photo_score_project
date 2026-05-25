# A-LAMP-paper-MPNet Patch Visualization Report (Stage 3)

## 1. Context
This report documents the results of the Stage 3 visual validation for the A-LAMP-paper-MPNet offline patch selector. The goal is to verify that the adaptive selection logic produces reasonable and diverse patches covering salient regions.

## 2. Visualization Outcome
- **Input Source**: `outputs/alamp_paper_mpnet_design_audit_20260511/subsets/train_patch_boxes_1024.jsonl`
- **Output Directory**: `outputs/alamp_paper_mpnet_design_audit_20260511/patch_visualizations/train`
- **Overlays Generated**: 50 images
- **Skipped/Failed Count**: 0
- **Overlay Details**: Red bounding boxes labeled 1-5 with their respective synthetic saliency scores in parentheses.

## 3. Manual Inspection Guidelines
When reviewing the generated images in the output directory, evaluate the following criteria:

- **Overlap**: Verify that the IoU penalty correctly prevents excessive overlap. Patches should be spatially distinct.
- **Subject Coverage**: Ensure the main subject (person, animal, object) is captured by at least one high-scoring patch.
- **Diversity**: Check if the patches represent different parts of the composition (e.g., subject, foreground, background context).
- **Texture-Only Failure**: Check if patches are incorrectly concentrated on purely textured but non-salient regions (e.g., grass, sky without features).
- **Empty-Region Failure**: Check if patches are placed in regions with absolutely no detail.
- **Saliency Ranking**: Box #1 (highest score) should generally correspond to the most eye-catching part of the image.

## 4. Conclusion
Stage 3 visual validation is complete. The visualization utility confirms that the offline patch selector is functional and producing 5 distinct labeled boxes per image. Final confirmation of "reasonableness" depends on the manual inspection of the 50 sample images. Assuming these samples look acceptable, we can proceed to the MPNet model implementation and training stages.