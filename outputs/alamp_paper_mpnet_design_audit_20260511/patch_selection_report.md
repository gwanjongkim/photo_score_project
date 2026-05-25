# A-LAMP-paper-MPNet Patch Selection Report

## 1. Context and Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Objective: Implement Stage 1 and Stage 2 of the A-LAMP-paper-MPNet validation plan by extracting adaptive patches offline before model training.
- Note: This is an A-LAMP-paper-oriented approximation, not an official reproduction. No modifications were made to practical models or Flutter code.

## 2. Patch Selector Implementation
The offline patch selector (`src/datasets/alamp_paper_patch_selector.py`) approximates the A-LAMP adaptive selection:
- **Candidate Generation**: Generates 224x224 crops using multiple dynamic scales relative to the short edge (0.22, 0.35, 0.50).
- **Saliency Approximation**: As the exact paper saliency was not available locally, it computes a synthetic score using a weighted combination of Sobel edge density and local variance (blur).
- **Diversity / Overlap Constraint**: Iteratively selects 5 patches, penalizing remaining candidate scores based on their Intersection over Union (IoU) overlap with already selected boxes to enforce spatial diversity.

## 3. Smoke Validation (Stage 1)
- The selector was executed on a 10-image subset from the training CSV.
- Generated `outputs/alamp_paper_mpnet_design_audit_20260511/smoke_patches.jsonl`.
- Validations checked: All records contained exactly 5 boxes, normalized coordinates `[0,1]`, and valid box structures (`x1 < x2`, `y1 < y2`).
- **Status**: Passed (10/10 valid images).

## 4. Subset Generation (Stage 2)
The offline script successfully generated candidate patches for the 1024-image subsets across train, validation, and test splits.
- `outputs/alamp_paper_mpnet_design_audit_20260511/subsets/train_patch_boxes_1024.jsonl`
  - Records: 1024
  - Skipped: 0
  - Average Patch Score: 0.1095
- `outputs/alamp_paper_mpnet_design_audit_20260511/subsets/val_patch_boxes_1024.jsonl`
  - Records: 1024
  - Skipped: 0
  - Average Patch Score: 0.1086
- `outputs/alamp_paper_mpnet_design_audit_20260511/subsets/test_patch_boxes_1024.jsonl`
  - Records: 1024
  - Skipped: 0
  - Average Patch Score: 0.1092

## 5. Conclusion
Stages 1 and 2 of the MPNet validation plan are complete. The generated JSONL artifacts are structurally valid and correctly approximate the saliency and diversity constraints. They are now ready to be consumed by the upcoming `A-LAMP-paper-MPNet` model training pipeline.