# Student Training Manifest Design Report

## 1. Executive Summary
This report outlines the design for the A-cut mobile technical IQA student training manifest. Following the successful 9,000-image dry-run for guarded TOPIQ teacher labels, we have audited the resulting data for integrity and leakage risks. We confirm that the dry-run set is free of overlap with previously used validation and holdout sets. We recommend a **group-based, stratified 70/15/15 split** policy for the student model training. The first training experiment will utilize the 9,000 dry-run images to establish a baseline before expanding to the full 159k dataset.

## 2. Input Files Inspected
- `outputs/topiq_teacher_label_dryrun_20260601/guarded_labels.csv` (Primary source for guarded targets)
- `outputs/topiq_teacher_validation_20260531/validation_manifest.csv` (Leakage check reference)
- `outputs/topiq_teacher_guard_holdout_20260601/holdout_manifest.csv` (Leakage check reference)

## 3. Guarded Label Dry-Run Status
- **Row Count**: 9,000 images.
- **Dataset Distribution**: Perfectly balanced (3,000 rows each for KonIQ-10k, SPAQ, and FLIVE).
- **Integrity**: 0 missing values, 0 duplicate image paths.
- **Score Ranges**:
  - `guarded_target_0_100`: [12.75, 87.60] (Mean: 62.00)
  - `calibrated_topiq_0_100`: [12.80, 81.90]
  - `quality_score_0_100` (MOS): [2.00, 94.70]
- **Observations**: The guarded target successfully blends the high-fidelity teacher (calibrated TOPIQ) with the ground-truth MOS, effectively regularizing the student labels.

## 4. Leakage Check
- **Validation Overlap**: 0 images.
- **Holdout Overlap**: 0 images.
- **Verdict**: The dry-run dataset is safely isolated from the calibration validation and holdout evaluation sets.

## 5. Group Leakage Risk Analysis
- **Finding**: 95 source groups were identified with multiple images (2 or 3 images per group).
- **Context**: This primarily affects the **FLIVE/PaQ-2-PiQ** dataset where multiple patches are extracted from a single source image.
- **Safe Group Key**:
  - For paths containing `patches/`: Extract the base image name (e.g., `patches/AVA__123.jpg_patch_1.jpg` -> `AVA__123.jpg`).
  - For others: Use the filename.
- **Requirement**: Splitting MUST be performed at the `source_group` level to prevent the model from seeing different parts of the same image in both training and test sets.

## 6. Recommended Split Policy
- **Ratio**: 70% Train / 15% Val / 15% Test.
- **Mechanism**: **Group-based Split** using the `source_group` key.
- **Stratification**:
  - **Dataset**: Ensure roughly equal representation of KonIQ, SPAQ, and FLIVE in each split.
  - **Score Bins**: Use decile-based stratification on `guarded_target_0_100` to ensure uniform quality distribution across splits.
- **Evaluation Priority**: The `test` split within this manifest will serve as the primary "Distillation Parity" metric. The 900-image `holdout_manifest.csv` remains the "Golden Truth" benchmark.

## 7. Recommended Student Manifest Schema
The final manifest (`student_training_manifest.csv`) should include:
- `image_path`: Absolute or repo-relative path.
- `dataset`: Source dataset name.
- `source_group`: Key used for splitting.
- `guarded_target_0_100`: The primary training label.
- `student_split`: 'train', 'val', or 'test'.
- `quality_score_0_100`: Reference MOS (for analysis only).
- `calibrated_topiq_0_100`: Calibrated teacher score (for analysis only).

## 8. Recommended First Training Dataset
- **Size**: 9,000 rows (approx. 6,300 train / 1,350 val / 1,350 test).
- **Rationale**: Training on the dry-run set is the fastest way to verify the student model architecture and convergence. Labeling 150k+ images is computationally expensive and should only be done after the student pipeline is proven effective on the 9k subset.

## 9. Evaluation Sets
1. **Student Internal Test (1.35k)**: Quick feedback during training.
2. **Standard Research Test Split**: Re-evaluate on KonIQ Run_01 Test, SPAQ Test, and FLIVE Image-Test.
3. **A-cut Hard-Case Test**: Evaluate on the 900-image holdout set to specifically measure "semantic bias" reduction.

## 10. Risks and Blockers
- **Small Dataset Size**: 6.3k training images might be too few for a high-capacity student to generalize. If SRCC on the internal test set is < 0.8, full dataset labeling becomes mandatory.
- **Input Size Sensitivity**: The teacher saw 384x384. If the student sees 224x224, there may be a performance drop.

## 11. Final Recommendation
**READY_FOR_CODEX_MANIFEST_IMPLEMENTATION**

The design is sound, leakage is verified as zero, and the split strategy is robust against group-based leakage.

## 12. Next Codex Prompt
"Implement the student training manifest generation script. Use `outputs/topiq_teacher_label_dryrun_20260601/guarded_labels.csv` as input. Derive a `source_group` key to handle patches. Perform a 70/15/15 group-based split, stratified by dataset and decile score bins. Output the result to `outputs/student_training_manifest_design_20260601/student_training_manifest.csv`."
