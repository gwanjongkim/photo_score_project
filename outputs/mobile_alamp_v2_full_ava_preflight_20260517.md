# Mobile A-LAMP v2 Full-AVA Preflight Audit

## 1. Summary
The project has successfully validated the Mobile A-LAMP v2 architecture on a 4,096-sample subset (ROC-AUC 0.75), and the resulting TFLite model is currently operational in the Flutter app. However, full-scale training on the ~230,000 image AVA dataset is currently **BLOCKED** by the absence of full-dataset patch JSONLs. 

The audit confirms that while raw AVA images and labels are present, multi-stage preprocessing (U2Net saliency generation and V4 patch selection) must be completed before training can commence.

## 2. Current Mobile A-LAMP v2 Status
- **Baseline**: Trained on 4,096 V4 subset.
- **TFLite Path**: `outputs/mobile_alamp_v2_pretrained_4096_tflite/mobile_alamp_v2_fp16.tflite`
- **In-App Status**: Active (Ensemble weight 3).
- **Metric Target**: Break ROC-AUC 0.80 on full AVA val split.

## 3. Existing Patch JSONL Status
- **1024 Subset**: Present (Train/Val/Test).
- **4096 Subset**: Present (Train/Val/Test).
- **Full AVA (~255k total)**: **MISSING**.

## 4. Full AVA Dataset Readiness
- **Raw Images**: Confirmed (255,509 images in `data/raw/ava/images`).
- **Processed Labels**: Confirmed (`data/processed/ava/train.csv`, `val.csv`, `test.csv`).
- **Splits**: Ready (Train ~204k, Val ~25k, Test ~25k).

## 5. Required Preprocessing
Two sequential stages are required to generate the training manifests:
1. **Stage 1 (U2Net Saliency)**: Run `tools/generate_u2net_saliency_maps.py` across all AVA images.
   - *Estimated Time*: ~2.5 hours on 4070 SUPER.
   - *Storage*: ~5.2 GB for saliency PNGs.
2. **Stage 2 (V4 Patch Selection)**: Run `src/datasets/alamp_paper_patch_selector.py` with `--selector_version v4`.
   - *Estimated Time*: ~6.5 hours.

## 6. Patch Generation Command (Proposed)
```bash
# 1. Saliency Generation (Example for Train)
./.venv_gpu/bin/python tools/generate_u2net_saliency_maps.py \
    --csv data/processed/ava/train.csv \
    --output_dir outputs/alamp_v4_full/saliency_maps/train

# 2. V4 Patch Selection
./.venv_gpu/bin/python src/datasets/alamp_paper_patch_selector.py \
    --csv data/processed/ava/train.csv \
    --saliency_map_jsonl outputs/alamp_v4_full/saliency_maps/train/saliency_metadata.jsonl \
    --selector_version v4 \
    --output_jsonl outputs/alamp_v4_full/subsets/train_patch_boxes_full_v4.jsonl
```

## 7. Validation Checks
After patch generation, verify the following:
- Line counts in `train_patch_boxes_full_v4.jsonl` match `data/processed/ava/train.csv` (204,407).
- Run `tools/visualize_alamp_patch_boxes.py` on 50 random samples to ensure subject centering.
- Check for zero-score patches (indicating saliency mask failures).

## 8. Smoke Training Command
```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py \
    --train_patch_jsonl outputs/alamp_v4_full/subsets/train_patch_boxes_full_v4.jsonl \
    --val_patch_jsonl outputs/alamp_v4_full/subsets/val_patch_boxes_full_v4.jsonl \
    --out_dir outputs/mobile_alamp_v2_full_ava_smoke \
    --max_train_samples 128 \
    --max_val_samples 64 \
    --epochs 1 \
    --smoke
```

## 9. Full Training Command
```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/train/train_mobile_alamp_v2.py \
    --train_patch_jsonl outputs/alamp_v4_full/subsets/train_patch_boxes_full_v4.jsonl \
    --val_patch_jsonl outputs/alamp_v4_full/subsets/val_patch_boxes_full_v4.jsonl \
    --out_dir outputs/mobile_alamp_v2_full_ava_20260517 \
    --epochs 20 \
    --batch_size 16 \
    --save_model
```

## 10. Success Criteria
- **Validation ROC-AUC**: > 0.77 (Exceeding the 4096-subset performance).
- **Precision/Recall Balance**: Healthy trade-off at 0.5 threshold.
- **TFLite Conversion**: Successful FP16 export with parity `< 1e-4`.

## 11. Risks
- **Data Corruption**: Ensure saliency generation completes without `cv2.imread` failures across 230k files.
- **Training Time**: Full AVA training (20 epochs) may take 24-48 hours.
- **Overfitting**: Early stopping at 20 epochs might be aggressive; monitor validation curve closely.

## 12. Final Go / No-Go Judgment
**CONDITIONAL GO after patch JSONL generation.**
The training pipeline is fully operational, but cannot start until the multi-stage saliency and patch extraction process is completed for the full dataset.
