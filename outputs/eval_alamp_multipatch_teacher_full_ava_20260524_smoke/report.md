# A-LAMP Multi-Patch Teacher Test Evaluation

## 1. Summary

- Notice: A-LAMP Multi-Patch teacher baseline, not full A-LAMP reproduction.
- Sample count: 128
- Positive ratio: 0.679688
- Accuracy: 0.835938
- F1 / F-measure: 0.886486

## 2. Model Source

- Checkpoint type: weights_only
- Checkpoint: outputs/alamp_multipatch_teacher_full_ava_20260524/best.weights.h5
- Model variant: A-LAMP Multi-Patch teacher baseline
- Model description: VGG16 shared-patch Multi-Patch teacher with orderless mean+max aggregation
- Reproduction claim: not full A-LAMP reproduction

## 3. Test Dataset

- JSONL: outputs/alamp_external_patch_full_conversion_20260524/alamp_external_patches_test.jsonl
- Patch inputs: 5 external adaptive patch selections per image, resized to 224x224 RGB.
- Preprocessing: tf.keras.applications.vgg16.preprocess_input_rgb_float_pixels_0_255
- Label rule: mean_score > 5.0 -> 1, else 0.
- Patch boxes are external adaptive patch selections, not ground-truth labels.

## 4. Metrics

- Accuracy: 0.835938
- F1 / F-measure: 0.886486
- Precision: 0.836735
- Recall: 0.942529
- ROC-AUC: 0.885618
- Average Precision: 0.942502
- Prediction min/max/mean/std: 0.006845 / 0.994494 / 0.664138 / 0.233058

## 5. Confusion Matrix

|  | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 25 | 16 |
| True 1 | 5 | 82 |

## 6. Comparison Targets

- Current Mobile A-LAMP v2 full available AVA test: Accuracy ≈ 0.7049, F1 ≈ 0.7647.
- Paper A-LAMP target: Accuracy ≈ 0.825, F-measure ≈ 0.92.
- This model is only a Multi-Patch teacher baseline, not full A-LAMP.

## 7. Judgment

- Accuracy is near the paper A-LAMP target, so this is paper-target-adjacent, but it is still not full A-LAMP without the Layout-Aware subnet.
