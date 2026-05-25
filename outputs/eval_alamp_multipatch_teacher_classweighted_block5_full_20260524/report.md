# A-LAMP Multi-Patch Teacher Test Evaluation

## 1. Summary

- Notice: A-LAMP Multi-Patch teacher baseline, not full A-LAMP reproduction.
- Sample count: 25443
- Positive ratio: 0.710922
- Accuracy: 0.754314
- F1 / F-measure: 0.825873

## 2. Model Source

- Checkpoint type: weights_only
- Checkpoint: outputs/alamp_multipatch_teacher_classweighted_block5_full_20260524/best.weights.h5
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

- Accuracy: 0.754314
- F1 / F-measure: 0.825873
- Precision: 0.832295
- Recall: 0.819549
- Specificity / negative recall: 0.593882
- Balanced accuracy: 0.706715
- ROC-AUC: 0.792218
- Average Precision: 0.894629
- Predicted positive ratio: 0.700035
- Prediction min/max/mean/std: 0.001182 / 1.000000 / 0.625077 / 0.239200

## 5. Confusion Matrix

|  | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 4368 | 2987 |
| True 1 | 3264 | 14824 |

## 6. Comparison Targets

- Current Mobile A-LAMP v2 full available AVA test: Accuracy ≈ 0.7049, F1 ≈ 0.7647.
- Paper A-LAMP target: Accuracy ≈ 0.825, F-measure ≈ 0.92.
- This model is only a Multi-Patch teacher baseline, not full A-LAMP.

## 7. Judgment

- Accuracy is above 0.75, so this is a meaningful teacher candidate.
