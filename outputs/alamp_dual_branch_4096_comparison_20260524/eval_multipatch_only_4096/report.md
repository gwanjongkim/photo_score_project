# A-LAMP Multi-Patch Teacher Test Evaluation

## 1. Summary

- Notice: A-LAMP Multi-Patch teacher baseline, not full A-LAMP reproduction.
- Sample count: 4081
- Positive ratio: 0.712815
- Accuracy: 0.762558
- F1 / F-measure: 0.848000

## 2. Model Source

- Checkpoint type: weights_only
- Checkpoint: outputs/alamp_multipatch_teacher_full_ava_20260524/best.weights.h5
- Model variant: A-LAMP Multi-Patch teacher baseline
- Model description: VGG16 shared-patch Multi-Patch teacher with orderless mean+max aggregation
- Reproduction claim: not full A-LAMP reproduction

## 3. Test Dataset

- JSONL: outputs/alamp_dual_branch_4096_comparison_20260524/multipatch_test_4096_matched.jsonl
- Patch inputs: 5 external adaptive patch selections per image, resized to 224x224 RGB.
- Preprocessing: tf.keras.applications.vgg16.preprocess_input_rgb_float_pixels_0_255
- Label rule: mean_score > 5.0 -> 1, else 0.
- Patch boxes are external adaptive patch selections, not ground-truth labels.

## 4. Metrics

- Accuracy: 0.762558
- F1 / F-measure: 0.848000
- Precision: 0.779862
- Recall: 0.929185
- ROC-AUC: 0.791372
- Average Precision: 0.900265
- Prediction min/max/mean/std: 0.000183 / 0.999918 / 0.728571 / 0.212657

## 5. Confusion Matrix

|  | Pred 0 | Pred 1 |
|---|---:|---:|
| True 0 | 409 | 763 |
| True 1 | 206 | 2703 |

## 6. Comparison Targets

- Current Mobile A-LAMP v2 full available AVA test: Accuracy ≈ 0.7049, F1 ≈ 0.7647.
- Paper A-LAMP target: Accuracy ≈ 0.825, F-measure ≈ 0.92.
- This model is only a Multi-Patch teacher baseline, not full A-LAMP.

## 7. Judgment

- Accuracy is above 0.75, so this is a meaningful teacher candidate.
