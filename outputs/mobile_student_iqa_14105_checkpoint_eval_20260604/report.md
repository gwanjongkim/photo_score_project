# Mobile Student IQA 14105 Checkpoint Evaluation Report

## 1. Executive Summary
Phase 11 checkpoint evaluation for the 14,105-row mobile student IQA model is complete. The `best_by_val_mae` checkpoint (identical to the final model) achieved an overall test SRCC of **0.6945** and MAE of **4.7963**, meeting the baseline quality targets. However, dataset-wise analysis reveals a significant performance gap: while KonIQ (0.72) and SPAQ (0.89) are strong, **FLIVE remains weak at 0.5176**. 

**Final Decision: BEST_CHECKPOINT_NEEDS_FLIVE_HYBRID_PLAN**

## 2. Sources Inspected
- `outputs/mobile_student_iqa_14105_full_controlled_20260604/report.md`
- `outputs/student_distillation_manifest_14105_20260603/test_manifest.csv`
- `outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/checkpoint_metrics.csv`

## 3. Checkpoints Evaluated
- `best_by_val_mae_model.keras` (Best MAE at final epoch 25)
- `final_model.keras` (Identical to above)

## 4. Overall Metrics
| checkpoint | split | n | mae_0_100 | rmse_0_100 | srcc | plcc |
| --- | --- | --- | --- | --- | --- | --- |
| best_by_val_mae | test | 1413 | 4.7963 | 6.4227 | 0.6945 | 0.8423 |
| best_by_val_mae | val | 1409 | 4.8010 | 6.4158 | 0.7147 | 0.8291 |

## 5. Dataset-wise Metrics (Test Set)
| dataset | n | mae_0_100 | rmse_0_100 | srcc | plcc |
| --- | --- | --- | --- | --- | --- |
| flive | 399 | 8.9626 | 9.7190 | 0.5176 | 0.5168 |
| koniq10k | 901 | 2.7239 | 3.4970 | 0.7230 | 0.7572 |
| spaq | 113 | 6.6091 | 9.2075 | 0.8984 | 0.8785 |

## 6. Safety and Failure Cases
- **Severe FP**: 0 (MOS < 40 and pred > 65)
- **FP**: 1 (SPAQ test: pred > 60 when MOS < 40)
- **FN**: 0
- **Failure Rows**: 0 (All images processed successfully)
- **Collapse**: False (Std dev of predictions is 8.08, indicating healthy variance)

## 7. Prediction Distribution
- **Pred Min**: 11.62
- **Pred Max**: 76.83
- **Pred Mean**: 61.11
- **Pred Std**: 8.08

## 8. Best Checkpoint Selection
`best_by_val_mae_model.keras` is selected as the primary candidate. It demonstrates high stability and captures KonIQ/SPAQ nuances extremely well.

## 9. FLIVE Weakness / Hybrid Runtime Consideration
The student model struggles with the FLIVE dataset (SRCC 0.51). Given the high performance on other technical datasets, this model is recommended for general-purpose technical quality scoring, but should likely be paired with the existing production model or a separate FLIVE robustness signal for crowdsourced content.

## 10. Final Decision
**BEST_CHECKPOINT_NEEDS_FLIVE_HYBRID_PLAN**

## 11. Recommended Next Step
Proceed to **Phase 12: TFLite Export and Hybrid Runtime Design**.
1. Export `best_by_val_mae_model.keras` to FP16 TFLite for mobile profiling.
2. Design a runtime strategy that combines this new student's precision on technical degradation with production-level robustness on diverse FLIVE-style imagery.
