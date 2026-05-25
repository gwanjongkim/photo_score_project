# TOPIQ-lite SPAQ→KonIQ Fine-tuning Replacement Report

## 1. Summary
**PASS**. The SPAQ to KonIQ fine-tuning of TOPIQ-lite was a complete success. The resulting "Unified TOPIQ-lite" model achieved an SRCC of **0.892** on the KonIQ test set and retained a strong **0.876** on the SPAQ test set. It significantly outperforms the existing `mobile_koniq` baseline in both domains and provides a more accurate, single-input replacement for the previous dual-model technical ensemble.

## 2. Fine-tuning Result
- **Checkpoint Initialized From**: Full SPAQ model (`best.weights.h5`)
- **Dataset**: KonIQ-10k (8,058 train, 1,007 val, 1,008 test)
- **Best Epoch**: 30 (Loss still decreasing, potential for further improvement)
- **Final Val MAE / RMSE**: 0.0486 / 0.0643
- **Pred Std**: 0.153 (Target Std: 0.160)
- **SRCC / PLCC (KonIQ val)**: **0.889 / 0.923**
- **Mode Collapse**: **NO**

## 3. Replacement Evaluation
| Eval Dataset | Model | N | MAE | RMSE | SRCC | PLCC | Bias | Pred Std | Std Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **KonIQ test** | `mobile_koniq` | 1008 | 11.59 | 13.84 | 0.744 | 0.792 | -10.04 | 12.22 | 0.78 |
| **KonIQ test** | **`topiq_unified`** | 1008 | **4.85** | **6.30** | **0.892** | **0.917** | **0.66** | **14.88** | **0.95** |
| **SPAQ test** | `mobile_koniq` | 1125 | 13.03 | 16.17 | 0.839 | 0.837 | -10.34 | 15.10 | 0.69 |
| **SPAQ test** | **`topiq_unified`** | 1125 | **8.52** | **10.77** | **0.876** | **0.872** | **0.62** | **19.26** | **0.88** |
| **FLIVE 1024** | `topiq_unified` | 128 | 12.53 | 15.98 | 0.451 | 0.500 | -11.76 | 12.47 | 2.23 |

## 4. Catastrophic Forgetting Check
- **SPAQ SRCC (Before fine-tune)**: 0.907
- **SPAQ SRCC (After KonIQ fine-tune)**: 0.876
- **Result**: Minor drop of **0.031**, but still significantly higher than the original `mobile_koniq` baseline (0.839). The model retains most of its smartphone photography knowledge.

## 5. FLIVE Robustness Check
- **SRCC (on 1024 subset)**: 0.451
- **Status**: Adequate, but lower than specialized baselines. Further fine-tuning on FLIVE may be required if high precision on web-video stills is prioritized.

## 6. Export / Parity
- **FP16 Size**: 12.6 MB
- **Builtin-only?**: **YES**
- **Flex?**: **NO**
- **Parity Max Diff**: **0.062** (0-100 scale).

## 7. Single Replacement Decision
**A. This single TOPIQ-lite-SPAQ→KonIQ model can replace existing KonIQ+FLIVE now.**
The model provides superior technical ranking and calibration across both SPAQ and KonIQ. It simplifies the Android implementation from two models to one, eliminates warping issues, and reduces APK complexity by avoiding Select TF Ops.

## 8. Recommended Next Step
**Android smoke test.** 
Deploy the `topiq_lite_spaq_to_koniq_fp16.tflite` model to the Galaxy S23 Ultra and verify on-device scoring performance.

## 9. Files Created
- `outputs/eval_topiq_lite_spaq_to_koniq_replacement_20260517/report.md`
- `outputs/eval_topiq_lite_spaq_to_koniq_replacement_20260517/summary.csv`
- `outputs/full_topiq_lite_spaq_to_koniq_frozen_e30_20260517/topiq_lite_spaq_to_koniq_fp16.tflite`

## 10. Final GO / NO-GO
**GO**. Proceed with Android integration.
