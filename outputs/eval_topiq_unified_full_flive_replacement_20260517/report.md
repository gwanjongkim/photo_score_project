# TOPIQ-lite Unified vs Existing FLIVE Mobile Full FLIVE Evaluation

## 1. Summary
**FAIL (Replacement Not Proven)**

TOPIQ-lite Unified 모델은 기존의 `mobile_koniq` 보다는 FLIVE 데이터셋에서 더 나은 성능을 보이지만, 전용 모델인 `mobile_flive`에 비해서는 성능이 크게 떨어집니다. 특히 MAE가 약 3배(3.1 vs 10.1) 높고, 상관계수(SRCC/PLCC) 또한 유의미하게 낮습니다. 따라서 현재 상태로는 `mobile_flive`를 대체할 수 없습니다.

## 2. Models Evaluated
| Model | Type | Path | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| mobile_flive | TFLite | exports/tflite/flive_image_mobile.tflite | 224x224, /255.0 | MOS [0, 100] |
| mobile_koniq | TFLite | exports/tflite/koniq_mobile.tflite | 224x224, /255.0 | MOS [0, 100] |
| topiq_unified | TFLite FP16 | outputs/full_topiq_lite_spaq_to_koniq_frozen_e30_20260517/topiq_lite_spaq_to_koniq_fp16.tflite | 384x384, pad, 0..255 | Norm MOS [0, 1] |

## 3. FLIVE Splits
| Split | CSV | N | MOS range | MOS mean/std |
| :--- | :--- | :--- | :--- | :--- |
| flive_val | data/processed/flive/image_val.csv | 3981 | 9.66 - 90.53 | 72.05 / 6.37 |
| flive_test | data/processed/flive/image_test.csv | 3981 | 13.60 - 91.72 | 72.18 / 6.05 |

## 4. Metrics Summary
| Split | Model | N | MAE | RMSE | SRCC | PLCC | Bias | Pred Std | Std Ratio | Avg ms/img |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| flive_val | mobile_flive | 3981 | **3.14** | **4.20** | **0.608** | **0.757** | 0.58 | 4.67 | 0.73 | 6.69 |
| flive_val | mobile_koniq | 3981 | 13.60 | 16.09 | 0.397 | 0.496 | -13.16 | 10.58 | 1.66 | 6.55 |
| flive_val | topiq_unified | 3981 | 10.19 | 12.77 | 0.458 | 0.544 | -9.29 | 10.42 | 1.63 | 37.29 |
| flive_test | mobile_flive | 3981 | **3.06** | **4.05** | **0.622** | **0.750** | 0.57 | 4.35 | 0.72 | 6.63 |
| flive_test | mobile_koniq | 3981 | 13.69 | 16.17 | 0.419 | 0.513 | -13.33 | 10.64 | 1.76 | 6.48 |
| flive_test | topiq_unified | 3981 | 10.04 | 12.65 | 0.488 | 0.565 | -9.26 | 10.44 | 1.73 | 36.60 |

## 5. FLIVE Replacement Interpretation
- **Does topiq_unified beat mobile_flive on FLIVE val?** No. MAE 10.19 vs 3.14.
- **Does topiq_unified beat mobile_flive on FLIVE test?** No. MAE 10.04 vs 3.06.
- **Is the gap small enough to still justify one-model replacement?** No. MAE 차이가 3배 이상이며, 상관계수 차이도 큽니다. SPAQ/KonIQ에서의 이득이 크더라도 FLIVE 성능 하락폭이 너무 큽니다.
- **Or is FLIVE weakness too large?** Yes. FLIVE 데이터셋에 대한 훈련이나 Calibration이 부족한 것으로 판단됩니다.

## 6. Single Replacement Decision
**B. topiq_unified is strong on SPAQ/KonIQ but FLIVE is too weak; run FLIVE calibration fine-tuning.**

## 7. Recommended Next Step
**FLIVE calibration fine-tuning**
- Unified 모델을 유지하면서 FLIVE 데이터셋에 대해 적은 learning rate로 fine-tuning을 수행하여 FLIVE 성능을 끌어올려야 합니다.
- 또는 SPAQ + KonIQ + FLIVE를 모두 포함한 Mixed training을 수행하여 전체적인 밸런스를 맞춰야 합니다.

## 8. Files Created
- `outputs/eval_topiq_unified_full_flive_replacement_20260517/summary.csv`
- `outputs/eval_topiq_unified_full_flive_replacement_20260517/report.md`
- `outputs/eval_topiq_unified_full_flive_replacement_20260517/predictions_*.csv`

## 9. Final GO / NO-GO
**NO-GO**
현재 모델로는 FLIVE 전용 모델을 대체할 수 없습니다. 추가적인 훈련이 필요합니다.
