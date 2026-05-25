# TOPIQ-lite FLIVE Calibration Replacement Evaluation

## 1. Summary
**PARTIAL PASS (Forgetting Detected)**

FLIVE 전용 calibration 결과, FLIVE 데이터셋에서의 성능은 기존 전용 모델 수준으로 극적으로 향상되었습니다. 그러나 KonIQ와 SPAQ 데이터셋에서 심각한 성능 저하(Catastrophic Forgetting, 특히 MAE와 Bias 측면)가 관찰되었습니다. 따라서 단일 모델 교체를 위해서는 FLIVE 전용 calibration이 아닌, 모든 데이터셋을 포함한 **Mixed Replay Fine-tuning**이 필수적입니다.

## 2. Calibration Training Result (outputs/full_topiq_lite_unified_to_flive_calib_frozen_e20_20260517)
- **Train/Val Loss:** 0.0017 -> 0.0010 (Val Loss 안정적 감소)
- **Best Epoch:** 20 (최종 에폭 근처에서 수렴)
- **Mode Collapse:** No (예측값의 표준편차가 유지됨)
- **Bias:** FLIVE val 기준 0.47로 매우 낮음 (성공적인 calibration)

## 3. FLIVE Improvement
| Split | Model | MAE | RMSE | SRCC | PLCC | Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| flive_test | mobile_flive (Baseline) | 3.06 | 4.05 | 0.622 | 0.750 | 0.57 |
| flive_test | topiq_unified_orig | 10.04 | 12.65 | 0.488 | 0.565 | -9.26 |
| flive_test | **topiq_flive_calib** | **3.34** | **4.57** | **0.564** | **0.660** | **0.45** |

- FLIVE MAE가 10.04에서 3.34로 대폭 개선되어 전용 모델(3.06)에 근접함.
- SRCC 또한 0.488에서 0.564로 유의미하게 향상됨.

## 4. Catastrophic Forgetting Check
| Split | Model | MAE | RMSE | SRCC | PLCC | Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| koniq_test | topiq_unified_orig | 5.62 | 7.03 | 0.871 | 0.904 | -2.07 |
| koniq_test | **topiq_flive_calib** | **13.53** | **17.43** | **0.761** | **0.807** | **+12.94** |
| spaq_test | topiq_unified_orig | 9.07 | 11.31 | 0.871 | 0.866 | -2.22 |
| spaq_test | **topiq_flive_calib** | **18.47** | **22.13** | **0.872** | **0.731** | **+15.54** |

- **KonIQ:** MAE가 5.6에서 13.5로 급증, SRCC 0.11 하락.
- **SPAQ:** MAE가 9.1에서 18.5로 두 배 이상 증가.
- **Bias:** FLIVE 데이터의 높은 평균 점수(72) 영향으로 KonIQ(58)와 SPAQ(48) 데이터에서 심각한 과대평가(+13~15) 발생.

## 5. Single Replacement Decision
**B. FLIVE improved but SPAQ/KonIQ degraded; need mixed replay fine-tuning.**

## 6. Recommended Next Step
**Mixed Replay Fine-tuning**
- SPAQ + KonIQ + FLIVE 데이터를 적절한 비율로 섞어 훈련 데이터셋을 구성해야 합니다.
- Calibration된 가중치에서 시작하거나, original unified 가중치에서 시작하여 Mixed training을 수행함으로써 모든 데이터셋에서 고른 성능을 내는 단일 모델을 도출해야 합니다.

## 7. Files Created
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/summary.csv`
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/report.md`
- `outputs/eval_topiq_lite_flive_calibration_replacement_20260517/predictions_*.csv`

## 8. Final GO / NO-GO
**NO-GO (As-is)**
현재의 calibration 방식은 FLIVE 성능은 잡았으나 다른 영역의 성능을 망가뜨렸습니다. Mixed Replay 전략으로 선회해야 합니다.
