# TOPIQ-lite mixed_112 Low-LR Unfreeze Replacement Evaluation

## 1. Summary
**PARTIAL PASS / FAIL (Ranking Improvement Not Achieved)**

`mixed_112_frozen` 가중치를 기반으로 Backbone을 unfreeze하고 극저 학습률(1e-6)로 Fine-tuning을 수행한 결과, MAE와 Bias 측면에서는 안정적인 성능을 유지했으나, 핵심 목표였던 **FLIVE 랭킹 성능(SRCC) 개선은 이루어지지 않았습니다.** 오히려 SRCC와 PLCC 지표가 미세하게 하락하는 경향을 보였습니다. 

## 2. Training Result (mixed_112_unfrozen_lr1e6)
- **Epochs:** 12
- **Best Epoch:** 12 (Val loss 0.0028)
- **Val Loss/MAE:** 0.0028 / 0.0595
- **Mode Collapse:** No (FLIVE test std ratio 1.16로 양호)

## 3. Replacement Evaluation (Test Sets)
| Eval Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLIVE** | mobile_flive (Ref) | 3.06 | 4.05 | 0.622 | 0.750 | 0.57 | 0.72 |
| | mixed_112_frozen | **4.62** | **6.26** | **0.459** | **0.561** | -0.84 | 1.17 |
| | mixed_112_unfrozen | 4.60 | 6.32 | 0.453 | 0.544 | **0.07** | 1.16 |
| **KonIQ** | mobile_koniq (Ref) | 8.84 | 10.66 | 0.783 | 0.830 | -6.13 | 0.82 |
| | mixed_112_frozen | **6.01** | **8.06** | **0.846** | **0.873** | 2.53 | 0.82 |
| | mixed_112_unfrozen | 6.47 | 8.61 | 0.831 | 0.857 | 3.05 | 0.83 |
| **SPAQ** | mixed_112_frozen | **8.22** | 10.33 | **0.894** | **0.888** | 1.05 | 0.80 |
| | mixed_112_unfrozen | 8.38 | **10.32** | 0.891 | 0.884 | **0.68** | 0.84 |

## 4. Improvement vs mixed_112 frozen
- **FLIVE improvement?** No. SRCC가 0.459에서 0.453으로 소폭 하락했습니다. 다만 Bias는 0.07로 거의 완벽하게 0에 수렴했습니다.
- **KonIQ retained?** Yes. SRCC 0.83 수준으로 유지되었으나 Frozen 모델(0.846)보다는 낮습니다.
- **SPAQ retained?** Yes. SRCC 0.89 수준을 견고하게 유지하고 있습니다.

## 5. Single Replacement Decision
**C. Unfreeze hurt stability; revert to mixed_112 frozen.**

1e-6의 매우 낮은 학습률임에도 불구하고 Backbone 전체를 unfreeze하는 것은 랭킹 성능(SRCC) 향상에 도움이 되지 않았습니다. 현재 데이터 구성과 아키텍처 하에서는 Frozen 상태의 `mixed_112`가 가장 안정적인 밸런스를 보여줍니다. 

## 6. Recommended Next Step
**TFLite export/parity of best model (mixed_112_frozen)**
- Unfreeze 시도가 실패했으므로, 현재 가장 성능이 좋은 `mixed_112_frozen` 모델을 최종 교체 후보로 확정합니다.
- 해당 모델을 FP16 TFLite로 export하고, 실제 모바일 배포 환경에서의 Parity 및 속도를 측정하는 단계로 넘어갈 것을 권장합니다.
- 만약 FLIVE SRCC(0.46)가 여전히 수용 불가능한 수준이라면, 단일 모델 교체 전략을 재검토하거나 FLIVE 데이터셋에 특화된 추가적인 손실 함수(Ranking Loss 등) 도입이 필요할 수 있습니다.

## 7. Files Created
- `outputs/eval_topiq_lite_mixed112_unfreeze_replacement_20260517/summary.csv`
- `outputs/eval_topiq_lite_mixed112_unfreeze_replacement_20260517/report.md`
- `outputs/mixed_replay_topiq_lite_112_unfrozen_lr1e6_e12_20260517/`

## 8. Final GO / NO-GO
**PARTIAL GO (Revert to Frozen)**
Unfrozen 모델은 성능 이득이 없으므로 채택하지 않습니다. Frozen `mixed_112` 모델로 TFLite 검증 단계에 진입합니다.
