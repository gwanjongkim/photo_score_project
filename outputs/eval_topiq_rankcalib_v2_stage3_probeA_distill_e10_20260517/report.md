# TOPIQ-RankCalib v2 Stage 3 Probe A Longer Evaluation

## 1. Summary
**PARTIAL PASS**

TOPIQ-RankCalib v2 Stage 3 (Unified Head Distillation) 실험 결과, **Distillation이 Unified Head의 성능을 모든 데이터셋에서 성공적으로 향상시킴**을 확인하였습니다. Stage 2의 Unified Head에 비해 SRCC 지표가 FLIVE(0.444 -> 0.470), KonIQ(0.827 -> 0.834), SPAQ(0.878 -> 0.881) 모두 개선되었습니다. 

하지만, 단일 출력 모델 기준으로는 여전히 기존 `TOPIQ-lite ranking_lam01` 모델(FLIVE SRCC 0.496)이 더 높은 순위 산정 능력을 보여주고 있으며, KonIQ/SPAQ의 절대적인 성능 또한 `mixed_112_frozen` 등의 베이스라인보다 소폭 낮습니다. 따라서 Distillation 강도를 높이거나 학습 시간을 늘리는 등의 추가 튜닝이 필요합니다.

## 2. Training Result Highlights
- **Best Epoch:** 4
- **Quick-FLIVE Unified SRCC:** 0.458 (Epoch 10 기준)
- **Quick-FLIVE Teacher SRCC:** 0.472 (FLIVE-specific head)
- **Mode Collapse:** 없음 (Std Ratio 0.85~1.29로 건강한 예측 분포 확인)

## 3. Full Test Metrics Summary
| Dataset | Model/Head | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLIVE** | mixed_112_frozen | 5.02 | 6.81 | 0.469 | 0.563 | -2.36 | 1.22 |
| **FLIVE** | ranking_lam01_gap05 | 5.17 | 7.04 | **0.496** | **0.586** | -2.27 | 1.33 |
| **FLIVE** | **RCv2_S3_Unified** | 5.13 | 6.98 | 0.470 | 0.556 | -1.89 | 1.29 |
| **FLIVE** | RCv2_S3_FliveHead | **4.27** | **5.60** | **0.510** | **0.575** | -1.25 | 0.96 |
| **KonIQ** | mixed_112_frozen | 5.80 | 7.73 | **0.863** | **0.884** | 2.45 | 0.85 |
| **KonIQ** | ranking_lam01_gap05 | 5.84 | 7.78 | **0.864** | **0.885** | 2.72 | 0.91 |
| **KonIQ** | **RCv2_S3_Unified** | 6.48 | 8.69 | 0.834 | 0.855 | 3.09 | 0.89 |
| **KonIQ** | RCv2_S3_KoniqHead | 5.70 | 7.39 | 0.856 | 0.882 | 0.17 | 0.93 |
| **SPAQ** | mixed_112_frozen | 7.91 | 10.04 | **0.899** | 0.893 | 1.30 | 0.83 |
| **SPAQ** | ranking_lam01_gap05 | 7.73 | 9.84 | 0.898 | **0.895** | 0.59 | 0.86 |
| **SPAQ** | **RCv2_S3_Unified** | 8.46 | 10.79 | 0.881 | 0.874 | 1.39 | 0.84 |
| **SPAQ** | RCv2_S3_SpaqHead | 8.20 | 10.57 | 0.877 | 0.878 | 0.52 | 0.83 |

## 4. Comparison to Stage 2
- **Unified Head 개선:** Stage 2 대비 모든 데이터셋에서 SRCC가 향상됨. (FLIVE: +0.026, KonIQ: +0.007, SPAQ: +0.003)
- **안정성:** MAE와 Bias가 안정적으로 유지되었으며, 큰 성능 하락 없이 Distillation이 진행됨.
- **Dataset-specific heads:** Teacher head들 또한 학습 과정에서 안정적으로 성능을 유지함.

## 5. Comparison to TOPIQ-lite ranking_lam01
- **Ranking 성능:** `ranking_lam01`이 여전히 단일 모델 기준 FLIVE SRCC 0.496으로 더 우세함.
- **배포 적합성:** 현재 상태로는 `ranking_lam01` 또는 `mixed_112_frozen`이 더 매력적인 후보임. 
- **잠재력:** Distillation을 통해 Unified Head가 전용 Head의 성능을 따라가고 있는 추세이므로, 훈련 Epoch를 늘리거나 Distillation 비중을 높이면 추월 가능성이 있음.

## 6. Candidate Decision
**B. Stage 3 improves but still needs tuning.**

Unified Head가 Distillation을 통해 명확한 성능 향상을 보였으나, 아직 기존 베스트 단일 모델을 압도하지는 못했습니다.

## 7. Recommended Next Step
**Another Stage 3 tuning run (Probe B)**
- `distill_lambda`를 0.3에서 **1.0**으로 상향하여 Unified Head가 전용 Head를 더 강하게 모사하도록 유도.
- 학습 Epoch를 10에서 **20**으로 연장.
- Backbone Unfreeze 실험을 통해 전체적인 용량(Capacity) 확보 시도.

## 8. Files Created
- `outputs/eval_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517/summary.csv`
- `outputs/eval_topiq_rankcalib_v2_stage3_probeA_distill_e10_20260517/report.md`
- `scripts/eval_topiq_rankcalib_v2_stage3.py`

## 9. Final GO / NO-GO
**PARTIAL GO**
단일 모델 통합을 위한 Distillation의 효용성은 입증되었으나, 최종 배포를 위한 성능 임계치에는 아직 도달하지 못했습니다. 추가 튜닝을 진행합니다.
