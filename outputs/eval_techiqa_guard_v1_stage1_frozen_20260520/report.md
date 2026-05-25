# TechIQA-Guard v1 Stage 1 Frozen Evaluation

## 1. Summary
TechIQA-Guard v1 Stage 1 (Frozen Backbone) 모델은 TOPIQ mixed112 대비 사용자 제보 False Positive 이미지들에 대해 약간의 점수 하락(안전성 개선)을 보였으나, 여전히 `koniq_mobile` 보다는 현저히 높은 점수를 부여하고 있습니다. 또한, 전반적인 기술적 품질 측정 성능(SRCC/PLCC)은 모든 데이터셋에서 하락하였으며, 특히 FLIVE 데이터셋에서의 성능 저하(SRCC 0.38)가 뚜렷합니다.

## 2. Training Result
- **Epochs:** 10 (Best at Epoch 9)
- **Validation Loss:** 0.015 (Initial 0.022)
- **Convergence:** 안정적으로 수렴하였으나, Frozen Backbone 제약으로 인해 성능 향상에 한계가 있음.
- **Mode Collapse:** 미발생 (Standard Deviation 비율이 0.9 수준으로 양호).

## 3. Test Metrics (N=500)

| Dataset | Model | MAE | RMSE | SRCC | PLCC | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **FLIVE** | techiqa_guard_v1_s1 | 5.91 | 7.97 | 0.382 | 0.486 | 1.50 |
| FLIVE | topiq_mixed112 | 4.93 | 6.81 | 0.458 | 0.528 | 1.28 |
| **KonIQ** | techiqa_guard_v1_s1 | 6.21 | 8.24 | 0.842 | 0.868 | 0.91 |
| KonIQ | topiq_mixed112 | 5.86 | 7.91 | 0.869 | 0.892 | 0.83 |
| **SPAQ** | techiqa_guard_v1_s1 | 8.08 | 10.37 | 0.881 | 0.887 | 0.92 |
| SPAQ | topiq_mixed112 | 7.20 | 9.02 | 0.918 | 0.914 | 0.85 |

## 4. Comparison to Existing Models
- **SPAQ/KonIQ:** 기존 프로덕션 모델(`koniq_mobile`) 수준의 성능을 유지하고 있으나, 최신 TOPIQ 베이스라인보다는 낮음.
- **FLIVE:** 기존 전용 모델(`flive_image_mobile`, SRCC 0.61) 및 TOPIQ 베이스라인(SRCC 0.46)보다 유의미하게 낮은 성능을 보임.
- **Unified Score:** 단일 헤드로 통합하는 과정에서 FLIVE 데이터의 특징을 충분히 학습하지 못한 것으로 판단됨.

## 5. False Positive Behavior
사용자 제보 이미지(실루엣, 저조도)에 대한 점수 분석:

| Image | TechIQA-Guard | mixed112 | koniq_mobile |
| :--- | :---: | :---: | :---: |
| 20230201_181300.jpg | 65.59 | 65.91 | 63.80 |
| 1675342165226-13.jpg | 69.84 | 70.18 | 63.13 |
| 1675342165226-3.jpg | 72.84 | 74.83 | 63.20 |

- **개선 정도:** mixed112 대비 약 0.5~2.0점 하락하였으나, 여전히 안전성 임계값(예: 60점 이하)에 도달하지 못함.
- **위험성:** Frozen Backbone이 해당 이미지들을 '고품질'로 인식하는 특징을 그대로 유지하고 있어, Head 학습만으로는 근본적인 해결이 어려움.

## 6. Decision
**C. Unfreeze partial layers**

이유: 
1. Head만 학습해서는 FLIVE 성능 저하를 막기 어렵고 False Positive 억제 효과도 미미함.
2. Backbone의 상위 레이어들을 Unfreeze하여 기술적 결함(블러, 노이즈)에 대한 Feature 재추출이 필요함.
3. FLIVE 데이터의 가중치를 높이거나 Ranking Loss를 추가하기 전에, Feature 공간 자체의 조정이 선행되어야 함.

## 7. Recommended Next Step
- **Stage 2 Plan:** 
  - Backbone의 마지막 2~3개 블록을 Unfreeze하여 학습.
  - Hard False Positive 데이터에 대해 높은 가중치(Weighting) 또는 낮은 타겟 점수를 부여하는 `False-Positive Guard Loss` 병행 검토.
  - FLIVE 데이터 비중을 현재보다 1.5배 강화하여 SRCC 복구 시도.
