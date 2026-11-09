# Mobile Student IQA 50k Label Bias and Dataset Weighting Analysis Report

## 1. Executive Summary
50k B0 모델은 14k 학생 모델 대비 공식 벤치마크(Track B)에서 Overall SRCC 0.703(vs 0.743)으로 약 0.04 하락하며 퇴보했습니다. 분석 결과, 이는 FLIVE 데이터셋의 과도한 비중(68%)과 교사 모델(MDIQA)의 KonIQ-10k 라벨 압축(Target Std 4.87 vs MOS Std 15.42)이 결합된 결과로 판단됩니다. B1/B2로 넘어가기 전, 데이터셋 가중치 조절 및 라벨 스케일 보정을 적용한 B0 재학습이 필수적입니다.

## 2. Sources Inspected
- 50k B0 Run: `outputs/mobile_student_iqa_50k_controlled_b0_20260608/`
- 14k Student Eval: `outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/`
- MDIQA Teacher: `outputs/official_protocol_mdiqa_signal_benchmark_20260604/`
- 50k Manifest: `outputs/mdiqa_distillation_50k_full_targets_20260608/`

## 3. Paired Official Benchmark Join
공식 Track B 이미지 4,967개에 대해 MDIQA 교사, 14k 학생, 50k B0 모델의 예측치를 결합하여 분석했습니다. (`official_paired_predictions.csv` 생성 완료)

## 4. Dataset-wise Degradation
| Dataset | Count | MOS SRCC (14k) | MOS SRCC (B0) | Delta | Bias (B0) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **KonIQ-10k** | 2,015 | 0.793 | 0.739 | **-0.054** | +1.76 |
| **SPAQ** | 1,125 | 0.866 | 0.854 | -0.012 | -3.22 |
| **FLIVE** | 1,827 | 0.432 | 0.507 | **+0.075** | **-9.95** |
| **OVERALL** | 4,967 | 0.743 | 0.703 | **-0.040** | -3.67 |

- FLIVE 자체 성능은 향상되었으나, KonIQ에서의 큰 하락이 전체 성능을 끌어내렸습니다.

## 5. Target Distribution and Scale Shift
| Dataset | Train Target Std | Official MOS Std | Compression Ratio |
| :--- | :--- | :--- | :--- |
| **FLIVE** | 10.59 | 7.81 | 1.35 |
| **KonIQ-10k** | 4.87 | 15.42 | **0.31** |
| **SPAQ** | 18.96 | 21.94 | 0.86 |

- **KonIQ 압축**: 교사 모델의 KonIQ 예측값이 매우 좁은 범위(Std 4.87)에 갇혀 있어, 학생 모델이 순위(Ranking) 정보를 충분히 학습하지 못했습니다.
- **FLIVE 편향**: 학습 데이터의 FLIVE Target Mean(63.8)이 공식 MOS Mean(71.8)보다 8점 이상 낮아, 모델이 전반적으로 낮게 예측하는 경향을 보입니다.

## 6. Dataset Weighting Analysis
현재 50k 학습 데이터셋 구성(Raw Weighting):
- **FLIVE**: 33,891 (67.8%) -> **지나치게 높음**
- **SPAQ**: 8,916 (17.8%)
- **KonIQ**: 7,193 (14.4%)

추천 가중치 정책 (W5):
- FLIVE 가중치 축소 (20~30%)
- KonIQ/SPAQ 가중치 확대 (각 35~40%)
- 이유: FLIVE는 교사 라벨의 신뢰도가 낮고 비중이 너무 커서 모델의 편향을 주도함.

## 7. Error Pattern Analysis
- **Underprediction**: FLIVE 고점 이미지들에서 모델이 -10점 이상의 오차를 보임.
- **KonIQ Failure**: KonIQ의 중위권 점수대 이미지들을 제대로 구분하지 못함.
- **Safety**: Severe FP(Pred > MOS + 25) 사례가 5건 발생했으며, 주로 어두운 실내 이미지에서 발생.

## 8. Internal Validation vs Official Benchmark Shift
- Internal Test SRCC: 0.801
- Official SRCC: 0.703
- **Gap: 0.098**
- 원인: 내부 테스트셋도 FLIVE 비중이 높아, FLIVE 교사 타겟을 잘 따라가는 것(Distillation Fidelity)이 높은 점수로 이어지지만, 실제 인간의 MOS와는 괴리가 있음.

## 9. Root Cause Assessment
1. **KonIQ 라벨의 심각한 압축**: 교사 모델이 KonIQ 데이터셋에서 변별력을 잃어 학생에게 잘못된 신호를 줌.
2. **FLIVE 데이터셋 지배**: 68%에 달하는 FLIVE 비중이 모델을 FLIVE의 편향된(Biased) 교사 라벨에 고착시킴.
3. **용량(Capacity) 부족 가능성**: B0로 50k를 소화하기에 부족할 수 있으나, 현재의 라벨/가중치 문제가 더 근본적임.

## 10. Recommended Next Training Policy
- **Strategy A: Weighted B0 Rerun**
    - FLIVE 가중치를 0.3으로 제한하고 KonIQ/SPAQ 비중을 높여 학습.
    - KonIQ 라벨의 Std를 강제로 확장(Scaling)하거나 Ranking Loss 도입 검토.
    - 50k 데이터 중 FLIVE 비중을 물리적으로 줄인 30k 데이터셋 구성도 고려.

## 11. Final Decision
`B0_50K_REGRESSION_NEEDS_WEIGHTED_B0_RERUN`

## 12. Next Actions
1. `configs/`에 가중치가 적용된 `student_distill_50k_weighted.json` 생성.
2. B0 기반 Weighted 재학습 수행.
3. 결과가 0.75 SRCC 이상으로 개선될 경우 B1/B2로 확장.
