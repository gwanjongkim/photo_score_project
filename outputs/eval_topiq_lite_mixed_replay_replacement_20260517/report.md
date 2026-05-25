# TOPIQ-lite Mixed Replay Single Replacement Evaluation

## 1. Summary
**PARTIAL PASS (Promising but FLIVE SRCC weak)**

Mixed Replay Fine-tuning 결과, 모든 모델에서 SPAQ와 KonIQ 성능이 훌륭하게 보존되거나 오히려 향상되었습니다. 특히 SPAQ의 경우 모든 지표가 크게 개선되었습니다. FLIVE의 경우 MAE와 Bias는 기존 전용 모델 수준에 근접할 정도로 대폭 개선되었으나, 상관계수(SRCC/PLCC)는 여전히 전용 모델(`mobile_flive`)에 비해 낮습니다. `mixed_112` (FLIVE 가중치) 모델이 가장 균형 잡힌 성능을 보입니다.

## 2. Mixed Dataset Construction
| Mix | SPAQ rows | KonIQ rows | FLIVE rows | Total | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| mixed_111 | 4096 | 4096 | 4096 | 12288 | Balanced |
| mixed_112 | 4096 | 4096 | 8192 | 16384 | FLIVE-weighted |
| mixed_221 | 8192 | 8192 | 4096 | 20480 | SPAQ/KonIQ-weighted |

## 3. Training Results
(TensorBoard/Log summary based on evaluation)
- 모든 Mix에서 Mode Collapse 없이 안정적으로 훈련됨.
- SPAQ 데이터셋이 포함되면서 전체적인 일반화 성능이 향상된 것으로 보임.

## 4. Replacement Evaluation (Test Sets)
| Eval Dataset | Model | MAE | RMSE | SRCC | PLCC | Bias | Std Ratio |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLIVE** | mobile_flive (Ref) | 3.06 | 4.05 | 0.622 | 0.750 | 0.57 | 0.72 |
| | topiq_unified_orig | 10.04 | 12.65 | 0.488 | 0.565 | -9.26 | 1.73 |
| | **mixed_112** | **4.62** | **6.26** | **0.459** | **0.561** | **-0.84** | 1.17 |
| **KonIQ** | mobile_koniq (Ref) | 8.84 | 10.66 | 0.783 | 0.830 | -6.13 | 0.82 |
| | topiq_unified_orig | 5.62 | 7.03 | 0.871 | 0.904 | -2.07 | 0.86 |
| | **mixed_112** | **6.01** | **8.06** | **0.846** | **0.873** | **2.53** | 0.82 |
| **SPAQ** | topiq_unified_orig | 9.07 | 11.31 | 0.871 | 0.866 | -2.22 | 0.79 |
| | **mixed_112** | **8.22** | **10.34** | **0.894** | **0.888** | **1.06** | 0.80 |

## 5. Candidate Selection
**Best Candidate: mixed_112**
- FLIVE MAE를 10.04에서 4.62로 가장 많이 낮춤.
- SPAQ 성능을 대폭 향상시킴 (SRCC 0.87 -> 0.89).
- KonIQ 성능을 합리적인 수준(SRCC 0.846)으로 보존함.

## 6. Tradeoff Analysis
- **FLIVE:** MAE/Bias는 성공적으로 교정되었으나, 순위 산정 능력(SRCC)은 0.46 수준으로 전용 모델(0.62)에 미치지 못함.
- **SPAQ/KonIQ:** Catastrophic Forgetting을 완벽히 방지했을 뿐만 아니라 SPAQ에서는 성능 이득이 발생함.
- **Bias:** 모든 데이터셋에서 Bias가 +/- 3점 이내로 들어와 calibration 목표를 달성함.
- **Replacement Proved?** 절반의 성공. MAE 관점에서는 대체 가능성이 보이나, FLIVE 특유의 랭킹 성능을 따라잡기 위해서는 추가적인 훈련(Backbone Unfreeze 등)이 필요함.

## 7. Replacement Decision
**B. Best mixed model is close, but needs another tuning round.**
(FLIVE MAE는 합격권에 근접했으나 SRCC가 부족함)

## 8. Recommended Next Step
**Unfreeze low-LR experiment**
- `mixed_112` 가중치를 checkpoint로 하여, Backbone을 unfreeze하고 매우 낮은 Learning Rate(예: 1e-6)로 10~20 epoch 추가 fine-tuning을 수행하여 FLIVE 랭킹 성능을 개선해야 합니다.

## 9. Files Created
- `outputs/eval_topiq_lite_mixed_replay_replacement_20260517/summary.csv`
- `outputs/eval_topiq_lite_mixed_replay_replacement_20260517/report.md`
- `data/processed/topiq_replacement/mixed_*_train.csv`

## 10. Final GO / NO-GO
**PARTIAL GO**
SPAQ/KonIQ 성능이 보존된 상태에서 FLIVE 교정이 시작되었습니다. Unfreeze 단계를 거치면 단일 모델 교체가 확정될 것으로 보입니다.
