# TechIQA-Guard v1 Dataset Design Audit

## 1. Summary
TechIQA-Guard v1은 `koniq_mobile.tflite`와 `flive_image_mobile.tflite`를 대체하기 위한 단일 출력(Single-output) 온디바이스 기술적 이미지 품질 측정 모델입니다. 본 설계 감사는 기존 TOPIQ mixed112 모델이 기술적 결함이 있는 이미지를 과대평가(False Positive)하는 문제를 해결하고, SPAQ, KonIQ, FLIVE 데이터를 통합하여 안정적인 성능을 확보하는 데 중점을 둡니다.

## 2. Available Data
기존 프로젝트 내에 가용한 데이터셋 및 매니페스트는 다음과 같습니다.
- **SPAQ:** `data/processed/spaq/labels_all.csv` (~11k)
- **KonIQ-10k:** `data/processed/koniq10k/labels_all.csv` (~10k)
- **FLIVE:** `data/processed/flive/labels_image_all.csv` (~40k)
- **Mixed Dataset:** `data/processed/topiq_replacement/mixed_112_train.csv` (이미 통합된 형태의 참조용 데이터)

## 3. Existing Evaluation Evidence
`outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/report.md` 분석 결과:
- **TOPIQ mixed112:** SPAQ(SRCC 0.90)와 KonIQ(SRCC 0.86)에서 우수한 성능을 보이나, FLIVE에서는 기존 전용 모델(SRCC 0.63)보다 낮은 성능(SRCC 0.47)을 보임.
- **Problem:** 기술적으로 열악한 이미지(저조도, 노이즈, 블러)에 대해 기존 모델보다 점수가 높게 나오는 경향이 확인됨 (+1.08 mean delta).

## 4. False-Positive Problem
사용자 및 성능 테스트에서 확인된 주요 False Positive 사례:
- `20230201_181300.jpg`, `1675342165226-13.jpg`, `1675342165226-3.jpg`
- **특성:** 실루엣, 드라마틱한 저조도 환경에서 TOPIQ 모델이 예술적 의도로 오인하여 기술 점수를 높게 부여함.
- **TechIQA-Guard Goal:** 이러한 이미지들을 명확히 '낮은 기술 품질'로 학습하여 기존 모델 수준의 안전성을 확보함.

## 5. Proposed Dataset Schema
`data/processed/techiqa_guard/` 하위에 생성될 CSV 파일들의 표준 스키마입니다.

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_path` | String | 절대 경로 |
| `dataset` | String | `spaq`, `koniq10k`, `flive_image`, `hard_fp` |
| `normalized_mos` | Float | 0.0 ~ 1.0 (Target Label) |
| `mos_100` | Float | 0.0 ~ 100.0 (Display용) |
| `split` | String | `train`, `val`, `test` |
| `hard_false_positive` | Boolean | True인 경우 가중 학습 또는 특수 레이블링 |
| `mixed112_score` | Float | 참조용 TOPIQ 점수 |
| `existing_avg_score` | Float | 참조용 기존 모델 평균 점수 |
| `source_note` | String | 특이사항 (예: "low_light_silhouette") |

## 6. Hard False-Positive Mining Plan
1. **Delta-based Mining:** `mixed112_score - existing_avg_score > 15` (100점 만점 기준)인 이미지 추출.
2. **Category Mining:** `PAQ2PIQ` 또는 `FLIVE` 데이터 중 `blur`, `noise` 태그가 높으나 TOPIQ 점수가 높은 샘플 수집.
3. **Manual Curation:** `test_vila/` 폴더 내 실패 사례 이미지를 `hard_fp` 데이터셋으로 명시적 추가.

## 7. Train/Val/Test Plan
- **Unified Training Set:** SPAQ, KonIQ, FLIVE 이미지를 1:1:2 비율로 혼합 (FLIVE 비중 강화).
- **Validation:** 각 데이터셋별 10% 할당.
- **Testing:** 
  - `test_flive.csv`, `test_koniq.csv`, `test_spaq.csv` 독립 운영.
  - `hard_false_positive.csv`를 통해 FP 억제력 별도 측정.

## 8. First Smoke Dataset Plan
빠른 검증을 위해 총 300~500장 규모의 Smoke Dataset 구성:
- SPAQ/KonIQ/FLIVE 각 100장 (고/중/저 품질 고루 분포).
- 모든 식별된 Hard False Positive 이미지 포함.
- `data/processed/techiqa_guard/smoke_v1.csv`로 저장.

## 9. Risks
- **Conservatism:** FP를 너무 강하게 잡을 경우, 어두운 분위기의 정상적인 사진 점수도 낮아질 위험 (Aesthetic 모델과의 조화 필요).
- **Direct Quantization:** Distillation 없이 직접 학습 후 양자화 시 정확도 손실 발생 가능성.

## 10. Codex Implementation Prompt
```text
Create a script 'src/data_prep/make_techiqa_guard_dataset.py' that:
1. Loads SPAQ, KonIQ, FLIVE manifests.
2. Normalizes KonIQ MOS to 0-1 range.
3. Integrates user-flagged false positive images from 'test_vila/' with normalized_mos=0.2.
4. Calculates 'mixed112_score' and 'existing_avg_score' if prediction CSVs are available.
5. Saves unified train/val/test/hard_fp CSVs to 'data/processed/techiqa_guard/'.
```

## 11. Final Recommendation
TechIQA-Guard v1은 **"Safety-first Technical IQA"**를 지향해야 합니다. TOPIQ의 높은 SPAQ 성능은 유지하되, FLIVE의 구체적인 결함 패턴을 더 잘 포착하도록 학습 데이터를 재구성하는 것이 핵심입니다. Multi-head를 포기하는 대신 단일 Head의 Robustness를 극대화하기 위해 Hard FP 마이닝에 더 많은 리소스를 투입할 것을 권장합니다.
