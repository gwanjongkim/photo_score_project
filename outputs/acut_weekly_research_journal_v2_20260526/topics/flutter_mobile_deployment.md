# Flutter 및 모바일 배포

## 1. 확인된 사실

- **2026-03-30 ~ 2026-04-05**: Stage 5 및 Firebase 초기 통합
- **2026-04-06 ~ 2026-04-12**: Acut 제품 백엔드 정비 및 증류 학습
- **2026-04-13 ~ 2026-04-19**: VILA 및 NVILA-8B 멀티모달 추론 실험
- **2026-04-20 ~ 2026-04-26**: 경량 미적 품질 모델 확정 및 검증
- **2026-04-27 ~ 2026-05-03**: Gemma 어댑터 학습 및 모바일 수출 조사
- **2026-05-04 ~ 2026-05-10**: RGNet 및 A-LAMP 베이스라인 수립
- **2026-05-11 ~ 2026-05-17**: 그래프 기반 레이아웃 퓨전 및 ICAA TF 구현
- **2026-05-18 ~ 2026-05-24**: TOPIQ-lite 벤치마크 및 DistortionGuard 개발
- **2026-05-25 ~ 2026-05-31**: NIMA 버그 수정 및 최종 성능 재검증

## 2. 주요 근거

- `docs/acut_product_integration.md` (신뢰도: high)
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/student_model.float16.metadata.json` (신뢰도: high)
- `outputs/distill_exec_20260411/stage5_student_aadb_subset/tflite_smoke_comparison.json` (신뢰도: high)
- `outputs/distill_gpu_runtime_check/runtime_info.txt` (신뢰도: high)
- `outputs/test_vila_nvila_resume_20260415/explanation_records.jsonl` (신뢰도: high)
- `outputs/test_vila_nvila_resume_20260415/ranked_results.csv` (신뢰도: high)
- `outputs/test_vila_nvila_resume_20260415/ranked_results.jsonl` (신뢰도: high)
- `outputs/acut_transparency_test_vila_20260416/explanation_records.jsonl` (신뢰도: high)
- `outputs/acut_transparency_test_vila_20260416/ranked_results.csv` (신뢰도: high)
- `outputs/acut_transparency_test_vila_20260416/ranked_results.jsonl` (신뢰도: high)

## 3. 현재 상태

성능 고도화 및 안정화 단계.
