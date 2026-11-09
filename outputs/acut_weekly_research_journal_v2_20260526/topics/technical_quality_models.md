# 기술적 품질 평가 모델 연구

## 1. 확인된 사실

- **2026-03-30 ~ 2026-04-05**: Stage 5 및 Firebase 초기 통합
- **2026-04-13 ~ 2026-04-19**: VILA 및 NVILA-8B 멀티모달 추론 실험
- **2026-04-20 ~ 2026-04-26**: 경량 미적 품질 모델 확정 및 검증
- **2026-04-27 ~ 2026-05-03**: Gemma 어댑터 학습 및 모바일 수출 조사
- **2026-05-04 ~ 2026-05-10**: RGNet 및 A-LAMP 베이스라인 수립
- **2026-05-11 ~ 2026-05-17**: 그래프 기반 레이아웃 퓨전 및 ICAA TF 구현
- **2026-05-18 ~ 2026-05-24**: TOPIQ-lite 벤치마크 및 DistortionGuard 개발
- **2026-05-25 ~ 2026-05-31**: NIMA 버그 수정 및 최종 성능 재검증

## 2. 주요 근거

- `configs/stage5_reference.json` (신뢰도: high)
- `outputs/stage5_runs/stage5_smoke_test/run_manifest.json` (신뢰도: high)
- `configs/stage5_student_server.json` (신뢰도: high)
- `outputs/test_vila_nvila_eval_20260415_214918/scores.csv` (신뢰도: high)
- `outputs/test_vila_nvila_eval_20260415_214918/scores.jsonl` (신뢰도: high)
- `outputs/test_vila_nvila_resume_20260415/ranked_results.csv` (신뢰도: high)
- `outputs/test_vila_nvila_resume_20260415/ranked_results.jsonl` (신뢰도: high)
- `outputs/acut_transparency_quick_export_check/top_k_summary.json` (신뢰도: high)
- `outputs/acut_transparency_test_vila_20260416/app_results.csv` (신뢰도: high)
- `outputs/acut_transparency_test_vila_20260416/app_results.json` (신뢰도: high)

## 3. 현재 상태

TOPIQ-lite 및 KonIQ-Mobile 기반의 강력한 기술 품질 지표 확보됨.
