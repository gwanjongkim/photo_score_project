# Gemma 온디바이스 LLM 실험

## 1. 확인된 사실

- **2026-04-27 ~ 2026-05-03**: Gemma 어댑터 학습 및 모바일 수출 조사
- **2026-05-04 ~ 2026-05-10**: RGNet 및 A-LAMP 베이스라인 수립
- **2026-05-11 ~ 2026-05-17**: 그래프 기반 레이아웃 퓨전 및 ICAA TF 구현
- **2026-05-25 ~ 2026-05-31**: NIMA 버그 수정 및 최종 성능 재검증

## 2. 주요 근거

- `outputs/gemini_api_teacher_aesthetic3_20260427/candidate_manifests_summary.json` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427/scored_pool_inspection.json` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427/test_vila.jsonl` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/GEMMA_282_HANDOFF.md` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/dataset_all.csv` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/dataset_all.jsonl` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/export_gemma4_282/export_summary.json` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/README.md` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/README.md` (신뢰도: high)
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/checkpoint-80/adapter_config.json` (신뢰도: high)

## 3. 현재 상태

어댑터 학습은 성공했으나 품질 문제로 배포 보류 상태.
