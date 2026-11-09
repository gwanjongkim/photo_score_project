# 리포트 생성 로그 (Report Generation Log)

## 1. 조사 및 발췌 과정
- `outputs/acut_model_training_code_audit_20260601/`의 결과를 바탕으로 대상 파일 선정.
- 각 모델의 `build` 함수, `preprocess` 관련 로직, `fit/compile` 호출부, `TFLiteConverter` 사용처를 정밀 분석.
- 10-40라인 내외의 핵심 연산 위주로 스니펫 구성.

## 2. 검토된 파일 및 추출된 기호
- `src/models/nima_distribution.py`: `build_nima_distribution_model`, `emd_loss`
- `src/models/rgnet_paper_v1.py`: `ASPPContextModule`, `V1RegionSimilarityAdjacency`
- `src/models/mobile_alamp_v2.py`: `build_mobile_alamp_v2_model`
- `src/datasets/native_size_dataset.py`: `prepare_alamp_inputs`
- `src/train/train_regression.py`: `main` (Regression Loop)
- `experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py`: `run` (Weight Transfer)
- `src/export/export_tflite.py`: `convert_keras_model`

## 3. 제외된 파일 및 사유
- `src/utils/`: 범용 유틸리티 코드로 모델 핵심 원리와는 무관함.
- `docs/`: 텍스트 설명 위주이며 코드가 포함되지 않음.
- 대규모 리서치용 스크립트: 프로덕션 모델 생성과 직접 연관되지 않은 실험적 코드 제외.

## 4. 불확실한 항목
- ICAA의 경우 `TFICAA17KDAT` 클래스의 내부 상세 구현은 `tf_models/` 하위 파일에 있으나, 발췌문에서는 전체 구조를 설명하기 위해 상위 레벨의 가중치 로드 로직에 집중함.

## 5. 생성된 파일 목록
- `report.md`
- `code_excerpt_index.csv`
- `presentation_code_appendix.md`
- `snippets/nima_core_code.md`
- `snippets/rgnet_core_code.md`
- `snippets/mobile_alamp_v2_core_code.md`
- `snippets/icaa_core_code.md`
- `snippets/koniq_core_code.md`
- `snippets/flive_core_code.md`
- `snippets/common_export_code.md`
- `report_generation_log.md`

## 6. 패치 내역 (2026-06-01)
- **ICAA**: "추정" 문구 제거 및 공식 가중치 재구축(Rebuild) 사실 명시. `TFICAA17KDAT` 내부 구현 확인 필요 마킹.
- **KonIQ/FLIVE**: `train_regression.py`의 `argparse` 및 태스크 선택 로직 추가. 각 모델의 데이터셋 및 프리셋 근거 강화.
- **RGNet**: 실무용 자산(`rgnet_pil_resize_aadb_fp16.tflite`)과 연구용 실험의 혼동 방지를 위한 경고 문구 추가.
