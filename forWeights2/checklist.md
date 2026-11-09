# forWeights2 작업 체크리스트

- [x] Gemini 계획서를 읽고 핵심 기준을 요약한다.
- [x] 기존 `forWeights` 구조와 `forWeights2` 존재 여부를 확인한다.
- [x] `forWeights2` 복사본을 생성한다.
- [x] 4개 모델 기준 config를 작성한다.
- [x] `vector_tflite`와 `score_index` 검증을 모델 레지스트리에 추가한다.
- [x] TFLite runner가 scalar와 vector output을 안전하게 처리하도록 확장한다.
- [x] HTML 리포트가 4개 모델을 동적으로 다루는지 확인한다.
- [x] `READ.md`를 독립 실행 문서로 작성한다.
- [x] 최소 검증 명령을 실행한다.
- [x] 기존 `forWeights`가 수정되지 않았는지 확인한다.
- [x] NIMA 고정 전처리 모델(`nima_mobile_fixed_preproc_fp16.tflite`) 및 메타데이터를 배포한다.
- [x] RGNet PIL 리사이즈 모델(`rgnet_pil_resize_aadb_fp16.tflite`) 및 메타데이터를 배포한다.
- [x] `configs/aesthetic_weight_lab.yaml`의 모델 경로 및 가중치(RGNet 0.4)를 업데이트한다.
- [x] `HANDOFF.md` 및 `MODEL_CARD.md` 문서를 업데이트한다.
- [x] 검증 명령을 통해 새 모델 로드 여부를 확인한다.
