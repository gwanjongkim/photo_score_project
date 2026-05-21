# forWeights2 컨텍스트 노트

## 2026-05-20

- Gemini 계획서는 기존 `forWeights`의 오프라인 HTML 리포트, 동적 모델 목록, `active_weight_sum` 정규화가 4개 모델에도 재사용 가능하다고 판단했다.
- 기존 코드 확인 결과 `model_registry.py`의 `normalized_weighted_score`와 `html_report.py`의 JavaScript는 모델 목록을 순회하므로 하드코딩된 3개 모델 제한은 보이지 않았다.
- ICAA는 `[MOS, Color]` 벡터 출력 가능성이 있으므로 스칼라 squeeze만 사용하는 기존 runner를 그대로 쓰면 안전하지 않다.
- `forWeights2`는 존재하지 않았으므로 새로 복사했다.
- 루트 저장소에서 변경 확인이 가능하도록 `forWeights/.git`은 복사하지 않았다.
- 요청된 RGNet, Mobile A-LAMP v2, ICAA TFLite 파일은 로컬에 존재해 `forWeights2/models/aesthetic/`로 복사했다.
- ICAA 후보 두 개 중 `outputs/icaa_tf_native_tflite_fp16_20260516_144023/icaa_dat_tf_native_fp16.tflite`는 변환 보고서가 `overall_pass: true`라 최신 통과 후보로 사용했다.
- Mobile A-LAMP v2 export metadata가 `mobilenetv3_include_preprocessing_float_pixels_0_255`를 명시하므로 config에 `normalization: zero_to_255`를 사용했다.
- ICAA config는 `type: vector_tflite`, `score_index: 0`, `score_column: icaa_score`로 설정했다.
- `READ.md`는 OS별 가상환경, 실행 명령, active weight sum 계산식, ICAA score_index, 오류 대응을 포함하도록 새로 작성했다.
- 복사본에 있던 legacy RGNet/A-LAMP 모델 파일과 오래된 `outputs/` 감사 산출물은 `forWeights2`에서 제거했다.
- `./.venv_gpu/bin/python` 기반 문법 검사, config 로드, checksum, help, fake artifact 생성, fake ICAA vector/scalar 추출 검증이 통과했다.
- `git -C forWeights status --short` 결과는 비어 있어 기존 `forWeights` 수정은 확인되지 않았다.
