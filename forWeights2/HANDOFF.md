# forWeights2 Handoff

## 구성

`forWeights2`는 기존 `forWeights`의 독립 실행 구조를 복사하고, 모델 구성을 4개로 확장한 로컬 실험 도구입니다.

- NIMA는 `nima_mobile.tflite`와 `nima_score`를 사용합니다.
- RGNet은 `rgnet_paper_aadb_fp16.tflite`와 `rgnet_score`를 사용합니다.
- A-LAMP는 `mobile_alamp_v2_fp16.tflite`와 `alamp_score`를 사용합니다.
- ICAA는 `icaa_dat_tf_native_fp16.tflite`와 `icaa_score`를 사용합니다.

## 주요 변경

- `configs/aesthetic_weight_lab.yaml`은 4개 모델과 0.25 기본 weight를 사용합니다.
- `model_registry.py`는 `vector_tflite`와 `score_index` 검증을 지원합니다.
- `tflite_model_runner.py`는 ICAA 같은 벡터 output에서 지정 index를 점수로 추출합니다.
- `html_report.py`는 모델 목록을 순회하며 4개 개별 점수와 weight slider/input을 렌더링합니다.
- `READ.md`는 OS별 가상환경, 실행 명령, 오류 대응, 검증 명령을 포함합니다.

## 실행 기준

팀원이 실행할 때는 `forWeights2/READ.md`를 기준으로 사용합니다. HTML 리포트에서 weight를 바꾸면 모델 재실행 없이 이미 계산된 raw score로 final score와 ranking을 다시 계산합니다.

## 주의

이 폴더는 qualitative weight lab입니다. 작은 사내 이미지 폴더 결과를 공식 benchmark나 제품 성능 결론으로 해석하지 마세요.
