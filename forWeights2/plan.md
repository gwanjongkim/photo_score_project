# forWeights2 구현 계획

## 기준

Gemini 계획서의 기준에 맞춰 기존 `forWeights`의 독립 실행 구조와 HTML 가중치 재계산 방식을 유지한다.

## 범위

1. `forWeights2` 내부 파일만 수정한다.
2. 모델 구성을 NIMA, RGNet, A-LAMP, ICAA 4개로 확장한다.
3. ICAA 벡터 출력 처리를 위해 `vector_tflite`와 `score_index`를 지원한다.
4. `active_weight_sum` 기반 최종 점수 계산은 유지한다.
5. 실행 안내는 `forWeights2/READ.md`에 새로 작성한다.

## 검증

1. Python 문법 검사를 수행한다.
2. YAML config가 로드되고 4개 모델 spec이 생성되는지 확인한다.
3. 설정의 모델 경로와 `score_index` 반영 여부를 확인한다.
4. 검색으로 4개 score key와 HTML 동적 처리 여부를 확인한다.
5. `git status`, `git diff --stat`, `git diff -- forWeights2`를 확인한다.
