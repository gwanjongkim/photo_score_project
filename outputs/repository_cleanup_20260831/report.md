# 저장소 정리 결과

## 결론

고신뢰 정리만 수행했다. 원천 데이터, 연구 코드, 최종 어댑터, 체크포인트, 평가표, 감사 보고서와 명령 기록은 유지했고, 동일한 기반 모델 캐시와 재생성 가능한 비모바일 파생 모델 본체, 실행 캐시와 명백한 잔재를 제거했다.

정리 전 프로젝트는 약 363GiB였고 정리 후 약 263GiB다. 논리 크기 합계로 최소 106,725,450,386바이트, 약 99.4GiB를 회수했다. `outputs`의 현재 논리 크기는 129,882,935,126바이트, 약 121GiB다.

## 핵심 유지 항목

- `data`, `checkpoints`, `models`, `weights`, `exports`의 연구 자산.
- `src`, `scripts`, `tools`, `configs`, `experiments`의 코드와 설정.
- `outputs/capstone_model_audit_20260711`의 논문 기준 및 구현 감사.
- `outputs/aesthetic_ensemble_weight_search_complete_20260603`의 최종 앙상블 결정 자료.
- `outputs/mdiqa_official_variant_sanity_20260601`의 공식 변형 검증 자료.
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter`와 평가 및 핸드오프.
- Gemma-4 E4B 기준 캐시 한 벌.
- 다른 Gemma 실험의 최종 어댑터와 평가 자료.
- Git으로 추적되는 파생 모델 설정, 토크나이저와 양자화 요약.

## 제거 항목

- 전체 SHA-256이 동일한 Gemma-4 E4B 기반 모델 캐시 네 벌.
- 모바일 형식이 아니며 보존 자료로 재생성 가능한 merged 및 BNB4 `model.safetensors` 세 벌.
- 모든 Python `__pycache__`.
- Windows `Zone.Identifier` 두 개.
- 잘못 생성된 `=0.45.0` 파일.
- 중첩된 빈 API 테스트 출력 디렉터리.

## 보수적으로 유지한 항목

아래 항목은 크지만 이번 정리에서 삭제하지 않았다.

- 여러 Python 가상환경. 실행 계보별 의존성이 달라 현재 참조 여부만으로 삭제하기 어렵다.
- `.git/lfs` 약 31GiB. 원격 복구 가능성을 확인하지 않은 상태에서 정리하면 체크아웃 복구가 깨질 수 있다.
- ALAMP, RGNet, ICAA, MDIQA의 개별 훈련 체크포인트와 실패 실험. 논문 재현과 비교 계보에 쓰일 수 있어 이름만으로 삭제하지 않았다.
- 원천 및 가공 데이터셋. 중복 여부보다 데이터 분할과 매니페스트 계보가 우선이다.

## 검증

- 삭제 대상 대형 경로의 부재를 확인했다.
- Python `__pycache__` 수가 0인지 확인했다.
- 기준 Gemma 캐시, 282행 최종 어댑터, 주요 감사 및 벤치마크 디렉터리의 존재를 확인했다.
- 사용자 소유의 기존 미커밋 코드 변경은 수정하지 않았다.
