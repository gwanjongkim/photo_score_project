# 삭제 대상 명세

## 중복 기반 모델 캐시

다음 네 디렉터리는 동일한 Gemma-4 E4B 기반 모델 blob을 포함한다. 기준 캐시 한 벌은 별도로 유지한다.

- `outputs/gemma4_e4b_aesthetic3_smoke/train_run/hf_cache` — 16,031,710,033바이트.
- `outputs/gemma4_e4b_aesthetic3_smoke/train_run_pass2/hf_cache` — 16,031,868,366바이트.
- `outputs/gemma4_e4b_aesthetic3_smoke/train_run_pass2b/hf_cache` — 16,031,867,256바이트.
- `outputs/gemma4_e4b_aesthetic3_expanded_20260427/train_run/hf_cache` — 16,031,833,786바이트.

## 재생성 가능한 비모바일 파생 모델 본체

- `outputs/gemma4_e4b_aesthetic3_smoke/export_pass2b/merged_model/model.safetensors` — 15,882,477,500바이트.
- `outputs/gemma4_mobile_export_probe_20260429/merged_gemini_teacher_282/model.safetensors` — 15,882,477,500바이트.
- `outputs/gemma4_mobile_export_probe_20260429/quantized_bnb4_merged_gemini_teacher_282/model.safetensors` — 9,301,680,673바이트.

이 파일들은 보존된 기반 모델 캐시, LoRA 어댑터, 설정, 핸드오프의 명령으로 재생성할 수 있다. 모바일용 `.litertlm`, `.task`, `.tflite`, `.gguf` 산출물은 아니다. 각 디렉터리의 Git 추적 설정, 토크나이저, 양자화 요약은 유지했다.

## 일반 캐시와 잔재

- 프로젝트 아래 모든 `__pycache__` 디렉터리 — 정리 전 합계 1,531,476,179바이트.
- `GEMINI - 복사본.md:Zone.Identifier`.
- `TROUBLESHOOTING - 복사본.md:Zone.Identifier`.
- `photo_score_project/test_outputs/validation_with_api_key` — 중첩된 임시 테스트 잔재.
- `=0.45.0` — 패키지 설치 출력이 파일로 잘못 저장된 잔재.

## 복구 방법

- Python 캐시는 실행 시 자동 생성된다.
- Gemma 기반 모델은 보존한 기준 캐시를 사용하거나 Hugging Face에서 다시 받을 수 있다.
- merged 모델과 BNB4 파생물은 `outputs/gemma4_mobile_export_probe_20260429/MOBILE_EXPORT_HANDOFF.md`의 명령으로 재생성할 수 있다.
- 나머지 작은 잔재는 연구 결과가 아니므로 별도 백업하지 않는다.

## 완료 결과

- 논리 크기 기준 최소 106,725,450,386바이트를 제거했다. 약 99.4GiB다.
- `outputs`는 약 220GiB에서 약 121GiB로 줄었다.
- 프로젝트 전체는 `du` 기준 약 363GiB에서 약 263GiB로 줄었다.
- 보존한 Gemma 기준 캐시 blob의 크기는 15,992,595,884바이트이며 SHA-256은 위 기록과 일치한다.
