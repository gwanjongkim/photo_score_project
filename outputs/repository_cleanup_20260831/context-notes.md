# 정리 판단 기록

- 저장소는 이미지 품질 및 미학 점수 연구와 모바일 배포 가능성 검증을 포함한다.
- 원천 데이터, 연구 코드, 체크포인트, 최종 LoRA 어댑터, 평가표, 보고서, 명령 기록은 보수적으로 유지한다.
- `google/gemma-4-E4B-it` 기반 모델 캐시의 15,992,595,884바이트 blob이 다섯 실험 폴더에 존재했다.
- 다섯 blob의 전체 SHA-256은 모두 `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`으로 일치했다.
- 기준 캐시는 `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/hf_cache`에 한 벌 유지한다. 이 경로는 모바일 내보내기 핸드오프에서 원본 캐시로 명시되어 있다.
- `outputs/gemini_api_teacher_aesthetic3_20260427_revised/gemma4_train_run_282/final_adapter`는 282행 Gemini 교사 기반의 핵심 학습 산출물이므로 유지한다.
- 두 merged 모델은 서로 다른 SHA-256이므로 중복 파일로 판정하지 않았다. 다만 각각 보존된 기본 모델과 해당 어댑터에서 재생성되는 파생물이고 모바일 실행 형식이 아니므로 대형 `model.safetensors`만 삭제 대상으로 분류했다.
- BitsAndBytes NF4 산출물은 약 8.7GB이며 Android, Flutter, LiteRT 산출물이 아니라는 기존 핸드오프 결론에 따라 삭제 대상으로 분류했다.
- 파생 모델 디렉터리의 Git 추적 설정, 토크나이저, 양자화 요약은 재현성 자료이므로 복구하여 유지했다.
- 사용자 소유의 미커밋 코드와 문서 변경은 수정하지 않는다.
