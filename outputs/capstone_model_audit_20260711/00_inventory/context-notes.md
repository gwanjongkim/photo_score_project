# 작업 맥락 기록

## 범위와 결정

- 이 단계는 완전한 저장소 인벤토리와 증거 수집만 수행한다.
- 논문 충실도에 대한 최종 판단은 유보한다.
- 학습·재학습·추가 추론을 실행하지 않는다.
- 생산 소스와 기존 아티팩트를 수정하지 않는다.
- 새 파일은 `outputs/capstone_model_audit_20260711/00_inventory/` 아래에만 작성한다.
- 분류는 파일명만으로 정하지 않고 코드 참조, 문서 주장, 런타임 연결, 산출 경로 등 둘 이상의 근거를 우선한다.
- 증명되지 않은 계보 전이는 `UNKNOWN_LINK`로 남긴다.
- 체크리스트와 이 문서는 프로젝트 `AGENTS.md` 지침에 따른 작업 연속성 보조 산출물이다.

## 진행 중 결정

- 2026-07-11. 첫 셸 명령은 지정 출력 디렉터리 생성이며, 명령 기록에 포함했다.
- 2026-07-11. 기본 Python Stage5 런타임, image-only 기술 평가 공식, mobile-oriented helper를 서로 다른 정책으로 기록했다.
- 2026-07-11. 외부 Flutter 체크아웃이 이 저장소에 없으므로 mobile handoff와 active app runtime 사이를 자동으로 PRODUCTION으로 승격하지 않았다.
- 2026-07-11. old NIMA는 기본 서버 config가 계속 참조하지만 preprocessing repair 보고서가 superseded 결함을 명시하므로 LEGACY로 분리했다.
- 2026-07-11. ICAA는 현재 canonical/copy FP16 파일이 모두 175,885,416 bytes이고 SHA-256이 같음을 확인했다. 기존 11.9 MB 주장, ImageNet normalization 대 zero-to-one, color index 1 대 score index 0 충돌은 미해결로 남겼다.
- 2026-07-11. 중요 아티팩트 289개를 현재 바이트 기준으로 해시했다. TSV 두 파일은 각각 header 포함 290행이며 중복 경로가 없다.
- 2026-07-11. `forWeights2/models/aesthetic/SHA256SUMS.txt`의 네 TFLite checksum은 모두 현재 파일과 일치했다.
- 2026-07-11. 최종 tracked diff는 감사 시작 시점의 14개 수정 파일과 동일하다. 새 감사 파일은 지정된 `00_inventory/` 하나의 untracked root 아래에만 있다.
- 2026-07-11. 비소스 감사 산출물만 작성했으므로 학습, 추론, export, production 변경, commit은 수행하지 않았다.
