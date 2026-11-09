# 리포트 생성 로그 (Report Generation Log)

## 1. 수행 명령어 (Commands Used)
- `grep -rn` 및 `rg`를 이용한 코드 전수 조사.
- `cat` 및 `read_file`을 이용한 핵심 소스 코드 내용 확인.
- `ls` 및 `glob`을 이용한 모델 파일 및 메타데이터 위치 식별.

## 2. 조사 디렉토리 (Directories Scanned)
- `src/`: 추론, 전처리, 모델 정의 로직.
- `configs/`: 파이프라인 및 모델 가중치 설정.
- `outputs/`: 리서치 보고서 및 벤치마크 결과.
- `forWeights2/`: TFLite 모델 번들 및 가이드라인.

## 3. 검색어 (Search Terms Used)
- `nima`, `rgnet`, `alamp`, `icaa`, `koniq`, `flive`, `topiq`, `4/7`, `0.571`, `score_index`, `0..255`, `divide_by_255`.

## 4. 확인된 주요 파일 (Key Files Verified)
- `src/infer/run_acut_pipeline.py`
- `src/infer/select_best_shots.py`
- `src/infer/predict_quality_bundle.py`
- `src/models/alamp.py`
- `forWeights2/models/aesthetic/MODEL_CARD.md`
- `outputs/research_protocol_production_technical_eval_20260530/report.md`

## 5. 불확실한 항목 (Uncertain Items)
- **Flutter 구현부**: WSL 환경에는 Dart 코드가 없으므로, 위에서 발견된 전처리 불일치(A-LAMP)가 실제 모바일 앱에서도 발생하는지 여부는 Flutter 프로젝트 조사가 병행되어야 함.
- **TOPIQ Integration**: 리서치 보고서에는 4/7, 3/7 공식이 명시되어 있으나, `src/infer/`의 실제 파이프라인 코드에는 반영 전인 상태임.
