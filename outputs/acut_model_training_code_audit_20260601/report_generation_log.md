# 리포트 생성 로그 (Report Generation Log)

## 1. 수행 명령어 (Commands Used)
- `grep -r`: 프로젝트 내 모델명 및 체크포인트 경로 검색.
- `ls -R`: 출력 및 체크포인트 디렉토리 구조 파악.
- `read_file`: `summary.json`, `report.md`, `metadata.json` 등 메타데이터 파일 분석.

## 2. 조사 디렉토리 (Directories Scanned)
- `src/train/`: 학습 스크립트 소스.
- `src/export/`: 내보내기 스크립트 소스.
- `checkpoints/`: 모델 가중치 및 Keras 파일.
- `outputs/`: 학습 결과 요약 및 평가 보고서.
- `scripts/`: 모델별 특화 내보내기 및 평가 스크립트.

## 3. 검색어 (Search Terms Used)
- `train_nima`, `train_rgnet`, `train_mobile_alamp`, `train_regression`, `export_savedmodel`, `export_tflite`, `best.weights.h5`.

## 4. 확인된 주요 근거 파일 (Key Evidence Files)
- `outputs/nima_preprocessing_fix_20260525/report.md`
- `outputs/rgnet_v1_aadb_pil_resize_retrain_20260524/train_summary.json`
- `outputs/mobile_alamp_v2_full_ava_20260519/summary.json`
- `outputs/icaa_tf_native_savedmodel_20260516_134439/stage_i_savedmodel_report.json`

## 5. 불확실성 (Uncertainty)
- KonIQ와 FLIVE의 경우 범용 학습 스크립트(`train_regression.py`)를 사용했음이 확인되었으나, 해당 모델의 특정 학습 설정값이 담긴 개별 파일이 분산되어 있어 "학습 당시의 환경"을 100% 재구성하기 위해서는 체크포인트 파일 내부의 메타데이터 조사가 추가로 필요할 수 있음.
