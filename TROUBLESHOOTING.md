# TROUBLESHOOTING

## 1. `ModuleNotFoundError`

### 증상

- `No module named 'tensorflow'`
- `No module named 'firebase_admin'`
- `No module named 'google.generativeai'`

### 원인

- 실행 모드에 맞는 requirements가 설치되지 않음.

### 해결

```bash
# 모드1
pip install -r requirements.txt

# 모드2
pip install -r requirements-gemini.txt

# 모드3
pip install -r requirements-firebase-worker.txt
```

## 2. 모델 파일 누락

### 증상

- preflight에서 `Model file missing ...`
- 파이프라인 실행 중 모델 로드 실패

### 원인

- `configs/stage5_reference.json`의 `bundle_args` 경로에 모델이 없음.

### 해결

1. 모델 아카이브(외부 전달물)를 받아 저장소 루트 기준 경로에 배치
2. 필요 시 `ACUT_AESTHETIC_CONFIG`를 사용자 정의 config로 변경
3. 재검사:

```bash
python scripts/preflight_check.py --mode minimal
```

## 3. Gemini 설명이 생성되지 않음

### 증상

- 실행은 성공하지만 Gemini 설명이 기존 기본 설명으로 유지됨

### 원인

- `ENABLE_GEMINI=false`
- `GEMINI_API_KEY` 미설정
- `google-generativeai` 미설치

### 해결

```bash
pip install -r requirements-gemini.txt
export ENABLE_GEMINI=true
export GEMINI_API_KEY=<YOUR_KEY>
python scripts/preflight_check.py --mode gemini
```

## 4. Firebase worker 인증 실패

### 증상

- worker 시작 후 Firebase 초기화 오류
- 권한 오류(`permission denied`)

### 원인

- `GOOGLE_APPLICATION_CREDENTIALS` 경로/권한 문제
- 잘못된 프로젝트/버킷 설정

### 해결

1. 서비스 계정 JSON 경로 확인
2. 환경변수 확인:
- `GOOGLE_APPLICATION_CREDENTIALS`
- `FIREBASE_PROJECT_ID`
- `FIREBASE_STORAGE_BUCKET`
3. preflight 재실행:

```bash
python scripts/preflight_check.py --mode worker
```

## 5. HEIC/HEIF 이미지 디코딩 실패

### 증상

- 이미지 디코딩 단계 오류

### 원인

- `pillow-heif` 미설치

### 해결

```bash
pip install pillow-heif
```

또는 worker 의존성 재설치:

```bash
pip install -r requirements-firebase-worker.txt
```

## 6. `PYTHONPATH` 관련 모듈 import 실패

### 증상

- `python -m src...` 실행 시 `No module named src`

### 해결

저장소 루트에서 실행하고:

```bash
export PYTHONPATH=.
```

## 7. 점수 결과가 비어 있음

### 증상

- `num_images=0`
- `No usable images could be scored from input_dir`

### 원인

- 입력 폴더에 지원 확장자 이미지가 없음
- 파일 손상/디코딩 불가

### 해결

1. 입력 폴더 확인 (`.jpg,.jpeg,.png,.webp,.bmp`)
2. 필요 시 `--recursive` 사용
3. `score_failures.json` 확인
