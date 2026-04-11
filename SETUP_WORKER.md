# SETUP_WORKER

이 문서는 Firebase queue + Python worker(모드 3) 재현 절차입니다.

## 1. 사전 준비

- Python 3.10~3.12
- Node.js 20 (`functions/` 빌드/배포 시)
- Firebase CLI 로그인 완료

## 2. Python worker 의존성

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-firebase-worker.txt
```

선택(VILA 점수까지 사용할 때):

```bash
pip install -r requirements-vila.txt
```

## 3. 환경변수

`.env.example` 기반으로 아래 항목을 채웁니다.

필수/권장:

- `PYTHONPATH=.`
- `FIREBASE_PROJECT_ID=<project-id>`
- `FIREBASE_STORAGE_BUCKET=<bucket-name>`
- `GOOGLE_APPLICATION_CREDENTIALS=/abs/path/service-account.json`

선택:

- `ACUT_AESTHETIC_CONFIG`
- `ACUT_DISABLE_AESTHETIC_SCORING`
- `ACUT_KEEP_FAILED_JOB_DIRS=true`
- `ACUT_VILA_PYTHON`, `ACUT_VILA_MODEL_NAME`, `ACUT_VILA_PROMPT_PRESET`
- `ENABLE_GEMINI=true`, `GEMINI_API_KEY` (설명 생성까지 필요할 때)

## 4. preflight

```bash
python scripts/preflight_check.py --mode worker
```

## 5. Firebase Functions(Queue API) 준비

```bash
cd functions
npm install
npm run build
# 필요 시
# firebase deploy --only functions
cd ..
```

## 6. 실행 방법

### 6-1) Firebase 없이 로컬 worker smoke

```bash
PYTHONPATH=. python -m src.firebase.acut_job_worker \
  --local_input_dir test_samples \
  --local_output_dir outputs/worker_local_smoke \
  --local_top_k 5
```

### 6-2) Firebase 연동 worker

```bash
PYTHONPATH=. python -m src.firebase.acut_job_worker \
  --job_collection jobs \
  --bucket "$FIREBASE_STORAGE_BUCKET"
```

## 7. Emulator 사용 시

터미널 1:

```bash
cd functions
npm run build
cd ..
firebase emulators:start --project demo-acut --only functions,firestore,storage
```

터미널 2:

```bash
export FIRESTORE_EMULATOR_HOST=127.0.0.1:8080
export FIREBASE_STORAGE_EMULATOR_HOST=127.0.0.1:9199
PYTHONPATH=. python -m src.firebase.acut_job_worker --once
```

## 8. 산출물

worker 완료 시 Storage/로컬 output에 아래 파일이 생성됩니다.

- `app_results.json`
- `top_k_summary.json`
- `review_sheet.csv`
- `ranked_results.jsonl`
- `ranked_results.csv`
- `score_failures.json` (실패 이미지가 있을 때)
