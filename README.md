# photo_score_project

`photo_score_project`를 다른 컴퓨터에서 재현 가능하게 실행하기 위한 기준 문서입니다.
이 문서는 아래 3가지 실행 모드를 분리해서 제공합니다.

- 모드 1: 로컬 점수 계산만
- 모드 2: 로컬 점수 계산 + Gemini 설명 생성
- 모드 3: Firebase worker 전체 실행

## 1) 권장 환경

- OS: Linux/macOS (Windows는 WSL 권장)
- Python: **3.12.x** (권장 범위: `>=3.10, <3.13`)
- Node.js: 20.x (`functions/` 사용 시)

## 2) 설치 (clone 이후 공통)

```bash
git clone <YOUR_REPO_URL> photo_score_project
cd photo_score_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

의존성은 목적별로 분리되어 있습니다.

- `requirements.txt`: 로컬 점수 계산/파이프라인 공통
- `requirements-gemini.txt`: Gemini 설명 생성 포함
- `requirements-firebase-worker.txt`: Firebase worker 포함
- `requirements-vila.txt`: 선택 기능(VILA 점수) 포함

예시:

```bash
pip install -r requirements.txt
```

## 3) 환경변수 템플릿

```bash
cp .env.example .env
# .env 내용을 본인 환경에 맞게 수정
```

코드에서 실제 참조하는 주요 키:

- `ACUT_AESTHETIC_CONFIG`
- `ACUT_DISABLE_AESTHETIC_SCORING`
- `GEMINI_API_KEY`
- `GEMINI_MODEL_NAME`
- `ENABLE_GEMINI`
- `FIREBASE_STORAGE_BUCKET`
- `FIREBASE_PROJECT_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `ACUT_VILA_*`

## 4) 모델 파일 배치

기본 설정(`configs/stage5_reference.json`) 기준 필수 경로:

- `checkpoints/composition_aadb_gpu/final_model.keras`
- `checkpoints/technical_koniq_gpu/final_model.keras`
- `checkpoints/technical_flive_image_gpu/final_model.keras`
- `checkpoints/technical_flive_patch_gpu/final_model.keras`
- `checkpoints/nima_ava_gpu/final_model.keras`
- `checkpoints/alamp_aadb_gpu/final_model.keras`
- `checkpoints/musiq_aadb_gpu/final_model.keras`
- `checkpoints/rgnet_aadb_gpu/final_model.keras`

선택(페어와이즈 rerank용):

- `checkpoints/pairwise_aadb_gpu/final_model.keras`
- `data/processed/aadb/val.csv`

모델 전달/검증 정책:

- 대형 모델 파일은 Git 외부 전달물(zip/tar/아티팩트 스토리지)로 받아도 됩니다.
- 전달 후 **저장소 루트 기준 상대경로를 그대로 유지**해야 합니다.
- 별도 metadata JSON이나 verify JSON은 필수 아님. 이 저장소는 `scripts/preflight_check.py`로 경로/파일 존재를 검증합니다.

## 5) preflight 검사

모드별 실행 전 아래 검사 권장:

```bash
python scripts/preflight_check.py --mode minimal
python scripts/preflight_check.py --mode gemini
python scripts/preflight_check.py --mode worker
```

## 6) 실행 모드

### 모드 1: 최소 실행 (로컬 점수 계산만)

Firebase/Gemini 없이 점수 산출(`scores.jsonl`, `scores.csv`)만 수행합니다.

```bash
PYTHONPATH=. python scripts/run_local_scoring.py \
  --input_dir test_samples \
  --output_dir outputs/mode1_scores
```

### 모드 2: 점수 + Gemini 설명 생성 (로컬)

로컬 파이프라인 실행 후 top-k에 대해 Gemini 멀티모달 설명을 생성합니다.

```bash
pip install -r requirements-gemini.txt
ENABLE_GEMINI=true GEMINI_API_KEY=<YOUR_KEY> PYTHONPATH=. python -m src.infer.run_acut_pipeline \
  --input_dir test_samples \
  --output_dir outputs/mode2_pipeline \
  --top_k 5 \
  --enable_gemini
```

### 모드 3: 전체 worker 실행

#### 3-a) Firebase 없이 로컬 worker smoke

```bash
pip install -r requirements-firebase-worker.txt
PYTHONPATH=. python -m src.firebase.acut_job_worker \
  --local_input_dir test_samples \
  --local_output_dir outputs/mode3_worker_local \
  --local_top_k 5
```

#### 3-b) Firebase 연동 worker

```bash
pip install -r requirements-firebase-worker.txt
PYTHONPATH=. python -m src.firebase.acut_job_worker \
  --job_collection jobs \
  --bucket "$FIREBASE_STORAGE_BUCKET"
```

## 7) 자주 발생하는 문제

- `No module named ...`: requirements 파일을 모드에 맞게 다시 설치
- `Model file missing ...`: 모델 파일 경로/이름 확인 (`configs/stage5_reference.json` 기준)
- `GEMINI_API_KEY ... not set`: `.env` 또는 셸 환경변수 설정
- `GOOGLE_APPLICATION_CREDENTIALS ... missing`: 서비스 계정 JSON 경로 확인
- HEIC 이미지 디코딩 실패: `pillow-heif` 설치 여부 확인

자세한 내용은 아래 문서 참고:

- [SETUP_LOCAL.md](SETUP_LOCAL.md)
- [SETUP_WORKER.md](SETUP_WORKER.md)
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
