# SETUP_LOCAL

이 문서는 **Firebase 없이 로컬에서 재현**하는 절차를 설명합니다.

## 1. Python/venv 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. 의존성 설치

### 모드 1 (점수 계산만)

```bash
pip install -r requirements.txt
```

### 모드 2 (점수 + Gemini)

```bash
pip install -r requirements-gemini.txt
```

## 3. 환경변수 준비

```bash
cp .env.example .env
```

최소 권장:

- `PYTHONPATH=.`
- `ACUT_AESTHETIC_CONFIG=configs/stage5_reference.json`

Gemini 모드 추가:

- `ENABLE_GEMINI=true`
- `GEMINI_API_KEY=<YOUR_KEY>`
- `GEMINI_MODEL_NAME=models/gemini-2.5-flash-image` (선택)

## 4. 모델 파일 검증

`configs/stage5_reference.json`의 `bundle_args` 경로에 모델이 실제로 있어야 합니다.

필수 모델:

- `checkpoints/composition_aadb_gpu/final_model.keras`
- `checkpoints/technical_koniq_gpu/final_model.keras`
- `checkpoints/technical_flive_image_gpu/final_model.keras`
- `checkpoints/technical_flive_patch_gpu/final_model.keras`
- `checkpoints/nima_ava_gpu/final_model.keras`
- `checkpoints/alamp_aadb_gpu/final_model.keras`
- `checkpoints/musiq_aadb_gpu/final_model.keras`
- `checkpoints/rgnet_aadb_gpu/final_model.keras`

## 5. preflight

### 모드 1

```bash
python scripts/preflight_check.py --mode minimal
```

### 모드 2

```bash
python scripts/preflight_check.py --mode gemini
```

## 6. 실행

### 모드 1: 점수 계산만

```bash
PYTHONPATH=. python scripts/run_local_scoring.py \
  --input_dir test_samples \
  --output_dir outputs/mode1_scores
```

출력:

- `outputs/mode1_scores/scores.jsonl`
- `outputs/mode1_scores/scores.csv`

### 모드 2: 점수 + Gemini 설명

```bash
ENABLE_GEMINI=true GEMINI_API_KEY=<YOUR_KEY> PYTHONPATH=. python -m src.infer.run_acut_pipeline \
  --input_dir test_samples \
  --output_dir outputs/mode2_pipeline \
  --top_k 5 \
  --enable_gemini
```

주요 출력:

- `app_results.json`
- `top_k_summary.json`
- `review_sheet.csv`
- `ranked_results.jsonl`
- `ranked_results.csv`
- `scores.jsonl`, `scores.csv`

## 7. 결과 확인 포인트

- 실행 stdout 마지막 JSON의 `num_images`, `num_skipped_images` 확인
- `score_failures.json` 생성 여부 확인 (실패 이미지가 있으면 생성)
- Gemini를 켠 경우 `app_results.json`의 `acut_short_reason` 등 문구 업데이트 확인
