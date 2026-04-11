#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_PATH="${1:-test_samples/KakaoTalk_20260330_180646779_10.jpg}"

cd "$ROOT_DIR"

if [[ ! -f "$IMAGE_PATH" ]]; then
  echo "Sample image not found: $IMAGE_PATH" >&2
  exit 1
fi

env MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv_gpu/bin/python src/export/export_tflite.py \
  --preset aadb \
  --export_source rebuild \
  --conversion_mode builtin \
  --output_dir exports/tflite \
  --output_name composition_aadb_gpu.tflite \
  --metadata_path exports/tflite/composition_aadb_gpu.metadata.json \
  --overwrite

env MPLCONFIGDIR=/tmp/matplotlib CUDA_VISIBLE_DEVICES='' PYTHONPATH=. ./.venv_gpu/bin/python src/export/verify_tflite.py \
  --preset aadb \
  --model_path exports/tflite/composition_aadb_gpu.tflite \
  --image_path "$IMAGE_PATH" \
  --output_json exports/tflite/composition_aadb_gpu.verify.json
