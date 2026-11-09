#!/bin/bash
set -e

export TF_CUDNN_USE_AUTOTUNE=0
export CUDA_VISIBLE_DEVICES=""
echo "=== Forward Smoke & Tiny Train ==="
./.venv_gpu/bin/python -m src.train.train_alamp_paper_ava_v2_graph_lite \
    --config configs/paper_benchmarks/alamp_paper_ava_v2_graph_lite.yaml \
    --out_dir outputs/alamp_paper_ava_v2_graph_lite_20260511/smoke_train \
    --max_train_samples 64 \
    --max_val_samples 32 \
    --epochs 1 \
    --batch_size 2 \
    --backbone_weights none \
    --allow_cpu_mid

echo "=== Mid Run Train ==="
./.venv_gpu/bin/python -m src.train.train_alamp_paper_ava_v2_graph_lite \
    --config configs/paper_benchmarks/alamp_paper_ava_v2_graph_lite.yaml \
    --out_dir outputs/alamp_paper_ava_v2_graph_lite_20260511/mid_train \
    --max_train_samples 1024 \
    --max_val_samples 1024 \
    --epochs 1 \
    --batch_size 4 \
    --backbone_weights imagenet \
    --allow_cpu_mid

echo "=== Mid Run Eval ==="
./.venv_gpu/bin/python src/eval/evaluate_alamp_paper_ava_v2_graph_lite.py \
    --config configs/paper_benchmarks/alamp_paper_ava_v2_graph_lite.yaml \
    --model_path outputs/alamp_paper_ava_v2_graph_lite_20260511/mid_train/final_model.keras \
    --output_dir outputs/alamp_paper_ava_v2_graph_lite_20260511/eval_mid \
    --batch_size 4
