# MPNet GPU Environment Audit Report (2026-05-13)

## 1. Executive Summary
The previous MPNet training/evaluation run accidentally executed on CPU because the environment variable `CUDA_VISIBLE_DEVICES` was set to an empty string (`""`). This state explicitly instructs CUDA-enabled libraries (TensorFlow, PyTorch) to ignore all available GPUs.

Current auditing confirms that the NVIDIA hardware is fully accessible via WSL, and both TensorFlow and PyTorch are correctly configured to use the GPU when the environment variable is properly set or unset.

## 2. Audit Findings

### Is CUDA_VISIBLE_DEVICES disabling GPU?
**Yes.** In the previous session reported by the user, `export CUDA_VISIBLE_DEVICES=""` was used. My tests confirm that this causes TensorFlow to see 0 GPUs and throw `CUDA_ERROR_NO_DEVICE`.

### Does WSL see the NVIDIA GPU?
**Yes.** `nvidia-smi` correctly identifies the **NVIDIA GeForce RTX 4070 SUPER** with Driver Version 591.55 and CUDA 13.1.

### Does PyTorch see GPU?
**Yes.** In `.venv_gpu`, `torch.cuda.is_available()` returns `True` and correctly identifies the RTX 4070 SUPER.

### Does TensorFlow see GPU?
**Yes.** In `.venv_gpu`, `tf.config.list_physical_devices("GPU")` returns `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]` provided that `CUDA_VISIBLE_DEVICES` is not empty.

### Root Cause Analysis
If TensorFlow does not see the GPU but PyTorch does, it is often due to mismatched `nvidia-cudnn-cu12` or `nvidia-cublas-cu12` versions inside the venv, or a hardcoded `CUDA_VISIBLE_DEVICES` in an activation script. However, in this case, the root cause was the manual export of an empty string to `CUDA_VISIBLE_DEVICES`.

## 3. Improvements Implemented

### Abort-if-no-GPU Guard
Modified `src/train/train_alamp_paper_mpnet_ava.py` to include a strict GPU check:
- Added `--require_gpu` flag (defaults to `True`).
- Added logic to `_setup_tensorflow` to log an error and exit if no GPUs are detected when `require_gpu` is active.
- This prevents silent CPU fallback, which is critical for expensive 1024/4096 sample training runs.

## 4. Recommended Commands for MPNet V4 1024 Training

To run the training on GPU, ensure the environment variable is corrected before execution:

```bash
# 1. Restore GPU visibility
unset CUDA_VISIBLE_DEVICES

# 2. Run training with GPU requirement
export PYTHONPATH=$PYTHONPATH:.
./.venv_gpu/bin/python src/train/train_alamp_paper_mpnet_ava.py \
    --config configs/aesthetic_weight_lab.yaml \
    --selector_name mpnet_v4_1024 \
    --max_train_samples 1024 \
    --require_gpu
```

If you wish to force CPU (not recommended), use `--no_gpu`.
