# TensorFlow GPU Visibility Fix Report

## Executive Summary
- Final decision: `TF_GPU_FIXED_AND_50K_SMOKE_PASS`.
- Correct rerun environment: `.venv_tf`.
- Required execution pattern: unsandboxed `env -u LD_LIBRARY_PATH`, with `LD_LIBRARY_PATH=/usr/lib/wsl/lib:<.venv_tf nvidia lib dirs>`.
- Root cause of original `GPU=[]`: current Codex sandbox hides WSL GPU device interface (`/dev/dxg` absent), so TensorFlow and PyTorch cannot initialize CUDA inside sandbox.
- `.venv_gpu` is not suitable for TensorFlow 2.21 training now because its runtime CuDNN is `9.1.0`, below TensorFlow build requirement `9.3.0`.
- `.venv_tf` has compatible NVIDIA wheel libs and completed the bounded 4000/1000/1000 smoke.

## Scope
- No full training.
- No TFLite export.
- No Flutter changes.
- No production `technical_score` changes.
- No commit.
- Student trainer was not rewritten.

## Trainer Check
- `scripts/technical_iqa/train_mobile_student_iqa_50k_smoke.py` supports `--out_dir`.
- Bounded reruns used new output dirs, so previous `outputs/mobile_student_iqa_50k_smoke_20260608/` was not overwritten.

## Hardware / Driver
- `nvidia-smi` sees GPU hardware.
- Driver Version: `591.55`.
- CUDA Version shown by driver: `13.1`.
- GPU memory at preflight: `1234 MiB / 12282 MiB`.

## Environment Findings

### Sandboxed `.venv_gpu`
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs: `[]`.
- Error: `failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: UNKNOWN ERROR (100)`.
- PyTorch CUDA check also failed: `cuda_available False`, `device_count 0`.
- `/dev/dxg` absent.

### Sandboxed `.venv_tf`
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- GPUs: `[]`.
- Same CUDA init error.

### Unsandboxed `.venv_gpu`
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- `/dev/dxg`: `True`.
- GPUs: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.
- Bounded smoke started but failed at training because runtime CuDNN was `9.1.0` while TensorFlow was compiled with `9.3.0`.

### Unsandboxed `.venv_tf`
- TensorFlow: `2.21.0`.
- CUDA build: `True`.
- `/dev/dxg`: `True`.
- GPUs: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`.
- NVIDIA CuDNN package: `nvidia-cudnn-cu12 9.23.0.39`.
- Selected as correct environment.

## Bounded Smoke Rerun
- Command used `.venv_tf`.
- Output path: `outputs/mobile_student_iqa_50k_smoke_gpu_fixed_venv_tf_20260608/`.
- Train/val/test counts: `4000 / 1000 / 1000`.
- Runtime: `155.24` seconds.
- GPU status: TensorFlow GPU visible, GPU count `1`.
- ImageNet weights: loaded.
- Final train loss: `0.002758`.
- Final val loss: `0.002619`.
- Prediction std: train `9.462186`, val `8.815579`, test `7.993017`.
- Failure count: `0`.
- NaN/Inf count: `0`.
- Collapse status: no collapse; `pred_std_too_low=False`, `pred_outside_0_100=False`.
- Trainer final decision: `STUDENT_50K_SMOKE_READY_FOR_CONTROLLED_TRAINING`.

## Final Decision
`TF_GPU_FIXED_AND_50K_SMOKE_PASS`

## Recommended Next Step
Use `.venv_tf` for the next controlled run with the same unsandboxed WSL CUDA launch pattern. Do not use `.venv_gpu` for TensorFlow training until its CuDNN runtime is upgraded.
