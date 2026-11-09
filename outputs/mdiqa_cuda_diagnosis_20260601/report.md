# MDIQA CUDA Diagnosis Report

## Summary
- Environment: `/home/omen_pc1/photo_score_project/.venv_mdiqa`.
- `nvidia-smi` sees one NVIDIA GeForce RTX 4070 SUPER.
- Installed torch is CUDA-enabled, not CPU-only: `torch==2.12.0+cu130`.
- `torch.version.cuda`: `13.0`.
- In the normal sandboxed shell, `torch.cuda.is_available()` is `False` and `torch.cuda.device_count()` is `0`.
- The normal shell inherits `.venv_gpu` CUDA/cuDNN paths in `LD_LIBRARY_PATH`, causing a cuDNN mismatch.
- With `LD_LIBRARY_PATH` unset and the CUDA probe run outside the sandbox, `.venv_mdiqa` CUDA works.

## Evidence
### nvidia-smi
```text
NVIDIA-SMI 590.48.01
Driver Version: 591.55
CUDA Version: 13.1
GPU 0: NVIDIA GeForce RTX 4070 SUPER
Memory: 12282 MiB
```

### Sandboxed `.venv_mdiqa` torch probe
```text
torch_version 2.12.0+cu130
torch_version_cuda 13.0
cuda_is_available False
cuda_device_count 0
cudnn_version_error RuntimeError cuDNN version incompatibility: PyTorch was compiled against (9, 20, 0) but found runtime version (9, 1, 0).
```

### Clean `LD_LIBRARY_PATH`, outside sandbox
```text
LD_LIBRARY_PATH None
torch 2.12.0+cu130
torch.version.cuda 13.0
cuda_available True
device_count 1
cudnn_version 92000
cuda_tensor_ok tensor([1.], device='cuda:0')
device_name NVIDIA GeForce RTX 4070 SUPER
```

## Interpretation
- Reinstalling torch is not needed.
- The installed torch build is CUDA-enabled and compatible with the visible GPU when its bundled CUDA/cuDNN libraries are not shadowed.
- For GPU MDIQA runs, use a clean library environment, for example:

```bash
env -u LD_LIBRARY_PATH ./.venv_mdiqa/bin/python scripts/technical_iqa/run_mdiqa_sanity_benchmark.py
```

## Recommendation
- Do not reinstall torch.
- Run the sanity benchmark with `LD_LIBRARY_PATH` unset and GPU access enabled.
