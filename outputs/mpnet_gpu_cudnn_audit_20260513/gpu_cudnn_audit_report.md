# MPNet GPU cuDNN Initialization Audit (2026-05-13)

## 1. Executive Summary
TensorFlow 2.20.0 was failing to initialize the cuDNN library during training/inference (`FAILED_PRECONDITION: DNN library initialization failed`). 

Diagnostics revealed that TensorFlow 2.20.0 was compiled against **cuDNN 9.3.0**, but the environment had **cuDNN 9.1.0** installed (via `nvidia-cudnn-cu12==9.1.0.70`). Upgrading the `nvidia-cudnn-cu12` package to version `9.3.0.75` resolved the issue without breaking PyTorch 2.5.1.

## 2. Audit Findings

### Exact Cause
The binary distribution of TensorFlow 2.20.0 has a hard requirement for cuDNN >= 9.3.0. The environment had a version mismatch:
- **Loaded runtime:** 9.1.0
- **Compiled source:** 9.3.0

This discrepancy caused `InvalidArgumentError: No DNN in stream executor` whenever a Convolution operation was attempted on the GPU.

### Does TensorFlow GPU Conv work?
- **Before Fix:** No. Failed with `InvalidArgumentError`.
- **After Fix:** **Yes.** Conv2D and single-batch training (XLA) are fully functional on the RTX 4070 SUPER.

### PyTorch Compatibility
Although upgrading cuDNN triggers a pip dependency warning for PyTorch 2.5.1 (which requests 9.1.0), local verification confirms that PyTorch remains functional for Convolution operations using the newer 9.3.0 library.

## 3. Fix Applied

The following command was executed to resolve the mismatch:
```bash
./.venv_gpu/bin/python -m pip install nvidia-cudnn-cu12==9.3.0.75
```

## 4. Final Recommendation
MPNet 1024 training should proceed on GPU. The environment is now verified for:
1. GPU Visibility
2. cuDNN Initialization
3. XLA Compilation
4. Basic Training Steps

**Do not use CPU fallback** for MPNet 1024/4096 training as the performance penalty is excessive and the GPU environment is now stable.
