# A-LAMP GPU Acceleration Audit Report

## 1. Current Runtime Path
- **Model**: `mobile_alamp_v2_fp16.tflite`
- **Execution**: TensorFlow Lite on CPU with **XNNPACK** delegate.
- **Batching**: Image-by-image (Serial). No batching is implemented in the current benchmark script.
- **Preprocessing Bottleneck**:
  - Adaptive patch selection (`select_adaptive_patches`) is computationally expensive on CPU. 
  - It involves:
    - Spectral Residual Saliency computation.
    - Feature extraction (LAB color histograms and Canny edge histograms) for hundreds of candidate boxes.
    - Greedy selection loop with IoU and similarity constraints.
  - Initial estimates suggest preprocessing takes a significant portion of the per-image time (~0.1-0.2s).

## 2. GPU Acceleration Options

### A. TFLite Linux GPU Delegate
- **Mechanism**: Use `libtensorflowlite_gpu_delegate.so` to run TFLite ops on GPU via OpenGL or OpenCL.
- **Status**: **NOT AVAILABLE**. The delegate library was not found on the system. 
- **Complexity**: **HARD**. Requires either finding a pre-built binary matching the glibc/driver version or building from source.

### B. Original Keras/SavedModel GPU Batch Inference
- **Mechanism**: Use the original `.keras` model with standard TensorFlow GPU backend and batching.
- **Source Found**: `outputs/mobile_alamp_v2_full_ava_20260519/best_val_auc_model.keras`
- **Status**: **POSSIBLE (Environment fix required)**. 
  - The physical GPU (RTX 4070) is detected via `nvidia-smi`.
  - However, TensorFlow in `.venv_gpu` and `.venv_tf` cannot currently load the CUDA/cuDNN libraries.
- **Complexity**: **EASY** (if environment is fixed).

### C. Android GPU Delegate
- **Mechanism**: TFLite GPU acceleration on mobile devices.
- **Status**: **NOT APPLICABLE** for this WSL benchmark audit.

## 3. Feasibility Verdict

| Option | Verdict | Complexity | Note |
| :--- | :--- | :--- | :--- |
| **TFLite CPU** | **RUNNING** | - | Current baseline. Correct but slow. |
| **TFLite GPU** | **HARD** | High | Delegate library missing. |
| **Keras GPU** | **POSSIBLE** | Medium | Requires fixing LD_LIBRARY_PATH for WSL. |

## 4. Recommended Path
1. **Finish current run**: Do not stop the CPU run if it is already progressing significantly (e.g., >20%).
2. **Fix GPU Environment**: For future benchmarks, ensure `LD_LIBRARY_PATH` includes the WSL-specific CUDA libraries (usually under `/usr/lib/wsl/lib`).
3. **Transition to Keras Batching**: If GPU is fixed, use the original Keras model. This allows true batching (e.g., 32-64 images at once) and likely 10x+ speedup.
4. **Parity Check**: Before using Keras for final reports, run the parity test below.

## 5. Proposed Parity Test (TFLite vs Keras)
Run a 100-image test to ensure the Keras model and the TFLite exported version produce identical scores.

### Metrics to Track:
- **Max Absolute Difference**: Expected < 1e-3.
- **Mean Absolute Difference**: Expected < 1e-4.
- **SRCC (Rank correlation)**: Expected > 0.999.

### Diagnostic Script (Mockup):
```python
# Load TFLite
interpreter = tf.lite.Interpreter(model_path=\"...fp16.tflite\")
# Load Keras
keras_model = tf.keras.models.load_model(\"...model.keras\", custom_objects=...)

# For 100 images:
# 1. Run tflite_pred
# 2. Run keras_pred (batched)
# 3. Compare
```

## 6. Conclusion
The current A-LAMP benchmark is limited by CPU serial execution and expensive adaptive preprocessing. While a GPU is present, it is not currently accessible to TensorFlow. Fixing the WSL library linking would enable a shift to **batched Keras GPU inference**, which is the most realistic path to acceleration.
