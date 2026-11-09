# TensorFlow GPU Environment Audit

## 1. Environments Checked
- `.venv_tf`: TensorFlow 2.21.0, cuDNN 9.23.0.39. GPU visible.
- `.venv_gpu`: TensorFlow 2.21.0, cuDNN 9.1.0 (mismatch with compiled 9.3.0). GPU visible to TF, but crashes on DNN initialization.

## 2. Selected Environment
- `.venv_tf` is the correct environment.
- The `LD_LIBRARY_PATH` must be dynamically set to include the `pip`-installed NVIDIA libraries within the virtual environment.

## 3. Decision
**PHASE1_READY_FOR_EXPANDED_PAIRED_RUN**
GPU is usable.
