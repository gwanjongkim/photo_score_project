# Production Performance Bottleneck Audit Report

## 1. Executive Summary
The production technical scoring script is slow (~10s per image initially, ~4s per image steady-state) because it executes a sequence of heavy Keras models on the **CPU**. While the system has an RTX 4070 SUPER, the TensorFlow environment in `.venv_gpu` has a CuDNN version mismatch (9.1.0 runtime vs 9.3.0 compiled), causing DNN initialization to fail and forcing a CPU fallback. MDIQA (PyTorch) is not affected and runs correctly on the GPU.

## 2. Component GPU Usage Status
| Component | Framework | GPU Status | Note |
|---|---|---|---|
| MDIQA | PyTorch | **ENABLED** | Uses CUDA correctly |
| Production Ensemble | TensorFlow/Keras | **FAILED (CPU Fallback)** | CuDNN version mismatch |
| Image Loading | PIL | CPU | standard |
| Preprocessing | NumPy | CPU | standard |

## 3. Bottleneck Analysis
- **Exact Bottleneck**: Sequence of 9+ model calls per image (KonIQ, FLIVE Image, 5x FLIVE Patches, AADB, MUSIQ).
- **Execution Mode**: Image-by-image (serial).
- **Dominant Costs**: 
  - AADB Predict: ~1.4s (CPU)
  - MUSIQ Predict: ~0.6s (CPU)
  - KonIQ Predict: ~0.7s (CPU)
  - FLIVE Image/Patch: ~1.5s total (CPU)
- **Total Overhead**: ~4.2s per image after initial warm-up.

## 4. Potential Optimizations
- **GPU Acceleration**: Possible if CuDNN versions are aligned in `.venv_gpu`. This would likely reduce inference time by 10-20x.
- **Batch Inference**: Currently not implemented in `run_paired_production_predictions_for_mdiqa.py`. Implementing batching for the backbone models (KonIQ, FLIVE) is possible and would reduce Python/NumPy overhead.
- **Selective Evaluation**: If only KonIQ/SPAQ baselines are needed, disabling FLIVE patches and MUSIQ would save ~50% of the time, but this would deviate from the "Production" score definition.

## 5. Safest Code Changes
1. **Fix TensorFlow GPU**: Align CuDNN libraries or use a compatible runtime environment.
2. **Batching**: Modify the prediction loop to pass image batches to the models.
3. **Resume Support**: Ensure the script can resume from a partial `paired_predictions.csv` to avoid re-evaluating 10k images on failure.

## 6. Production Semantic Impact
The production technical scoring logic is defined in `src/infer/select_best_shots.py` using specific weights. Any optimization (batching, GPU) will maintain these semantics as long as the model weights and normalization steps remain unchanged.
