# MUSIQ WSL/Kaggle Audit Report

## 1. Executive Summary
The project currently utilizes a "repo-practical" MUSIQ implementation (simplified Transformer architecture) which has been trained locally on KonIQ-10k, SPAQ, and FLIVE. However, recent benchmarks of the TFLite-exported versions of these local models show near-zero SRCC, suggesting a failure in either the training pipeline or the inference normalization. Public Google MUSIQ models on Kaggle provide a high-quality alternative as teachers. These models are official research-grade Transformers that process images at multiple scales and arbitrary aspect ratios.

## 2. Existing WSL MUSIQ Artifacts
*   **Architecture**: Defined in `src/models/musiq.py`. Features multi-scale patch token processing but simplified token budgeting.
*   **Checkpoints**:
    *   `checkpoints/musiq_aadb_gpu/`: Aesthetic variant (AADB).
    *   `outputs/probe1024_musiq_koniq/`: Technical variant (KonIQ-10k).
    *   `outputs/probe1024_musiq_spaq/`: Technical variant (SPAQ).
    *   `outputs/probe1024_musiq_flive/`: Technical variant (FLIVE).
*   **Exports**: TFLite versions exist in `exports/tflite/` (fp16 and fp32).
*   **Status**: Locally trained "practical" models appear unreliable based on `outputs/eval_tflite_technical_compare/summary.csv` (SRCC < 0.1).

## 3. Kaggle Model Availability
Official Google MUSIQ models are available on Kaggle and can be loaded via `tensorflow_hub`. They are not currently downloaded locally.
*   **Base URL**: `https://www.kaggle.com/models/google/musiq`
*   **Connectivity**: Verified reachable from WSL via `curl`.

## 4. Candidate Variants
For technical IQA teacher roles, we prioritize the following variants:
1.  **KonIQ-10k**: Technical quality focus. Paper SRCC: 0.927.
2.  **SPAQ**: Technical quality focus. Paper SRCC: 0.911.
3.  **PaQ-2-PiQ**: Technical quality focus. Paper SRCC: 0.881.
4.  **AVA**: Aesthetic focus. (Priority: Low - for reference only).

## 5. Environment Requirements
*   **Environment**: `.venv_tf` (currently has TensorFlow 2.21.0).
*   **Required Packages**: `tensorflow-hub` and `kagglehub` (missing, must be installed).
*   **GPU**: RTX 4070 SUPER is available and recommended for these large Transformer models.

## 6. Smoke Test Plan
1.  Activate `.venv_tf`.
2.  Install `tensorflow-hub` and `kagglehub`.
3.  Create `scripts/smoke_test_kaggle_musiq.py`:
    *   Load `google/musiq/tf2/koniq-10k/1`.
    *   Run inference on 5 local images (`test_samples/`).
    *   Print raw scores (expected range: 1..10).
4.  Confirm the model handles different image dimensions correctly.

## 7. Benchmark Plan Against Production
Once verified, evaluate the Kaggle variants against the research protocols:
*   **Manifests**: Use `outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
*   **Baselines**:
    *   **KonIQ-10k**: Target > 0.934 (Current production baseline is very high).
    *   **SPAQ**: Target > 0.873.
    *   **FLIVE**: Target > 0.574.
*   **Goal**: Determine if any official MUSIQ variant provides a statistically significant improvement over the current scoring stack.

## 8. Risks
*   **Performance vs. Baseline**: The production KonIQ-10k baseline (0.934) is extremely high; official MUSIQ might not exceed it on that specific dataset.
*   **Inference Latency**: Being full Transformers, these models are significantly slower than the "practical" versions or the current baseline.
*   **Normalization**: Differences in MOS scaling (1..10 vs 0..100) must be carefully mapped.

## 9. Final Decision
**Decision: MUSIQ_READY_FOR_SMOKE_SETUP**

## 10. Exact Next Commands
```bash
# 1. Activate environment
source .venv_tf/bin/activate

# 2. Install required packages
pip install tensorflow-hub kagglehub

# 3. Verify variant loading (example: KonIQ)
python3 -c "import tensorflow_hub as hub; model = hub.load('https://www.kaggle.com/models/google/musiq/TensorFlow2/koniq-10k/1'); print('Model loaded successfully')"
```
