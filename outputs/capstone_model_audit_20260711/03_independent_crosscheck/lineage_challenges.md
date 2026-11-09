<!-- 체크포인트부터 TFLite 파일로의 학습 이력 추적성 및 패리티 검증의 한계를 분석하는 문서. -->
# Lineage Challenges

This report evaluates the traceability gaps and lineage challenges between Keras training checkpoints and mobile TFLite models in the repository.

---

## 1. Unverified training invocations

For almost all local models (KonIQ mobile, FLIVE image, legacy NIMA, practical RGNet, and practical A-LAMP), the original training commands, seeds, exact target column references, and execution stdout logs are missing. The repository contains the final Keras checkpoints (`final_model.keras`) and weights (`best.weights.h5`), but we cannot trace how these weights were generated:
- **KonIQ mobile**: The csv builder selects `MOS_zscore` but the training configuration refers to MSE/MAE on raw MOS. It is unproven whether the model was trained on raw MOS or z-scores.
- **RGNet practical**: The exact training command is undocumented.
- **Legacy NIMA**: The training run is completely unrecorded in git history.

---

## 2. Checkpoint-to-TFLite matching gaps

The repository contains multiple copies of TFLite files in different folders:
- `exports/tflite/`
- `forWeights/`
- `forWeights2/`
- `models/`

For example:
- `exports/tflite/koniq_mobile.tflite`
- `forWeights2/models/aesthetic/nima_mobile_fixed_preproc_fp16.tflite`

While the verification script `verify_tflite.py` reports successful execution, there is no build-system proof that the TFLite file in `forWeights2/` was compiled from the *exact* Keras checkpoint currently stored in `checkpoints/`. In several cases, the TFLite file creation date/mtime is older than the Keras checkpoint mtime, suggesting that the TFLite file was built from an earlier version of the weights or copied from an external source, introducing a potential mismatch.

---

## 3. Insufficient Python-to-TFLite Parity Checks

The parity verification script (`src/export/verify_tflite.py` and `verify_student_tflite.py`) computes parity metrics (e.g. max absolute difference) over a very small set of validation samples (usually 32 or 50 samples) using a simplified data loader.
- **Risks**:
  - Parity is not accuracy. A TFLite model that has low absolute difference on 32 images could still fail or diverge on the full test distribution.
  - The parity checks do not use the same image loading pipeline as the mobile app. For instance, the mobile app (written in Dart) may use different JPEG decoding, bilinear interpolation, or pixel-value divisions. Checking python-to-TFLite parity on Python-decoded images does not guarantee parity on Dart-decoded images.
  - No parity check was run for the FLIVE patch, TechIQA-Guard, or 50k MDIQA student models (the latter was intentionally rejected before export).

---

## 4. Discrepancies in ICAA Mobile Weight Files

In `outputs/icaa_performance_reverification_20260525/report.md`, the ICAA FP16 TFLite model is described as being approximately 11.9 MB. However, the actual file found at `forWeights2/models/aesthetic/icaa_dat_tf_native_fp16.tflite` is **175,885,416 bytes** (approximately 167 MB).
- **Implication**: This massive size discrepancy indicates that the model was either not fully quantized to FP16 (or it contains large unquantized layers/weights), or the file checked into the repository is a different, unquantized candidate. A 167 MB file is generally unsuitable for standard mobile deployments, contradicting the production claims in the project files.
- **Contract Conflict**: The weight-lab configuration in `forWeights2/configs/aesthetic_weight_lab.yaml` uses zero-to-one normalization and output index 0 for ICAA, while the validated export pipeline uses ImageNet normalization and output index 1. This represents a complete break in the deployment contract.
