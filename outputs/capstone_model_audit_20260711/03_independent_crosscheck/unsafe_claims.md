<!-- 안전성 검토에 따라 지원되지 않거나 상충되는 프로젝트 주장들을 문서화하는 보고서. -->
# Unsafe Claims Review

This report identifies project and repository claims that are unsafe, unsupported by evidence, or technically contradicted by the source code.

---

## 1. Naming mobile models after paper architectures

- **Unsafe Claim**: Describing the local MobileNetV2 regressors as "KonIQ Mobile" (or implying equivalence to "KonCept512") and "FLIVE Mobile" (or implying equivalence to "PaQ-2-PiQ").
- **Why it is unsafe**: Training a simple MobileNetV2 on an IQA dataset does not make the model a faithful reproduction of the corresponding paper architectures. `KonCept512` requires Inception-ResNet-v2 at 512x384, and `PaQ-2-PiQ` requires a multi-scale local-to-global ROI architecture with joint patch/picture loss. These claims inflate the reproducibility status of the models.

---

## 2. Preprocessing-Fixed Checkpoint Deployed in Production

- **Unsafe Claim**: Claiming that the double-rescaling preprocessing fix for NIMA has been deployed.
- **Why it is unsafe**: The default server configuration (`configs/stage5_reference.json`) still points to the old legacy checkpoint `checkpoints/nima_ava_gpu/final_model.keras` which contains the double-rescaling bug. The fixed preprocessing candidate (`checkpoints/nima_ava_gpu_fixed_preproc_20260525/`) is not loaded by default, meaning that the production server continues to run the defective model.

---

## 3. ICAA DAT Deployment and Color Score Contract

- **Unsafe Claim**: Claiming ICAA DAT has an active mobile deployment or is ready for deployment.
- **Why it is unsafe**:
  1. The local model file `src/models/icaa_dat.py` is missing entirely.
  2. The checked-in FP16 native TFLite file size is **175,885,416 bytes**, which is far too large for standard mobile deployments.
  3. The validated export pipeline expects ImageNet normalization and outputs color score at index 1, while the mobile-lab config (`aesthetic_weight_lab.yaml`) specifies zero-to-one normalization and reads index 0 (MOS). This contract contradiction would lead to incorrect scores if loaded in runtime.

---

## 4. Distilled Student Models are Deployed

- **Unsafe Claim**: Claiming that the MDIQA 14k/50k student models or the Stage 5 student are active in the application.
- **Why it is unsafe**: While Keras and TFLite files exist for these student models, later design/thesis reports explicitly classify the student distillation lines as **REJECTED** or "no-go". No active production runtime configs load these files.

---

## 5. Verified late-fusion correlation

- **Unsafe Claim**: Reporting that the late-fusion aesthetic and technical ensemble correlations have been independently verified.
- **Why it is unsafe**: The audit prediction tables do not contain the raw ground-truth column, preventing the independent calculation of PLCC or SRCC.

---

## 6. Late fusion of incompatible units

- **Unsafe Claim**: Presenting the late-fusion formula in `configs/stage5_reference.json` as a valid aesthetic score.
- **Why it is unsafe**: The formula linearly combines scores from AADB composition `[0, 1]`, legacy NIMA `[1, 10]`, A-LAMP `[0, 1]`, and RGNet `[0, 1]` without first aligning their scales. This lack of calibration makes the fused output mathematically inconsistent and dominated by NIMA scores.
