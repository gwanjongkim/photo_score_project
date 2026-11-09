# A-CUT Public Model Repositories

#### A-CUT Capstone Project — Mobile Image Quality and Aesthetic Assessment

This repository index lists the separate, independently runnable evaluation/export repositories for the various models explored during the A-CUT project.

Each repository is modularized to contain its own scripts, configurations, and documentation. Large model artifacts (checkpoints, TFLite files >50MB) and raw datasets are **excluded** due to GitHub file size limitations and licensing. They should be downloaded via GitHub Releases or Git LFS as documented in each repository's `models/download.md`.

## Repositories

| Repository | Task | Status | Main Evidence | Upload Decision |
|------------|------|--------|---------------|-----------------|
| [acut-nima-mobile](./acut-nima-mobile) | Aesthetic | mobile-optimized aesthetic assessment model | `nima_mobile_fixed_preproc_fp16.tflite` | Production |
| [acut-rgnet-mobile](./acut-rgnet-mobile) | Aesthetic | mobile-optimized or adapted aesthetics model | `rgnet_pil_resize_aadb_fp16.tflite` | Production |
| [acut-mobile-alamp](./acut-mobile-alamp) | Aesthetic | mobile practical multi-patch aesthetics model | `mobile_alamp_v2_fp16.tflite` | Production |
| [acut-icaa-color-aesthetics](./acut-icaa-color-aesthetics) | Aesthetic | color aesthetics model / TF-native or TFLite adaptation | `icaa_dat_tf_native.tflite` parity evidence | Evaluated / Converted |
| [acut-technical-iqa](./acut-technical-iqa) | Technical IQA | umbrella benchmark repo for technical quality assessment | KonIQ + FLIVE baseline benchmark | Evaluated / Baseline |
| [acut-topiq-mixed112](./acut-topiq-mixed112) | Technical IQA | TOPIQ-lite mixed replay candidate | Parity/recovery evidence | Evaluated (Not for prod) |
| [acut-topiq-flive-hybrid](./acut-topiq-flive-hybrid) | Technical IQA | best offline/log-only technical IQA candidate | `topiq_existing_flive_57_43` results | Best Log-Only |
| [acut-c6-topiq-koniq](./acut-c6-topiq-koniq) | Technical IQA | evaluated but not selected | `C6` cap results | Negative Result |
| [acut-mdiqa-benchmark](./acut-mdiqa-benchmark) | Technical IQA | teacher-only benchmark / not mobile production | Official protocol results | Benchmark Only |
| [acut-musiq-probe](./acut-musiq-probe) | Technical IQA | probe / failed or weak local attempt | Probe reports | Probe/Negative Result |
| [acut-student-iqa](./acut-student-iqa) | Technical IQA | student/mobile technical IQA attempt | `mobile_student_iqa_14105` evidence | Evaluated |

## Note on Artifacts

If you require access to the weights, please see `models/download.md` within the specific model's repository.
