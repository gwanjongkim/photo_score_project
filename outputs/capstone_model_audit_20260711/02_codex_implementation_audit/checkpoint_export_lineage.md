# Checkpoint and export lineage

`UNKNOWN_LINK` denotes an unproven transition, even when filenames appear related.

## Proven or partial chains

- KonIQ mobile: KonIQ raw scores → `make_koniq_csv.py` random manifests → `train_regression.py` → `checkpoints/technical_koniq_gpu/final_model.keras` → metadata says rebuild export → `exports/tflite/koniq_mobile.tflite` SHA-256 `67faf8…be56` → active server config → external mobile asset **UNKNOWN_LINK**.
- FLIVE image: `labels_image.csv` → random manifests → generic trainer → `technical_flive_image_gpu/final_model.keras` → rebuild export → `flive_image_mobile.tflite` SHA-256 `58aee1…5ebdb` → server runtime → mobile runtime **UNKNOWN_LINK**.
- NIMA legacy: AVA processed CSV → historical training invocation **UNKNOWN_LINK** → `checkpoints/nima_ava_gpu/final_model.keras` → `nima_mobile.tflite` SHA-256 `48ed5a…f8800` → default Stage5 reference. This is legacy, not fixed preprocessing.
- NIMA fixed: processed AVA → `train_nima.py` → `checkpoints/nima_ava_gpu_fixed_preproc_20260525/final_model.keras` → fixed FP32/FP16 exports/handoff → active default runtime **UNKNOWN_LINK**.
- RGNet practical: AADB MAT split → manifests → `train_rgnet.py` → `rgnet_aadb_gpu/final_model.keras` → `rgnet_aadb_gpu.tflite` SHA-256 `b81461…cdeb2` → copy under `models/aesthetic/` → active external app **UNKNOWN_LINK**.
- A-LAMP practical: AADB manifests + heuristic patch generation → trainer → checkpoint → `alamp_aadb_gpu.tflite` SHA-256 `8b579a…88c7` → external runtime **UNKNOWN_LINK**.
- Mobile A-LAMP v2: AVA patch JSONL → `best_val_auc_model.keras` → exact export command in `export_metadata.json` → FP32/FP16 → parity on 1,024 samples. FP16 SHA-256 `70cbc5…fa43`; app deployment **UNKNOWN_LINK**.
- Stage 5 student: teacher-target manifest provenance **UNKNOWN_LINK** → student training → Keras/checkpoint → `stage5_student…tflite` SHA-256 `0e983d…63940` → deployed runtime **UNKNOWN_LINK**; later project decision classifies it rejected.

## Absent export chains

MDIQA teacher, 50k MDIQA student, TechIQA-Guard, and DistortionGuard have no proven TFLite/mobile chain. ICAA’s local FP16 candidate has a claimed contract conflict and active app link is unproven. MUSIQ probe exports are experimental and cannot establish official-model lineage.
