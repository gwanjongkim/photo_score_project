# Mobile Student IQA 50k Smoke Training Report

## 1. Executive Summary
- Final decision: `STUDENT_50K_SMOKE_FAILED`.
- Run type: bounded smoke training only; no full training, no TFLite export, no Flutter edits, no production `technical_score` edits.
- Runtime seconds: `53.55`.
- Usable train/val/test counts: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Final train loss: `nan`.
- Final val loss: `nan`.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv`
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_distillation_50k_full_targets_20260608/report.md`
- `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py`
- `scripts/technical_iqa/train_mobile_student_iqa_v4a.py`
- `scripts/technical_iqa/train_mobile_student_iqa_v4b.py`
- `scripts/technical_iqa/make_student_training_manifest_50k.py`

## 3. Manifest and Split Construction
- Input rows: `50000`.
- Official Track B rows excluded before split: `0`.
- Duplicate image paths in input: `0`.
- Full split counts before caps: `{'train': 40000, 'val': 5004, 'test': 4996}`.
- Capped counts before validation: `{'train': 4000, 'val': 1000, 'test': 1000}`.
- Image-path overlap across splits: `0`.
- Source-group overlap across splits: `0`.
- Official Track B rows in train: `0`.

## 4. Environment / GPU Preflight
- TensorFlow: `2.21.0`.
- GPUs visible to TensorFlow: `["PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')"]`.
- GPU count: `1`.
- Execution mode: `GPU`.

## 5. Smoke Training Configuration
- Backbone: `efficientnetv2b0`.
- Image size: `256`.
- Batch size: `16`.
- Head epochs: `1`.
- Finetune epochs: `1`.
- Loss: `huber`.
- target_norm: `teacher_target_0_100 / 100`.
- ImageNet weights requested: `imagenet`.
- ImageNet weights loaded: `True`.
- Backbone weights error: ``.
- Include preprocessing: `True`.
- Finetune trainable backbone layers: `None`.

## 6. Training History
_No rows._

## 7. Prediction and Metric Summary
_No rows._

## 8. Dataset-wise Metrics
_No rows._

## 9. Failure Cases
- Failure CSV: `outputs/mobile_student_iqa_50k_smoke_gpu_fixed_20260608/failure_cases.csv`.
- Failure rows: `1`.

## 10. Collapse / Safety Checks
- Collapse check: `{'not_run': True}`.

## 11. Comparison Against 14k Student Baseline
- Previous 14k student official SRCC: `0.7436`.
- MDIQA teacher official SRCC: `0.8227`.
- Prior gap decision: `STUDENT_GAP_MAINLY_CAPACITY_AND_COVERAGE`.
- This smoke is not a final official benchmark; compare only for pipeline sanity.

## 12. Final Decision
STUDENT_50K_SMOKE_FAILED

## 13. Recommended Next Step
- Inspect failures, target ranges, prediction spread, and dataloader logs before retry.
