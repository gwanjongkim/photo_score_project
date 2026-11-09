# Phase 8 context notes

- Scope stayed Track A: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.
- Reused `scripts/technical_iqa/train_mobile_student_iqa_14105_smoke.py` functions to avoid duplicating data loading, model construction, metrics, and collapse checks.
- Created `scripts/technical_iqa/train_mobile_student_iqa_14105_short.py` only as a short controlled wrapper with required CLI flags.
- Sandbox TensorFlow preflight did not expose GPU, but escalated required GPU-safe command did expose RTX 4070 SUPER through TensorFlow 2.21.0.
- Short run used ImageNet EfficientNetV2B0 weights, 256px images, batch 16, Huber loss, 3 head epochs, 3 finetune epochs, max 4000 train, 1000 val, 1000 test.
- Final decision was `STUDENT_14105_SHORT_READY_FOR_LONGER_TRAINING` because GPU and ImageNet loaded, losses were finite, predictions did not collapse, MAE/SRCC improved over smoke, and severe FP decreased.
- This remains short controlled training only. No full distillation, TFLite export, Flutter change, production scoring change, production model asset, or commit was performed.
