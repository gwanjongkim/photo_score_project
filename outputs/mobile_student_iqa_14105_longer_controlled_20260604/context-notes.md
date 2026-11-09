# Phase 9 context notes

- Scope remains Track A Option A: validated known-dataset teacher-cache subset, not MDIQA official full-dataset relabeling.
- Phase 8 short controlled training succeeded and is baseline for Phase 9 comparison.
- FLIVE remains weakest from Phase 8: test SRCC 0.2875 versus KonIQ 0.5645 and SPAQ 0.7822.
- Phase 9 should reuse smoke/short helper interfaces and avoid duplicate image loading, model, metrics, and collapse logic.
- Longer controlled run uses EfficientNetV2B0 ImageNet weights, 256px, batch 16, Huber loss, all manifest splits, 5 head epochs, 10 finetune epochs.
- This remains longer controlled training only. No final full distillation, TFLite export, Flutter change, production scoring change, production model asset change, or commit.
- Result: GPU visible, TensorFlow 2.21.0, ImageNet EfficientNetV2B0 loaded, 11283/1409/1413 train/val/test rows used, image failures 0, NaN/inf 0.
- Longer controlled decision: `STUDENT_14105_LONGER_READY_FOR_FULL_CONTROLLED_TRAINING`.
- Key movement versus short: val SRCC 0.6436 -> 0.6876, test SRCC 0.6105 -> 0.6743, test severe FP 1 -> 0, collapse false.
- FLIVE caveat: val/test FLIVE SRCC improved to 0.5213/0.5470, but FLIVE MAE worsened to 9.5486/9.4273.
