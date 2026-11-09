# Unresolved findings

1. Which independent baseline directory was intended by the user—the absent generic path or the available Codex baseline?
2. Do serialized MobileNetV2/EfficientNetV2 checkpoints contain internal rescaling, and what exact tensors did training/export/mobile provide?
3. Which NIMA artifact is active in the actual mobile application, and is it byte-identical to the fixed export?
4. What are the official filename/content overlaps for FLIVE and KonIQ relative to the project random splits?
5. Are there content-hash or perceptual near duplicates across any train/validation/test or teacher/student manifests?
6. Where are immutable raw predictions with ground truth for every reported model and ensemble metric?
7. Is calibrated aesthetic ensemble weighting applied in active runtime, and if so where are scale transforms encoded?
8. What local file implements the ICAA DAT TensorFlow/mobile port, its output heads, label mapping and normalization?
9. Can every MDIQA checkpoint be loaded against the exact upstream commit/options and run once deterministically?
10. Were 14k/50k student targets generated exclusively from teacher predictions, ground-truth MOS, or a mixture, and before or after split assignment?
11. Are any rejected Stage5/guard/student exports present in external application assets despite local no-go reports?
12. Can Python-to-TFLite-to-mobile parity be replayed on the exact same image bytes and preprocessing for all production candidates?
