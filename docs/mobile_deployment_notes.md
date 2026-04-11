# Mobile Deployment Notes

## Recommended First TFLite Targets

- `checkpoints/technical_koniq_gpu/final_model.keras`
  Reason: single-input MobileNetV2 regressor, smallest practical quality model in the repo.
- `checkpoints/technical_flive_image_gpu/final_model.keras`
  Reason: single-input MobileNetV2 regressor, same deployment profile as KonIQ.
- `checkpoints/composition_aadb_gpu/final_model.keras`
  Reason: single-input EfficientNetV2B0 regressor, heavier than MobileNetV2 but still a reasonable non-real-time on-device target.
- `checkpoints/nima_ava_gpu/final_model.keras`
  Reason: single-input EfficientNetV2B0 with a 10-bin softmax head; plausible for on-device batch scoring if latency is acceptable.

## Models Better Kept Off-Device

- `checkpoints/technical_flive_patch_gpu/final_model.keras`
  Blocker: requires multiple patch crops and repeated inference per image.
- `checkpoints/alamp_aadb_gpu/final_model.keras`
  Blocker: multi-input global plus patch model with custom layers and preprocessing.
- `checkpoints/musiq_aadb_gpu/final_model.keras`
  Blocker: multi-input token pipeline, Lambda-serialized artifact, SavedModel fallback needed in this repo.
- `checkpoints/rgnet_aadb_gpu/final_model.keras`
  Blocker: custom graph layers and composition graph construction.
- `checkpoints/pairwise_aadb_gpu/final_model.keras`
  Blocker: pair input, relative comparison, and quadratic reference cost for reranking workflows.

## Practical Export Checklist

1. Start with KonIQ and FLIVE-image from their SavedModel exports, not the heavier selector stack.
2. Validate one-image latency and memory on the actual target phone before adding AADB.
3. Add NIMA only after the single-pass technical models are stable.
4. Keep A-Lamp, MUSIQ, RGNet, and pairwise in desktop/server or offline best-shot flows until there is a separate export effort for each.
5. Treat SavedModel availability as a deployment asset; there is no current repo TFLite pipeline yet.
