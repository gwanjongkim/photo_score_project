# RGNet TFLite Retry Context Notes

- Date: 2026-05-07.
- Working directory: `/home/omen_pc1/photo_score_project`.
- Output directory: `outputs/rgnet_tflite_retry_20260507/`.
- Input checkpoint: `outputs/ava_pretrain_aadb_finetune_20260507/rgnet_ava_pretrain_aadb_finetune/final_model.keras`.
- Reference image: `data/raw/ava/images/85837.jpg`.
- Constraints: no Flutter copy, no previous TFLite overwrite, no `src/` changes unless absolutely necessary.
- Prior RGNet retry inputs to diagnose:
  - `outputs/ava_pretrain_aadb_finetune_20260507/tflite/rgnet_ava_aadb_finetune.metadata.json`.
  - `outputs/ava_pretrain_aadb_finetune_20260507/tflite/rgnet_ava_aadb_finetune.verify.json`.
  - `outputs/ava_pretrain_aadb_finetune_20260507/tflite/rgnet_ava_aadb_finetune_rebuild_diagnostic.verify.json`.
- Known prior failure: SavedModel export required Select TF ops and local allocation failed at `FlexMul`; builtin rebuild diagnostic ran but failed parity by about 0.17775 on the smoke image.
- Direct Keras builtin conversion failed because the loaded checkpoint still contains mixed-float16 TensorFlow ops that require Flex.
- Pure float32 strict rebuild copied all 373 weights, but Keras-vs-Keras parity failed before conversion; this reproduces the old builtin rebuild mismatch as a dtype-policy/compute-path problem rather than a missing-weight problem.
- Mixed-policy rebuild matched the original Keras checkpoint exactly on 21 images before conversion, but builtin conversion failed because the mixed-float16 graph requires Select TF ops.
- Select TF ops diagnostic was created but local allocation failed at `FlexMul`; it is not deployment-ready.
- Final retry conclusion: no deployment-ready builtin RGNet TFLite artifact was produced.

