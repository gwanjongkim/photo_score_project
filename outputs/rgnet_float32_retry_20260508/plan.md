# RGNet Float32 Retry Plan

Assumptions.
- The output directory for this run is `outputs/rgnet_float32_retry_20260508/`.
- Existing dirty git files are unrelated and will not be modified.
- `src/train/train_rgnet.py` is not suitable because it enables `mixed_float16` on GPU.
- The experiment will use only experiment-local scripts under `outputs/rgnet_float32_retry_20260508/scripts/`.

Plan.
1. Create float32-only training/export/evaluation scripts that reuse the local RGNet model and CSV dataset code.
2. Run a 1-epoch AVA smoke train using the fixed smoke CSVs.
3. Export the smoke model to builtin-only TFLite and require Keras-vs-TFLite parity before full training.
4. If smoke passes, train full AVA pretrain under float32.
5. Fine-tune the AVA model on AADB under float32.
6. Evaluate baseline, previous mixed fine-tune, new AVA pretrain, and new AVA-AADB fine-tune on the fixed AVA/AADB 512 subsets.
7. Export the new fine-tuned model to builtin-only TFLite and verify parity.
8. Only if parity passes, evaluate the TFLite model on the fixed subsets.
9. Write the markdown report, summary JSON, and command log.

Success criteria.
- Builtin-only TFLite conversion succeeds without Select TF ops.
- Standard `tf.lite.Interpreter` loads and allocates the model.
- TFLite input is `[1, 256, 256, 3]` and output is `[1, 1]`.
- Keras-vs-TFLite max absolute diff is `<= 1e-4` on the reference image and at least 20 images.
- AADB val512 SRCC beats the original RGNet AADB baseline of `0.4983`.
- No Flutter copy is performed.
