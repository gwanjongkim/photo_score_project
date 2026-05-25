# RGNet Float32 Retry Checklist

- [x] Record environment discovery.
- [x] Confirm output directory does not overwrite previous `rgnet_float32_retry_*` outputs.
- [x] Inspect existing RGNet training/export paths.
- [x] Create experiment-only float32 scripts.
- [x] Smoke train RGNet float32 for 1 epoch.
- [x] Export smoke model to builtin-only TFLite.
- [x] Verify smoke Keras-vs-TFLite parity.
- [x] Full AVA float32 pretrain.
- [x] Full AADB float32 fine-tune.
- [x] Keras fixed-subset evaluation.
- [x] Builtin-only TFLite export of fine-tuned model.
- [x] TFLite parity validation.
- [x] TFLite fixed-subset evaluation if parity passes.
- [x] Write final report and summary JSON.
- [x] Verify final JSON parses.
