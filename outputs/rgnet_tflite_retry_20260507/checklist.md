# RGNet TFLite Retry Checklist

- [x] Create isolated retry output directory.
- [x] Run and log environment checks.
- [x] Load fine-tuned RGNet Keras checkpoint and record reference image score.
- [x] Read and summarize existing failed RGNet export metadata and verify JSONs.
- [x] Attempt direct Keras built-in TFLite conversion.
- [x] Verify direct Keras TFLite with standard `tf.lite.Interpreter` if conversion succeeds; skipped because conversion failed.
- [x] Run smoke and 20-image parity if direct Keras conversion succeeds; skipped because conversion failed.
- [x] Attempt strict float32 rebuild only if direct conversion fails.
- [x] Evaluate a passing built-in TFLite candidate on fixed AVA/AADB subsets if available.
- [x] Write `rgnet_tflite_retry_summary.json`.
- [x] Write `rgnet_tflite_retry_report.md`.
- [x] Confirm no Flutter copy was performed.
