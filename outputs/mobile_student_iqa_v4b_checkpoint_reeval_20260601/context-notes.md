# Mobile Student IQA v4b Checkpoint Re-evaluation Context Notes

- 2026-06-01: Task is evaluation-only. No training, TFLite export, Flutter edits, production scoring edits, or modification of existing v3/v4a/v4b training outputs.
- 2026-06-01: v4b trainer emits RGB float32 images in 0..255, resizes with PIL bilinear, and relies on Keras application preprocessing inside the saved model when available.
- 2026-06-01: v4b report validates image rows before splitting and reports 7,507 validation rows and 7,495 test rows, so re-evaluation should validate manifest rows the same way.
- 2026-06-01: v4b uses custom loss and metrics only for compile-time training. Checkpoint loading should first use `tf.keras.models.load_model(..., compile=False)`.
- 2026-06-01: All five requested checkpoints loaded with input size 256. Report generation originally hit missing optional `tabulate`, so markdown rendering was changed to a local table formatter and the report was regenerated from saved CSV outputs without rerunning inference.
- 2026-06-01: Final raw recommendation is `final_model`, but no raw checkpoint satisfies readiness because SRCC remains far below 0.85, severe FP is nonzero, and FLIVE SRCC remains below 0.50.
