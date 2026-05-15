# Stage H Full DAT Parity Plan

## Scope
- Assemble a full inference-only TensorFlow/Keras ICAA17K DAT model.
- Compare final PyTorch-vs-TensorFlow `[B, 2]` outputs first, then save intermediate drift diagnostics.
- Keep Stage G-2 and Stage G-3 full strict parity noted as unresolved.
- Do not train, export SavedModel, convert to TFLite, implement deployment code, or touch official/A-cut/Flutter files.

## Steps
1. Confirm PyTorch `DAT.forward` order and checkpoint key groups.
   - Verify: Stage H report lists the exact forward order and mapped checkpoint groups.
2. Implement `tf_dat_model.py`.
   - Verify: model accepts NHWC `[B, 224, 224, 3]`, reuses verified components, implements down projections, `cls_norm`, pooling, histogram head, and final sigmoid heads.
3. Add `stage_h_full_dat_parity.py`.
   - Verify: script strict-loads PyTorch, maps TensorFlow weights, runs deterministic random input and 16 real ICAA17K test images, writes JSON/CSV/log/intermediate summary/Markdown.
4. Run Stage H.
   - Verify: final MOS/color/full output diffs are evaluated against score-level thresholds and the report states whether Stage I SavedModel export or later TFLite conversion is safe.
