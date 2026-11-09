# ICAA Benchmark Plan

## Targets
1. **Baseline:** `outputs/icaa_tf_native_tflite_fp16_20260516_143721/icaa_dat_tf_native_fp16.tflite` (Current Production equivalent)
2. **Candidate 1:** Dynamic Range PTQ TFLite
3. **Candidate 2:** Full Integer INT8 PTQ TFLite
4. **Candidate 3 (Future):** QAT INT8 TFLite

## Offline WSL Benchmark Protocol
- **Dataset:** 50-image A-cut test set.
- **Metrics:**
  - **MAE / MaxAE:** Absolute errors vs FP16 TFLite output.
  - **SRCC / PLCC:** Ranking correlation against FP16 TFLite output.
  - **Ranking diff:** Compute actual A-cut ranking on offline pipeline and count swaps.
  - **Runtime compatibility:** Check if TFLite Interpreter can allocate and invoke successfully without CPU fallback errors on WSL.
  - **Error cases:** Check for NaN/Inf outputs.

## Device Benchmark Protocol (Only if WSL passes)
- **Artifact:** Copy candidate to Flutter `assets/models/` as an experimental file (e.g., `icaa_experimental_int8.tflite`).
- **Metadata:** Do NOT overwrite production metadata. Use a separate `experimental_metadata.json` if needed.
- **Metrics:**
  - **Latency:** Android CPU inference time (target < 1.0s).
  - **Delegates:** Ensure it does not crash when GPU delegate is attempted (even if it falls back to CPU).
  - **iOS:** Test iOS CPU inference latency and compatibility.