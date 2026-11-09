# ICAA INT8/QAT Audit and Acceleration Plan

## 1. 결론
- existing INT8 model: No existing INT8 models were found.
- existing QAT model: No existing QAT models were found.
- source model found: Yes, a TensorFlow SavedModel used for export is available at `outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`.
- representative dataset found: Yes, `data/raw/icaa17k` is available and mapped via `external/icaa_official_repo/ICAA17K_code/dataset/ICAA17K/1test.csv`.
- immediate next action: Create a dynamic-range and full-integer PTQ INT8 candidate using the existing TensorFlow SavedModel and representative dataset to benchmark offline parity.

## 2. Artifact Inventory
Please refer to `artifact_inventory.csv` for the full list of `.tflite` and metadata files found in the project. Notable artifacts include:
- Production FP16 model: `/home/omen_pc1/photo_score_project/forWeights2/models/aesthetic/icaa_dat_tf_native_fp16.tflite`
- Production Metadata: `/home/omen_pc1/photo_score_project/outputs/flutter_aesthetic_metadata_20260523/icaa_dat_tf_native.metadata.json`
- Latest Conversion Scripts: `experiments/icaa_tf_native/scripts/stage_k_convert_tflite_fp16.py`

## 3. Current ICAA Export Lineage
The current `icaa_dat_tf_native_fp16.tflite` was exported from a TensorFlow SavedModel (`outputs/icaa_tf_native_savedmodel_20260516_134439/saved_model`). The conversion script `stage_k_convert_tflite_fp16.py` utilizes the standard TFLiteConverter with `tf.float16` supported types. The process includes a representative dataset check for parity validation but does not apply INT8 quantization. The source model lineage traces back to a PyTorch implementation that was likely exported to ONNX and then converted to TF native.

## 4. INT8/PTQ Feasibility
- dynamic range PTQ: Highly feasible. Can be done directly from the existing SavedModel by applying `tf.lite.Optimize.DEFAULT` without representative data.
- full integer PTQ: Feasible. The TF SavedModel and the required ICAA17K representative dataset are both available.
- required representative dataset: Available at `data/raw/icaa17k`.
- expected speed: Dynamic range may show little improvement on CPU. Full integer PTQ is expected to improve Android CPU latency noticeably.
- parity risk: High for full integer PTQ due to potential activation quantization errors which could impact the final A-cut aesthetic score ranking.

## 5. QAT Feasibility
- source model requirement: Requires original PyTorch model code and pre-trained FP32 checkpoint.
- training/fine-tuning requirement: Requires setting up a QAT fine-tuning pipeline on the ICAA17K dataset.
- expected accuracy preservation: Highest chance to preserve score parity with FP16/FP32 baseline under INT8 quantization.
- cost/risk: High implementation cost and complexity compared to PTQ.

## 6. Benchmark Plan
Target comparisons will be conducted offline first between the candidate PTQ models and the baseline `icaa_dat_tf_native_fp16.tflite` using a 50-image A-cut set. Metrics will include MAE, MaxAE, SRCC, and ranking changes. For detailed steps, see `benchmark_plan.md`.

## 7. Android/iOS Considerations
- Portability is critical. The full-integer PTQ model MUST retain FLOAT32 input/output interfaces (`inference_input_type`/`inference_output_type`) to seamlessly drop into the existing Flutter A-cut pipeline without code changes.
- Android GPU delegate might reject INT8 operations or fall back to CPU, which is acceptable since the baseline GPU delegate failed anyway.
- iOS CPU inference might have different INT8 acceleration profiles (e.g., CoreML backend behavior) which must be verified.

## 8. Recommended Next Step
Choose one: B. No INT8/QAT found; create dynamic-range/PTQ INT8 candidate first.