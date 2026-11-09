# ICAA Conversion Feasibility

## Dynamic Range Quantization (PTQ)
- **Feasibility:** High.
- **Process:** Can be applied directly during TFLite conversion without a representative dataset by setting `optimizations = [tf.lite.Optimize.DEFAULT]`.
- **Expected Speedup:** Uncertain. Android CPU fallback might perform dequantization at runtime which can negate speedups, but it's the lowest risk approach for model parity.

## Full Integer Quantization (PTQ INT8)
- **Feasibility:** Medium.
- **Requirements:** Needs a representative dataset. We found `data/raw/icaa17k` with 16k+ images, and the script `stage_k_convert_tflite_fp16.py` successfully loads real images from this path.
- **Contract Impact:** Must ensure input/output tensors remain FLOAT32 (`inference_input_type=tf.float32`, `inference_output_type=tf.float32`) to adhere to the Flutter pipeline contract `[1, 224, 224, 3] pixel/255 -> ImageNet mean/std -> float32 output`.
- **Expected Speedup:** High chance of meaningful improvement on CPU.
- **Parity Risk:** Moderate to high. The activations and weights will be INT8, which may degrade A-cut ranking quality.

## QAT (Quantization Aware Training)
- **Feasibility:** Low to Medium.
- **Requirements:** Requires retraining/fine-tuning the original PyTorch model with simulated quantization.
- **Process:** Convert PyTorch model to QAT, train, export to ONNX -> TF SavedModel -> TFLite INT8.
- **Parity Risk:** Lowest risk to accuracy if QAT converges well, but highest implementation cost.