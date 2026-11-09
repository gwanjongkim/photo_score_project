# ICAA QAT Plan

## Level 3: Quantization Aware Training
1. **Prerequisites:** Locate the original PyTorch training script and dataset loaders.
2. **Baseline Validation:** Verify PyTorch FP32 model achieves original validation metrics.
3. **QAT Setup:**
   - Use PyTorch Quantization (`torch.quantization`) or specialized QAT libraries (like AIMET or NNCF).
   - Insert fake-quantize modules in the model graph.
4. **Training:**
   - Initialize with the current best FP32/FP16 weights.
   - Fine-tune on the ICAA17K dataset with a low learning rate.
   - Evaluate standard metrics (SRCC/PLCC) during training.
5. **Export:**
   - Export QAT model to ONNX.
   - Convert ONNX to TF SavedModel using `onnx2tf`.
   - Convert TF SavedModel to INT8 TFLite.
6. **Validation:** Compare QAT INT8 model outputs with original FP16 TFLite model on the 50-image A-cut test set and check ranking changes.