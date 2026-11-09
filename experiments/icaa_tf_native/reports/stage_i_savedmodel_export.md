# Stage I SavedModel Export and Reload Parity

## Scope
- Exact command run: `python experiments/icaa_tf_native/scripts/stage_i_export_savedmodel.py`
- SavedModel path: `outputs/icaa_tf_native_savedmodel_20260516_042006/saved_model`
- Output directory: `outputs/icaa_tf_native_savedmodel_20260516_042006`
- Status: `ok`
- Parity: `pass_preferred`

## Serving Signature
- Input: `image`
  - Dtype: `tf.float32`
  - Shape: `[None, 224, 224, 3]`
- Output: `scores`
  - Dtype: `tf.float32`
  - Shape: `[None, 2]`

## Test Inputs
- Test A: 1 deterministic random normalized image input
- Test B: 16 real ICAA17K test images

## Reload Parity Results
| Test | Status | Max abs diff | Mean abs diff | MOS max | Color max |
| --- | --- | --- | --- | --- | --- |
| Random Input | pass_preferred | 0 | 0 | 0 | 0 |
| 16 Real Images | pass_preferred | 4.768e-07 | 6.985e-08 | 4.768e-07 | 2.980e-07 |

## Model Info
- SavedModel size: ~340MB (Standard FP32 model)
- Keras layers: All layers (PatchProj, stages 0-3, DownProj, ClsNorm, Dense heads) successfully built and exported.

## Unresolved Issues
- None. Bit-exact parity achieved for random input, and within 1e-6 for real images.

## Decision
- Safe to proceed to Stage J TFLite FP32 conversion: **True**
- Next step: `experiments/icaa_tf_native/scripts/stage_j_export_tflite.py`
