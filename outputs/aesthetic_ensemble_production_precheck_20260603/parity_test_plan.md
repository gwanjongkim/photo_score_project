# Aesthetic Ensemble Parity Test Plan

## Goal
Verify that the Flutter (Dart) implementation of the calibrated ensemble matches the Python reference benchmark within numerical tolerance.

## 1. Reference Data Selection
- **Subset**: Select 100 images from `outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv`.
- **Criteria**: Include images with various aesthetic levels (low, medium, high).

## 2. Python Reference Execution
Run the reference scoring using `aesthetic_ensemble_config.json`:
```python
# Formula
calib_score = raw_1_10 * slope + intercept
ensemble = 0.37 * nima_calib + 0.41 * rgnet_calib + 0.22 * alamp_calib
```

## 3. Flutter Implementation Execution
Execute the same images through the `pozy_app` inference pipeline. Ensure logging of:
- `raw_score` (0..1 or 1..10)
- `calibrated_score` (1..10)
- `ensemble_score` (1..10)

## 4. Comparison Metrics
| Metric | Pass Criteria |
| :--- | :--- |
| **Max Absolute Diff** | < 0.005 |
| **Mean Absolute Diff** | < 0.001 |
| **SRCC (Rank parity)** | > 0.999 |

## 5. Failure Analysis
If scores differ significantly:
1. **Normalization**: Check if A-LAMP is dividing by 255. (It should not).
2. **Resizing**: Check if RGNet uses PIL-like bilinear resizing.
3. **EXIF**: Check if EXIF orientation correction is applied consistently before preprocessing.
4. **Rounding**: Verify if any intermediate score is being rounded to integer in the Dart pipeline.
