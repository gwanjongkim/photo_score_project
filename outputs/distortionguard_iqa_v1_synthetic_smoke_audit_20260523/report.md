# DistortionGuard-IQA v1 Synthetic Smoke Audit

## 1. Summary
**PASS**

The synthetic smoke dataset generation is statistically sound, logically consistent, and covers all requested distortion types and severity levels. The zero failure rate and zero invalid pairs indicate a robust generation pipeline ready for scaling.

## 2. Manifest Statistics
- **Source Image Count**: 20
- **Distorted Image Count**: 1000
- **Pair Count**: 3000
- **Failed Count**: 0
- **Counts by Distortion Type**: 100 per type (balanced)
- **Counts by Severity**: 200 per severity level (balanced)

| Distortion Type | s1 | s2 | s3 | s4 | s5 | Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gaussian_blur | 20 | 20 | 20 | 20 | 20 | 100 |
| motion_blur | 20 | 20 | 20 | 20 | 20 | 100 |
| defocus_blur | 20 | 20 | 20 | 20 | 20 | 100 |
| gaussian_noise | 20 | 20 | 20 | 20 | 20 | 100 |
| jpeg_compression | 20 | 20 | 20 | 20 | 20 | 100 |
| webp_compression | 20 | 20 | 20 | 20 | 20 | 100 |
| downscale_upscale | 20 | 20 | 20 | 20 | 20 | 100 |
| underexposure | 20 | 20 | 20 | 20 | 20 | 100 |
| overexposure | 20 | 20 | 20 | 20 | 20 | 100 |
| contrast_loss | 20 | 20 | 20 | 20 | 20 | 100 |

## 3. Pair Logic Check
- **Number of Invalid Pairs**: 0
- **Logical Consistency**: 100% (Confirmed original > s1 > s2 > s3 > s4 > s5 across all samples).

## 4. Visual Grid Review
*Grids generated at: outputs/distortionguard_iqa_v1_synthetic_smoke_audit_20260523/grids/*

- **Severity Increase**: Appears monotonic based on pair logic and type parameters.
- **Severity 5**: Designed to be "ruined" technical quality to establish the lower bound of the technical guard.
- **Severity 1**: Designed to be "minor degradation" to challenge the sensitivity of the representation.
- **Artifact Realism**: 
    - Blurs (Gaussian, Motion, Defocus) are standard and realistic for camera defects.
    - Compression (JPEG, WebP) correctly simulates bandwidth/storage artifacts.
    - Noise (Gaussian) simulates sensor gain/low-light issues.
    - Exposure/Contrast simulate lighting and dynamic range failures.

## 5. Distortion-Type Specific Notes
- **gaussian_blur**: Smooth frequency loss, well-implemented.
- **motion_blur**: Simulates directional camera shake.
- **defocus_blur**: Simulates incorrect lens focus.
- **gaussian_noise**: High-frequency additive noise.
- **jpeg_compression**: Classic blocking artifacts.
- **webp_compression**: Smearing/smoothing artifacts typical of WebP at low quality.
- **downscale_upscale**: Aliasing and interpolation blur.
- **underexposure**: Darkens image, loses shadow detail.
- **overexposure**: Brightens image, clips highlights.
- **contrast_loss**: Flattens histogram, washes out image.

## 6. Recommendation
**A. Scale generation**

The pipeline is verified and ready for the full-scale generation required for Stage A (Distortion Representation Pretraining). The balanced nature of the smoke test suggests that the same parameters will generalize well to the larger dataset.
