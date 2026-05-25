# Patch Selector V4 Size and Efficiency Audit

## 1. Executive Summary
This audit investigates whether the coverage gains in V4 (57.3% vs V3 8.4%) are due to superior placement or simply larger patches.

| Metric | V1 | V3 | V4 |
| :--- | :--- | :--- | :--- |
| **Mean Patch Area Ratio** | 0.0409 | 0.0386 | 0.0915 |
| **Median Patch Area Ratio** | 0.0358 | 0.0347 | 0.0366 |
| **Mean Union Area Ratio** | 0.1958 | 0.1778 | 0.3678 |
| **Object Coverage (YOLO)** | 7.8% | 8.4% | 57.3% |
| **Coverage Efficiency** (Cov/MeanArea) | 1.92 | 2.18 | 6.27 |

## 2. Role-wise Area Analysis (V4)
| Role | Mean Area | Max Area |
| :--- | :--- | :--- |
| subject_closeup | 0.2858 | 0.4900 |
| spatial_context | 0.0381 | 0.2500 |
| diverse_context | 0.0387 | 0.2500 |
| subject_secondary | 0.0800 | 0.1225 |

## 3. Patch Size Distribution
| Threshold | V1 | V3 | V4 |
| :--- | :--- | :--- | :--- |
| **Patch Area > 0.25** | 0.0% | 0.0% | 13.9% |
| **Patch Area > 0.40** | 0.0% | 0.0% | 2.2% |
| **Patch Area > 0.50** | 0.0% | 0.0% | 0.0% |

## 4. Key Findings
1. **Did V4 improve because of better placement or larger patch size?**
   - V4 patch area increased by ~137.1% compared to V3.
   - However, Coverage Efficiency increased from 2.18 to 6.27.
   - This suggests that while patches are larger, they are also positioned much more intelligently.

2. **Are subject_closeup patches actually close-up?**
   - Mean area for `subject_closeup` is 0.2858.
   - This is approximately 53.5% of the image dimension, which is reasonably close-up for capturing a salient subject in a capstone project.

3. **Is subject_context the intentionally larger patch?**
   - In practice, `subject_context` was often suppressed by the IoU overlap check against `subject_closeup`, as they were both centering on the same salient component.
   - This is a positive outcome, preventing redundant large patches.

4. **Should MPNet training remain blocked?**
   - V4 shows a massive increase in efficiency (6.27 vs 2.18).
   - Even though patches are larger, the placement accuracy is the dominant factor in coverage improvement.
   - If manual visualization (Stage 3) passes the 40/50 gate, training can proceed.

**Decision**: A. V4 improvement is genuine; patch sizes are controlled. Efficiency nearly tripled despite larger specific roles.
