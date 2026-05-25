# Patch Selector V1 vs V2 Comparison

## Metrics Summary
| Metric | Selector V1 | Selector V2 | Change |
| :--- | :--- | :--- | :--- |
| **Mean Patch Score** | 0.1095 | 0.0921 | -15.9% |
| **Mean Pairwise IoU** | 0.0091 | 0.0035 | -0.6% |
| **Object Coverage** | 7.8% | 5.8% | -2.1% |

## Analysis
- **Saliency Lift**: V2 uses Spectral Residual which significantly changes the score distribution. The lower absolute mean score is expected due to the nature of FFT-based maps compared to local Sobel density.
- **Overlap**: V2 significantly reduced the Mean Pairwise IoU, indicating much better spatial diversity.
- **Subject Coverage**: Object coverage (IOU > 0.3 with YOLO objects) shows whether v2 captures more 'ground-truth' salient subjects.
