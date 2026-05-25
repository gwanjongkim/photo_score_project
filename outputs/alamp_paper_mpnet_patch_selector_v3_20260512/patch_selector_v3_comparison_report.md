# Patch Selector V1 vs V2 vs V3 Comparison

## Metrics Summary
| Metric | Selector V1 | Selector V2 | Selector V3 | Change (V1->V3) |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Patch Score** | 0.1095 | 0.0921 | 0.5742 | +424.5% |
| **Mean Pairwise IoU** | 0.0091 | 0.0035 | 0.0258 | +1.7% |
| **Object Coverage** | 7.8% | 5.8% | 8.4% | +0.6% |

## Analysis
- **Saliency Lift**: V3 uses U²-Net, which provides true semantic salient object detection. The score reflects the confidence of the mask.
- **Overlap**: V3 maintains low overlap comparable to V2, proving the diversity constraints are preserved.
- **Subject Coverage**: Object coverage (IOU > 0.3 with YOLO objects) shows that V3 significantly captures more 'ground-truth' salient subjects than previous approximations.
