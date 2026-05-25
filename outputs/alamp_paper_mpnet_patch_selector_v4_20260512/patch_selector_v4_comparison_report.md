# Patch Selector V1 vs V2 vs V3 vs V4 Comparison

## Metrics Summary
| Metric | V1 | V2 | V3 | V4 | Change (V3->V4) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean Patch Score** | 0.1095 | 0.0921 | 0.5742 | 0.6168 | +7.4% |
| **Mean Pairwise IoU** | 0.0091 | 0.0035 | 0.0258 | 0.0589 | +128.6% |
| **Object Coverage** | 7.8% | 5.8% | 8.4% | 57.3% | +48.9% |
| **Main Object Coverage** | 6.9% | 4.9% | 8.0% | 64.3% | +56.3% |

## Analysis
- **V4 Strategy**: Directly targets salient components identified by U²-Net, allocating roles for close-ups and context.
- **Subject Coverage**: V4 is expected to show significant improvement in Main Object Coverage as it explicitly places patches on the largest salient contours.
- **Diversity**: V4 maintains spatial diversity by filling remaining slots with context-aware patches while avoiding subject-overlap.
