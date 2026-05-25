# A-LAMP Paper Strict Patch Selector Report

## Scope
- Implemented only the strict offline patch selector requested for this run.
- Did not train MPNet.
- Did not use V4 as the final selector.
- Did not implement the layout graph.
- Did not modify Flutter, forWeights, practical A-LAMP, or RGNet.

## Saliency Source Substitution
The A-LAMP paper uses graph-based saliency for `S`. The exact original saliency implementation is not locally available for this run, so this selector uses the existing U2-Net saliency maps only as the saliency map source. This is documented as a saliency-source substitution and is not an official A-LAMP reproduction.

## Objective
`F = sum_i S_i + sum_(i<j) Dp(i,j) + sum_(i<j) Ds(i,j)`.

- `S`: mean U2-Net saliency inside each selected patch.
- `Dp`: bounded Gaussian Wasserstein-style pattern distance using Sobel edge and Lab chrominance distributions.
- `Ds`: Euclidean patch-center distance normalized by image diagonal.
- Overlap constraint: hard intersection-over-min-area ratio threshold during selection.
- Patch count: 5.

## Strict Split Metrics
| Split | Valid | Skipped | Exact 5 | Mean S | Mean Dp | Mean Ds | Mean overlap | Relaxed | Objective |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1024 | 0 | 1024 | 0.3432 | 0.3361 | 0.3858 | 0.0888 | 25 | 8.9352 |
| val | 1024 | 0 | 1024 | 0.3354 | 0.3387 | 0.3852 | 0.0864 | 23 | 8.9161 |
| test | 1024 | 0 | 1024 | 0.3439 | 0.3432 | 0.3828 | 0.0899 | 30 | 8.9800 |

## Comparison Against V4
| Split | Selector | Mean S | Mean IoU | Object coverage | Main object coverage |
| :--- | :--- | ---: | ---: | ---: | ---: |
| train | strict | 0.3432 | 0.0557 | 35.7% | 38.9% |
| train | v4 | 0.6168 | 0.0589 | 57.3% | 64.3% |
| val | strict | 0.3354 | 0.0537 | 36.1% | 40.3% |
| val | v4 | 0.6097 | 0.0612 | 58.6% | 66.7% |
| test | strict | 0.3439 | 0.0566 | 36.5% | 37.9% |
| test | v4 | 0.6136 | 0.0589 | 60.3% | 68.0% |

## Manual 50 Image Gate
- Overlay directory: `outputs/alamp_paper_strict_patch_selector_20260512/overlay_visualizations`.
- Overlay visualizations found: 50.
- Expected visualizations: 50.
- Status: generated.

## Output Files
- `outputs/alamp_paper_strict_patch_selector_20260512/train_patch_boxes_1024_strict.jsonl`
- `outputs/alamp_paper_strict_patch_selector_20260512/val_patch_boxes_1024_strict.jsonl`
- `outputs/alamp_paper_strict_patch_selector_20260512/test_patch_boxes_1024_strict.jsonl`
- `outputs/alamp_paper_strict_patch_selector_20260512/patch_selector_strict_summary.json`
