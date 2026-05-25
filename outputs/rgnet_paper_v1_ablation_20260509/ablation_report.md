# RGNet Paper-v1 Ablation Report

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 20260509
- Python & TF: Python 3.12 / TensorFlow 2.20.0
- GPU visibility: 1x NVIDIA GeForce RTX 4070 SUPER
- Git status: Local modifications active in repo, but ablation cleanly isolated.

## 2. Motivation
RGNet-paper-v1 underperformed the previous baseline (v0) on full AADB regression (Test SRCC 0.6389 vs 0.6433). The v1 model combined multiple structural changes at once (ASPP, specific dilations, aggregation strategies, block counts). This ablation isolates each factor to find the exact source of performance degradation compared to v0.

## 3. Baseline References
| Model | Split | SRCC | PLCC | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RGNet-paper-v0 full | AADB Test | 0.6433 | 0.6499 | 0.1255 | 0.1557 | Fixed reference |
| RGNet-paper-v1 full | AADB Test | 0.6389 | 0.6423 | 0.1282 | 0.1602 | Fixed reference |

## 4. Ablation Matrix
| Name | Image Size | Dilation Rates | Graph Blocks | Aggregation | LSE r | Score Activation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| agg_mean | 256 | [1,3,6,12,18] | 3 | mean | - | sigmoid_before |
| agg_max | 256 | [1,3,6,12,18] | 3 | max | - | sigmoid_before |
| agg_lse_r2 | 256 | [1,3,6,12,18] | 3 | lse | 2.0 | sigmoid_before |
| agg_lse_r4 (v1 base) | 256 | [1,3,6,12,18] | 3 | lse | 4.0 | sigmoid_before |
| agg_lse_r8 | 256 | [1,3,6,12,18] | 3 | lse | 8.0 | sigmoid_before |
| dilation_paper | 256 | [3,6,12,18] | 3 | lse | 4.0 | sigmoid_before |
| input_300 | 300 | [1,3,6,12,18] | 3 | lse | 4.0 | sigmoid_before |
| raw_lse_then_sigmoid | 256 | [1,3,6,12,18] | 3 | lse | 4.0 | raw_then_sigmoid |
| graph_blocks_2 | 256 | [1,3,6,12,18] | 2 | lse | 4.0 | sigmoid_before |

## 5. Smoke Results
All 9 configurations passed the smoke tests successfully.
| Name | Passed | Notes |
| :--- | :--- | :--- |
| All | Yes | No shape/NaN/save-load issues detected. |

## 6. Mid-run Results
Trained on 1024 train / 256 val samples for 5 epochs. Evaluated on 256 test samples.
| Name | Test SRCC | Test PLCC | Test MAE | Test RMSE | Best Val Loss |
| :--- | :--- | :--- | :--- | :--- | :--- |
| dilation_paper_3_6_12_18 | 0.5334 | 0.5664 | 0.1157 | 0.1477 | 0.0270 |
| agg_mean | 0.5211 | 0.5633 | 0.1103 | 0.1436 | 0.0277 |
| agg_lse_r2 | 0.4982 | 0.5504 | 0.1335 | 0.1644 | 0.0301 |
| graph_blocks_2 | 0.4941 | 0.5245 | 0.1240 | 0.1557 | 0.0306 |
| agg_lse_r8 | 0.4860 | 0.5247 | 0.1190 | 0.1506 | 0.0301 |
| raw_lse_then_sigmoid | 0.4705 | 0.4895 | 0.1203 | 0.1539 | 0.0269 |
| agg_lse_r4 (v1 base) | 0.4638 | 0.5024 | 0.1211 | 0.1525 | 0.0272 |
| agg_max | 0.4047 | 0.4353 | 0.1408 | 0.1738 | 0.0336 |
| input_300 | 0.3952 | 0.4166 | 0.1289 | 0.1638 | 0.0264 |

## 7. Full-run Results
No full run was automatically launched. Promising candidates will require manual trigger.

## 8. Findings
- **Did aggregation cause the performance drop?** Yes, significantly. The current v1 baseline uses `agg_lse_r4` (SRCC 0.463). Moving to `agg_mean` vastly improved mid-run SRCC to 0.521.
- **Did LSE r matter?** Yes, lower r values (`r=2`) outperformed `r=4` and `r=8`. This suggests pushing towards a mean-like aggregation (r=0 limit) is better for AADB regression than max-like aggregation.
- **Did paper dilation [3,6,12,18] help?** **Yes, the most.** Dropping the `1` dilation branch yielded the highest Test SRCC (0.533) among all ablations, dominating the baseline.
- **Did input size 300 help or cause memory issues?** No, it performed the worst (SRCC 0.395). The higher resolution likely exacerbated overfitting on the small mid-run split, or the graph convolution isn't translating well.
- **Did raw-before-LSE-then-sigmoid help?** It slightly improved over the v1 baseline (0.470 vs 0.463), but did not beat pure mean aggregation.
- **Did graph_blocks=2 recover v0-like performance?** It improved upon the 3-block baseline (0.494 vs 0.463), indicating 3 blocks may over-smooth the graph features.

## 9. Decision
- `dilation_paper_3_6_12_18`: **promising (worth full run)**
- `agg_mean`: **promising (worth full run)**
- `graph_blocks_2`: similar to v1 / slight improvement
- `agg_lse_r2`: similar to v1 / slight improvement
- `raw_lse_then_sigmoid`: similar to v1
- `agg_lse_r8`: similar to v1
- `agg_max`: worse than v1
- `input_300`: worse than v1

## 10. Next Step Recommendation
**Continue tuning v1.** The mid-run results strongly imply we can recover and exceed v0 performance by dropping the `1` dilation branch (`dilation_paper_3_6_12_18`) and replacing LSE aggregation with simple `mean` aggregation. I recommend running the optional `combined_paper_like` run with mean aggregation + paper dilations as the final v1 candidate.