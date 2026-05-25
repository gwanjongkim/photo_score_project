# RGNet Paper-v1 Full Candidate Validation Report

## 1. Environment
- **pwd:** /home/omen_pc1/photo_score_project
- **date:** Sun May 10 2026
- **Python:** 3.12.3
- **TensorFlow:** 2.20.0
- **GPU visibility:** [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
- **git status summary:** Multiple modified files including `src/models/rgnet_paper_v1.py` and `src/train/train_rgnet_paper_v1_aadb.py`. Several untracked files including configuration files for v1 ablations.

## 2. Motivation
The mid-run ablation identified paper dilation `[3, 6, 12, 18]` and mean aggregation as promising factors compared to the default v1 configuration. This validation evaluates these factors under full training over 20 epochs on the AADB dataset to determine if they can surpass the v0 baseline.

## 3. Candidate Configs

| Candidate | Image Size | Dilation Rates | Graph Blocks | Aggregation | LSE r | Score Activation | Batch Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| combined_paper_dilation_mean | 256 | `[3, 6, 12, 18]` | 3 | mean | N/A | sigmoid_before_aggregation | 8 |
| dilation_paper_3_6_12_18_full | 256 | `[3, 6, 12, 18]` | 3 | lse | 4.0 | sigmoid_before_aggregation | 8 |
| agg_mean_full | 256 | `[1, 3, 6, 12, 18]` | 3 | mean | N/A | sigmoid_before_aggregation | 8 |

## 4. Training Results

| Candidate | Epochs Completed | Best Epoch | Best Val Loss | Best Val MAE | Batch Size | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| combined_paper_dilation_mean | 5 | 2 | 0.0226 | 0.1182 | 8 | Stopped early, no OOM |
| dilation_paper_3_6_12_18_full | 5 | 2 | 0.0227 | 0.1163 | 8 | Stopped early, no OOM |
| agg_mean_full | 4 | 1 | 0.0224 | 0.1161 | 8 | Stopped early, no OOM |

## 5. Evaluation Results

| Candidate | Split | Samples | SRCC | PLCC | MAE | RMSE | seconds_per_image |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| combined_paper_dilation_mean | test | 1000 | 0.6263 | 0.6362 | 0.1261 | 0.1570 | 0.0241 |
| combined_paper_dilation_mean | val | 846 | 0.5754 | 0.5813 | 0.1182 | 0.1505 | 0.0128 |
| dilation_paper_3_6_12_18_full | test | 1000 | 0.6401 | 0.6466 | 0.1241 | 0.1553 | 0.0241 |
| dilation_paper_3_6_12_18_full | val | 846 | 0.5699 | 0.5799 | 0.1163 | 0.1506 | 0.0128 |
| agg_mean_full | test | 1000 | 0.6819 | 0.6878 | 0.1198 | 0.1486 | 0.0242 |
| agg_mean_full | val | 846 | 0.5956 | 0.5999 | 0.1161 | 0.1496 | 0.0128 |

## 6. Comparison Against Baselines

| Model | Split | SRCC | PLCC | MAE | RMSE | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| v0 full test | test | 0.6433 | 0.6499 | 0.1255 | 0.1557 | Baseline |
| v1 default full test | test | 0.6389 | 0.6423 | 0.1282 | 0.1602 | Underperformed v0 |
| combined_paper_dilation_mean | test | 0.6263 | 0.6362 | 0.1261 | 0.1570 | Underperformed v0 and v1 |
| dilation_paper_3_6_12_18_full | test | 0.6401 | 0.6466 | 0.1241 | 0.1553 | Beat v1 but not v0 |
| agg_mean_full | test | 0.6819 | 0.6878 | 0.1198 | 0.1486 | **Beat v0**, new best |

## 7. Findings
- **Did mean aggregation beat LSE?** Yes, significantly. The `agg_mean_full` candidate achieved an SRCC of 0.6819 compared to the LSE default's 0.6389.
- **Did paper dilation survive full training?** On its own, paper dilation `[3, 6, 12, 18]` (SRCC 0.6401) performed slightly better than the default `[1, 3, 6, 12, 18]` with LSE (SRCC 0.6389), but not enough to beat the v0 baseline (0.6433).
- **Did the combined candidate beat individual candidates?** No, combining mean aggregation with the paper dilation rates resulted in worse performance (SRCC 0.6263).
- **Did any candidate beat v0?** Yes, the `agg_mean_full` candidate achieved an SRCC of 0.6819, comfortably beating the v0 baseline of 0.6433.
- **Should v1 continue or should v0 remain the baseline?** The v1 direction should continue using the mean aggregation configuration, as it has established a new best baseline.

## 8. Decision
- promote mean candidate

## 9. Next Step Recommendation
- move to RGNet-paper-AVA-classification
