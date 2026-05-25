# RGNet Model Lineage Audit Report

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: 2026-05-13
- Python: 3.12.3
- TensorFlow: 2.20.0
- GPU: NVIDIA GeForce RTX 4070 SUPER

## 2. Model Inventory

### 2.1 Practical App RGNet (AADB Regression)
- **Keras Path**: `checkpoints/rgnet_aadb_gpu/final_model.keras`
- **TFLite Path**: `models/aesthetic/rgnet_aadb_gpu.tflite`
- **Configuration**: `configs/stage5_reference.json`
- **Usage**: Currently deployed/deployable in the practical A-cut ranking stack.
- **Lineage**: Standard practical track using AVA pretraining followed by AADB finetuning (confirmed by `rgnet_float32_retry_20260508` which mirrors this lineage).

### 2.2 Paper-Oriented AADB RGNet (Regression)
- **Keras Path**: `outputs/rgnet_paper_v1_ablation_full_candidates_20260510/full_train/agg_mean_full/final_model.keras`
- **Label**: `RGNet-paper-v1 agg_mean_full`
- **Metrics**: SRCC 0.6819, PLCC 0.6878 (AADB full test)
- **Initialization**: `imagenet` (Separate track).

### 2.3 Paper-Oriented AVA RGNet (Classification)
- **Keras Path**: `outputs/rgnet_paper_ava_classification_20260510/full_train/lse_r4/final_model.keras`
- **Label**: `RGNet-paper-AVA lse_r4`
- **Metrics**: Accuracy 0.7697, ROC-AUC 0.7979 (AVA test)
- **Initialization**: `imagenet` (Separate track).

## 3. Lineage Investigation: AVA vs AADB
**Question**: Was the AVA classification model initialized from the AADB regression model?

**Answer**: **No.** 
Evidence from `outputs/rgnet_paper_ava_classification_20260510/full_train/lse_r4/train_summary.json` confirms that the AVA classification model was initialized using `"backbone_weights_requested": "imagenet"`. 

The paper-oriented models were trained as isolated tracks from ImageNet to establish clean benchmarks for their respective tasks (AADB regression and AVA classification). This differs from the **Practical App Model**, which uses a Transfer Learning lineage (AVA -> AADB) to maximize scalar regression performance on small datasets.

## 4. Comparison Table

| Model Variant | Task | Dataset | Start Weights | ROC-AUC / SRCC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Practical RGNet** | Regression | AADB | AVA Pretrain | 0.57* (SRCC) | Deployed |
| **Paper AADB v1** | Regression | AADB | ImageNet | 0.6819 (SRCC) | Benchmark |
| **Paper AVA lse_r4**| Classification | AVA | ImageNet | 0.7979 (AUC) | Benchmark |

*\*Note: Practical metrics often quoted on val512 subsets; paper metrics on full test sets.*

## 5. Final Findings
- **AVA RGNet lse_r4 Path**: `outputs/rgnet_paper_ava_classification_20260510/full_train/lse_r4/final_model.keras`
- **Practical AADB RGNet Path**: `checkpoints/rgnet_aadb_gpu/final_model.keras`
- **Lineage Status**: AVA RGNet is a **separate training track** from ImageNet. It is NOT a continuation from AADB.
- **Application Recommendation**: Use the **Practical RGNet** (`rgnet_aadb_gpu.tflite`) for the mobile app as it is optimized for scalar scoring.
- **Paper Recommendation**: Use the **Paper AVA lse_r4** model for paper-comparable AVA classification benchmarks.
