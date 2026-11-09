# RGNet Paper Reproduction Preflight

## 1. Executive Summary
This preflight audit evaluates the readiness of the `photo_score_project` to perform a formal paper reproduction of the RGNet model ("Composition-Aware Image Aesthetics Assessment", WACV 2020) on external GPU infrastructure. The project has mature code for both AVA binary classification and AADB regression tracks. AVA is the primary benchmark for the original paper and is highly feasible given the current dataset readiness (255,509 images available). AADB is also ready but serves as a smaller proxy. The implementation captures key paper elements (DenseNet-121, ASPP, Region Graph, LSE pooling) with some modern enhancements (Residual GCN). The main gap for reproduction is the optimizer choice (Adam 1e-4 vs. Paper SGD 1e-2) and training duration (20 epochs vs. Paper 120 epochs). Rented GPU infrastructure (Vast.ai) is strongly recommended for the 120-epoch AVA run.

## 2. Current RGNet Code Inventory
The codebase is well-structured with isolated paper-oriented scripts:

- **Model Files:**
  - `src/models/rgnet_paper_ava.py`: AVA-specific classification model.
  - `src/models/rgnet_paper_v1.py`: Enhanced v1 model for AADB/General regression.
  - `src/models/rgnet_paper.py`: Original v0 approximation.
- **Training Scripts:**
  - `src/train/train_rgnet_paper_ava.py`: Dedicated script for AVA classification.
  - `src/train/train_rgnet_paper_v1_aadb.py`: Multi-variant script for AADB regression.
- **Evaluation Scripts:**
  - `src/eval/evaluate_rgnet_paper_ava.py`: Classification metrics (Accuracy, AP, AUC).
  - `src/eval/evaluate_rgnet_paper_v1_aadb.py`: Regression metrics (SRCC, PLCC, MAE).
- **Configuration Files:**
  - `configs/paper_benchmarks/rgnet_paper_ava_classification.yaml`
  - `configs/paper_benchmarks/rgnet_paper_v1_aadb_regression.yaml`
- **Export Scripts:**
  - `scripts/export_rgnet_paper_aadb_tflite.py`

## 3. Dataset Readiness
### AVA Dataset
- **Availability:** Fully available via symlink to `/home/omen_pc1/ava`.
- **Image Count:** 255,509 JPG files confirmed.
- **Metadata:** `data/processed/ava/train.csv`, `test.csv`, `val.csv` exist.
- **Labels:** `mean_score` and vote distributions available.

### AADB Dataset
- **Availability:** 9,958 images in `data/raw/aadb/images`.
- **Labels:** 9,459 labels in `data/processed/aadb/train.csv` (range [0, 1]).
- **Splits:** 7,613 train / 847 val / 1,001 test.

## 4. AVA Reproduction Feasibility
- **Feasibility:** **High.** This is the primary target for paper reproduction.
- **Partitions:** Current splits (~204k train, 25k val, 25k test) are close to the paper's standard partition (~230k train, 20k test). 
- **Binary Label Rule:** Follows paper rule: `mean_score >= 5.0` (High) vs. `< 5.0` (Low).
- **Evaluation Metrics:** `evaluate_rgnet_paper_ava.py` reports Accuracy and Average Precision (AP), matching paper requirements.

## 5. AADB Reproduction Feasibility
- **Feasibility:** **Moderate.** Useful for rapid iteration before AVA.
- **Normalization:** Scores are already in [0, 1], compatible with the Sigmoid output layer.
- **Evaluation Metrics:** Reports SRCC and PLCC, matching paper requirements.
- **Baseline:** Existing best SRCC is 0.6819; paper target is ~0.7104.

## 6. Paper Gap Analysis
| Component | Paper Specification | Current Implementation (v1/AVA) | Status |
| :--- | :--- | :--- | :--- |
| **Backbone** | DenseNet-121 (FCN) | DenseNet-121 (FCN) | **Match** |
| **Context** | ASPP (DeepLab style) | ASPP and Cascaded DenseASPP | **Exceeds** |
| **Graph Nodes** | Spatial features as nodes | Spatial features as nodes | **Match** |
| **Adjacency** | Cosine Similarity + Softmax | Cosine Similarity + Softmax | **Match** |
| **Graph reasoning** | Residual GCN | V1ResidualGraphConvolution | **Match** |
| **Aggregation** | LSE pooling (r=4) | LSE pooling (r=4) | **Match** |
| **Optimizer** | SGD (M=0.9, WD=5e-4) | Adam (WD optional) | **Gap** |
| **Schedule** | Polynomial decay (120 ep) | Polynomial (20 ep default) | **Gap** |
| **Batch Size** | 32 | Configurable (Default 16) | **Config** |

## 7. Blocking Issues Before Renting GPU
1. **Optimizer Config:** The `AVA` config should be updated to support SGD with momentum if a strict reproduction is desired.
2. **Epoch Extension:** Create a `configs/paper_benchmarks/rgnet_reproduction_long.yaml` with 120 epochs.
3. **Data Path Portability:** Ensure the training script handles relative data paths correctly for Colab/Vast.ai environments where the local home directory differs.

## 8. Recommended First Rented-GPU Experiment
- **Dataset:** AVA Binary Classification (Full).
- **Script:** `src/train/train_rgnet_paper_ava.py`
- **Config:** `configs/paper_benchmarks/rgnet_paper_ava_classification.yaml` (updated to 120 epochs).
- **Expected Runtime:** 18-24 hours on a single high-end GPU.
- **Required VRAM:** 24GB+ (for Batch Size 32 and DenseNet-121 + ASPP).
- **Recommended GPU:** NVIDIA RTX 3090, 4090, or A100 (80GB).

## 9. Colab vs Vast.ai Recommendation
- **Colab:** Use for **Short Tests** and code validation (1-2 epochs). Colab's 12-24 hour timeout and unstable filesystem make it unsuitable for a 120-epoch AVA run.
- **Vast.ai:** Use for **Long Training**. Allows persistent storage and long-running processes (screen/tmux). Cost-effective for 24-hour dedicated GPU tasks.

## 10. Vast.ai Migration Checklist
1. **Files to Sync:**
   - Entire `src/` directory.
   - `configs/paper_benchmarks/` directory.
   - `requirements.txt`.
2. **Dataset to Sync:**
   - `data/processed/ava/*.csv`.
   - `data/raw/ava/images/` (recommend zipping or using `rsync`).
3. **Environment Setup:**
   - `pip install -r requirements.txt`.
   - Verify GPU visibility: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
4. **Command to Run:**
   ```bash
   python -m src.train.train_rgnet_paper_ava \
     --config configs/paper_benchmarks/rgnet_paper_ava_classification.yaml \
     --epochs 120 \
     --batch_size 32 \
     --out_dir outputs/vast_rgnet_ava_repro
   ```
5. **Result Files to Copy Back:**
   - `final_model.keras`
   - `best.weights.h5`
   - `train_summary.json`
   - `training_history.csv`

## 11. Final Recommendation
Proceed with **Vast.ai** for a full 120-epoch reproduction of the **AVA Classification** task. Before migration, perform a 1-epoch smoke test on WSL using the full AVA dataset to ensure the `tf.data` pipeline handles the 255k image volume without memory leaks or I/O bottlenecks.
