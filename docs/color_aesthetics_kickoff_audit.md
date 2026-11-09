# Color Aesthetics Kickoff Audit

## 1. Executive Summary
This audit reviews the official implementation of "Thinking Image Color Aesthetics Assessment" (ICCV 2023). The repository contains the Delegate Transformer (DAT) model and the ICAA17K dataset logic. The model is designed for scalar regression of both holistic MOS and color-specific aesthetic scores. While the core architecture is innovative (using deformable attention), it presents significant challenges for mobile deployment (TFLite) and requires specific environment setup.

## 2. Confirmed Facts
*   **Repository Location:** `external/icaa_official_repo/ICAA17K_code/`
*   **Dataset Zip:** Found at `/mnt/c/Users/OMEN PC1/Downloads/ICKK17K_dataset.zip` (2.6GB).
*   **CSV Files:** `1all.csv`, `1test.csv`, `1train.csv` exist in `dataset/ICAA17K/`.
*   **Model Architecture:** `DAT` class in `models/dat.py`. Uses `SoftHistogram` for color feature extraction and Deformable Attention blocks.
*   **Output Format:** Returns a tuple `(x, positions, references)`. `x` has shape `(batch, 2)` (if `num_classes=1`).
    *   `x[:, 0]`: MOS (Aesthetic Score) scaled to [0, 1].
    *   `x[:, 1]`: Color Aesthetic Score scaled to [0, 1].
*   **Loss Function:** `MSELoss` (Mean Squared Error) between prediction and `score/10`.
*   **Metrics:** 
    *   SRCC (Spearman Rank Correlation Coefficient)
    *   PLCC/LCC (Pearson Linear Correlation Coefficient)
    *   Accuracy (Binary classification at threshold 0.50).
*   **Preprocessing:** 
    *   Training: Resize(256) -> RandomCrop(224) -> RandomHorizontalFlip.
    *   Inference: Resize(224).
    *   Normalization: ImageNet mean/std.
    *   Color Space: RGB (via PIL `default_loader`).

## 3. Open Questions
*   **Score Distribution:** The `get_score` function using `torch.linspace(1, 10, 10)` is defined but currently unused in `main_nni.py`. The active code uses scalar regression. Is there a distribution-based head intended for other tasks?
*   **Pre-trained Weights:** The README is contradictory about weight availability. No `.pth` files are currently in the repository. We may need to train from scratch or find the external link mentioned.
*   **EXIF Handling:** The `default_loader` does not explicitly handle EXIF orientation, which might lead to orientation-induced color assessment errors if not pre-processed.

## 4. Model Output and Metric Path
The model uses two heads (`class_head` and `class_head2`) which are sigmoid-activated scalars.
*   **Prediction:** `torch.cat([x1, x2], 1)` -> `(Batch, 2)`.
*   **Target:** `y = [MOS/10, color/10]`.
*   **Primary Metric Target:** `main_nni.py` focuses evaluation on `y_pred[:, 1]` (the Color score).
*   **Accuracy Threshold:** 0.50 (corresponds to a raw score of 5.0/10).

## 5. Preprocessing Path
*   **Input size:** 224x224.
*   **Normalizer:** standard ImageNet `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
*   **Library:** `torchvision.transforms`.
*   **Loader:** `torchvision.datasets.folder.default_loader` (PIL).

## 6. Dataset Path Plan
*   **Target Directory:** `data/raw/icaa17k/`
*   **Extraction Command:** `unzip "/mnt/c/Users/OMEN PC1/Downloads/ICKK17K_dataset.zip" -d data/raw/icaa17k/`
*   **Structure Compatibility:** Zip contains category folders (e.g., `complementary/`) at root. Extraction to `data/raw/icaa17k/` will match the `ID` paths in the CSV (e.g., `complementary/orange_blue/image.jpg`).

## 7. Environment Risk
*   **Python Version:** Current project uses Python 3.12. Official README suggests older versions (PyTorch 1.7.1).
*   **Missing Dependencies:** `timm`, `einops`, `nni` are required and NOT currently in `.venv_gpu`.
*   **Import Risk:** `DAT` depends on `timm.models.layers.DropPath`.
*   **Recommendation:** Use `.venv_gpu` but install missing packages via a dedicated `requirements-icaa.txt`.

## 8. Standalone Smoke Test Plan
A minimal test script `scripts/smoke_test_icaa.py` should:
1.  Add `external/icaa_official_repo/ICAA17K_code/` to `sys.path`.
2.  Import `DAT` from `models.dat`.
3.  Instantiate `DAT` with `num_classes=1` and standard config.
4.  Run a forward pass on a dummy `(1, 3, 224, 224)` tensor.
5.  Verify output shape `(1, 2)`.
6.  Perform one backward pass with a dummy loss.

## 9. D0 Color Feature Baseline Plan
To establish a non-deep baseline:
1.  Extract global color features from ICAA17K images:
    *   Mean and Std of H, S, V channels.
    *   Mean and Std of L, a, b channels.
    *   3D Color Histogram (coarse, e.g., 4x4x4).
2.  Train a simple regressor (e.g., SVR or Random Forest) to predict the `color` score.
3.  Evaluate using SRCC and PLCC to compare with the deep model's claimed performance.

## 10. TFLite / Mobile Risk
*   **Critical Ops:** `F.grid_sample` and `torch.einsum`.
*   **TFLite Status:** 
    *   `grid_sample` is NOT supported by standard TFLite and requires `SELECT_TF_OPS` or custom implementation.
    *   `einsum` often requires conversion to multiple `transpose` and `matmul` ops, which can be inefficient or fail during conversion.
*   **Conclusion:** Direct TFLite conversion of `Delegate Transformer` is **High Risk**. Mobile deployment may require significant architecture modification (e.g., replacing deformable attention with standard windowed attention).

## 11. Recommended Next Commands
```bash
# 1. Install dependencies (if permitted)
# pip install timm einops nni

# 2. Extract dataset (requires ~3GB space)
# mkdir -p data/raw/icaa17k
# unzip "/mnt/c/Users/OMEN PC1/Downloads/ICKK17K_dataset.zip" -d data/raw/icaa17k/

# 3. Run smoke test
# python scripts/smoke_test_icaa.py
```
