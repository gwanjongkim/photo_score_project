<!-- 재현성 오딧 과정에서 이전 에이전트가 놓쳤거나 다루지 않은 주요 기술적 결함과 위험 요소를 기록하는 문서. -->
# Missed Issues

This report documents the significant technical issues and reproduction risks that were overlooked or insufficiently analyzed in the previous Codex audits.

---

## 1. MobileNetV2 Normalization Discrepancy

- **Issue**: Mismatch between ImageNet backbone initialization and input scale.
- **Details**: `build_technical_regressor` (`src/models/technical_regressor.py`) loads `MobileNetV2` with `weights="imagenet"`. A pre-trained ImageNet model expects inputs in the range `[-1, 1]` (scaled from `[0, 255]` using `preprocess_input`). However, the data loader (`src/datasets/csv_dataset.py`) decodes the JPEG, converts it to float32 (which scales it to `[0, 1]`), and feeds it directly to the model.
- **Impact**: Because the input is not normalized to `[-1, 1]`, all input pixel values are shifted and compressed (e.g. what should have been mapped to 0 is fed as 0.5; what should have been 1 is fed as 1). This distorts the activations of the first convolutional layer and degrades the quality of the pre-trained ImageNet feature representation, rendering the "pretrained weights" initialization mostly counterproductive or requiring extensive fine-tuning.
- **Codex Handling**: Codex listed this as a "high-risk preprocessing mismatch" but did not highlight its mathematical impact on feature extraction or downgrade the paper fidelity of the MobileNetV2-based models appropriately.

---

## 2. EfficientNetV2B0 Double-Scaling Bug in RGNet

- **Issue**: Double scaling of input tensors inside the RGNet model.
- **Details**: The default `tf.keras.applications.EfficientNetV2B0` layer includes an internal preprocessing block that rescales raw `[0, 255]` inputs by dividing by 255.0 and standardizing. Unlike `nima_distribution.py` which explicitly disables this by passing `include_preprocessing=False`, `rgnet.py` builds the backbone using:
  ```python
  tf.keras.applications.EfficientNetV2B0(
      include_top=False,
      weights=weights,
      input_shape=input_shape,
  )
  ```
  Since the dataset pipeline already scales the image to `[0, 1]`, the model rescales the inputs a second time (dividing by 255 again inside the backbone).
- **Impact**: The inputs to the feature extractor are compressed to the range `[0, 0.0039]`. This extreme scale compression leads to vanishing activations and gradients, crippling the model's ability to extract composition or region features.
- **Codex Handling**: Codex failed to diagnose this as an active bug, grouping it merely as a generic "duplicate/under-scaling risk".

---

## 3. Database-Level Data Leakage in FLIVE Splitting

- **Issue**: Shuffling image and patch rows together leading to split leakage.
- **Details**: In `src/preprocess/make_flive_csv.py`, `all_df` is built by concatenating `image_df` and `patch_df`, and then calling `train_test_split(all_df, test_size=0.2)`. Because patches are cropped directly from the full images in the dataset, the parent image and its corresponding patches share identical visual features.
- **Impact**: By randomly shuffling the concatenated dataset, patches from image `X` can easily end up in the training split while the full image `X` ends up in the test or validation split (or vice-versa). This introduces information leakage across splits, inflating validation and test scores artificially and invalidating claims of generalization.
- **Codex Handling**: Codex did not identify or discuss this cross-split leakage risk in its dataset integrity reviews.

---

## 4. AVA Protocol Deviation in NIMA Splitting

- **Issue**: Evaluation on custom matched subsets instead of the paper's partition.
- **Details**: The NIMA paper trains on 250,000 images and tests on a fixed set of 5,000 images. The local AVA preprocessing script `make_ava_csv.py` matching mechanism filters out missing JPEG files (which is common for AVA downloaders due to broken links). It then applies a random 80/10/10 split on the locally matched subset.
- **Impact**: This split contains far fewer than 250,000 images and has a different distribution. Reporting results on this random subset makes direct comparison with NIMA's official results (e.g. SRCC, LCC, Accuracy) mathematically invalid.
- **Codex Handling**: Codex accepted the local random split as paper-faithful, noting only that the paper "does not publish a reusable AVA filename list".

---

## 5. Missing Logistic Fitting in PLCC Calculations

- **Issue**: Directly calculating PLCC on raw predictions.
- **Details**: In standard IQA evaluation protocols (e.g. KonIQ-10k, FLIVE), objective model predictions must undergo a non-linear mapping (such as a 4-parameter logistic function) to match subjective MOS scale non-linearities before calculating the Pearson Linear Correlation Coefficient (PLCC). The local benchmark scripts (e.g. `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`) directly call `scipy.stats.pearsonr` on raw outputs.
- **Impact**: direct PLCC calculation underestimates the linear correlation of the model predictions and makes them incompatible with paper-reported PLCC values.
- **Codex Handling**: Codex noted "no stated logistic fitting" in paper baselines but failed to highlight that omitting this step in local evaluation code invalidates benchmark comparisons.

---

## 6. Omission of Joint AADB Attribute Losses

- **Issue**: Omission of attribute supervision in AADB models.
- **Details**: The AADB paper requires predicting 11 aesthetic attributes (composition, color, lighting, etc.) jointly with the overall aesthetic score using joint regression losses. The local implementations of RGNet and AADB models only predict the overall score.
- **Impact**: The model loses the regularization and feature guidance provided by the multi-task attribute loss, violating a core design requirement of the paper.
- **Codex Handling**: Codex marked the omission of attributes in `paper_requirements.md` but did not penalize the paper fidelity scores of the practical models for this omission.
