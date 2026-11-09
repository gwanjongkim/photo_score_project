<!-- 오딧 발견 사항의 결함을 해결하기 위해 실행해야 할 검증 테스트 코드를 제안하는 문서. -->
# Required Resolution Tests

This document defines the concrete executable tests required to resolve the disputes and technical issues identified in this cross-check.

---

## 1. Preprocessing Normalization Verification Test

- **Goal**: Verify if feeding `[0, 1]` floats vs `[-1, 1]` or `[0, 255]` rescaled inputs causes feature corruption in pre-trained backbones.
- **Test Code Outline**:
  Create a script `scripts/test_preprocessing_impact.py` that:
  1. Loads a pre-trained `MobileNetV2` model directly from `tf.keras.applications`.
  2. Generates a set of raw image bytes, decodes them, and rescales them to:
     - Range A: `[0, 1]` floats (current training pipeline).
     - Range B: `[-1, 1]` floats (standard Keras `mobilenet_v2.preprocess_input` range).
  3. Extracts the features from the global average pooling layer for both ranges.
  4. Computes the activation statistics (mean, std, min, max, fraction of zero activations).
  5. Verifies if Range A suffers from activation compression or saturation compared to Range B.

---

## 2. RGNet Double-Scaling Resolution Test

- **Goal**: Verify and resolve the double-scaling bug in the practical RGNet model.
- **Test Code Outline**:
  Create a script `scripts/test_rgnet_doublescaling.py` that:
  1. Instantiates `build_rgnet_model` from `src/models/rgnet.py`.
  2. Inspects the layers of the model and asserts that the backbone contains no `Rescaling` layer if the model expects `[0, 1]` inputs, or verifies that input pixels are in `[0, 255]` range if preprocessing is enabled.
  3. Proposes the following code fix in `src/models/rgnet.py`:
     ```python
     # Force include_preprocessing=False when constructing the backbone
     base = tf.keras.applications.EfficientNetV2B0(
         include_top=False,
         weights=weights,
         input_shape=input_shape,
         include_preprocessing=False,  # <--- Fix
     )
     ```
  4. Loads the weights, runs inference on a test batch of images, and verifies if the predictions are stable and cover the full `[0, 1]` output interval.

---

## 3. FLIVE Cross-Split Leakage Audit Test

- **Goal**: Identify and quantify the parent-image overlap leakage between training and validation/test splits in FLIVE.
- **Test Code Outline**:
  Create a script `scripts/audit_flive_leakage.py` that:
  1. Loads `data/processed/paq2piq/all_train.csv` and `all_test.csv` (or `image_train.csv` and `patch_test.csv`).
  2. Parses the file paths to extract the unique image identifier (stem/hash) of the parent image.
  3. Computes the intersection of parent image IDs between the training split and the test/validation splits.
  4. Reports the percentage of test/validation samples whose parent image was leaked into the training set.
  5. Re-splits the dataset using a grouped split strategy (`GroupKFold` or `GroupShuffleSplit` on the parent image ID) to guarantee zero leakage, and exports the clean split manifests.

---

## 4. Logistic Fitting for IQA PLCC

- **Goal**: Implement standard non-linear mapping before computing PLCC for technical models.
- **Test Code Outline**:
  Implement a 4-parameter logistic fitting function in `src/utils/metrics.py`:
  ```python
  import numpy as np
  from scipy.optimize import curve_fit

  def logistic_func(x, beta1, beta2, beta3, beta4):
      return beta1 * (0.5 - 1.0 / (1.0 + np.exp(beta2 * (x - beta3)))) + beta4 * x

  def fit_and_calculate_plcc(predictions, ground_truth):
      # Initialize parameters
      beta_init = [np.max(ground_truth), 1.0, np.mean(predictions), 1.0]
      try:
          popt, _ = curve_fit(logistic_func, predictions, ground_truth, p0=beta_init, maxfev=10000)
          fitted_predictions = logistic_func(predictions, *popt)
          plcc = np.corrcoef(fitted_predictions, ground_truth)[0, 1]
          return plcc, popt
      except Exception:
          # Fallback to linear PLCC if fitting fails
          return np.corrcoef(predictions, ground_truth)[0, 1], None
  ```
  Apply this function in the technical evaluation scripts and verify if the PLCC matches the published paper results.
