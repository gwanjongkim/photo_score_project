<!-- 데이터 누수 및 평가 지표 계산 결함 등 실험적 타당성 한계를 지적하는 문서. -->
# Experimental Validity Challenges

This report evaluates the experimental and validation flaws in the photo score project, highlighting data leakage, metric calculation defects, and semantic scale discrepancies in ensembles.

---

## 1. FLIVE Image/Patch split leakage

In `src/preprocess/make_flive_csv.py`, the dataset splits are constructed randomly by shuffling image-level and patch-level annotations together.
- **Why this is a flaw**: Since patches are cropped directly from the full parent images, they share identical visual features, camera profiles, objects, and textures. Shuffling them randomly allows a patch of image `X` to enter the training set while the full image `X` itself is placed in the test or validation set.
- **Consequence**: The model memorizes parent-image features during training, inflating its performance on the test set. This visual leakage invalidates the reported test performance and exaggerates the model's blind quality assessment capabilities.

---

## 2. Incompatible semantic units in late fusion

The default server ensemble (`configs/stage5_reference.json`) combines multiple predictions using a weighted linear combination:
- **Weights**: AADB composition `0.28`, NIMA `0.24`, A-LAMP `0.22`, RGNet `0.18`, pairwise AADB `0.08`.
- **The discrepancy**: These models predict scores on entirely different scales:
  - **AADB / RGNet / A-LAMP**: Predict sigmoid scalar scores in the range `[0, 1]`.
  - **NIMA**: Predicts expected scores in the range `[1, 10]`.
  - **KonIQ / FLIVE**: Predict raw MOS-like scores in the range `[0, 100]` (which are divided by 100 only in some mobile scripts).
- **Consequence**: Performing a linear late fusion without first normalizing all models to a common semantic scale (e.g. mapping NIMA's `[1, 10]` to `[0, 1]` or scaling technical models appropriately) means that models with larger scales (like NIMA or raw technical scores) dominate the fusion output. This violates basic late-fusion design principles.

---

## 3. Omission of Non-Linear Logistic Fitting in PLCC

In image quality assessment (IQA) literature, it is standard practice to apply a 4-parameter or 5-parameter non-linear logistic function to map objective predicted scores to subjective Mean Opinion Scores (MOS) before calculating the Pearson Linear Correlation Coefficient (PLCC).
- **Why it matters**: Subjective human ratings are non-linear, especially at the extremes of the quality scale. The logistic function fits the predictions to a linear scale with MOS.
- **The flaw**: The local benchmark code (e.g. `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`) directly computes `scipy.stats.pearsonr` on raw predictions.
- **Consequence**: The reported PLCC values are deflated and cannot be compared to the published paper PLCC benchmarks, which always include this non-linear mapping.

---

## 4. Missing Ground Truth for Independent Verification

In the audit prediction tables (e.g. `outputs/research_protocol_production_technical_eval_20260530/production_technical_predictions.csv`), the ground-truth columns are omitted.
- **Consequence**: An independent auditor cannot compute SRCC, PLCC, or RMSE directly from the prediction tables. The metrics are untraceable and cannot be certified as correct without re-running the full evaluation pipeline from the raw datasets.
