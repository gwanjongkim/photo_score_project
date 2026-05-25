# A-cut DistortionGuard-IQA v1 Design Report

## 1. Executive Summary
A new distortion-aware technical image quality assessment (IQA) model, DistortionGuard-IQA v1, is fully justified. Previous architectures (TOPIQ, MUSIQ, TechIQA-Guard v1) failed as standalone replacements for the production `koniq_mobile` guard because they inherently entangle high-level aesthetic/semantic features with low-level technical defects. This leads to dangerous over-scoring of technically ruined images that happen to contain visually pleasing subjects. By explicitly forcing the model to learn and separate synthetic and authentic technical distortions from content semantics, DistortionGuard-IQA v1 aims to become a single, robust, on-device technical guard.

## 2. Why TOPIQ/MUSIQ/TechIQA-Guard v1 Failed for This Product
Local evaluation on the expanded confirmed hard false-positive v2 dataset (`n=44`) provides clear evidence of architectural limitations in the previous approaches:
- **TechIQA Stage 4/4B/5 Failure**: `techiqa_stage4` (mean: 68.07), `techiqa_stage4b` (mean: 68.82), and `techiqa_stage5_v2` (mean: 68.53) completely failed to generalize false-positive protection from the initial 3 manual images to the expanded 44-image set.
- **Mixed112 Over-scoring**: The `topiq_mixed112` candidate over-scored these defective images with a mean of 65.35.
- **Production Baseline Contrast**: The existing `existing_avg` mean is 61.02, with `koniq_mobile` proving to be the strongest technical guard with a highly suppressive mean of 47.84. Conversely, `flive_mobile` acts as a weak guard here (mean: 74.19).
- **Interpretation**: Single-head models trained directly on mixed MOS data fail to isolate technical flaws. The semantic/aesthetic features learned from massive datasets overpower the sparse examples of severe technical defects, resulting in over-scoring.

## 3. Paper Analysis

### 3.1 DBCNN
*Blind Image Quality Assessment Using a Deep Bilinear Convolutional Neural Network*
- **Core Idea**: Uses a two-stream network fusing a synthetic distortion-aware stream (pretrained on synthetic datasets) and an authentic content-aware stream (pretrained on ImageNet) using bilinear pooling.
- **What to borrow**: The conceptual two-stream approach. Forcing one branch to explicitly understand synthetic technical distortions (blur, noise, compression) is critical to building a robust guard.
- **What not to borrow**: Heavy backbones (VGG16) and standard Bilinear Pooling, which squares feature dimensions (e.g., $O(c^2)$).
- **Mobile Adaptation**: Use a shared lightweight backbone branching into two lightweight heads, fused via simple concatenation or compact bilinear pooling. *(Note: Exact parameter counts of DBCNN are omitted pending local paper verification, but standard bilinear pooling is universally anti-mobile).*

### 3.2 HyperIQA
*Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network*
- **Core Idea**: Uses semantic content features to dynamically generate the weights for a quality prediction sub-network, adapting the quality assessment to the specific image content.
- **What to borrow**: The concept that the severity of a distortion is context-dependent (e.g., blur in the background is fine, blur on the main subject is bad).
- **Mobile Adaptation**: Full hyper-networks generate weights dynamically and break TFLite compilation/acceleration. We will adapt this by using lightweight Feature-wise Linear Modulation (FiLM) or cross-attention, where content features scale and shift the distortion features, fully supported by TFLite.

### 3.3 CONTRIQUE
*Image Quality Assessment using Contrastive Learning*
- **Core Idea**: Employs contrastive learning on image patches with varying distortion types and levels to learn a rich, distortion-aware representation space before mapping to MOS.
- **What to borrow**: Pretraining the network to separate images based on distortion *type* and *severity level* without needing MOS labels.
- **Synthetic Distortion Pretraining Plan**: We will build an automated pipeline to apply specific synthetic distortions to high-quality authentic images and pretrain the network to classify the distortion type and regress the severity level.

### 3.4 Re-IQA
*Unsupervised Learning for Image Quality Assessment in the Wild*
- **Core Idea**: Explicitly separates content-aware representations from quality-aware representations using unsupervised learning to avoid feature entanglement.
- **Content/Quality Separation Plan**: We will design the architecture to extract mid-level features (sensitive to textures/defects) for the quality branch, and high-level features (sensitive to semantics) for the content branch, explicitly forcing them apart before final fusion.

### 3.5 RankIQA
*Learning from Rankings for No-reference Image Quality Assessment*
- **Core Idea**: Trains a network using pairwise ranking on synthetically distorted images (where distortion levels establish a strict ranking), drastically augmenting training data.
- **Why ranking should be auxiliary only**: Local results from TechIQA Stage 5 showed that strong pairwise ranking (especially on authentic datasets like FLIVE) overpowered the hard-FP guard loss. Ranking will be used as a weak auxiliary objective (`lambda ~ 0.01`) during fine-tuning strictly to order synthetic severity levels, not to dictate the final authentic MOS scale.

## 4. Proposed Model: DistortionGuard-IQA v1
- **Input Signature**: `(224, 224, 3)`, float32, normalized `[-1, 1]`.
- **Backbone**: `MobileNetV3-Large` or `EfficientNet-Lite0` to guarantee on-device CPU/GPU acceleration.
- **Branch Structure**: Shared backbone splitting at mid-to-high levels.
- **Distortion Branch**: Taps into early/mid-level features (e.g., `block3`, `block4`). Extracts texture, noise, and edge degradation.
- **Content Branch**: Taps into final convolution layers. Extracts semantic context.
- **Authentic Quality Branch**: The final regression head mapping fused features to MOS.
- **Lightweight Fusion**: Uses Spatial Attention to let Content Features mask Distortion Features (simulating HyperIQA without dynamic weights), followed by concatenation.
- **Patch-Risk Pooling**: Instead of pure Global Average Pooling (GAP) which washes out local severe defects, we will use a concatenation of GAP and Global Max Pooling (GMP) to preserve signals of extreme local distortions.
- **Single Output**: Final output is a single float `technical_score`. (Intermediate classification outputs for distortion type/severity will be detached during TFLite export).
- **TFLite Compatibility Constraints**: Strict avoidance of custom ops, dynamic tensor shapes, unrolled dynamic loops, and standard hyper-networks.

## 5. Dataset Plan
- **SPAQ / KonIQ / FLIVE**: Standard authentic IQA regression datasets.
- **Synthetic Distortions**: Generated offline from top-10% MOS SPAQ images.
- **Hard-FP Confirmed v2**: The 44 manually/algorithmically confirmed false-positive images (oversampled).
- **Splits**: Standard 80/20 train/val split based on image identifiers to prevent data leakage across crops/pairs.
- **Metadata Columns**: `dataset`, `normalized_mos`, `distortion_type_id`, `distortion_severity` (0.0 to 1.0), `is_hard_fp`.

## 6. Synthetic Distortion Pretraining
- **Distortion Types**: 
  1. Gaussian Blur
  2. Motion Blur
  3. Gaussian Noise
  4. JPEG Compression
  5. WebP Compression
- **Severity Levels**: 5 discrete levels (mapped to a continuous 0.2 to 1.0 scale).
- **Pair Generation**: Pairs generated dynamically: `(Original, Distorted)` and `(Distorted_L1, Distorted_L2)` for weak ranking.
- **Auxiliary Labels**: Multi-class target for `type`, continuous target for `severity`.
- **Output Manifests**: A massive CSV of synthetically generated image paths and their exact perturbation parameters.

## 7. Training Stages
- **Stage A**: Synthetic distortion representation pretraining. Model trained to predict distortion type and severity.
- **Stage B**: Authentic IQA direct regression. Freeze early backbone, train fusion and regression heads on KonIQ/SPAQ/FLIVE.
- **Stage C**: Hard-FP guard fine-tuning using `koniq_mobile` as guard reference. Heavy oversampling of `hard_fp_v2` dataset.
- **Stage D**: Weak FLIVE ranking fine-tuning to stabilize the SRCC curve without breaking the Stage C guard.
- **Stage E**: FP16 TFLite export and strict parity evaluation.

## 8. Loss Design
- **Synthetic Distortion Type Loss**: Categorical Crossentropy.
- **Distortion Level Loss**: MSE / Huber.
- **MOS Huber Loss**: Main authentic regression loss (robust to outliers).
- **PLCC Loss**: Differentiable Pearson Correlation loss to maximize linear correlation.
- **Hard-FP Guard Loss**: Asymmetric ReLU penalty predicting scores strictly below the `koniq_mobile` threshold + margin.
- **Weak Pairwise Ranking Loss**: Softplus margin loss applied to synthetic/authentic pairs (`lambda=0.01`).
- **Dataset Calibration Loss**: Learned per-dataset scalar bias to align datasets to a single scale.

## 9. Success Criteria
- **FLIVE Metrics**: SRCC > 0.50
- **KonIQ Metrics**: SRCC > 0.85
- **SPAQ Metrics**: SRCC > 0.88
- **Hard-FP v2 Metrics**: Mean Score **< 50.0** (Crucial criteria: must match or approach `koniq_mobile` guard strength).
- **Android Latency**: < 20ms on modern Snapdragon CPU/GPU.

## 10. Implementation Plan
- **Gemini Tasks**: Implement `DistortionGuard-IQA v1` Keras architecture, build multi-stage training loops, setup evaluation scripts, and handle TFLite graph optimization.
- **Codex Tasks**: Develop the high-throughput synthetic distortion pipeline, dataset augmentation scripts, and tf.data generators.

## 11. First Codex Implementation Prompt
```text
Task: Build a Synthetic Distortion Generator for BIQA Pretraining.
Goal: Create a Python script (`tools/generate_synthetic_distortions.py`) that reads a manifest of pristine images (e.g., top 10% SPAQ) and outputs specific synthetic distortions.
Rules:
1. Do NOT implement any model training.
2. Implement 5 distortions: Gaussian Blur, Motion Blur, Gaussian Noise, JPEG Compression, WebP Compression.
3. For each image, generate 5 severity levels for each distortion type.
4. Use `imgaug` or `cv2` / `PIL` for efficient generation.
5. Save the resulting images to a specified output directory and generate a CSV manifest containing: `original_path`, `distorted_path`, `distortion_type_id`, `distortion_severity` (0.0 to 1.0).
6. Ensure the code is modular, well-commented, and supports multiprocessing for speed.
```

## 12. Risks and Stop Conditions
- **Risk**: The multi-branch fusion architecture might exceed the strict 20ms latency budget when compiled via XNNPACK in TFLite.
- **Risk**: Hard-FPs might still over-score if the semantic features from the content branch are too strongly correlated with the regression head.
- **Stop Condition**: Abandon this architectural path if Stage A/B latency exceeds 30ms on device, or if Stage B authentic training fails to achieve > 0.8 SRCC on KonIQ prior to Stage C hard-FP tuning.

## 13. Final Recommendation
**A. Proceed with DistortionGuard-IQA v1 (Stage A dataset builder first).**

*Action Plan*: The architectural direction is sound based on BIQA literature. However, no model code should be written until the synthetic pretraining dataset is physically generated and validated. The immediate next step is executing the Codex prompt for the Synthetic Distortion Generator.
