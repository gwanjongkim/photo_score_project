<!-- 재현성 오딧 교차 검증에 대한 고수준 요약 및 주요 오딧 불일치 사항을 보고하는 문서. -->
# Independent Crosscheck Summary

This document presents a high-level summary of the independent adversarial cross-check of the Codex reproducibility audit for the Photo Score project.

## Purpose

The purpose of this review is to verify the technical soundness of the previous audits (`01_paper_baseline_codex` and `02_codex_implementation_audit`) by re-evaluating primary source files, models, splits, preprocessing routines, and metric calculations. We identify unsupported conclusions, incorrect paper interpretations, and critical implementation details that were missed or overly prioritized.

## Findings Classification

We group the findings into four categories:

### 1. CONFIRMED_BY_BOTH
- **NIMA 10-bin distribution head and expected score expectation**: Implementing `Dense(10, activation="softmax")` and the expected score calculation `sum(p * s)` is a faithful translation of the NIMA paper's output representation.
- **NIMA Earth Mover's Distance (EMD) loss**: The local implementation of EMD `r=2` is mathematically equivalent to the normalized squared EMD loss in the paper.
- **RGNet practical region-based graph convolutional reasoning**: The presence of `RegionGraphBuilder` and `GraphConvolution` in `src/models/rgnet.py` implements a genuine relational graph neural network, distinguishing it from a standard global pooling CNN.
- **A-LAMP multi-patch input**: The model accepts 5 patches and 1 global view as inputs, preserving the multi-patch input format of the A-LAMP baseline.

### 2. DISPUTED
- **NIMA fixed candidate score (72)**: We dispute the paper fidelity score of 72. The pre-trained MobileNetV2/EfficientNetV2 backbone weights were loaded with ImageNet initialization, but the training script feeds `[0, 1]` floats. This causes a representation mismatch because ImageNet models expect `[-1, 1]` or `[0, 255]` input (which degrades visual feature extraction). The corrected score is **68**.
- **RGNet practical paper fidelity score (48)**: We dispute the paper fidelity score of 48. The practical implementation swaps DenseNet-121/DenseASPP for EfficientNetV2B0, replaces learned projected inner-product affinity with fixed cosine similarity, and replaces region LSE aggregation with learned attention. Most importantly, it contains a double-rescaling bug that compresses input ranges. The corrected score is **18**.
- **KonIQ mobile paper fidelity score (15)**: We dispute the score of 15. The model is a simple MobileNetV2 regressor trained on a random split, whereas the paper uses KonCept512 (Inception-ResNet-v2) on the official split. Due to the complete architecture swap and random splitting, the corrected score is **5**.
- **FLIVE image mobile paper fidelity score (10)**: We dispute the score of 10. The paper uses PaQ-2-PiQ (ResNet-18-based local-global ROI architecture). The local model is a simple MobileNetV2. Thus the architecture is completely different. The corrected score is **5**.

### 3. REFUTED
- **ICAA local line paper fidelity score (25)**: We refute the positive paper fidelity score of 25. The local file `src/models/icaa_dat.py` does not exist in the repository. The model is completely unimplemented in the local source tree. The corrected score is **0 (NOT_IMPLEMENTED)**.
- **Ensemble correlation recomputation (Metric Validity)**: The claim that the ensemble correlations were independently verified is refuted because the inspected prediction table lacks a ground-truth column. No independent recomputation of PLCC/SRCC can be validated from the available files.

### 4. STILL_UNVERIFIED
- **Active mobile/runtime application**: The active TFLite weights and policy loaded in the external Flutter app remain unverified.
- **Teacher-student distillation deployment**: Whether the distilled MDIQA student models (14k and 50k) are deployed or active in the app remains unverified.

## Summary Scorecard Comparison

| Model | Codex Paper Fidelity | Corrected Paper Fidelity | Codex Exp Validity | Corrected Exp Validity | Adversarial Verdict |
|---|---:|---:|---:|---:|---|
| KonIQ mobile | 15 | **5** | 58 | **45** | Overly generous; complete architecture swap & random split |
| FLIVE image mobile | 10 | **5** | 50 | **40** | Overly generous; complete architecture swap & database leakage |
| RGNet practical | 48 | **18** | 63 | **45** | Overly generous; double-scaling bug & missing essential math |
| NIMA fixed candidate | 72 | **68** | 64 | **64** | Near-faithful; data-to-backbone normalization mismatch |
| ICAA/DAT local line | 25 | **0** | 35 | **0** | Refuted; file does not exist (NOT_IMPLEMENTED) |
| Mobile A-LAMP v2 | 36 | **30** | 68 | **68** | Inspired-only; coordinates/attention replace object graphs |
