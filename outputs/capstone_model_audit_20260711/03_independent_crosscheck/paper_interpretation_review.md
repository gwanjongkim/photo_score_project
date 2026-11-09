<!-- 논문 요구사항 분석 과정에서 발견된 이전 에이전트의 잘못된 해석이나 누락을 검토하는 문서. -->
# Paper Interpretation Review

This document evaluates the paper baseline interpretations established in `01_paper_baseline_codex` and flags instances where requirements were interpreted incorrectly or too loosely.

---

## 1. NIMA: Split and Evaluation Protocol

- **Codex Interpretation**: Codex states that "the paper does not publish a reusable AVA filename list or validation split," and accepts a random 80/20 or 80/10/10 split as paper-faithful.
- **Correction**: While the AVA dataset does not have a single official validation list built-in, subsequent literature settled on a standard division (approximately 230,000 images for training and 20,000 for testing, based on specific image ID splits). More importantly, the NIMA paper explicitly states: *"We trained our models on 250,000 images and tested on the remaining 5,000 images."* Testing on a random 10% split (~25,000 images) matches neither the training size nor the test size of the paper.

---

## 2. RGNet: Graph construction and attribute supervision

- **Codex Interpretation**: Codex accepted the practical RGNet model as functionally equivalent because it implements a graph convolutional network (`RegionGraphBuilder` and `GraphConvolution`).
- **Correction**: This interpretation is too loose. The WACV 2020 paper describes a composition-aware network where region nodes are built by dilated CNN features combined with a multi-scale DenseASPP block (rates 3, 6, 12, 18), and graph edges are computed using learned projected inner products. The local implementation uses fixed cosine similarity over EfficientNet features. Swapping every mathematical component (ASPP, learned edge weights, LSE aggregation) and calling it a MATCH or functionally equivalent because "a graph exists" ignores the specific composition-modeling contributions of the paper.

---

## 3. A-LAMP: Saliency and Layout-Aware Graphs

- **Codex Interpretation**: Codex classified the local A-LAMP models as inspired-only but accepted the multi-patch attention as a reasonable approximation.
- **Correction**: The core contribution of A-LAMP (CVPR 2017) is the combination of:
  1. An adaptive patch proposal subnet that optimizes saliency, pattern diversity, and spatial non-overlap (via Wasserstein distance on color/edge distributions).
  2. A layout-aware subnet that builds a spatial coordinate/attribute graph over 4 local object nodes and 1 global node.
  The local implementation replaces the adaptive patch proposal with a deterministic grid heuristic and replaces the object layout graph with coordinate/mask channels (`LayoutCueAugmentation`). These are not just approximations; they are entirely different heuristics.

---

## 4. KonIQ-10k and FLIVE: Architecture Mappings

- **Codex Interpretation**: Codex accepted the local KonIQ and FLIVE mobile regressors as "PRODUCTION_ADAPTED_MODEL" but assigned positive paper fidelity scores (15 and 10) despite them being simple MobileNetV2 regressors.
- **Correction**: An independent review should classify these models as **0** paper fidelity (NOT_REPRODUCED). The KonIQ-10k paper evaluates `KonCept512` (Inception-ResNet-v2 at 512x384), and the PaQ-2-PiQ paper proposes a complex local-global ROI architecture. Training a MobileNetV2 on IQA datasets yields a dataset-trained model, not a paper reproduction. Assigning non-zero paper fidelity scores conflates "training on the same dataset" with "implementing the paper's method."

---

## 5. ICAA: Color Attributes and DAT

- **Codex Interpretation**: Codex assigned a paper fidelity of 25 to the ICAA DAT local line.
- **Correction**: This is an incorrect interpretation of the codebase. A model that has no local implementation code in the repository cannot be graded on paper fidelity. The PyTorch codebase of ICAA is vendored under `external/icaa_official_repo/` but has not been ported or integrated into the local `src/` modules.
