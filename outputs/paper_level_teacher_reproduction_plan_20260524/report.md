# Paper-Level Teacher Reproduction Plan

## 1. Summary
This plan outlines a technical roadmap to reach paper-level aesthetic assessment targets (RGNet: 0.7104 SRCC; A-LAMP: 82.5% Accuracy) by implementing "Faithful Teacher" models. Current project models are mobile-oriented approximations that established a strong baseline (RGNet 0.66 SRCC, A-LAMP 70.5% Accuracy) but lack the high-capacity components required to hit paper-level targets. This strategy prioritizes building heavy, academic-grade teachers first, which will then serve as the source of truth for mobile distillation.

## 2. Paper-Level Targets
| Model | Paper Source | Benchmark | Metric | Target |
| :--- | :--- | :--- | :--- | :--- |
| **RGNet** | WACV 2020 | AADB | SRCC | **0.7104** |
| **RGNet** | WACV 2020 | AVA | Accuracy | **83.59%** |
| **A-LAMP** | CVPR 2017 | AVA | Accuracy | **82.50%** |
| **A-LAMP** | CVPR 2017 | AVA | F-measure | **0.92** |

## 3. Gap Analysis: Missing Components

### RGNet (WACV 2020)
| Component | Paper Specification | Current Implementation | Status |
| :--- | :--- | :--- | :--- |
| **ASPP** | DenseASPP (Cascaded Dilations) | Simple Parallel ASPP | **Gap** |
| **Graph Edges**| Spatial + Semantic Relationships | Cosine Similarity only | **Major Gap** |
| **Backbone** | DenseNet121 | DenseNet-style backbone appears partially aligned; full paper-equivalent DenseASPP/feature pipeline remains unverified. | **Assumption** |
| **Training** | SGD + Momentum, Multi-step LR | Adam (Fixed/Early Stop) | **Gap** |

### A-LAMP (CVPR 2017)
| Component | Paper Specification | Current Implementation | Status |
| :--- | :--- | :--- | :--- |
| **Architecture** | Dual-subnet: Multi-Patch + Layout-Aware | Single-path/MobileNet approximations | **Major Gap** |
| **Patching** | **Adaptive** Layout-Aware Selection | Fixed Top-5 Saliency | **Major Gap** |
| **Subnets**| Multi-Patch and Layout-Aware subnets | Flattened dense layers / attention | **Gap** |
| **Resolution** | Arbitrary-size support | Fixed (384/224) | **Gap** |
| **Aggregation**| Specific aggregation layer | Global pooling / attention | **Gap** |

## 4. Reusable Repo Artifacts
- **A-LAMP Adaptive Selection:** `external/A-Lamp_external_audit/A-Lamp/multi_patch_subnet/adaptive_patch_selection/patch_selection.py` contains the logic for adaptive cropping.
- **RGNet v1 Base:** `src/models/rgnet_paper_v1.py` provides the residual graph convolution framework.
- **Data Pipelines:** Existing pipelines include TF-native paths and newly added PIL/mobile-like paths, but full Python–Flutter tensor parity is not yet proven for all models.

## 5. Strategy: Faithful Paper Teacher First
We will build **Faithful Paper Teachers** (Option A). Building a CLIP-based teacher (Option B) is powerful but risks moving away from the specific "Composition-Aware" and "Layout-Aware" goals of the project.

### Teacher Success Criteria:
- **RGNet Teacher:** SRCC >= 0.70 on AADB Test (TF-native or PIL).
- **A-LAMP Teacher:** Accuracy >= 80% on Full AVA Test Set.

## 6. Staged Roadmap

### Phase 1: Heavy Teacher Implementation (The "Ceiling")
- **RGNet-Faithful:**
  - Implement cascaded DenseASPP.
  - Upgrade `V1RegionSimilarityAdjacency` to include spatial coordinate edges.
  - Switch to SGD optimizer with `1e-4` weight decay.
- **A-LAMP-Faithful:**
  - Integrate `patch_selection.py` into the training pipeline for adaptive patching.
  - Implement dedicated Multi-Patch and Layout-Aware subnets.
  - Implement paper-specific aggregation layer and handle arbitrary-size inputs if feasible.

### Phase 2: Distillation to Mobile Student (The "Bridge")
- **Student Candidates:**
  - `MobileNetV3Large` (A-LAMP)
  - `EfficientNet-Lite0` (RGNet)
- **Distillation Loss:**
  - `L = hard_label_loss + alpha * teacher_soft_mse + beta * ranking_loss`.
- **Target:** Maintain 95% of the Teacher's performance at <10% of the parameter count.

### Phase 3: Export & On-Device Verification
- FP16 TFLite export.
- Alignment check: Quantify the "Resize Gap" again to ensure retraining fully recovered it.
- Flutter integration of the final distilled models.

## 7. Immediate Next Experiments
1. **RGNet faithful architecture audit:** Deep dive into `rgnet_paper_v1.py` and comparison with WACV 2020 paper to verify exact layer configurations.
2. **A-LAMP external code integration feasibility audit:** Verify compatibility of `patch_selection.py` with existing AVA manifests and data loading pipeline.

## 8. Final Judgment
RGNet paper-level performance may be approachable but is not guaranteed. A faithful teacher must first exceed SRCC 0.70 on AADB before mobile distillation is justified. Reaching paper-level SOTA for A-LAMP (target 82.5% vs current 70.5%) remains highly difficult and is contingent on successful implementation of complex layout-aware subnets and adaptive patching.

## 9. Before Implementation: Required Verification
- Verify exact RGNet architecture against paper before modifying layers.
- Verify whether current DenseASPP is truly simple parallel ASPP.
- Verify whether spatial and semantic graph edges in the paper are implemented or missing.
- Verify A-LAMP external `patch_selection.py` compatibility with AVA manifests.
- Verify if A-LAMP layout-aware subnet code is empty or incomplete.
- Verify full AVA training/evaluation data availability.
- Verify expected runtime and GPU memory requirements for "Heavy" teachers.
