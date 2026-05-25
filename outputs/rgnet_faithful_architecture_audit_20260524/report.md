# RGNet Faithful Architecture Audit

## 1. Summary
This audit compares the current `rgnet_paper_v1.py` implementation against the WACV 2020 paper "Composition-Aware Image Aesthetics Assessment". The current implementation is a high-quality "paper-oriented approximation" that achieves strong results (AADB SRCC 0.66) but lacks two critical architectural features and a rigorous training recipe required to hit the paper target (0.7104). The most significant gaps are the absence of **spatial graph edges** and the use of a simplified **Parallel ASPP** instead of the paper's **Cascaded DenseASPP**.

## 2. Paper Architecture Requirements
| Component | Paper Specification |
| :--- | :--- |
| **Backbone** | DenseNet121 (pre-trained on ImageNet) |
| **Feature Extraction** | Fully convolutional features from late DenseNet layers. |
| **Context Module** | **DenseASPP**: Cascaded dilated convolutions for multi-scale context. |
| **Region Nodes** | Spatial locations in the feature map represent local regions. |
| **Graph Construction** | **Hybrid Edges**: Semantic (feature similarity) AND Spatial (geometric layout). |
| **Graph Reasoning** | Graph Convolutional Networks (GCN) to model region dependencies. |
| **Aggregation** | Image-level score prediction from region-level reasoning. |
| **Training** | SGD with Momentum, weight decay, multi-step LR schedule. |

## 3. Current Implementation
- **Status:** `RGNet-paper-v1 approximation`
- **Backbone:** DenseNet121 (Match).
- **ASPP:** Parallel branch `ASPPContextModule` (5 branches).
- **Graph Edges:** `V1RegionSimilarityAdjacency` (Cosine similarity only).
- **Reasoning:** `V1ResidualGraphConvolution` (3 blocks default).
- **Optimizer:** Adam (1e-4).

## 4. Component-by-Component Comparison
| Component | Match? | implementation Note |
| :--- | :--- | :--- |
| **Input Size** | Match | 256x256 (Default). |
| **Backbone** | Match | DenseNet121. |
| **DenseASPP** | **GAP** | Current uses parallel ASPP; paper specifies cascaded DenseASPP for denser scale coverage. |
| **Region Nodes** | Match | Spatial positions in the 8x8 or 16x16 feature map. |
| **Semantic Edges**| Match | Cosine similarity with softmax normalization. |
| **Spatial Edges** | **GAP** | **Missing.** Paper models geometric layout between regions; current models only feature similarity. |
| **GCN Blocks** | Match | 3 blocks (Paper typically uses 2-3). |
| **Aggregation** | Match | LSE (Log-Sum-Exp) with r=4. |
| **Training Loss** | Match | MSE for AADB regression. |

## 5. Confirmed Matches
- **DenseNet121 Backbone:** Correctly implemented as the feature extractor.
- **Region-Level Scoring:** Model predicts scores at the region level and aggregates them, aligning with the "Composition-Aware" philosophy.
- **Log-Sum-Exp Aggregation:** Stable implementation found in `RegionScoreAggregation`.

## 6. Confirmed Gaps
- **Spatial Graph Edges:** This is the most critical missing piece for "Composition-Awareness." The paper explicitly models the relative spatial positions of regions to capture global layout.
- **DenseASPP vs Parallel ASPP:** Current implementation uses `ASPPContextModule` where branches are parallel. The paper's "DenseASPP" cascades dilated layers to achieve a much larger and denser receptive field.
- **Optimizer/Schedule:** Switching from Adam to SGD with Momentum and specific weight decay is a known "rigor" requirement for matching academic aesthetic benchmarks.

## 7. DenseASPP / Feature Pipeline Check
- **Current:** Parallel branches with rates [1, 3, 6, 12, 18].
- **Paper Requirement:** Cascaded layers where the output of rate=3 is fed into rate=6, and so on. This creates "dense" receptive field coverage rather than "sampled" coverage.

## 8. Graph Edge Modeling Check
- **Current:** Adjacency matrix $A$ is purely $Softmax(Similarity / T)$.
- **Paper Requirement:** $A = \alpha \cdot A_{semantic} + (1-\alpha) \cdot A_{spatial}$.
- **Impact:** Without $A_{spatial}$, the model lacks the ability to learn that "a red region at the top-left" vs "a red region at the center" implies different compositions.

## 9. Training Recipe Gap
- **Current:** Adam, early stopping (patience 3), restored best weights.
- **Paper:** SGD (Momentum 0.9), Weight Decay (1e-4), specific multi-step decay (e.g., at 30, 60 epochs).
- **Assumption:** Retraining with SGD is necessary to reach the final 2-3% SRCC improvement.

## 10. Likelihood of Reaching SRCC 0.7104
- **Verdict:** **UNLIKELY** without architectural changes.
- **Reasoning:** We have recovered the "Resize Gap" (0.61 -> 0.66), but the remaining 0.05 SRCC gap represents the "Reasoning Gap." Feature similarity alone cannot fully model composition.

## 11. Recommended Next Implementation
**Experiment T1: Spatial Graph Edges & SGD Rigor**
1.  Update `V1RegionSimilarityAdjacency` (or create `V2`) to include a learnable or fixed spatial distance penalty/edge.
2.  Switch the training script to use SGD with Momentum and weight decay.
3.  Retrain on AADB to verify if "Composition Reasoning" improves ranking.

## 12. Risks
- **Complexity:** Spatial graph modeling increases the number of parameters and edge calculations.
- **Convergence:** SGD is more sensitive to hyperparameters than Adam; finding the correct multi-step schedule for AADB may require more trials.
