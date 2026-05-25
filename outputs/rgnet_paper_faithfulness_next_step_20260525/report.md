# RGNet Paper-Faithfulness Next Step

## 1. Summary
The RGNet paper-faithfulness track is currently leading with the `rgnet_paper_v1` (ImageNet-initialized, Mean-aggregation) variant, achieving **0.6819 SRCC** on AADB. This outperforms the original `v0` baseline (0.6433) but remains below the paper's reported target of **0.7104**. The A-LAMP paper-capacity experiments are currently paused due to underperformance, making RGNet the primary path for improving on-device aesthetic performance. This audit identifies critical architectural and procedural gaps that prevent matching the paper's results.

## 2. Existing RGNet Files
- **Models:**
  - `src/models/rgnet_paper_v1.py`: Current best "paper-oriented" proxy (ASPP + GCN blocks + Region scores).
  - `src/models/rgnet_paper.py`: Original `v0` baseline (Global pool + GCN blocks).
  - `src/models/rgnet.py`: Practical version used for production.
- **Scripts:**
  - `src/train/train_rgnet_paper_v1_aadb.py`: Standard AADB regression trainer.
  - `src/eval/evaluate_rgnet_paper_v1_aadb.py`: Evaluation suite.
- **Configs:**
  - `configs/paper_benchmarks/rgnet_paper_v1_aadb_regression.yaml`: Best config (with `agg_mean` modification).
  - `configs/paper_benchmarks/rgnet_paper_ava_classification.yaml`: AVA classification variant.

## 3. Existing RGNet Results
- **AADB SRCC:** **0.6819** (v1 `agg_mean_full`).
- **AADB PLCC:** 0.6878.
- **AADB MAE:** 0.1198.
- **Status:** The `v1` approximation with mean aggregation is the current record holder. LSE aggregation (r=4) underperformed in local tests (0.6389), likely due to the absence of the paper's full feature pipeline.

## 4. Paper Requirements
Based on the original paper *"Composition-Aware Image Aesthetics Assessment"* (WACV 2020):
- **Input Size:** 300x300 (current project uses 256x256 by default).
- **DenseASPP:** Cascaded dilated convolutions (current is parallel ASPP).
- **Spatial Graph Edges:** Adjacency matrix should include a geometric layout bias (current is pure content similarity).
- **Training Recipe:** SGD with Momentum, weight decay (1e-5), polynomial learning rate decay (starting at 1e-4), 80 epochs.
- **Aggregation:** Log-Sum-Exp (LSE) with $r=4$.

## 5. Current Implementation Gaps
| Gap | Description | Impact |
| :--- | :--- | :--- |
| **Spatial Graph Edges** | Missing geometric distance bias in the adjacency matrix. | **High** |
| **DenseASPP** | Current `ASPPContextModule` is parallel; paper uses cascaded branches. | Medium |
| **Optimizer/Schedule**| Current uses Adam; paper requires SGD + Momentum + PolyLR. | Medium-High |
| **Input Resolution** | Current 256x256; paper 300x300. | Low-Medium |

## 6. Highest-Impact Missing Component
**Spatial Graph Edges.** 
The paper's core claim is "Composition-Awareness." Content similarity alone (current implementation) captures which regions look alike but ignores where they are relative to each other. Adding spatial edges allows the GCN to explicitly reason about global layout rules (symmetry, balance, rule of thirds) by weighing connections based on geometric proximity.

## 7. Recommended First Experiment
**Experiment: "RGNet-v1-Hybrid-Spatial"**
1. **Model Change:** Update `src/models/rgnet_paper_v1.py` to include a `SpatialGeometryAdjacency` layer.
2. **Hybrid Adjacency:** $A = (1-\alpha) A_{semantic} + \alpha A_{spatial}$, where $A_{spatial}(i, j) = \exp(-\|p_i - p_j\|^2 / (2 \sigma^2))$.
3. **Task:** Implement this as `V1HybridSpatialAdjacency` to maintain backward compatibility.

## 8. Validation Plan
- **Smoke Run:** 1024 samples, 5 epochs to verify convergence and no NaNs.
- **Full Run:** 20 epochs on AADB using `agg_mean` (known best aggregation) + Hybrid Spatial Edges.
- **Commands:**
  ```bash
  python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/rgnet_spatial_experiment_20260525
  ```

## 9. Success Criteria
- **SRCC Baseline:** Must exceed **0.6819** on the AADB test split.
- **Faithfulness:** Adjacency visualization should show meaningful spatial weighting.

## 10. Final Recommendation
The next step should be the **implementation of Spatial Graph Edges**. It is a surgical architectural improvement that directly aligns with the paper's theoretical framework and has a high likelihood of closing the remaining 0.0285 SRCC gap to the paper target.
