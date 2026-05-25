# A-LAMP MPNet 4096 Graph Fusion Experiment Final Report

This report summarizes the layout-aware graph fusion experiments (GraphLite and GraphGCN) conducted on the 4096-sample subset of the AVA dataset, using the improved A-LAMP-paper-MPNet backbone.

## 1. MPNet-only Baseline (Backbone)
The fusion models were built using frozen features from the following verified backbone:
- **Model Path**: `outputs/alamp_paper_mpnet_4096_expansion/v4_4096/final_model.keras`
- **Val ROC-AUC**: **0.7161**
- **Test ROC-AUC**: 0.7000
- **Average Precision**: 0.8445
- **BCE Loss**: 0.6205

## 2. GraphLite Fusion Results
GraphLite represents the layout graph as a flattened vector of bounding box coordinates and object class IDs.
- **Best Val ROC-AUC**: 0.7067 (Epoch 3)
- **Final Val ROC-AUC**: 0.6971 (Epoch 10)
- **Final Val BCE Loss**: 1.3820
- **Delta vs Baseline**: **-0.0190** (Regression)

## 3. GraphGCN Fusion Results
GraphGCN uses a Graph Convolutional Network to process the actual adjacency structure and node features.
- **Best Val ROC-AUC**: 0.7062 (Epoch 2)
- **Final Val ROC-AUC**: 0.6897 (Epoch 10)
- **Final Val BCE Loss**: 1.4601
- **Delta vs Baseline**: **-0.0264** (Regression)

## 4. Technical Status
- **Feature Cache**: MPNet features were successfully extracted and cached in `features/*.npz` for both runs, ensuring identical backbone inputs.
- **Graph Alignment**: 100% alignment (4096/4096 samples) was achieved between patch metadata and object graph records.
- **Implementation**: Both GraphLite and GraphGCN were successfully implemented and trained for 10 epochs using chunked processing to avoid OOM.

## 5. Overfitting Evidence
Both fusion models exhibited extreme overfitting almost immediately:
- **Train AUC**: Reached **~0.9978** in both cases.
- **Val AUC**: Plateaued early (Epoch 2 or 3) and then declined.
- **Val Loss**: Increased monotonically from ~0.65 to ~1.46, indicating the model was "memorizing" the training set's graph features rather than learning generalizable layout patterns.

## 6. Interpretation: Why Fusion Failed
- **Backbone Dominance**: The MPNet backbone (V4 selector) already implicitly captures multi-patch compositional cues. The additional graph data may be redundant.
- **Data Scarcity**: 4096 samples are sufficient for a CNN backbone but appear insufficient for a high-capacity MLP/GCN fusion head to distinguish signal from noise in object detector outputs.
- **Graph Quality**: The layout graphs were generated from a general-purpose object detector. The resulting nodes/edges may not be discriminative enough for fine-grained aesthetic assessment compared to raw image features.

## 7. Recommendation and Next Direction
This experiment is a paper-oriented approximation and **not an official A-LAMP reproduction**.

**Final Judgment:**
- **Stop GraphLite/GraphGCN fusion branch for now.**
- The **MPNet-only 4096** backbone remains the stronger candidate for the A-LAMP logic branch.
- No further time should be spent tuning the graph fusion head unless a significantly larger dataset (e.g., 20k+ samples) or a much more aggressive regularization strategy (dropout > 0.5, weight decay) is proposed.

---
*Report generated on 2026-05-16.*
