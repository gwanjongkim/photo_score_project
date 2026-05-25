# Paper-Faithful RGNet/A-LAMP Audit

## 1. Environment
- CWD: `/home/omen_pc1/photo_score_project`
- Date: `20260509`
- Git Status: Dirty tree but correctly positioned in main repository.

## 2. Dataset Inventory
AVA and AADB datasets are available in `data/processed/`.
- AVA has ~255k images across splits. Mean scores span 1.8 to 8.6. Binary labels are creatable from `mean_score`.
- AADB has ~9.5k images with normalized scores [0, 1]. Splits appear preserved.
See `dataset_inventory.json` for details.

## 3. Existing Experiment Results
Recent retry experiments for TFLite conversion and float32 baselines show:
- Best app-oriented RGNet model (`rgnet_aadb_gpu`) and float32 baseline achieved ~0.57 SRCC on AADB val set.
- Best app-oriented A-LAMP achieved ~0.58 SRCC on AADB val set.
- TFLite artifacts for RGNet showed parity in `rgnet_float32_retry` outputs, preserving inference capability.
- RGNet and A-LAMP are currently being used as app-candidate models via a weighted ensemble in the Flutter application. Unrelated to paper-faithful reproduction, the current implementations focus on runtime practicality (MobileNet/EfficientNet backbones, approximations of graph/layout constructions).

## 4. Paper Protocols
- **RGNet**:
  - AVA task: typically binary classification (good vs bad based on mean > 5) or multi-class.
  - AADB task: regression.
  - Backbone: DenseNet/VGG/ResNet.
  - Architecture: Local region nodes constructed from feature maps, Region Composition Graph, Graph Convolution.
- **A-LAMP**:
  - AVA task: typically classification or distribution regression.
  - Patch strategy: Saliency-map-based adaptive patch selection with diversity constraint.
  - Backbone: Shared VGG16.
  - Architecture: Multi-patch subnet, Layout-aware subnet with Object/Global attribute graphs.

## 5. Existing Implementation Gap Analysis
Current implementations (`src/models/rgnet.py`, `src/models/alamp.py`) are practical approximations designed for mobile inference.
See `implementation_gap_matrix.json` for exact gaps (e.g., EfficientNet backbones instead of paper backbones, missing exact layout/object graphs, missing AVA classification implementations).

## 6. RGNet Paper Experiment Plan
**Track A: RGNet-paper-AVA-classification**
- Manifest: `data/processed/ava/{train,val,test}.csv`
- Label: Binary (mean_score >= 5.0)
- Architecture: DenseNet or VGG backbone, exact RegionGraph construction.
- Loss: CrossEntropy
- Metrics: Accuracy, F1
- Commands: `python src/train/train_rgnet_paper.py --dataset ava --task classification`
- Expected Output: `outputs/rgnet_paper_ava_class`
- Criteria: Reproduce paper accuracy on AVA.

## 7. A-LAMP Paper Experiment Plan
**Track C: A-LAMP-paper-AVA-classification**
- Manifest: `data/processed/ava/{train,val,test}.csv`
- Label: Binary (mean_score >= 5.0)
- Architecture: VGG16 shared backbone, exact adaptive patch proposal and layout-aware subnet.
- Loss: CrossEntropy
- Metrics: Accuracy, F1
- Commands: `python src/train/train_alamp_paper.py --dataset ava --task classification`
- Expected Output: `outputs/alamp_paper_ava_class`
- Criteria: Reproduce paper accuracy on AVA.

## 8. Risks and Blockers
- **Missing Object/Saliency Proposals**: Exact generation logic for bounding boxes and saliency may be hard to replicate without original authors' code.
- **Architecture Ambiguity**: Paper implementation details for RGNet graph edges and A-LAMP layout graphs are often omitted from texts.
- **GPU Memory**: VGG16/DenseNet with full graphs and multi-patches will consume significantly more memory than the current MobileNet/EfficientNet approximations.
- **Framework Differences**: Caffe/PyTorch (common for older papers) paradigms don't always translate 1:1 to TensorFlow.
- **Overclaiming Risk**: We must clearly mark these new models as "attempted paper reproductions" and distinguish them from "official weights."

## 9. Recommended Implementation Order
1. Track B: RGNet-paper-AADB-regression (easiest to adapt from current AADB regression).
2. Track A: RGNet-paper-AVA-classification (requires adding classification head and label conversion).
3. Track C: A-LAMP-paper-AVA-classification (hardest, requires complex preprocessing reproduction).

## 10. What Not To Claim
- Do not claim these implementations are the official models.
- Do not deploy these unoptimized, memory-heavy paper-faithful models to the Flutter app.

## 11. Next Codex Prompt Recommendation
"Implement the `RGNet-paper-AADB-regression` track. Create a new model script `src/models/rgnet_paper.py` strictly using the DenseNet/VGG backbone and exact paper RegionGraph definitions. Setup the training script to output to `outputs/rgnet_paper_aadb_regression/` without overriding current practical models."
