<!-- 이전 오딧 보고서의 주장과 증거를 대조하여 불일치 및 반박 사항을 기록하는 문서. -->
# Disputed Findings

This report details the specific disputes and corrections raised against the Codex reproducibility audit.

---

## 1. RGNet Practical Paper Fidelity

- **Model**: RGNet Practical (`src/models/rgnet.py`)
- **Codex conclusion**: Paper fidelity = 48, classified as "PAPER_INSPIRED_IMPLEMENTATION". Codex acknowledged the backbone swap and loss/pooling differences but treated the model as functionally equivalent for relational graph reasoning.
- **Evidence cited by Codex**: `src/models/rgnet.py:55-70` (RegionGraphBuilder), lines 76-105 (GraphConvolution), and output checks showing correct execution.
- **Independent verification**: Re-evaluation of the model file shows the following critical deviations from WACV 2020:
  1. **ASPP / DenseASPP omission**: The WACV 2020 paper requires a dilated CNN backbone combined with DenseASPP (atrous rates 3, 6, 12, 18) to build spatial region nodes at multiple fields-of-view. The local implementation uses `EfficientNetV2B0` followed by a simple 1x1 projection conv (`region_proj`), completely losing multi-scale dilation.
  2. **No learned projections**: The paper requires a learned projected inner-product affinity `phi(x_i)^T * theta(x_j)` to compute edge weights. The local `RegionGraphBuilder` uses a fixed, parameter-free cosine similarity matrix.
  3. **No LSE Aggregation**: The paper aggregates region scores using Log-Sum-Exp (LSE) with a validated $r=4$ followed by a two-class softmax. The local implementation uses learned attention pooling.
  4. **The Double-Scaling Bug**: The model uses the default `tf.keras.applications.EfficientNetV2B0` which has internal preprocessing enabled. Since the training/evaluation dataset feeds `[0, 1]` scaled images, this leads to double scaling (dividing by 255 again inside the network), compressing inputs to `[0, 0.0039]`.
- **Missing or contradictory evidence**: The paper-reproducing dilated convolutions and learned edge projections are completely absent from the source code.
- **Corrected conclusion**: The model is a highly simplified GCN prototype with severe input-scaling defects and lacks the defining ASPP/learned-affinity mathematical structures of the paper. It should be downgraded to an experimental prototype.
- **Corrected score**: Paper Fidelity = **18** (was 48).
- **Confidence**: HIGH.
- **Required resolving test**: Implement a model with `include_preprocessing=False` in the backbone, replace cosine similarity with a bilinear projection layer `tf.matmul(x, W) @ tf.transpose(tf.matmul(x, V))`, and evaluate performance on AADB.

---

## 2. ICAA Local Line Port

- **Model**: ICAA / DAT Local Line (`src/models/icaa_dat.py` or equivalent)
- **Codex conclusion**: Paper fidelity = 25, classified as "UNVERIFIED".
- **Evidence cited by Codex**: Mention of a TF-native port and official PyTorch checkpoints.
- **Independent verification**: We searched the repository and confirmed that `src/models/icaa_dat.py` is absent. There is no local TensorFlow or Keras port of ICAA DAT in the source tree. Only the official PyTorch code (`external/icaa_official_repo/`) is present.
- **Missing or contradictory evidence**: No local source file implementing ICAA DAT exists in the codebase.
- **Corrected conclusion**: The local port is completely non-existent (NOT_IMPLEMENTED). A model with no local code cannot receive a paper fidelity score of 25.
- **Corrected score**: Paper Fidelity = **0** (was 25), Experimental Validity = **0** (was 35).
- **Confidence**: HIGH.
- **Required resolving test**: Verify that `src/models/icaa_dat.py` is indeed missing and remove it from any deployment or active benchmark references.

---

## 3. KonIQ Mobile Paper Fidelity

- **Model**: KonIQ Mobile (`src/models/technical_regressor.py`)
- **Codex conclusion**: Paper fidelity = 15, classified as "PRODUCTION_ADAPTED_MODEL".
- **Evidence cited by Codex**: MobileNetV2 architecture in `src/models/technical_regressor.py`.
- **Independent verification**: The original KonIQ-10k paper specifies the `KonCept512` model, which uses an Inception-ResNet-v2 backbone at a resolution of 512x384. The local implementation uses `MobileNetV2` at 224x224. Furthermore, the dataset loader (`csv_dataset.py`) feeds `[0, 1]` float images to a MobileNetV2 loaded with pre-trained ImageNet weights (which expect `[-1, 1]` normalization). This creates a data-to-backbone normalization mismatch that degrades feature extraction.
- **Missing or contradictory evidence**: The Inception-ResNet-v2 backbone and the official 512x384 split files are completely omitted.
- **Corrected conclusion**: The model shares nothing with the paper model except the dataset name. It is a simple MobileNetV2 regressor trained on a custom random split.
- **Corrected score**: Paper Fidelity = **5** (was 15).
- **Confidence**: HIGH.
- **Required resolving test**: Train a true `KonCept512` model on the official split and compare the SRCC/PLCC performance against the MobileNetV2 regressor.

---

## 4. FLIVE Image Mobile Paper Fidelity

- **Model**: FLIVE Image Mobile (`src/models/technical_regressor.py`)
- **Codex conclusion**: Paper fidelity = 10, classified as "PRODUCTION_ADAPTED_MODEL".
- **Evidence cited by Codex**: MobileNetV2 architecture in `src/models/technical_regressor.py`.
- **Independent verification**: The PaQ-2-PiQ paper requires a local-global ROI architecture using a ResNet-18 backbone with RoI pooling to jointly predict patch-level and picture-level quality. The local model is a simple MobileNetV2 regressor trained on image-level MOS. It completely lacks the local-to-global ROI features.
- **Missing or contradictory evidence**: The ResNet-18 ROI network and patch-level joint loss are absent from the local training path.
- **Corrected conclusion**: The model shares nothing with the paper model except the dataset name. It is a simple MobileNetV2 regressor.
- **Corrected score**: Paper Fidelity = **5** (was 10).
- **Confidence**: HIGH.
- **Required resolving test**: Run the official PyTorch PaQ-2-PiQ model on the test set and compare the results with the local MobileNetV2.
