# TOPIQ-lite Paper Fidelity Audit

## 1. Summary
The current A-cut technical IQA model (`topiq_lite`) is a lightweight, mobile-optimized adaptation of the original TOPIQ architecture. It successfully preserves the core "top-down semantics to distortions" philosophy and multi-scale guidance, but significantly simplifies the backbone and attention modules to meet TensorFlow Lite (TFLite) mobile deployment constraints.

## 2. Paper Core Requirements
The original TOPIQ paper ("TOPIQ: A Top-down Approach from Semantics to Distortions for Image Quality Assessment") defines several core requirements:
1. Top-down IQA from semantics to distortions
2. Multi-scale feature extraction
3. High-level semantic feature guides low-level/local distortion feature
4. Cross-scale attention
5. Coarse-to-fine / CFANet-like information propagation
6. NR-IQA regression support
7. ResNet50 backbone
8. FR-IQA and NR-IQA support

## 3. Current Implementation Facts
Based on the code in `src/models/topiq_lite.py`:
1. **Backbone name:** EfficientNetV2B0
2. **Input size:** 384x384x3
3. **Input value range:** 0..255 `float32` (EfficientNetV2 preprocessing handles scaling internally)
4. **Feature layer names:** `block3b_add` (low), `block5c_add` (mid), `top_activation` (high)
5. **Low/mid/high feature map shapes:** `low`: [48, 48, 48], `mid`: [24, 24, 112], `high`: [12, 12, 1280]
6. **Attention layers:** 1x1 `Conv2D` with `sigmoid` activation to generate spatial attention maps, followed by bilinear `Resizing` and `Multiply`.
7. **Aggregation method:** `GlobalAveragePooling2D` on each guided feature map, followed by `Concatenate`.
8. **Regression head:** `Dense` (256 units, swish activation) -> `Dropout` (0.3) -> `Dense` (1 unit).
9. **Output activation and scale:** `sigmoid` activation, producing a normalized MOS in the range [0, 1] (multiplied by 100 externally).

## 4. Component Mapping Table
| TOPIQ paper component | Implemented? | Evidence in code | Difference |
| :--- | :---: | :--- | :--- |
| multi-scale features | Yes | Extracts from `block3b_add`, `block5c_add`, `top_activation`. | Uses EfficientNetV2 stages instead of ResNet stages. |
| top-down semantic guidance | Yes | `high_feature` guides `mid_feature`; `mid_guided` guides `low_feature`. | - |
| cross-scale attention | Yes | 1x1 `Conv2D` (sigmoid) + `Resizing` + `Multiply`. | Simplified spatial attention map. |
| high-level-to-low-level attention map | Yes | `high_to_mid_attention` and `mid_to_low_attention`. | - |
| coarse-to-fine propagation | Yes | Sequential multiplication (`high` -> `mid` -> `low`). | - |
| NR-IQA regression head | Yes | GAP + Concat + Dense(256) + Dense(1). | - |
| ResNet50 backbone | No | Uses `tf.keras.applications.EfficientNetV2B0`. | Swapped for mobile efficiency. |
| CFANet-equivalent module | Partial | The attention map mechanism acts as a highly simplified CFANet. | Lacks complex channel/spatial attention blocks; uses simple 1x1 Conv to ensure TFLite compatibility. |

## 5. Differences from Original TOPIQ
- **Backbone:** Swapped from ResNet50 to EfficientNetV2B0 for parameter efficiency and mobile speed.
- **Framework:** Implemented in TensorFlow/Keras rather than PyTorch.
- **CFANet details:** The complex Coarse-to-Fine Attention Network (CFANet) from the paper is abstracted into a highly simplified spatial attention map (1x1 Conv + Sigmoid) and a bilinear resize operation.
- **Attention module details:** No advanced paired channel/spatial attention; purely spatial 1x1 convolution to reduce latency.
- **FR/NR support:** The FR-IQA branch is completely removed as A-cut only requires NR-IQA.
- **Dataset/training strategy:** Trained directly on a mixed replay dataset (SPAQ, KonIQ, FLIVE) instead of the specific multi-stage training detailed in the original paper.
- **Mobile/TFLite constraints:** Fixed spatial shapes are enforced for `Resizing` layers to ensure the exported TFLite model uses only Builtin ops without Flex/Select TF Ops.

## 6. Claim Safety

**One short sentence:**
The deployed model is a lightweight, mobile-optimized adaptation of the TOPIQ architecture, utilizing an EfficientNetV2 backbone and simplified top-down attention mechanisms.

**One technical paragraph:**
We implemented a "TOPIQ-lite" model that preserves the core "semantics-to-distortions" philosophy of the original TOPIQ paper. It extracts multi-scale features (low, mid, high) from an EfficientNetV2B0 backbone and applies a coarse-to-fine information propagation strategy. High-level semantic features generate spatial attention maps to guide mid-level features, which in turn guide low-level features. These guided multi-scale features are globally pooled, concatenated, and passed through a multi-layer perceptron regression head to predict the final NR-IQA score.

**One limitation paragraph:**
While directly inspired by TOPIQ, this model is not a strict reproduction. To ensure efficient on-device inference via TensorFlow Lite, the ResNet50 backbone was replaced with EfficientNetV2, the complex CFANet modules were simplified to basic 1x1 convolutional attention maps, and FR-IQA support was omitted. Consequently, it trades some of the original model's theoretical capacity and architectural complexity for practical mobile deployment feasibility and speed.

## 7. Final Classification
**Paper-oriented TOPIQ-lite**

*Explanation:* The code is more than just a "generic attention CNN" because it explicitly models the top-down, semantic-to-distortion guidance flow that defines TOPIQ. However, it cannot be called a "strict reproduction" because the backbone, attention modules, and training strategies have been deliberately modified to favor mobile inference speed and TFLite compatibility. "Paper-oriented TOPIQ-lite" perfectly describes a model that adheres to the paper's architectural philosophy while aggressively optimizing its implementation for production.