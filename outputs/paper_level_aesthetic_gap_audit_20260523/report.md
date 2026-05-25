# Paper-Level Aesthetic Performance Gap Audit

## 1. Summary
This audit identifies technical and architectural gaps between our current on-device aesthetic models (paper-oriented RGNet, Mobile A-LAMP v2) and the official paper-level targets reported in literature. While our models achieve competitive results, a significant "Resize Gap" and architectural simplifications prevent them from reaching paper-level targets such as 82.5% AVA Accuracy or 0.71 AADB SRCC. This report outlines a staged plan to close these gaps through preprocessing alignment, faithful teacher reproduction, and knowledge distillation.

## 2. Paper-Level Targets
| Model | Paper Source | Benchmark | Metric | Paper-Level Target |
| :--- | :--- | :--- | :--- | :--- |
| **RGNet** | WACV 2020 | AVA | Accuracy | **83.59%** |
| **RGNet** | WACV 2020 | AADB | SRCC | **0.7104** |
| **A-LAMP** | CVPR 2017 | AVA | Accuracy | **82.50%** |
| **A-LAMP** | CVPR 2017 | AVA | F-measure | **0.92** |

## 3. Current Project Results
| Model | Implementation | Benchmark | Metric | Result (TFLite/Mobile-like) | Result (TF-Native) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RGNet** | paper-oriented RGNet (v1) | AADB Test (Mobile-like) | SRCC | **0.6139** | **0.6819** |
| **RGNet** | paper-oriented RGNet (AVA) | AVA | Accuracy | **TBD** | **76.97%** |
| **A-LAMP**| Mobile A-LAMP v2 full-AVA, A-LAMP-inspired | AVA Proxy (2,000 samples) | Accuracy | **69.95%** | **N/A** |
| **A-LAMP**| A-LAMP-inspired (v0) | AVA Proxy (1,000 samples) | Accuracy | **N/A** | **72.36%** |

*Note: The difference between RGNet's TF-Native result (0.6819) and Mobile-like result (0.6139) is primarily due to the **Resize Gap** (interpolation differences between TensorFlow and PIL/Flutter).*

## 4. RGNet Gap Analysis
| Component | Paper Requirement | Current Implementation | Evidence Path | Expected Impact | Fix Difficulty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ASPP** | DenseASPP (cascaded) | Simple ASPP (parallel) | `src/models/rgnet_paper_v1.py` | Medium (Feature diversity) | Low |
| **Graph** | RegionGraph (Sophisticated edges) | Cosine Similarity Adjacency | `RegionSimilarityAdjacency` | Medium (Relation modeling) | Medium |
| **Preprocessing** | Likely SGD-optimized pipeline | `tf.image.resize` (TF-Native) | `src/train/train_rgnet_paper_v1_aadb.py` | **High (Resize Gap)** | Low |
| **Backbone** | DenseNet121 | DenseNet121 | Matches | Low | N/A |

## 5. A-LAMP Gap Analysis
| Component | Paper Requirement | Current Implementation | Evidence Path | Expected Impact | Fix Difficulty |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Architecture** | Dual-subnet Multi-Patch + Layout-Aware | MobileNetV3 + Region Attention | `src/models/mobile_alamp_v2.py` | **Very High** | High |
| **Backbone** | Heavier CNN backbone | MobileNetV3Small (Light) | Matches | Medium | Low |
| **Patching** | Adaptive selection | Fixed Top-5 from saliency | `src/train/train_mobile_alamp_v2.py` | High (Composition) | Medium |
| **Resolution** | Arbitrary-size handling | Fixed 384x384 / 224x224 | Matches | Medium | High |

## 6. Resize / Preprocessing Gap
The **Resize Gap** is the most immediate technical debt. 
- **Finding:** RGNet SRCC drops from **0.6819** (TF-Native) to **0.6139** (Mobile-like) due to interpolation differences.
- **Root Cause:** Training uses `tf.image.resize`, while mobile deployment/benchmarking uses `PIL.Image.resize(BILINEAR)`.
- **Conclusion:** On-device models are being evaluated on out-of-distribution resized artifacts relative to their training environment.

## 7. Benchmark Split and Metric Gap
- **A-LAMP Subset:** Current benchmarks use a 2,000-sample proxy. While representative, it is not the full 25,551-sample AVA test set used in the paper.
- **Label Imbalance:** AVA is ~70% positive. Raw accuracy is inflated by high recall. We need balanced accuracy and F1 focus to match paper rigor.

## 8. Architecture Gap
Our models are mobile-oriented/paper-inspired approximations, not official reproductions.
- **RGNet:** The Graph Convolution implementation is a single-head residual block. The paper uses more complex graph reasoning.
- **A-LAMP:** The "Layout-Aware" subnet in the paper specifically models spatial relationships between patches. Our mobile version uses 1D global attention over region tokens, losing explicit spatial layout.

## 9. Training Recipe Gap
- **Optimizer:** We use Adam (1e-4) with early stopping. SOTA papers often use SGD with Momentum, weight decay, and multi-step learning rate schedules over 100+ epochs.
- **Augmentation:** Current scripts only use random horizontal flips. Papers often use color jitter, rotation, and multi-scale cropping.

## 10. Mobile Deployment Gap
- **FP16 Quantization:** Validated in previous audits to have minimal impact (<0.001 diff), so quantization is NOT a primary gap source.

## 11. Can Mobile-Resize Retraining Reach Paper SOTA?
**Likely no.**
It is necessary to reduce the deployment gap and recover the ~0.07 SRCC drop in RGNet (bringing it back to ~0.68), but it is insufficient to reach paper-level targets without architecture/training/teacher improvements. Reaching 0.71+ (RGNet) or 82%+ (A-LAMP) requires the architectural depth and heavier backbones used in the papers, which exceed current mobile-oriented capacity.

## 12. Staged Roadmap to Paper-Level Performance

### Stage 1: Full Benchmark and Preprocessing Parity
- Evaluate A-LAMP and RGNet on the **Full 25,551 AVA Test Set**.
- Align resizing/cropping/normalization between Python training and Flutter/Mobile deployment.
- Retrain models using mobile-like preprocessing (PIL-Bilinear) to close the Resize Gap.

### Stage 2: Official-Like/Heavy Teacher Reproduction
- Implement and train faithful "Heavy" versions of the papers (VGG/DenseNet backbones, full Graph/Layout subnets).
- Evaluate these teachers against paper-level targets to establish the performance ceiling.

### Stage 3: Teacher-Student Distillation
- Distill knowledge from the Heavy Teachers into mobile students (MobileNetV3Large or EfficientNet-Lite).
- Use combined losses: Hard label loss + soft score distillation + pairwise ranking loss.

### Stage 4: Deployment and Verification
- Export FP16 TFLite models.
- Perform 1024-sample parity checks and full on-device benchmarks to verify final performance.

## 13. Final Judgment
Current models are high-quality, paper-inspired implementations optimized for mobile deployment, but they are not official reproductions. The 10-12% accuracy gap in A-LAMP is primarily architectural, while the 0.07 SRCC gap in RGNet is primarily a preprocessing artifact. Closing these gaps requires the rigorous alignment and distillation roadmap outlined above.
