# Aesthetic Teacher Reproduction Decision Report

## 1. Executive Summary
This report concludes the teacher reproduction track for A-LAMP and RGNet. Despite successful architectural implementations (Multi-Patch, Cascaded DenseASPP, GCN), both tracks failed to match the state-of-the-art benchmarks reported in their respective original papers (A-LAMP accuracy 0.825; RGNet AADB SRCC 0.7104). The primary blockers were hyperparameter sensitivity, class imbalance defaults in layout branches, and the inferiority of Log-Sum-Exp (LSE) aggregation compared to simple Mean aggregation in the current implementation.

**Recommendation:** Cease further paper reproduction attempts for A-LAMP/RGNet. Utilize the current best-performing "approximation" candidates as teachers for distillation, or transition to higher-performing modern architectures like MUSIQ or TOPIQ.

## 2. A-LAMP Results
| Variant | Status | ROC-AUC | Accuracy | F1 | Notes |
|---|---|---|---|---|---|
| **Multi-Patch Teacher (Best)** | **Stable** | **0.7877** | **0.7633** | **0.8482** | Best stable teacher. Uses mean+max patch aggregation. |
| Dual-Branch GCN | FAILED | 0.6635 | 0.7121 | 0.8281 | Extreme positive bias (96%). Driven by class imbalance. |
| Mobile A-LAMP v2 | Baseline | 0.7049 | 0.7049 | 0.7647 | The teacher significantly improves over the mobile version. |
| *Paper Target* | *Goal* | *N/A* | *~0.825* | *~0.92* | **Not reached.** |

### Key Findings:
- The Dual-Branch (Layout) branch suffered from "shortcut learning," predicting positive for almost every sample to minimize loss on the imbalanced AVA dataset.
- High-capacity graph fusion (4096 subset) led to immediate overfitting (Train AUC ~0.99) without validation gain.

## 3. RGNet Results
| Variant | Status | SRCC | PLCC | MAE | Notes |
|---|---|---|---|---|---|
| **V1 (Mean Aggregation) (Best)** | **Stable** | **0.6819** | **0.6878** | **0.1198** | Best ranking performance. Uses parallel ASPP. |
| Cascaded DenseASPP + Paper Recipe | Stable | 0.6683 | 0.6721 | 0.1206 | 300x300, PolyLR, but used LSE aggregation. |
| Cascaded DenseASPP only | Stable | 0.6050 | 0.6107 | 0.1299 | Parallel ASPP → Cascaded DenseASPP regression. |
| Hybrid-Spatial Adjacency | Stable | N/A | N/A | N/A | Implementation verified but no proven gain over baseline. |
| *Paper Target (AADB)* | *Goal* | *~0.7104* | *N/A* | *N/A* | **Not reached.** |

### Key Findings:
- **Aggregation is critical**: Switching from LSE (paper default) to Mean aggregation provided the single largest boost in SRCC (+0.043).
- **Architecture vs. Optimization**: Implementing the "correct" Cascaded DenseASPP alone degraded performance until the specific 300x300 input size and polynomial decay were added.

## 4. Why Paper-Level Performance Was Not Reached
1. **Implementation Success**: Both models were successfully ported to Keras 3 with faithful architectural components (ASPP, GCN, Multi-patch).
2. **Training Success**: Training was stable and converged.
3. **Benchmark Failure**:
   - **Hyperparameter Sensitivity**: The academic SGD/momentum recipes underperformed Adam in local experiments.
   - **Class Imbalance**: The high positive ratio in AVA (71%+) caused high-capacity models (A-LAMP Dual-Branch) to collapse into positive-only predictors.
   - **Aggregation Mismatch**: The numerically stable LSE implementation used in the project consistently underperforms Mean aggregation for ranking tasks, despite being the academic standard.

## 5. Current Best Candidates
- **Teacher A (Classification/AVA)**: `outputs/alamp_multipatch_teacher_full_ava_20260524/best.weights.h5`
  - *Best for distinguishing high/low aesthetics.*
- **Teacher B (Ranking/AADB)**: `outputs/rgnet_paper_v1_ablation_full_candidates_20260510/full_train/agg_mean_full/best.weights.h5`
  - *Best for relative scoring and order preservation.*

## 6. Deployment/Distillation Readiness
- The **Multi-Patch VGG16 Teacher** is ready for distillation. It provides a solid ~0.7877 ROC-AUC target, significantly higher than the mobile A-LAMP v2 (0.7049).
- The **RGNet V1 Mean Teacher** is ready for ranking distillation into mobile RGNet variants.

## 7. Recommended Next Step
**Decision: STOP Paper Reproduction.**
Further effort to close the ~0.03 SRCC (RGNet) or ~0.06 Accuracy (A-LAMP) gap to academic papers is likely subject to diminishing returns. 

**Proposed Pivot:**
1. **Distillation**: Use the identified "Best" candidates to distill smaller, on-device models (MobileNetV3 or custom lightweights).
2. **Model Transition**: Move toward **TOPIQ** or **MUSIQ** for teacher-level performance. Preliminary scripts suggest these modern transformers outperform the graph-based RGNet/A-LAMP approach.

## 8. Final Decision
- **Archive** the failed GCN layout branches and SGD recipes.
- **Adopt** the Multi-Patch (Mean+Max) and RGNet-v1 (Mean) as the final teacher artifacts for the 20260526 capstone checkpoint.
- **Proceed** to the distillation phase using these verified teachers.

---
*Report generated on 2026-05-26.*
