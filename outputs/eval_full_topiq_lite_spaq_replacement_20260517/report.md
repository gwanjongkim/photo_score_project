# TOPIQ-lite Full SPAQ Training and Single Replacement Evaluation

## 1. Summary
**PASS**. The full SPAQ training for TOPIQ-lite was highly successful. The resulting model significantly outperforms the existing `mobile_koniq` and `mobile_flive` baselines on the SPAQ test set, achieving a superior SRCC of **0.907** (vs 0.839). While it is currently the best single model for smartphone-oriented technical quality, it requires further fine-tuning on KonIQ to truly replace the entire existing technical set with maximum cross-dataset robustness.

## 2. Training Result
- **Epochs completed**: 10 (Early stopped)
- **Best epoch**: 5
- **Final train MAE / RMSE**: 0.0609 / 0.0778
- **Final val MAE / RMSE**: 0.0723 / 0.0916
- **Best val loss**: 0.0036
- **Pred min/max/mean/std**: 0.133 / 0.836 / 0.500 / 0.204
- **Target min/max/mean/std**: 0.048 / 0.898 / 0.517 / 0.224
- **SRCC / PLCC**: **0.945 / 0.954** (on validation subset)
- **Mode collapse**: **NO** (Healthy variance)

## 3. SPAQ Test Comparison (N=1125)
| Model | N | MAE | RMSE | SRCC | PLCC | Bias | Pred Std | Std Ratio | Avg ms/img |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `mobile_koniq` | 1125 | 13.03 | 16.17 | 0.839 | 0.837 | -10.34 | 15.10 | 0.69 | 5.81 |
| `mobile_flive` | 1125 | 20.19 | 24.23 | 0.808 | 0.654 | 17.08 | 9.87 | 0.45 | 5.79 |
| **`topiq_full_spaq`** | 1125 | **7.17** | **9.14** | **0.907** | **0.911** | **0.93** | **19.25** | **0.88** | 14.39 |

## 4. Cross-Dataset Replacement Check
| Eval Dataset | Model | N | MAE | RMSE | SRCC | PLCC | Bias | Pred Std | Std Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **spaq_1024_val** | `topiq_full_spaq` | 128 | 5.45 | 6.98 | 0.945 | 0.954 | -1.70 | 20.37 | 0.91 |
| **koniq_1024_val** | `topiq_full_spaq` | 128 | 8.69 | 10.95 | 0.674 | 0.765 | 3.26 | 15.22 | 1.00 |
| **flive_1024_val** | `topiq_full_spaq` | 128 | 14.07 | 17.34 | 0.438 | 0.452 | -13.00 | 12.87 | 2.30 |

## 5. Export Result
- **FP32 size**: 24.9 MB
- **FP16 size**: 12.6 MB
- **Builtin-only?**: **YES** (Confirmed via TFLite conversion)
- **Flex?**: **NO**
- **FP16 parity**: **0.038** (Excellent; Max absolute diff on 0-100 scale).

## 6. Replacement Decision
**C. TOPIQ-lite-SPAQ is promising but needs SPAQ→KonIQ fine-tuning.**
The model is already vastly superior to `mobile_koniq` on smartphone photos (SPAQ test) and performs respectably on KonIQ (SRCC 0.67). To reach the production goal of a single technical model that replaces both `koniq_mobile` and `flive_mobile` without any performance regression on non-smartphone images, we should proceed to a multi-stage training or fine-tuning phase on KonIQ.

## 7. Recommended Next Step
**SPAQ→KonIQ fine-tuning.** 
Use the weights from this full SPAQ model as a starting point and train on the full KonIQ dataset. This "Unified TOPIQ-lite" model is expected to be the final single replacement candidate.

## 8. Files Created
- summary.csv
- predictions_spaq_test_mobile_koniq.csv
- predictions_spaq_test_mobile_flive.csv
- predictions_spaq_test_topiq_full_spaq.csv
- predictions_koniq_1024_topiq_full_spaq.csv
- predictions_flive_1024_topiq_full_spaq.csv
- report.md
- commands.txt
- artifacts.txt

## 9. Final GO / NO-GO
**GO**. The replacement track is numerically and technically sound. Proceed to full KonIQ fine-tuning.
