# Mobile Student IQA 50k Label Scaling and Ranking Plan Report

## 1. Executive Summary
- Final decision: `PHASE20_RECOMMEND_WITHIN_DATASET_RANKING_SMOKE`.
- Phase 20 ran analysis and bounded smoke design only; no training smoke was run.
- Dataset weighting failure is now confirmed not to be just contribution math: W5-v2 hit 30/35/35 effective contribution and still fell to official SRCC `0.516337`.
- Target simulation says KonIQ x2 expansion is plausible as a fallback, but it cannot prove MOS rank recovery and risks amplifying teacher noise.
- Recommended Phase 21: one B0 smoke with Huber + within-dataset pairwise ranking loss (R2), margin `0.05`, lambda `0.10` first.

## 2. Sources Inspected
| source | status |
| --- | --- |
| outputs/mobile_student_iqa_50k_controlled_b0_20260608/report.md | present |
| outputs/mobile_student_iqa_50k_controlled_b0_20260608/official_benchmark_predictions.csv | present |
| outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/report.md | present |
| outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/official_paired_predictions.csv | present |
| outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/target_distribution_shift_summary.csv | present |
| outputs/mobile_student_iqa_50k_label_bias_weighting_analysis_20260608/datasetwise_degradation_summary.csv | present |
| outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5lite_20260609/report.md | present |
| outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/report.md | present |
| outputs/mobile_student_iqa_50k_weighted_b0_smoke_w5v2_ifreq_20260609/dataset_weight_effective_contribution.csv | present |
| outputs/official_protocol_mdiqa_signal_benchmark_20260604 | present |
| outputs/mobile_student_iqa_14105_checkpoint_eval_20260604 | present |
| outputs/mdiqa_distillation_50k_full_targets_20260608/final_50k_mdiqa_distillation_manifest.csv | present |
| outputs/mobile_student_iqa_14105_checkpoint_eval_20260604/official_benchmark_predictions.csv | missing; pred_14k used from official_paired_predictions.csv |

## 3. Why Dataset Weighting Failed
- Phase 18 W5-lite used raw per-sample weights and remained FLIVE-dominated, official SRCC `0.530639`.
- Phase 19 W5-v2 corrected effective contribution to FLIVE `0.300353`, KonIQ `0.349743`, SPAQ `0.349904`.
- Despite corrected contribution, official SRCC dropped to `0.516337`, below W5-lite and far below Phase 16 unweighted B0 `0.703290`.
- This makes weighted controlled training a bad next step.

## 4. Label Scaling Policy Simulation
| policy_id | koniq_std_ratio_vs_official | max_abs_mean_delta_vs_official | risk_assessment | recommendation |
| --- | --- | --- | --- | --- |
| S2_koniq_x2 | 0.631926 | 8.045130 | FLIVE remains strongly low-biased | candidate |
| S4_koniq_x2_flive_p4 | 0.631926 | 4.045130 | low distribution risk; still does not prove MOS rank alignment | candidate |
| S4_koniq_x2_flive_p6 | 0.631926 | 2.410609 | low distribution risk; still does not prove MOS rank alignment | diagnostic |
| S1_dataset_std_cap2 | 0.631926 | 8.045130 | FLIVE remains strongly low-biased | diagnostic |
| S2_koniq_x2.5 | 0.789907 | 8.045130 | FLIVE remains strongly low-biased | candidate |
| S2_koniq_x1.5 | 0.473944 | 8.045130 | FLIVE remains strongly low-biased | diagnostic |
| S4_koniq_x1.5_flive_p4 | 0.473944 | 4.045130 | low distribution risk; still does not prove MOS rank alignment | diagnostic |
| S4_koniq_x1.5_flive_p6 | 0.473944 | 2.410609 | low distribution risk; still does not prove MOS rank alignment | diagnostic |
| S3_flive_shift_p8 | 0.315963 | 2.410609 | KonIQ remains compressed | diagnostic |
| S3_flive_shift_p4 | 0.315963 | 4.045130 | KonIQ remains compressed | diagnostic |
| S3_flive_shift_p6 | 0.315963 | 2.410609 | KonIQ remains compressed | diagnostic |
| S0_original | 0.315963 | 8.045130 | FLIVE remains strongly low-biased; KonIQ remains compressed | diagnostic |
| S5_rank_quantile_stretch | 0.591147 | 12.402188 | FLIVE remains strongly low-biased | diagnostic |

- Best fallback label-only smoke policy: `S2_koniq_x2`, because it doubles KonIQ variance without touching FLIVE/SPAQ or adding cross-dataset assumptions.
- `S2_koniq_x2.5` and S4 combined policies widen KonIQ more but raise artificial-label/noise risk.
- `S3` FLIVE shifts reduce mean bias but do not address KonIQ ranking compression.
- `S5` rank quantile stretch preserves rank but changes distribution mechanically; useful diagnostic, not first training smoke.

## 5. Official Label Agreement Analysis
| dataset | comparison | n | srcc | plcc | mae | bias_pred_minus_ref | std_ratio_pred_ref |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flive | b0_50k_vs_mos | 1827 | 0.507083 | 0.590562 | 10.555370 | -9.955895 | 0.877643 |
| flive | mdiqa_teacher_vs_mos | 1827 | 0.569242 | 0.663699 | 8.332064 | -7.223510 | 1.299752 |
| flive | student_14k_vs_mdiqa_teacher | 1827 | 0.581799 | 0.646851 | 6.343870 | -1.102013 | 0.391898 |
| flive | student_14k_vs_mos | 1827 | 0.432862 | 0.529382 | 9.347689 | -8.325523 | 0.509370 |
| flive | unweighted_b0_vs_mdiqa_teacher | 1827 | 0.772426 | 0.804811 | 5.376652 | -2.732385 | 0.675239 |
| koniq10k | b0_50k_vs_mos | 2015 | 0.739097 | 0.737586 | 8.759144 | 1.763683 | 0.462039 |
| koniq10k | mdiqa_teacher_vs_mos | 2015 | 0.931115 | 0.944773 | 8.742361 | 1.127204 | 0.314625 |
| koniq10k | student_14k_vs_mdiqa_teacher | 2015 | 0.851643 | 0.854151 | 2.217181 | 1.573250 | 0.918796 |
| koniq10k | student_14k_vs_mos | 2015 | 0.793329 | 0.802688 | 9.542131 | 2.700454 | 0.289076 |
| koniq10k | unweighted_b0_vs_mdiqa_teacher | 2015 | 0.789415 | 0.787157 | 3.366182 | 0.636479 | 1.468540 |
| overall | b0_50k_vs_mos | 4967 | 0.703290 | 0.778297 | 9.623013 | -3.675822 | 0.705524 |
| overall | mdiqa_teacher_vs_mos | 4967 | 0.822749 | 0.822184 | 8.408399 | -3.177986 | 0.783338 |
| overall | student_14k_vs_mdiqa_teacher | 4967 | 0.803788 | 0.897159 | 4.330368 | 0.977489 | 0.816789 |
| overall | student_14k_vs_mos | 4967 | 0.743636 | 0.778156 | 9.393720 | -2.200496 | 0.639823 |
| overall | unweighted_b0_vs_mdiqa_teacher | 4967 | 0.809681 | 0.902799 | 4.462038 | -0.497836 | 0.900663 |
| spaq | b0_50k_vs_mos | 1125 | 0.854897 | 0.850860 | 9.656148 | -3.219741 | 0.800257 |
| spaq | mdiqa_teacher_vs_mos | 1125 | 0.913615 | 0.918710 | 7.934205 | -4.319128 | 0.854871 |
| spaq | student_14k_vs_mdiqa_teacher | 1125 | 0.949561 | 0.945111 | 4.845392 | 3.287527 | 0.943418 |
| spaq | student_14k_vs_mos | 1125 | 0.866105 | 0.856504 | 9.202652 | -1.031601 | 0.806501 |
| spaq | unweighted_b0_vs_mdiqa_teacher | 1125 | 0.931341 | 0.932320 | 4.939504 | 1.099387 | 0.936115 |

- Key read: if official paired teacher-MOS rank is available but student rank lags, use ranking loss before more dataset-weight sweeps.
- If teacher-MOS itself is weak in a dataset, target scaling cannot invent reliable rank signal.

## 6. Ranking Loss Design
| design_id | complexity_risk | expected_memory_cost | needs_custom_train_step | works_with_current_keras_fit | recommendation |
| --- | --- | --- | --- | --- | --- |
| R0_huber_only | low | O(batch) | False | True | baseline only, not next smoke |
| R1_huber_pairwise_batch | medium | O(batch^2), batch 16 = 256 pair matrix | True | not as plain loss unless y_true carries dataset ids; custom Model.train_step cleaner | do not test first |
| R2_within_dataset_pairwise | medium | O(batch^2), with dataset mask; batch 16 safe | True | requires dataset id in y_true/sample dict or custom dataset output | recommended Phase 21 smoke |
| R3_official_like_hard_pairs | medium-high | O(batch^2), sparse useful pairs | True | requires custom train_step and likely dataset-balanced sampler | second-stage after R2 or label scaling |

- R2 is safest first ranking smoke: within-dataset pairs avoid cross-dataset scale mismatch.
- R1 all-pairs can amplify FLIVE/KonIQ scale conflict.
- R3 hard-pair mining likely needs label scaling first because compressed KonIQ targets produce too few high/low pairs.

## 7. Recommended Phase 21 Experiment
- Run one B0 smoke only: Huber + within-dataset pairwise logistic ranking loss.
- Config: 4000/1000/1000, EfficientNetV2B0, image 256, batch 16, ImageNet, head 1, finetune 2.
- First rank settings: normalized margin `0.05`, `lambda_rank=0.10`, within-dataset pair mask only.
- Keep sample weights neutral at first unless code support is already clean; contribution weighting is not the bottleneck based on Phase 19.
- Official Track B image-only eval remains unchanged.

## 8. Optional Smoke Result
- Optional smoke was not run. Reason: ranking smoke requires a custom `train_step` / dataset-id plumbing, and Phase 20 goal is diagnosis and smoke planning.

## 9. Final Decision
PHASE20_RECOMMEND_WITHIN_DATASET_RANKING_SMOKE

## 10. Next Actions
| priority | action | status |
| --- | --- | --- |
| 1 | Implement one bounded R2 within-dataset pairwise ranking B0 smoke; no B1/B2 and no full controlled training. | recommended |
| 2 | Keep S2_koniq_x2 as fallback target-scaling smoke if ranking implementation is blocked. | fallback |
| 3 | Do not run weighted controlled rerun from W5-lite or W5-v2 results. | blocked |
