# Phase 4 Hybrid Teacher Label 2k Dry-run and Range Calibration Audit Report

## 1. Executive Summary
- Final decision: `HYBRID_LABEL_2K_NEEDS_RANGE_CALIBRATION`.
- Valid selected Policy A rows: `2000` / `2000`.
- Selected policy: `Policy A strict hybrid`.
- Safety counts: severe FP `0`, FP `0`, FN `0`.
- No new MDIQA inference or production inference was run.

## 2. Sources Inspected
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_20260603/report.md`
- `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_20260603/hybrid_teacher_label_spec.md`
- `/home/omen_pc1/photo_score_project/outputs/paired_production_mdiqa_significance_expanded_20260603/paired_predictions_expanded.csv`
- `/home/omen_pc1/photo_score_project/outputs/paired_production_mdiqa_significance_expanded_20260603/paired_significance_results_expanded.csv`
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_correct_protocol_full_benchmark_20260601/predictions.csv`
- `/home/omen_pc1/photo_score_project/outputs/mdiqa_correct_protocol_full_benchmark_20260601/dataset_metrics.csv`
- `/home/omen_pc1/photo_score_project/outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_flive.csv`
- `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`
- `/home/omen_pc1/photo_score_project/src/infer/select_best_shots.py`

## 3. Confirmed Label Policy
- KonIQ-10k -> MDIQA `seed1`.
- SPAQ -> MDIQA `xd`.
- FLIVE -> Production technical score.
- Fallback/Unknown -> Production technical score safety fallback.

## 4. 2k Dry-run Manifest Construction
- Target allocation: KonIQ 500, SPAQ 500, FLIVE image 700, fallback proxy 300.
- Fallback/unknown real mobile rows were unavailable; fallback uses a documented production-scored FLIVE image proxy group not duplicated with primary FLIVE rows.
- Output manifest: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_dryrun_2k_manifest.csv`.

## 5. Dataset and Source Counts
- fallback_proxy / production_safety_fallback: 300
- flive / production: 700
- koniq10k / mdiqa_seed1: 500
- spaq / mdiqa_xd: 500

## 6. Target Distribution Summary
- fallback_proxy / production_safety_fallback: n=300, mean=73.4244, std=4.2177, p05=65.3779, p50=74.0703, p95=78.0131
- flive / production: n=700, mean=73.3952, std=4.2710, p05=66.4207, p50=74.0757, p95=78.7193
- koniq10k / mdiqa_seed1: n=500, mean=59.8889, std=4.8928, p05=50.2697, p50=60.8417, p95=65.8606
- spaq / mdiqa_xd: n=500, mean=41.6721, std=18.3236, p05=17.7413, p50=39.5416, p95=70.5389

## 7. MOS Alignment
- fallback_proxy: n=300, SRCC=0.5678, PLCC=0.7382, MAE=3.1363, aligned MAE=3.0792
- flive: n=700, SRCC=0.6212, PLCC=0.7551, MAE=3.1214, aligned MAE=2.9831
- koniq10k: n=500, SRCC=0.9487, PLCC=0.9586, MAE=6.4625, aligned MAE=2.3837
- spaq: n=500, SRCC=0.9086, PLCC=0.9169, MAE=7.6043, aligned MAE=6.3458

## 8. Safety Cases
- Severe FP: `0`
- FP: `0`
- FN: `0`
- Detailed cases: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_2k_20260603/hybrid_label_safety_cases_2k.csv`.

## 9. Policy A vs Policy B
- fallback_proxy: Policy A SRCC=0.5678, Policy B SRCC=0.5678, delta=0.0000; MAE delta A-B=0.0000; severe FP A/B=0/0
- flive: Policy A SRCC=0.6212, Policy B SRCC=0.6212, delta=0.0000; MAE delta A-B=0.0000; severe FP A/B=0/0
- koniq10k: Policy A SRCC=0.9487, Policy B SRCC=0.9393, delta=0.0095; MAE delta A-B=0.0236; severe FP A/B=0/0
- spaq: Policy A SRCC=0.9086, Policy B SRCC=0.9084, delta=0.0002; MAE delta A-B=0.3155; severe FP A/B=0/0

## 10. Range Calibration Audit
- mdiqa_vs_production_source_range mdiqa_sources vs production_sources: mean_shift=22.6235, std_ratio=3.8111, gap_mismatch=34.9940, flags=flag_mean_shift_gt10,flag_std_ratio,flag_gap_mismatch_gt20
- dataset_boundary_mean_gap fallback_proxy vs flive: mean_shift=0.0292, std_ratio=0.9875, gap_mismatch=0.3366, flags=none
- dataset_boundary_mean_gap fallback_proxy vs koniq10k: mean_shift=13.5355, std_ratio=0.8620, gap_mismatch=2.9558, flags=flag_mean_shift_gt10
- dataset_boundary_mean_gap fallback_proxy vs spaq: mean_shift=31.7523, std_ratio=0.2302, gap_mismatch=40.1623, flags=flag_mean_shift_gt10,flag_std_ratio,flag_gap_mismatch_gt20
- dataset_boundary_mean_gap flive vs koniq10k: mean_shift=13.5063, std_ratio=0.8729, gap_mismatch=3.2924, flags=flag_mean_shift_gt10
- dataset_boundary_mean_gap flive vs spaq: mean_shift=31.7231, std_ratio=0.2331, gap_mismatch=40.4990, flags=flag_mean_shift_gt10,flag_std_ratio,flag_gap_mismatch_gt20
- dataset_boundary_mean_gap koniq10k vs spaq: mean_shift=18.2168, std_ratio=0.2670, gap_mismatch=37.2065, flags=flag_mean_shift_gt10,flag_std_ratio,flag_gap_mismatch_gt20
- boundary_and_clipping policy_a vs : mean_shift=nan, std_ratio=nan, gap_mismatch=nan, flags=none
- selected_policy_safety selected vs : mean_shift=nan, std_ratio=nan, gap_mismatch=nan, flags=none
- Recommendations: `/home/omen_pc1/photo_score_project/outputs/hybrid_teacher_label_design_review_2k_20260603/range_calibration_recommendations.csv`.

## 11. Dataset-aware Label Risks
- SPAQ remains lower-mean than KonIQ/FLIVE, but this is checked against MOS alignment and safety cases rather than automatically calibrated away.
- Fallback proxy is production-scored FLIVE image data, not real mobile data; it is suitable for policy mechanics but not a representativeness claim.
- FLIVE primary uses production scores and image rows only; patches are excluded.

## 12. Failure Cases
- None.

## 13. Final Decision
HYBRID_LABEL_2K_NEEDS_RANGE_CALIBRATION

## 14. Recommended Next Step
- Run a calibration design pass before larger dry-runs; do not create full labels yet.
