# Aesthetic Ensemble Weight Search 20260603

## Data Availability
|dataset|model|available_primary|rows_in_aligned_available_table|path|note|
|---|---|---|---|---|---|
|AVA|nima|True|25548|outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv|primary 20260603 full NIMA AVA|
|AVA|rgnet|False|25548||missing primary prediction CSV|
|AVA|alamp_adaptive|True|25548|outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_predictions.csv|primary 20260603 full adaptive A-LAMP AVA|
|AVA|icaa|False|25548||missing primary prediction CSV|
|AADB|nima|False|1000||missing primary prediction CSV|
|AADB|rgnet|True|1000|outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv|primary 20260603 full RGNet AADB|
|AADB|alamp_adaptive|False|1000||missing primary prediction CSV|
|AADB|icaa|False|1000||missing primary prediction CSV|
|ICAA17K|icaa|True|1617|outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_predictions.csv|primary 20260603 full ICAA17K color prediction|
|AADB|nima|False||outputs/distill_full_aadb/teacher_labels_train.csv|non-primary candidate exists: teacher-label cache, not a benchmark prediction CSV; not used for primary optimization|
|AADB|alamp|False||outputs/distill_full_aadb/teacher_labels_train.csv|non-primary candidate exists: teacher-label cache, non-adaptive A-LAMP score; not used for primary optimization|
|AADB|rgnet|False||outputs/distill_full_aadb/teacher_labels_train.csv|non-primary candidate exists: teacher-label cache, not used because primary RGNet AADB exists|
|AVA|rgnet|False||outputs/paper_level_stage1_benchmark_alignment_20260523/rgnet_pil_preprocess_predictions.csv|non-primary candidate exists: older protocol candidate, not mixed with 20260603 adaptive A-LAMP|
|ICAA17K|icaa|False||outputs/icaa_performance_reverification_20260525/icaa_predictions.csv|non-primary candidate exists: older ICAA reverification candidate; primary 20260603 ICAA17K exists|

## Model Score Scales
|dataset|model|raw_range|scaled_range|calibration|fit_fold|
|---|---|---|---|---|---|
|AVA|nima|2.60778..6.9995|2.60778..6.9995|y=0.820016x+1.41432|val|
|AVA|alamp_adaptive|0.000708302..1|0.00708302..10|y=0.164274x+4.65203|val|
|AADB|rgnet|0.0555421..0.93149|1.49988..9.38341|y=0.835215x+0.892776|val|
|ICAA17K|icaa|0.374126..0.794766|3.74126..7.94766|y=0.970838x+0.189069|val|

## Best Weights Without ICAA
_None._

## Best Weights With ICAA Unconstrained
_None._

## Best Weights With ICAA Capped At 0.20
_None._

## Best Weights With ICAA Capped At 0.25
_None._

## Per-Dataset Test Metrics For Selected Weight Sets
_None._

## Bootstrap Robustness
_None._

## ICAA Holistic-Aesthetic Decision
ICAA should remain a separate color subscore in this run. The primary files do not provide same-row NIMA, RGNet, adaptive A-LAMP, and ICAA predictions on AVA or AADB, so an observed holistic improvement cannot be established without substituting protocols.

## Final Recommendation For Production
No production ratio should be changed from this run because no complete primary same-dataset alignment exists for the requested holistic ensemble. Keep ICAA separate as a color subscore.

Safest holistic weight ratio: unavailable from primary 20260603 same-row evidence.

Optional color-aware ratio: unavailable from primary 20260603 same-row evidence.

Keep ICAA separate: yes.

## Caveats
- Do not tune on official test if no validation split exists.
- NIMA AVA split is local full, not official paper split.
- A-LAMP is paper-inspired adaptive, not full paper protocol.
- This run did not rerun inference and did not use non-primary protocol-mixed CSVs for optimization.
- Group A complete primary alignment missing: True
- Group B complete primary alignment missing: True
