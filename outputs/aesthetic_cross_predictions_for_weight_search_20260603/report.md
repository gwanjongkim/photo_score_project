# Aesthetic Cross Predictions For Weight Search 20260603

## Executive Summary
Generated missing same-row primary cross-predictions where requested and created complete alignment tables for the next weight-search step. This report does not tune weights and does not change production scoring.

## Model Artifacts
|kind|dataset|model|requested_asset_exists|requested_asset_path|actual_path|actual_exists|note|
|---|---|---|---|---|---|---|---|
|model||nima|False|assets/models/nima_mobile_fixed_preproc_fp16.tflite|forWeights2/models/aesthetic/nima_mobile_fixed_preproc_fp16.tflite|True|using equivalent forWeights2 artifact because requested assets path is absent|
|model||rgnet|False|assets/models/rgnet_pil_resize_aadb_fp16.tflite|forWeights2/models/aesthetic/rgnet_pil_resize_aadb_fp16.tflite|True|using equivalent forWeights2 artifact because requested assets path is absent|
|model||alamp_adaptive|False|assets/models/mobile_alamp_v2_fp16.tflite|forWeights2/models/aesthetic/mobile_alamp_v2_fp16.tflite|True|using equivalent forWeights2 artifact because requested assets path is absent|
|model||icaa|False|assets/models/icaa_dat_tf_native_fp16.tflite|forWeights2/models/aesthetic/icaa_dat_tf_native_fp16.tflite|True|using equivalent forWeights2 artifact because requested assets path is absent|

## Generated Missing Predictions
|kind|dataset|model|requested_asset_exists|requested_asset_path|actual_path|actual_exists|note|
|---|---|---|---|---|---|---|---|
|prediction_generated|ava|rgnet|||outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_rgnet_predictions.csv|True|{"inference_ms_max": 72.27743000839837, "inference_ms_mean": 51.69449463975058, "inference_ms_min": 44.383831991581246, "pred_raw_max": 0.956913948059082, "pred_raw_mean": 0.5795344506755267, "pred_raw_min": 0.0369777828454971, "pred_scaled_1_10_max": 9.56913948059082, "pred_scaled_1_10_mean": 5.795344506755267, "pred_scaled_1_10_min": 0.3697778284549713, "preprocess_ms_max": 24.423864000709727, "preprocess_ms_mean": 4.2389857769451, "preprocess_ms_min": 0.9653400047682226, "rows": 25550}|
|prediction_generated|ava|icaa|||outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_icaa_predictions.csv|True|{"inference_ms_max": 258.33681599760894, "inference_ms_mean": 157.88647939541653, "inference_ms_min": 145.78528200217988, "pred_raw_max": 0.7753280401229858, "pred_raw_mean": 0.6009470895101879, "pred_raw_min": 0.3612465262413025, "pred_scaled_1_10_max": 7.753280401229858, "pred_scaled_1_10_mean": 6.009470895101879, "pred_scaled_1_10_min": 3.612465262413025, "preprocess_ms_max": 17.834191006841138, "preprocess_ms_mean": 3.8639942005782664, "preprocess_ms_min": 1.1023139959434047, "rows": 25550}|
|prediction_generated|aadb|nima|||outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_nima_predictions.csv|True|{"inference_ms_max": 32.982576012727804, "inference_ms_mean": 5.207209413361852, "inference_ms_min": 4.577118990710005, "pred_raw_max": 5.925373547710478, "pred_raw_mean": 4.2562082847479035, "pred_raw_min": 3.1265920132864267, "pred_scaled_1_10_max": 5.925373547710478, "pred_scaled_1_10_mean": 4.2562082847479035, "pred_scaled_1_10_min": 3.1265920132864267, "preprocess_ms_max": 20.688520002295263, "preprocess_ms_mean": 6.662289715983206, "preprocess_ms_min": 3.334672990604304, "rows": 1000}|
|prediction_generated|aadb|alamp_adaptive|||outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_alamp_adaptive_predictions.csv|True|{"inference_ms_max": 15.902678002021275, "inference_ms_mean": 5.174960604024818, "inference_ms_min": 4.4782199984183535, "pred_raw_max": 0.9937121868133544, "pred_raw_mean": 0.3318626619912684, "pred_raw_min": 0.003074346575886, "pred_scaled_1_10_max": 9.937121868133543, "pred_scaled_1_10_mean": 3.318626619912684, "pred_scaled_1_10_min": 0.0307434657588601, "preprocess_ms_max": 420.76941099367104, "preprocess_ms_mean": 201.07888028610614, "preprocess_ms_min": 84.91566999873612, "rows": 1000}|
|prediction_generated|aadb|icaa|||outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_icaa_predictions.csv|True|{"inference_ms_max": 321.22129399795085, "inference_ms_mean": 162.62282484782918, "inference_ms_min": 148.42438799678348, "pred_raw_max": 0.7519247531890869, "pred_raw_mean": 0.5884567168354988, "pred_raw_min": 0.3918131291866302, "pred_scaled_1_10_max": 7.519247531890869, "pred_scaled_1_10_mean": 5.884567168354988, "pred_scaled_1_10_min": 3.9181312918663034, "preprocess_ms_max": 12.385135007207282, "preprocess_ms_mean": 6.576211057807086, "preprocess_ms_min": 3.318839997518808, "rows": 1000}|

## Reused Existing Predictions
|kind|dataset|model|requested_asset_exists|requested_asset_path|actual_path|actual_exists|note|
|---|---|---|---|---|---|---|---|
|prediction_reused|ava|nima|||outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv|True|primary existing benchmark prediction|
|prediction_reused|ava|alamp_adaptive|||outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_predictions.csv|True|primary existing benchmark prediction|
|prediction_reused|aadb|rgnet|||outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv|True|primary existing benchmark prediction|
|prediction_reused|icaa17k|icaa|||outputs/aesthetic_mobile_paper_benchmark_20260603/full/icaa_icaa17k/icaa_predictions.csv|True|primary existing benchmark prediction|

## Runtime Summary
|dataset|model|prediction_path|successful_prediction_rows|preprocess_ms_mean|preprocess_ms_median|preprocess_ms_max|inference_ms_mean|inference_ms_median|inference_ms_max|estimated_total_seconds_from_row_timings|
|---|---|---|---|---|---|---|---|---|---|---|
|ava|rgnet|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_rgnet_predictions.csv|25550|4.2389857769451|3.64854049985297|24.423864000709727|51.69449463975058|51.1942130033276|72.27743000839837|1429.1004246465743|
|ava|icaa|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_icaa_predictions.csv|25550|3.8639942005782655|3.639482500148006|17.834191006841138|157.88647939541653|157.2355134994723|258.33681599760894|4132.724600377666|
|aadb|nima|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_nima_predictions.csv|1000|6.662289715983206|6.530283004394732|20.688520002295263|5.207209413361852|4.864617498242296|32.982576012727804|11.869499129345058|
|aadb|alamp_adaptive|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_alamp_adaptive_predictions.csv|1000|201.0788802861061|193.50147199293133|420.76941099367104|5.174960604024818|5.035250505898148|15.902678002021275|206.25384089013096|
|aadb|icaa|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_icaa_predictions.csv|1000|6.576211057807086|6.447595507779624|12.385135007207282|162.62282484782918|160.7040499948198|321.22129399795085|169.19903590563626|

## Command Runtime Summary
_None._

## Row Counts And Alignment
|dataset|split_rows|aligned_complete_rows|output_path|nima_rows|rgnet_rows|alamp_adaptive_rows|icaa_rows|nima_missing_vs_split|nima_path|nima_reused_existing|rgnet_missing_vs_split|rgnet_path|rgnet_reused_existing|alamp_adaptive_missing_vs_split|alamp_adaptive_path|alamp_adaptive_reused_existing|icaa_missing_vs_split|icaa_path|icaa_reused_existing|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|ava|25551|25548|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv|25549|25550|25548|25550|2|outputs/aesthetic_mobile_paper_benchmark_20260603/full/nima_ava/nima_predictions.csv|True|1|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_rgnet_predictions.csv|False|3|outputs/aesthetic_rgnet_alamp_paper_protocol_followup_20260603/full_adaptive/alamp_adaptive_predictions.csv|True|1|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_icaa_predictions.csv|False|
|aadb|1000|1000|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_aadb_complete_primary.csv|1000|1000|1000|1000|0|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_nima_predictions.csv|False|0|outputs/aesthetic_mobile_paper_benchmark_20260603/full/rgnet_aadb/rgnet_predictions.csv|True|0|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_alamp_adaptive_predictions.csv|False|0|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_icaa_predictions.csv|False|

## Failure Summary
|dataset|model|failure_rows|failure_path|first_failure|
|---|---|---|---|---|
|ava|rgnet|1|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_rgnet_failures.csv|not a TIFF file (header b'\xff\xd8\xff\xdb\x00C\x00\x07' not valid)|
|ava|icaa|1|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_icaa_failures.csv|not a TIFF file (header b'\xff\xd8\xff\xdb\x00C\x00\x07' not valid)|
|aadb|nima|0|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_nima_failures.csv||
|aadb|alamp_adaptive|0|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_alamp_adaptive_failures.csv||
|aadb|icaa|0|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_icaa_failures.csv||

## Score Range Sanity Checks
|dataset|model|actual_path|note|
|---|---|---|---|
|ava|rgnet|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_rgnet_predictions.csv|{"inference_ms_max": 72.27743000839837, "inference_ms_mean": 51.69449463975058, "inference_ms_min": 44.383831991581246, "pred_raw_max": 0.956913948059082, "pred_raw_mean": 0.5795344506755267, "pred_raw_min": 0.0369777828454971, "pred_scaled_1_10_max": 9.56913948059082, "pred_scaled_1_10_mean": 5.795344506755267, "pred_scaled_1_10_min": 0.3697778284549713, "preprocess_ms_max": 24.423864000709727, "preprocess_ms_mean": 4.2389857769451, "preprocess_ms_min": 0.9653400047682226, "rows": 25550}|
|ava|icaa|outputs/aesthetic_cross_predictions_for_weight_search_20260603/ava_icaa_predictions.csv|{"inference_ms_max": 258.33681599760894, "inference_ms_mean": 157.88647939541653, "inference_ms_min": 145.78528200217988, "pred_raw_max": 0.7753280401229858, "pred_raw_mean": 0.6009470895101879, "pred_raw_min": 0.3612465262413025, "pred_scaled_1_10_max": 7.753280401229858, "pred_scaled_1_10_mean": 6.009470895101879, "pred_scaled_1_10_min": 3.612465262413025, "preprocess_ms_max": 17.834191006841138, "preprocess_ms_mean": 3.8639942005782664, "preprocess_ms_min": 1.1023139959434047, "rows": 25550}|
|aadb|nima|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_nima_predictions.csv|{"inference_ms_max": 32.982576012727804, "inference_ms_mean": 5.207209413361852, "inference_ms_min": 4.577118990710005, "pred_raw_max": 5.925373547710478, "pred_raw_mean": 4.2562082847479035, "pred_raw_min": 3.1265920132864267, "pred_scaled_1_10_max": 5.925373547710478, "pred_scaled_1_10_mean": 4.2562082847479035, "pred_scaled_1_10_min": 3.1265920132864267, "preprocess_ms_max": 20.688520002295263, "preprocess_ms_mean": 6.662289715983206, "preprocess_ms_min": 3.334672990604304, "rows": 1000}|
|aadb|alamp_adaptive|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_alamp_adaptive_predictions.csv|{"inference_ms_max": 15.902678002021275, "inference_ms_mean": 5.174960604024818, "inference_ms_min": 4.4782199984183535, "pred_raw_max": 0.9937121868133544, "pred_raw_mean": 0.3318626619912684, "pred_raw_min": 0.003074346575886, "pred_scaled_1_10_max": 9.937121868133543, "pred_scaled_1_10_mean": 3.318626619912684, "pred_scaled_1_10_min": 0.0307434657588601, "preprocess_ms_max": 420.76941099367104, "preprocess_ms_mean": 201.07888028610614, "preprocess_ms_min": 84.91566999873612, "rows": 1000}|
|aadb|icaa|outputs/aesthetic_cross_predictions_for_weight_search_20260603/aadb_icaa_predictions.csv|{"inference_ms_max": 321.22129399795085, "inference_ms_mean": 162.62282484782918, "inference_ms_min": 148.42438799678348, "pred_raw_max": 0.7519247531890869, "pred_raw_mean": 0.5884567168354988, "pred_raw_min": 0.3918131291866302, "pred_scaled_1_10_max": 7.519247531890869, "pred_scaled_1_10_mean": 5.884567168354988, "pred_scaled_1_10_min": 3.9181312918663034, "preprocess_ms_max": 12.385135007207282, "preprocess_ms_mean": 6.576211057807086, "preprocess_ms_min": 3.318839997518808, "rows": 1000}|

## Readiness
AVA complete alignment ready for weight search: True.

AADB complete alignment ready for weight search: True.

ICAA should still be treated as a separate color subscore until the next step tests its holistic AVA/AADB contribution.

## Next Weight-Search Command Or Prompt
Run the weight-search step using `outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_ava_complete_primary.csv` and `outputs/aesthetic_cross_predictions_for_weight_search_20260603/aligned_aadb_complete_primary.csv`, without using non-primary candidate predictions and while keeping ICAA separately reported.
