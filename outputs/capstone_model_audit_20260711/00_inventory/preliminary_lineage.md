# Preliminary lineage graphs

## Legend

- `PROVEN` means a direct code/config/metadata reference or a same-run output bundle proves the transition.
- `PARTIAL` means the endpoints and likely implementation are present, but the exact invocation or immutable provenance record is absent.
- `UNKNOWN_LINK` is used whenever the repository cannot prove the transition.
- These graphs are preliminary inventory aids. They do not decide paper fidelity.

## Technical IQA

### KonIQ Mobile

```text
data/raw/koniq10k/koniq10k_scores_and_distributions.csv
  → [PROVEN builder code] src/preprocess/make_koniq_csv.py
  → data/processed/koniq10k/labels_all.csv
  → [PROVEN seed-42 split code] data/processed/koniq10k/{train,val,test}.csv
  → [PARTIAL; exact CLI absent] src/train/train_regression.py --task technical
  → UNKNOWN_LINK exact run configuration
  → checkpoints/technical_koniq_gpu/{best.weights.h5,final_model.keras,saved_model/}
  → [PROVEN preset/metadata] src/export/{tflite_presets.py,export_tflite.py}
  → exports/tflite/koniq_mobile.tflite
  → UNKNOWN_LINK byte-identical external Flutter asset
  → UNKNOWN_LINK external mobile interpreter
  → mobile KonIQ score
```

Independent in-repository server path.

```text
checkpoints/technical_koniq_gpu/final_model.keras
  → [PROVEN] configs/stage5_reference.json
  → [PROVEN] src/infer/run_acut_pipeline.py
  → [PROVEN] src/infer/predict_quality_bundle.py
  → koniq_score
  → [PROVEN] src/infer/select_best_shots.py
  → runtime technical component
```

### FLIVE Image Mobile

```text
data/raw/paq2piq/labels_image.csv
  → [PROVEN] src/preprocess/make_flive_csv.py
  → data/processed/paq2piq/labels_image_all.csv
  → [PROVEN custom seed-42 split] image_{train,val,test}.csv
  → [PARTIAL; exact CLI absent] src/train/train_regression.py --task technical
  → UNKNOWN_LINK exact run configuration
  → checkpoints/technical_flive_image_gpu/{best.weights.h5,final_model.keras,saved_model/}
  → [PROVEN preset/metadata] src/export/{tflite_presets.py,export_tflite.py}
  → exports/tflite/flive_image_mobile.tflite
  → UNKNOWN_LINK byte-identical external Flutter asset
  → UNKNOWN_LINK external mobile interpreter
  → mobile FLIVE image score
```

Server Keras path is directly proven by `stage5_reference.json` and the inference bundle.

### FLIVE Patch

```text
data/raw/paq2piq/labels_patch.csv
  → src/preprocess/make_flive_csv.py
  → data/processed/paq2piq/patch_{train,val,test}.csv
  → [PARTIAL] src/train/train_regression.py
  → UNKNOWN_LINK exact invocation
  → checkpoints/technical_flive_patch_gpu/final_model.keras
  → [PROVEN server reference] configs/stage5_reference.json
  → [PROVEN five-crop path] predict_quality_bundle.py
  → flive_patch_mean / flive_patch_min / flive_patch_std
  → select_best_shots.py
  → server technical score

checkpoint
  → UNKNOWN_LINK export
  → UNKNOWN_LINK TFLite
  → UNKNOWN_LINK mobile asset
  → UNKNOWN_LINK mobile runtime
```

### Default technical ensemble

```text
KonIQ Keras ───────────────┐
FLIVE image Keras ─────────┤
FLIVE patch Keras ─────────┤
AADB MUSIQ Keras ──────────┤
                           ↓
configs/stage5_reference.json
  → run_acut_pipeline.py
  → predict_quality_bundle.py
  → select_best_shots.py
  → available-signal renormalization
  → server technical component
```

Image-only benchmark lineage.

```text
cached KonIQ and FLIVE image predictions
  → official_test_compatible_manifest.csv
  → benchmark_technical_iqa_signals_20260604.py
  → (0.32*K + 0.24*F) / 0.56
  → official_signal_predictions.csv
  → official_signal_metrics_overall.csv
  → official_signal_recommendation.json
```

This does not prove the external Flutter implementation.

### TOPIQ-lite mixed112 and C6

```text
SPAQ + KonIQ + FLIVE processed manifests
  → scripts/create_mixed_topiq_datasets.py
  → data/processed/topiq_replacement/mixed_112_{train,val}.csv
  → src/train/train_topiq_lite.py
  → UNKNOWN_LINK historical --init_weights behavior absent from current parser
  → outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517/{best.weights.h5,final_model.keras}
  → outputs/final_topiq_lite_mixed112_export_20260517/export_script.py
  → topiq_lite_mixed112_frozen_{fp32,fp16}.tflite
  → parity CSVs/report
  → UNKNOWN_LINK external shadow asset
  → UNKNOWN_LINK shadow runtime score
```

C6 evidence path.

```text
cached TOPIQ + KonIQ scores
  → output-scoped C6 scripts
  → min(0.7*TOPIQ + 0.3*KonIQ, KonIQ + 8)
  → WSL/Android parity reports
  → final_c6_topiq_koniq_technical_score_decision_20260527/report.md
  → REJECTED as production replacement
```

### Local MUSIQ technical probes

```text
KonIQ/SPAQ/FLIVE processed data
  → UNKNOWN_LINK subset generator
  → data/processed/smoke/*_1024_{train,val}.csv
  → UNKNOWN_LINK exact train_musiq.py invocation
  → outputs/probe1024_musiq_*/final_model.keras
  → UNKNOWN_LINK export implementation
  → exports/tflite/musiq_*_probe1024/*_fp16.tflite
  → benchmark_technical_iqa_signals_20260604.py
  → technical_iqa_signal_comparison_20260604 outputs
  → EXPERIMENTAL
```

### MDIQA teacher

```text
official MDIQA dataset metadata and split material
  → MDIQA/MDIQA/train_MDIQA.py + train_shared.py
  → official YAML configuration
  → UNKNOWN_LINK local training; checkpoints were downloaded
  → MDIQA/checkpoints/*.pth
  → scripts/technical_iqa/run_mdiqa_official_variant_sanity_benchmark.py
  → variant_sanity_manifest.csv
  → predictions.csv / dataset_metrics.csv / report.md
  → teacher/reference score
  → UNKNOWN_LINK export
  → UNKNOWN_LINK mobile asset/runtime
```

### MDIQA mobile students

14,105-row branch.

```text
known-dataset cached teacher predictions
  → prepare_student_distillation_manifest_14105.py
  → student_distillation_manifest.csv
  → train_mobile_student_iqa_14105_longer.py
  → mobile_student_iqa_14105_full_controlled_20260604/*.keras
  → export_student_iqa_14105_tflite.py
  → student_iqa_14105_{fp32,fp16}.tflite
  → parity predictions/metrics
  → UNKNOWN_LINK mobile runtime
  → mdiqa_student_final_no_go_report_20260611
  → REJECTED
```

50k branch.

```text
MDIQA cached/new teacher predictions
  → final_50k_mdiqa_distillation_manifest.csv
  → grouped/stratified split inside train_mobile_student_iqa_50k_smoke.py
  → train/val/test manifests in controlled output
  → train_mobile_student_iqa_50k_controlled.py
  → best/final Keras + history + predictions
  → weighting/ranking/label-scaling variants
  → final no-go report
  → REJECTED
  → UNKNOWN_LINK TFLite because export was intentionally stopped
```

### TechIQA-Guard v1

```text
KonIQ/SPAQ/FLIVE + mined hard-FP material
  → src/data/build_techiqa_guard_dataset.py
  → data/processed/techiqa_guard/*
  → src/train/train_techiqa_guard.py
  → stage checkpoints/logs/summaries
  → scripts/eval_techiqa_guard_v1_stage1.py
  → evaluation predictions/reports
  → final_techiqa_guard_v1_nogo_20260522/report.md
  → REJECTED
  → UNKNOWN_LINK export/mobile runtime
```

### DistortionGuard-IQA

```text
source images + synthetic distortion generator
  → synthetic manifests and pair manifests
  → train_distortionguard_stageA_synthetic.py
  → Stage A checkpoint
  → train_distortionguard_stageB_authentic.py
  → Stage B → B2 → B3 checkpoints/logs
  → eval_distortionguard_stageB_authentic.py
  → held-out and hard-FP predictions/reports
  → B3 diagnosis
  → REJECTED for export/Flutter
  → UNKNOWN_LINK TFLite/mobile runtime
```

## Aesthetic IAA

### NIMA fixed preprocessing

```text
AVA raw annotation expected by make_ava_csv.py [ABSENT]
  → UNKNOWN_LINK
  → data/processed/ava/{train,val,test}.csv
  → ava_distribution_dataset.py
  → train_nima.py
  → checkpoints/nima_ava_gpu_fixed_preproc_20260525/{best.weights.h5,final_model.keras}
  → export_tflite.py + fixed metadata
  → nima_mobile_fixed_preproc_{fp32,fp16}.tflite
  → parity/metrics/diagnostics
  → forWeights2/models/aesthetic/nima_mobile_fixed_preproc_fp16.tflite
  → UNKNOWN_LINK active Flutter asset/runtime
```

Legacy path still referenced by the default server.

```text
checkpoints/nima_ava_gpu/final_model.keras
  → configs/stage5_reference.json
  → default server runtime
  → LEGACY NIMA score
```

### Practical RGNet AADB

```text
data/raw/aadb/AADBinfo.mat
  → src/preprocess/make_aadb_csv.py
  → data/processed/aadb/{train,val,test}.csv
  → src/datasets/csv_dataset.py
  → [PARTIAL] src/train/train_rgnet.py
  → UNKNOWN_LINK exact invocation
  → checkpoints/rgnet_aadb_gpu/final_model.keras
  → configs/stage5_reference.json
  → predict_quality_bundle.py
  → rgnet_score
  → server aesthetic component

checkpoint
  → tflite preset/export
  → exports/tflite/rgnet_aadb_gpu.tflite
  → UNKNOWN_LINK external mobile runtime
```

### RGNet paper-v1 PIL candidate

```text
AADBinfo.mat
  → make_aadb_csv.py
  → AADB manifests
  → rgnet_paper_v1 config + trainer with PIL backend
  → rgnet_v1_aadb_pil_resize_retrain_20260524/{best.weights.h5,final_model.keras}
  → evaluate_rgnet_paper_v1_aadb.py
  → raw test/validation predictions
  → export_rgnet_paper_aadb_tflite.py
  → rgnet_pil_resize_aadb_fp16.tflite
  → parity report/mobile-like predictions
  → forWeights2 candidate
  → UNKNOWN_LINK active Flutter runtime
```

### Practical A-LAMP AADB

```text
AADB manifests
  → native_size_dataset.py
  → [PARTIAL] train_alamp.py
  → UNKNOWN_LINK exact invocation
  → checkpoints/alamp_aadb_gpu/final_model.keras
  → configs/stage5_reference.json
  → predict_quality_bundle.py
  → alamp_score
  → server aesthetic component

checkpoint
  → export_alamp_tflite.py
  → exports/tflite/alamp_aadb_gpu.tflite
  → UNKNOWN_LINK external mobile runtime
```

### Mobile A-LAMP v2

```text
AVA manifests
  → generate_u2net_saliency_maps.py
  → UNKNOWN_LINK exact full generation command
  → V4 patch selector
  → alamp_v4_full_ava_20260517/subsets/*_patch_boxes_full_v4.jsonl
  → train_mobile_alamp_v2.py
  → mobile_alamp_v2_full_ava_20260519/{best_val_auc_model.keras,final_model.keras}
  → export_mobile_alamp_v2_tflite.py
  → FP32/FP16 TFLite + parity
  → forWeights2/models/aesthetic/mobile_alamp_v2_fp16.tflite
  → UNKNOWN_LINK active Flutter runtime score
```

### ICAA / ICAA17K

```text
ICAA17K raw files
  → official 1train.csv / 1test.csv
  → external official dataset.py
  → UNKNOWN_LINK official training invocation/revision
  → weights/icaa_official/e_30_...pth
  → evaluate_icaa_official_weight.py
  → TF-native state mapping and parity stages
  → stage_i_export_savedmodel.py
  → icaa_tf_native_savedmodel_.../saved_model
  → stage_k_convert_tflite_fp16.py
  → icaa_dat_tf_native_fp16.tflite
  → forWeights2 candidate copy
  → UNKNOWN_LINK because runtime normalization and output index conflict
  → UNKNOWN_LINK active Flutter score
```

PTQ/QAT branches terminate at explicit rejection reports and do not join this lineage.

### Stage 5 AADB aesthetic student

```text
AADBinfo.mat
  → make_aadb_csv.py
  → AADB train/val manifests
  → stage5_reference teacher bundle
  → generate_teacher_labels.py
  → outputs/distill_full_aadb/teacher_labels_trainval.csv
  → configs/student_distill_full_aadb.json
  → train_student.py
  → checkpoints/student_full_aadb/{best.weights.h5,final_model.keras,saved_model/}
  → export_student_tflite.py
  → stage5_student_full_aadb_aesthetic_mobile_fp32.tflite
  → handoff document
  → UNKNOWN_LINK Flutter asset/runtime
  → later REJECTED classification
```

### Default Stage5 aesthetic ensemble

```text
AADB composition Keras ─────┐
legacy NIMA Keras ──────────┤
practical A-LAMP Keras ─────┤
practical RGNet Keras ──────┤
pairwise AADB Keras ────────┤
                            ↓
configs/stage5_reference.json
  → stage5_reference.py
  → run_acut_pipeline.py
  → predict_quality_bundle.py
  → select_best_shots.py
  → server aesthetic component
```

### Calibrated three-model candidate

```text
AVA/AADB manifests
  → reused NIMA/RGNet/A-LAMP/ICAA cross-predictions
  → aligned_ava_complete_primary.csv + aligned_aadb_complete_primary.csv
  → run_weight_search.py
  → final_recommendation.json
  → aesthetic_ensemble_config.json
  → UNKNOWN_LINK production source reads config
  → UNKNOWN_LINK active mobile/server runtime weights
```

## Adjacent scorers

### AADB composition and pairwise comparator

```text
AADB manifests
  → train_regression.py --task aadb
  → checkpoints/composition_aadb_gpu/final_model.keras
  → stage5_reference.json
  → aadb_score
  → server aesthetic component
```

```text
AADB train/validation pairs
  → train_pairwise.py
  → checkpoints/pairwise_aadb_gpu/final_model.keras
  → stage5_reference.json
  → optional pairwise rerank
  → runtime score only when explicitly enabled
```

Exact original invocations remain `UNKNOWN_LINK`.

### VILA/Gemma

```text
multimodal model checkpoints or external teacher API
  → VILA/Gemma inference/teacher code
  → cached scores or explanations
  → optional rerank/explanation path
  → UNKNOWN_LINK core mobile IQA/IAA asset
```

