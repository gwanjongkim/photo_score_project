# A-cut Capstone Project Evidence Index

This file lists the primary evidence artifacts used for the final project report.

| Output Directory | Primary Model | Key Metrics Found | Confidence | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `outputs/alamp_multipatch_teacher_full_ava_20260524` | A-LAMP Multi-Patch | Acc 0.7633, ROC-AUC 0.7877 | High | Best practical A-LAMP candidate. |
| `outputs/rgnet_paper_consolidated_20260511` | RGNet (Paper Approx) | AADB SRCC 0.6819, AVA Acc 0.7697 | High | Consolidated baseline for paper reproduction track. |
| `outputs/nima_fixed_preproc_eval_20260525` | NIMA (Fixed) | SRCC 0.5894, AUC 0.8018 | High | Fixed double-preprocessing bug in EfficientNetV2. |
| `outputs/icaa_performance_reverification_20260525` | ICAA (Color) | SRCC 0.8791, PLCC 0.8958 | High | Verified against ICAA17K test split. |
| `outputs/student_conservative_aesthetic_full_aadb_20260423` | Stage 5 Student | Pairwise Acc 0.9057 | High | Mobile distillation of RGNet/A-LAMP ensemble. |
| `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520` | TOPIQ-lite | SPAQ SRCC 0.899, KonIQ SRCC 0.863 | High | Technical quality benchmark results. |
| `outputs/alamp_paper_fusion_4096_final_report.md` | A-LAMP Graph Fusion | Val ROC-AUC 0.7067 | High | Documented failure of graph-based fusion on 4k subset. |
| `outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526` | RGNet (DenseASPP) | SRCC 0.6518 | High | Latest ablation showing DenseASPP didn't beat baseline. |

## Important Data Files
- `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`: Normalized summary of all key metrics.
- `data/processed/ava/test.csv`: Primary evaluation dataset for AVA-based models.
- `data/processed/aadb/test.csv`: Primary evaluation dataset for AADB-based models.

## Verification TFLite Metadata
- `outputs/nima_fixed_preproc_eval_20260525/nima_mobile_fixed_preproc_fp16.metadata.json`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/export/student_conservative_aesthetic_full_aadb_20260423.metadata.json`
- `outputs/icaa_tf_native_tflite_fp16_.../export_metadata.json`
