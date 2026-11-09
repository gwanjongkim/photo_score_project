# A-cut Capstone Project Report Generation Log

## Methods Used
- **Exploration**: Scanned `outputs/` and `experiments/` for report files, CSV summaries, and training logs.
- **Data Extraction**: Manually parsed key metrics from the latest and best-performing model reports.
- **Synthesis**: Built a normalized `model_metrics_summary.csv` using a Python script to ensure data integrity.
- **Visualization**: Generated 5+ PNG figures using `matplotlib` and `pandas` from extracted data.
- **Reporting**: Wrote a comprehensive Korean report based on the gathered evidence.

## Scanned Directories (Selected)
- `outputs/alamp_multipatch_teacher_full_ava_20260524/`
- `outputs/rgnet_paper_consolidated_20260511/`
- `outputs/nima_fixed_preproc_eval_20260525/`
- `outputs/icaa_performance_reverification_20260525/`
- `outputs/student_conservative_aesthetic_full_aadb_20260423/`
- `outputs/eval_final_topiq_candidates_vs_existing_technical_20260520/`

## Files Parsed
- `alamp_paper_fusion_4096_final_report.md`
- `rgnet_paper_oriented_summary.md`
- `nima_fixed_preproc_eval_20260525/report.md`
- `icaa_performance_reverification_20260525/report.md`
- `student_conservative_aesthetic_full_aadb_20260423/handoff.md`
- `eval_final_topiq_candidates_vs_existing_technical_20260520/report.md`
- `rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/report.md`

## Generated Outputs
- `outputs/acut_capstone_final_report_20260526/report.md`
- `outputs/acut_capstone_final_report_20260526/evidence_index.md`
- `outputs/acut_capstone_final_report_20260526/model_metrics_summary.csv`
- `outputs/acut_capstone_final_report_20260526/figures/*.png`

## Uncertainties / Missing Data
- **NIMA training history**: Epoch-level CSV was not found in the checkpoint or output dir; used summary metrics from the evaluation report.
- **ICAA training history**: Focus was on re-verification of official weights; no local training log used for figures.
- **Official Reproduction**: Clearly marked as "not achieved" or "in progress" for A-LAMP and RGNet to avoid overstatement.

## Recommended Next Action
- Integrate the fixed NIMA and verified ICAA models into the Flutter app.
- Use the Stage 5 student as the primary aesthetic scorer for mobile testing.
- Continue the paper reproduction track separately with focus on DenseASPP and RegionGraph implementation.
