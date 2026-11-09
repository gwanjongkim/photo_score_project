# Capstone Artifact Map (2026-06-05)

This document maps the important experiments, benchmarks, and outputs in this repository to their purpose, providing evidence for the A-cut capstone project.

## 1. Technical Quality Models (IQA)
This section covers the distillation and benchmarking of models assessing technical image quality (noise, blur, exposure).

*   **Student Distillation & Training:**
    *   `outputs/mobile_student_iqa_v*`: Training logs and outputs for various iterations of the mobile student IQA model.
    *   `outputs/mobile_student_iqa_14105_*`: Final controlled training runs for the mobile student.
*   **Benchmarking & Evaluation:**
    *   `outputs/official_protocol_technical_signal_benchmark_20260604/`: Benchmarks comparing official protocol technical signals.
    *   `outputs/technical_iqa_signal_comparison_20260604/`: Signal comparison reports.
    *   `outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524/`: Benchmark results for TOPIQ and KonIQ models.
*   **Parity Checks (WSL/Android):**
    *   `outputs/c6_android_wsl_parity_20260527/` & `outputs/c6_android_wsl_parity_exif_fixed_20260529/`: Critical evidence that the model executes identically on mobile (Android) and backend (WSL).

## 2. Aesthetic Quality Models
This section details experiments related to overall image aesthetics and composition.

*   **Model Training & Distillation:**
    *   `outputs/alamp_*`: Training logs for various A-LAMP model configurations (multipatch teacher, graph lite, etc.).
    *   `outputs/aesthetic_ensemble_weight_search_complete_20260603/`: Results of hyperparameter tuning and weight searching for the aesthetic ensemble.
*   **Benchmarking & Evaluation:**
    *   `outputs/aesthetic_mobile_paper_benchmark_20260603/`: Mobile aesthetic paper benchmark results.
    *   `outputs/acut_product_gold_weight_eval_20260603/`: Evaluation of the chosen "gold weights" for the A-cut product.

## 3. Capstone Reports & Journals
These directories contain the core narrative and weekly progress reports for the capstone.

*   `outputs/acut_capstone_final_report_20260526/`: The drafted final capstone report.
*   `outputs/acut_weekly_research_journal_20260526/`: Weekly research journals detailing the iterative process.
*   `outputs/acut_model_core_code_audit_20260601/`: Audits of the core model training pipelines.

## 4. TFLite & Mobile Deployment
Evidence of successful compression and conversion of models for on-device execution.

*   `exports/tflite/`: The successfully exported mobile `.tflite` files (e.g., `stage5_student_full_aadb_aesthetic_mobile_fp32`).
*   `outputs/icaa_export_safe*`: Documentation and artifacts from the process of making the ICAA model export-safe (converting PyTorch to ONNX to TFLite).
*   `experiments/icaa_tf_native/reports/`: Reports detailing the complex reimplementation of the PyTorch ICAA model into native TensorFlow to bypass conversion bugs.