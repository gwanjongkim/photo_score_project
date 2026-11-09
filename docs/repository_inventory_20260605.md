# Repository Inventory (2026-06-05)

## 1. Top-Level Directories

*   **`src/`**: Main source code containing model architectures (`models/`), training loops (`train/`), evaluation logic (`eval/`), data processing (`data/`), and distillation code (`distill/`, `gemma_distill/`).
*   **`configs/`**: Configuration files (YAML/JSON) for various experiments, teacher models, student distillation, and aesthetic weight lab setups.
*   **`scripts/`**: Executable Python scripts for running specific pipelines, benchmarks, sanity checks, and TFLite parity checks (especially within `technical_iqa/`).
*   **`tools/`**: Utility scripts for visualization, patch extraction, threshold calibration, and exporting models to TFLite/ONNX.
*   **`docs/`**: Project documentation, handoff notes, audit reports, and architecture plans.
*   **`data/`**: (Git-ignored) Contains raw datasets, processed CSVs, and technical distillation outputs. Size: ~33GB.
*   **`experiments/`**: Contains sub-projects or specific exploratory code, such as `icaa_tf_native/` (reimplementing PyTorch ICAA in TensorFlow) and various safe export experiments.
*   **`outputs/`**: (Partially Git-ignored) Extensive collection of experiment logs, generated reports, output CSVs, and large checkpoints/merged models. Size: ~219GB.
*   **`models/` & `checkpoints/`**: (Git-ignored) Large downloaded or trained model weights (e.g., VILA, NVILA, Gemma, TFLite exports). Size: ~24GB+.
*   **`exports/tflite/`**: TFLite models exported for mobile deployment, explicitly allowed in `.gitignore` for Flutter integration.
*   **`MDIQA/`**: External or sub-module repository containing MDIQA baseline code and checkpoints.
*   **`android_gemma_latency_harness/`**: Android application for testing Gemma latency.
*   **`forWeights/` & `forWeights2/`**: Directories containing handoff files, checklists, and metadata for model weight deliverables.
*   **Virtual Environments**: `.venv`, `.venv_gpu`, `.venv_vila`, `.venv_tf`, `.venv_mdiqa`, `.venv_test` (All Git-ignored).

## 2. Key Components

### Training Scripts
*   `src/train/train_alamp_multipatch_teacher.py`
*   `src/train/train_alamp_paper_ava_v2_graph_lite.py`
*   `src/train/train_mobile_alamp_v2.py`
*   `src/train/train_topiq_rankcalib_stage1.py`
*   `scripts/technical_iqa/train_mobile_student_iqa_*.py`
*   `src/gemma_distill/train_gemma.py`

### Evaluation / Benchmark Scripts
*   `scripts/evaluate_acut_product_gold_weights.py`
*   `scripts/eval_old_vs_new_alamp_rgnet_paper_benchmark.py`
*   `scripts/technical_iqa/benchmark_technical_iqa_signals_20260604.py`
*   `scripts/technical_iqa/run_mdiqa_correct_protocol_full_benchmark.py`

### TFLite Export / Conversion Scripts
*   `tools/export_mobile_alamp_v2_tflite.py`
*   `scripts/export_rgnet_paper_aadb_tflite.py`
*   `scripts/technical_iqa/export_student_iqa_14105_tflite.py`

### Important Outputs (Capstone Evidence)
*   `outputs/acut_capstone_final_report_20260526/`
*   `outputs/aesthetic_mobile_paper_benchmark_20260603/`
*   `outputs/mobile_student_iqa_14105_full_controlled_20260604/`
*   `outputs/official_protocol_technical_signal_benchmark_20260604/`
*   `outputs/technical_iqa_signal_comparison_20260604/`
*   Various Markdown reports generated in `outputs/` and `experiments/icaa_tf_native/reports/`.

## 3. Large Files & Ignored Artifacts
*   **Total Size:** ~362GB.
*   **HuggingFace Caches:** Found inside `outputs/` and `checkpoints/`.
*   **Massive Safetensors/Binaries:** Gemma 4 E4B, VILA 1.5, NVILA 8B weights.
*   **Libraries:** CUDA, PyTorch, TensorFlow binaries inside `.venv*`.
*   All of these are correctly targeted by the current `.gitignore` rules.