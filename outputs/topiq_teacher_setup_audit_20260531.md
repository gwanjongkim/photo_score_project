# TOPIQ Teacher Setup Audit

## 1. Existing TOPIQ Artifacts
- **Student Models (TFLite)**: `topiq_lite_mixed112` models (FP16/FP32) exist in `outputs/final_topiq_lite_mixed112_export_20260517/`.
- **Student Models (Keras)**: `final_model.keras` and `best.weights.h5` are in `outputs/mixed_replay_topiq_lite_112_frozen_e15_20260517`.
- **Internal Model Code**: `src/models/topiq_lite.py` and `src/models/topiq_rankcalib.py` are available for student implementation.
- **Previous Predictions**: `topiq_score` and `topiq_ranking_score` columns are present in benchmark CSVs within `outputs/technical_iqa_combo_benchmark_20260524/`.
- **Empty Directory**: `external/topiq_teacher` exists but is currently empty, suggesting it is a planned location for the research repository.

## 2. Python Environment Status
- **.venv_gpu**:
  - `torch` (2.5.1+cu121) and `torchvision` (0.20.1+cu121) are installed.
  - **MISSING**: `pyiqa` and `IQA-PyTorch` libraries.
- **Other venvs**: `.venv`, `.venv_test`, and `.venv_vila` also lack IQA-specific research libraries.

## 3. Dataset/Manifest Availability
- **Standardized Manifests**: `outputs/research_protocol_iqa_manifests_20260530/` contains four key artifacts:
  - `benchmark_all_research_protocol_iqa.csv`: Combined KonIQ-10k, SPAQ, and FLIVE test sets.
  - `benchmark_koniq10k_musiq_80_20_10runs.csv`
  - `benchmark_spaq_official_split.csv`
  - `benchmark_flive_paq2piq_toolbox_split.csv` (159,240 rows total).
- **Raw Data**: `data/raw/` contains the corresponding images for KonIQ, SPAQ, and PaQ-2-PiQ.

## 4. Previous Benchmark Outputs
- **Benchmark Reports**:
  - `outputs/technical_iqa_topiq_koniq_cap_benchmark_20260524`: Confirms `topiq_mixed112` reaches SRCC 0.899 on SPAQ.
  - `outputs/technical_iqa_combo_benchmark_20260524`: Highlights "semantic bias" in TOPIQ teacher, where it over-scores technical failures.
- **Decision Record**: Combination B (KonIQ + FLIVE) was previously favored for safety, but TOPIQ remains the target for high-fidelity ranking improvement.

## 5. Missing Dependencies
- **Research Library**: `pyiqa` (PyIQA) is needed to run the official TOPIQ teacher.
- **Weights**: Official research-grade PyTorch weights (e.g., `topiq_nr_resnet50` or `topiq_nr_vit`) must be downloaded (usually via `pyiqa` auto-download).
- **Teacher Script**: No existing script in `src/` or `scripts/` currently imports `pyiqa` to generate labels from the research-grade model.

## 6. Recommended Minimal Teacher Smoke Test
1. **Environment Setup**:
   ```bash
   # In .venv_gpu
   pip install pyiqa
   ```
2. **Teacher Initialization**:
   Create `scripts/smoke_test_topiq_teacher.py` to verify model loading:
   ```python
   import pyiqa
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   model = pyiqa.create_predictor('topiq', device=device)
   ```
3. **Smoke Execution**:
   Run inference on the first 10 rows of `outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
4. **Parity Check**:
   Compare results with `topiq_score` in `outputs/technical_iqa_combo_benchmark_20260524/fusion_predictions_spaq.csv` to confirm the version and scale of the previously used teacher.

## 7. Risks
- **Version Drift**: `pyiqa` may have updated the default TOPIQ weights since the May 2024 benchmarks.
- **Scale Mismatch**: Teacher output scale (usually 0..1 or 1..5) must be strictly mapped to the 0..100 A-cut project scale.
- **Computational Cost**: Generating labels for the full 159k FLIVE rows using a research-grade teacher (especially ViT variants) will require significant GPU time.
