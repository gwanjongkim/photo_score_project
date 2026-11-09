# Production Scoring Source Audit

## 1. Core Implementation
- **File**: `src/infer/select_best_shots.py`
- **Function**: `score_component(record, weights, component_name="technical")`
- **Weights** (`DEFAULT_SELECTOR_WEIGHTS`):
  - `koniq_score`: 0.32
  - `flive_image_score`: 0.24
  - `flive_patch_mean`: 0.22
  - `flive_patch_min`: 0.12
  - `musiq_score`: 0.10
- **Total Technical Weight**: 1.0 (relative to 0.4 tech / 0.6 aesthetic mix).

## 2. Models and Preprocessing
- **Models**:
  - KonIQ-Mobile: `checkpoints/technical_koniq_gpu/final_model.keras`
  - FLIVE-Image-Mobile: `checkpoints/technical_flive_image_gpu/final_model.keras`
  - FLIVE-Patch-Mobile: `checkpoints/technical_flive_patch_gpu/final_model.keras`
  - MUSIQ-AADB: `checkpoints/musiq_aadb_gpu/final_model.keras`
- **Input Preprocessing**:
  - Technical images: RGB, 224x224 bilinear resize (stretch).
  - Normalization: `float32` pixel values divided by 255.0.

## 3. Score Scaling
- Raw model outputs are assumed to be in `0..100` range (or `1..10` or `0..1`).
- `normalize_unit_score` maps these to a `0..1` unit scale.
- Final `technical_component` is a weighted average of normalized scores.
- For this audit, we will report production scores on the `0..100` scale for direct comparison with MOS and MDIQA.

## 4. Parity Status
- The production baseline SRCC (KonIQ 0.934, FLIVE 0.574, SPAQ 0.873) recorded in `outputs/research_protocol_baseline_eval_audit_20260530/report.md` uses these same models.
- Flutter runtime uses a MobileNetV2-based technical fallback with weights 0.60 KonIQ / 0.40 FLIVE-Image when only those two are available.
- The WSL selector bundle (Stage 5 Reference) is the source of truth for the "Production" baseline in our benchmark comparisons.
