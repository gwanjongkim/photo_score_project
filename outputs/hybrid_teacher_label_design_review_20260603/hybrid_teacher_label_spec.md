# Hybrid Teacher Label Design Specification

## 1. Goal
Define a hybrid teacher target for technical IQA distillation that leverages MDIQA's superior performance on KonIQ-10k and SPAQ while retaining the production technical score's stability on FLIVE and real-world mobile content.

## 2. Hybrid Teacher Rule (Policy A: Source-aware Strict Hybrid)
This is the primary candidate for training labels.

| Dataset | Teacher Source | Logic |
| :--- | :--- | :--- |
| **KonIQ-10k** | `mdiqa_seed1` | Use MDIQA `seed1` prediction (SRCC ~0.948) |
| **SPAQ** | `mdiqa_xd` | Use MDIQA `xd` prediction (SRCC ~0.914) |
| **FLIVE** | `production` | Use Production Technical Score (SRCC ~0.574) |
| **Unknown / Mobile** | `production_fallback` | Use Production Technical Score as safety fallback |

## 3. Alternative Policy (Policy B: 80/20 Blend)
Evaluated for range smoothing and safety.

- **KonIQ / SPAQ**: `0.8 * MDIQA + 0.2 * Production`
- **FLIVE / Unknown**: `Production`

## 4. Range and Clipping
- All targets must be in `[0, 100]`.
- Final target = `clip(selected_target, 0, 100)`.

## 5. Required Manifest Columns
- `image_path`
- `dataset`
- `split`
- `mdiqa_pred_0_100`
- `production_pred_0_100`
- `quality_score_0_100` (if available, for validation)
- `teacher_target_0_100`
- `teacher_source`

## 6. Implementation Notes
- This hybrid rule is for **offline label generation only**.
- The student model will be trained on these labels to produce a single, dataset-agnostic technical score at runtime.
- No dataset-aware logic should exist in the Flutter/C++ production code.

## 7. Safety Gates
- **Severe FP**: MOS < 40 and Teacher > 65.
- **Range Inconsistency**: Mean shift between MDIQA and Production > 10 points.
- **FLIVE Parity**: Student must match or exceed Production SRCC on FLIVE test set.
