# Aesthetic Mobile Model Origin Audit

## Executive Summary

| Model | Confirmed training dataset | Confirmed training script | Confirmed checkpoint/export source | Confirmed metrics | Confidence |
|---|---|---|---|---|---|
| `nima_mobile.tflite` | AVA (via `data/processed/ava/train_cleaned.csv`) | `src/train/train_nima.py` | `checkpoints/nima_ava_gpu/final_model.keras` | Evaluated against teacher: Spearman 0.3238, MAE 0.0468 | PARTIALLY CONFIRMED |
| `rgnet_aadb_gpu.tflite` | AADB (via `data/processed/aadb/train.csv`) | `src/train/train_rgnet.py` | `checkpoints/rgnet_aadb_gpu/final_model.keras` | Evaluated against teacher: Spearman 0.7979, MAE 0.0394 | PARTIALLY CONFIRMED |
| `alamp_aadb_gpu.tflite` | AADB (via `data/processed/aadb/train.csv`) | `src/train/train_alamp.py` | `checkpoints/alamp_aadb_gpu/best.weights.h5` | NOT CONFIRMED | PARTIALLY CONFIRMED |

## Model 1: nima_mobile.tflite
### Confirmed evidence
- **File properties:** Located at `exports/tflite/nima_mobile.tflite` (Size: 24,765,068 bytes, Last Modified: Apr 1 11:45).
- **Metadata (`exports/tflite/nima_mobile.metadata.json`):**
  - Confirms the source checkpoint path is `checkpoints/nima_ava_gpu/final_model.keras` and weights path `checkpoints/nima_ava_gpu/best.weights.h5`.
  - Confirms output shape `[1, 10]` representing a 10-bin aesthetic score distribution.
- **Verify JSON (`exports/tflite/nima_mobile.verify.json`):** Confirms exact numeric matching against the checkpoint.
- **Training scripts:** `src/train.sh` and `src/sequential_train.sh` both execute `src/train/train_nima.py` mapping to `--train_csv data/processed/ava/train_cleaned.csv` and outputting to directories named with `nima_ava`.
- **Metrics documentation:** `docs/pozy_app_stage5_aesthetic_student_handoff.md` lists `nima_mobile` with Spearman `0.3238` and MAE `0.0468` when evaluated on 846 AADB validation images against a teacher score.

### Dataset conclusion
It is highly likely that NIMA was trained on the AVA dataset, as strongly suggested by directory naming (`nima_ava_gpu`) and explicit mappings in `src/train.sh` and `src/sequential_train.sh` pointing to the AVA data paths. 

### Training/export reconstruction
- **Training:** Handled by `src/train/train_nima.py` with `data/processed/ava/train_cleaned.csv`.
- **Export:** Rebuilt to TFLite as detailed in the metadata JSON via a `builtin` conversion mode from the `final_model.keras` path.

### Missing evidence
- Explicit training metric logs (e.g. original validation performance on the AVA split) are not found locally (no `history.json` or model-specific evaluation logs in the checkpoint directory).

## Model 2: rgnet_aadb_gpu.tflite
### Confirmed evidence
- **File properties:** Located at `exports/tflite/rgnet_aadb_gpu.tflite` (Size: 26,085,072 bytes, Last Modified: Apr 4 21:41).
- **Metadata (`exports/tflite/rgnet_aadb_gpu.metadata.json`):**
  - Confirms source checkpoint is `checkpoints/rgnet_aadb_gpu/final_model.keras`.
  - Output is a single scalar `[1, 1]` indicating an RGNet aesthetic score in `[0, 1]`.
- **Verify JSON (`exports/tflite/rgnet_aadb_gpu.verify.json`):** Verifies the inference difference vs Keras model on `KakaoTalk_20260330_180646779_10.jpg`.
- **Training script:** `src/sequential_train.sh` launches `src/train/train_rgnet.py` using `--train_csv data/processed/aadb/train.csv`.
- **Metrics documentation:** `docs/pozy_app_stage5_aesthetic_student_handoff.md` lists `rgnet_aadb_gpu` with Spearman `0.7979` and MAE `0.0394` on the AADB validation split.

### Dataset conclusion
It is highly likely that RGNet was trained on the AADB dataset based on the directory identifier (`rgnet_aadb_gpu`) and the exact dataset paths fed into `src/train/train_rgnet.py` in `src/sequential_train.sh`.

### Training/export reconstruction
- **Training:** Run via `src/train/train_rgnet.py` using `data/processed/aadb/train.csv`.
- **Export:** Converted from the checkpoint `checkpoints/rgnet_aadb_gpu/final_model.keras` via rebuild and `builtin` TFLite mode.

### Missing evidence
- Explicit training log files recording original training metrics for RGNet are missing. The only documented metrics represent transfer validation against a teacher score on the AADB split.

## Model 3: alamp_aadb_gpu.tflite
### Confirmed evidence
- **File properties:** Located at `exports/tflite/alamp_aadb_gpu.tflite` (Size: 39,158,340 bytes, Last Modified: Apr 4 21:17).
- **Metadata (`exports/tflite/alamp_aadb_gpu.metadata.json`):**
  - Confirms the source weights path is `checkpoints/alamp_aadb_gpu/best.weights.h5`.
  - Explains the export used `rebuild_from_weights` because a standard SavedModel conversion failed due to a `BroadcastTo` op on float16 layout cues.
- **Training script:** `src/sequential_train.sh` runs `src/train/train_alamp.py` using `--train_csv data/processed/aadb/train.csv`.

### Dataset conclusion
It is highly likely that A-Lamp was trained on the AADB dataset. Evidence comes directly from `alamp_aadb_gpu` directory naming and the shell script `src/sequential_train.sh` passing `data/processed/aadb/train.csv` to the script.

### Training/export reconstruction
- **Training:** Triggered via `src/train/train_alamp.py` acting on `data/processed/aadb/train.csv`.
- **Export:** It failed standard conversion and was exported using `rebuild_from_weights` using `checkpoints/alamp_aadb_gpu/best.weights.h5` and a `float32` global policy to bypass a TFLite conversion bug with float16 tensors.

### Missing evidence
- Validation/Test metrics are not confirmed. They are neither documented in the mobile deployment docs nor available in local log files.

## Cross-check against paper datasets
Based purely on project file paths and configurations:
- **NIMA** models in literature are commonly trained on the AVA dataset, which aligns perfectly with this project's NIMA model using `data/processed/ava/train_cleaned.csv`.
- **A-Lamp** and **RGNet** are trained locally using AADB datasets, which aligns with common academic pipelines that report composition/attribute learning results on AADB.

## Final judgment
- **Which models are confirmed AADB-trained?**
  RGNet and A-Lamp are highly likely (partially confirmed) trained on AADB due to the `sequential_train.sh` script inputs and target directories.
- **Which models are confirmed AVA-trained?**
  NIMA is highly likely (partially confirmed) trained on AVA, based on `src/train.sh` script parameters and `nima_ava_gpu` directory usage.
- **Which models are not confirmed?**
  None of the models are completely missing provenance, but final original dataset test metrics cannot be definitively confirmed from local sources for any of them. A-Lamp lacks all validation metric records.
- **What exact files should be inspected next if we want stronger proof?**
  To achieve full confidence, the missing raw training logs (e.g., `history.json` or TensorBoard metrics inside the checkpoint output directories) or the original dataset split metadata files (like a `manifest.json` stored right after training) would need to be procured or made available locally. Exploring cloud storage references (e.g. Firebase or Weights & Biases) might also provide the raw training run data.