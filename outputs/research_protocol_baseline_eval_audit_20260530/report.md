# Research-Protocol Baseline Evaluation Audit

## 1. Summary
Existing technical scoring baselines can be evaluated against the new research-protocol manifests, but the exact smoke shape requested here is not exposed by the current benchmark scripts alone. I added a split-aware smoke prep helper so the next step can stay bounded at 5 images per dataset/split and avoid accidental full evaluation.

## 2. Manifests Confirmed
- KonIQ-10k MUSIQ-style manifest: `outputs/research_protocol_iqa_manifests_20260530/benchmark_koniq10k_musiq_80_20_10runs.csv` (`100,730` rows).
- SPAQ official split manifest: `outputs/research_protocol_iqa_manifests_20260530/benchmark_spaq_official_split.csv` (`11,125` rows).
- FLIVE/PaQ-2-PiQ standardized manifest: `outputs/research_protocol_iqa_manifests_20260530/benchmark_flive_paq2piq_toolbox_split.csv` (`159,240` rows).
- Combined manifest: `outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv` (`271,095` rows).

## 3. Scoring Entrypoints Found

### KonIQ standalone
- File path: `scripts/eval_final_topiq_candidates_vs_existing_technical.py`
- Callable / CLI: `PYTHONPATH=. ./.venv_gpu/bin/python scripts/eval_final_topiq_candidates_vs_existing_technical.py --out_dir ... --max_samples 5`
- Input requirements: RGB image paths from a CSV with `image_path` and `mos`.
- Output scale: `0..100` raw KonIQ score.
- Preprocessing contract: 224x224 bilinear resize, float32, divide by 255.0.
- Model artifact path: `exports/tflite/koniq_mobile.tflite`.
- Metadata file: `exports/tflite/koniq_mobile.metadata.json`, `exports/tflite/koniq_mobile.verify.json`.
- Confidence: high.

### FLIVE standalone
- File path: `scripts/eval_final_topiq_candidates_vs_existing_technical.py`
- Callable / CLI: `PYTHONPATH=. ./.venv_gpu/bin/python scripts/eval_final_topiq_candidates_vs_existing_technical.py --out_dir ... --max_samples 5`
- Input requirements: RGB image paths from a CSV with `image_path` and `mos`.
- Output scale: `0..100` raw FLIVE full-image score.
- Preprocessing contract: 224x224 bilinear resize, float32, divide by 255.0.
- Model artifact path: `exports/tflite/flive_image_mobile.tflite`.
- Metadata file: `exports/tflite/flive_image_mobile.metadata.json`, `exports/tflite/flive_image_mobile.verify.json`.
- Confidence: high.

### Production technical_score
- File path: `src/infer/run_acut_pipeline.py` and `src/infer/select_best_shots.py`.
- Callable / CLI: `PYTHONPATH=. ./.venv_gpu/bin/python src/infer/run_acut_pipeline.py --input_dir ... --output_dir ... --top_k 5`.
- Input requirements: image folder or precomputed score exports; optional VILA inputs.
- Output scale: technical model outputs are `0..100`, then normalized to `0..1` inside the selector; final A-cut score is `0..1`.
- Preprocessing contract: 224x224 bilinear resize, float32, divide by 255.0 for the technical branches.
- Model artifact path: `configs/stage5_reference.json` resolving `checkpoints/technical_koniq_gpu/final_model.keras` and `checkpoints/technical_flive_image_gpu/final_model.keras`; optional patch/MUSIQ/RGNet/AADB models are also listed in the stage5 bundle.
- Metadata file: `configs/stage5_reference.json`.
- Confidence: medium-high.

### C6 candidate
- File path: `outputs/c6_android_wsl_parity_exif_fixed_20260528/run_wsl_c6_parity_exif_fixed.py`
- Callable / CLI: `PYTHONPATH=. ./.venv_gpu/bin/python outputs/c6_android_wsl_parity_exif_fixed_20260528/run_wsl_c6_parity_exif_fixed.py`
- Input requirements: KonIQ and TOPIQ TFLite outputs on the same image set.
- Output scale: `0..100` combined C6 score.
- Preprocessing contract: KonIQ uses 224x224 stretch + `/255.0`; TOPIQ uses 384x384 `resize_with_pad` with no normalization.
- Model artifact path: `exports/tflite/koniq_mobile.tflite`; `outputs/final_topiq_lite_mixed112_export_20260517/topiq_lite_mixed112_frozen_fp16.tflite`.
- Metadata file: `outputs/topiq_mixed112_artifact_recovery_20260527/topiq_lite_mixed112.metadata.json`.
- Confidence: high.

## 4. Missing or Risky Pieces
- The current benchmark script supports `--max_samples`, but that is not the same as the requested split-aware smoke sample of 5 per dataset/split.
- FLIVE manifests include both image and patch rows; the standalone FLIVE image baseline should only consume image rows.
- Production scoring can auto-resolve more than the bare technical pair if `stage5_reference.json` is left fully enabled; smoke runs should keep the scope narrow.
- C6 remains comparison-only and is still blocked by the Android vs. WSL parity caveat recorded in the earlier C6 audit trail.

## 5. Metrics to Use
Primary paper metrics:
- SRCC.
- PLCC.

Secondary A-cut diagnostics:
- MAE.
- RMSE.
- Max AE.
- low-quality false positive rate.

These secondary metrics are for safety/debugging and do not replace the paper metrics.

## 6. Smoke Evaluation Plan
Sample shape:
- KonIQ: 5 images from `run_01/train` and 5 images from `run_01/test`.
- SPAQ: 5 images from `train` and 5 images from `test`.
- FLIVE: 5 image rows from `train`, `val`, and `test`.
- C6: reuse the same sampled KonIQ and FLIVE image rows.

Expected outputs:
- sampled smoke CSVs under `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/sampled_manifests/`
- command note file under `outputs/research_protocol_baseline_eval_audit_20260530/smoke_prep/smoke_eval_commands.txt`

Runtime risk:
- low for the smoke-prep step.
- medium if the existing benchmark scripts are invoked directly because they do not naturally preserve the exact split-aware sampling shape without the new prep step.

How to avoid full 271k inference:
- run only the smoke-prep helper first.
- cap the sample at 5 per split.
- do not run the combined manifest directly without sampling.

## 7. Decision
A. Ready to run smoke evaluation.

## 8. Next Action
Run `outputs/research_protocol_baseline_eval_audit_20260530/run_research_protocol_smoke_eval.py` with `--sample_size 5` to materialize the split-aware smoke CSVs, then execute the smallest benchmark harnesses against those sampled inputs.

## 9. Forbidden Actions
- Do not train.
- Do not run full benchmark yet.
- Do not replace production technical_score.
- Do not use C6 as production.
