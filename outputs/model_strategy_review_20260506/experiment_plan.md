# A-cut Model Strategy Experiment Plan

This plan was not executed. No full training was started during the audit.

## Experiment 1. Fix A-LAMP Preprocessing Mismatch And Compare Before/After

- Purpose: make Flutter A-LAMP runtime preprocessing match WSL training/export preprocessing before judging model quality.
- Required dataset: a fixed local image subset from AADB validation and/or the app's representative mobile photos.
- Local evidence: WSL uses `prepare_alamp_inputs` with aspect-preserving global input and adaptive/saliency patch proposals in `/home/omen_pc1/photo_score_project/src/datasets/native_size_dataset.py:43`, `:60`, `:84`, `:156`; Flutter uses square resize and fixed anchors in `/mnt/c/Users/OMEN PC1/StudioProjects/pozy_app/lib/feature/a_cut/layer/inference/image_preprocessor.dart:84`, `:114`.
- Command for WSL reference output on one image:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/infer/predict_alamp.py --image_path <image_path> --model checkpoints/alamp_aadb_gpu/final_model.keras --include_debug
```

- Flutter comparison command: not verifiable from local evidence. A Dart/Flutter test command should be selected after source changes are explicitly allowed.
- Expected output path: `outputs/alamp_preprocess_parity_20260506/`
- Required time estimate: not verifiable from local evidence.
- Metrics: per-image absolute score difference between WSL Keras and Flutter TFLite, selected patch box similarity if exposed, top-k agreement, pairwise ranking agreement, and mobile latency.
- Success criterion: Flutter A-LAMP scores are close to WSL reference scores on the fixed subset and rankings do not regress against the current baseline.
- Rollback criterion: preprocessing change increases Keras/TFLite disagreement, breaks A-LAMP loading, or worsens top-k/pairwise ranking on the fixed subset.

## Experiment 2. Train And Evaluate RGNet On AVA If Scalar Compatibility Is Confirmed

- Purpose: test whether an AVA-trained RGNet-style scalar model improves ranking and calibration compared with current AADB RGNet.
- Required dataset: AVA train/val/test images and a normalized scalar manifest with a 0..1 target, for example `(mean_score - 1) / 9`.
- Local evidence: AVA manifests/images exist; `train_rgnet.py` accepts CSV and `--target_col`, but current output is sigmoid 0..1.
- Precondition: create or verify a normalized AVA scalar CSV. The currently inspected AVA CSV exposes `mean_score` on 1..10, which should not be used directly with the current sigmoid output.
- Training command template after normalized CSV exists:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/train/train_rgnet.py --train_csv data/processed/ava/train_unit.csv --val_csv data/processed/ava/val_unit.csv --target_col aesthetic_unit_score --image_size 256 --batch_size 16 --epochs 10 --out_dir outputs/rgnet_ava_unit_20260506
```

- Export command template after training:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/export/export_tflite.py --preset rgnet --model_path outputs/rgnet_ava_unit_20260506 --output_dir outputs/rgnet_ava_unit_20260506/tflite --output_name rgnet_ava_unit.tflite --metadata_path outputs/rgnet_ava_unit_20260506/tflite/rgnet_ava_unit.metadata.json --overwrite
```

- Expected output path: `outputs/rgnet_ava_unit_20260506/`
- Required time estimate: not verifiable from local evidence.
- Metrics: SRCC, PLCC, MAE, MSE/RMSE, pairwise ranking agreement, top-k agreement, TFLite parity, and mobile latency.
- Success criterion: statistically meaningful improvement over current `rgnet_aadb_gpu` on the same validation/test protocol, with TFLite parity acceptable for app scoring.
- Rollback criterion: scale mismatch, failed TFLite conversion, worse ranking/top-k metrics, or unacceptable mobile latency.

## Experiment 3. Train And Evaluate A-LAMP On AVA If Scalar Compatibility Is Confirmed

- Purpose: test whether AVA improves the A-LAMP-style model after preprocessing parity is addressed.
- Required dataset: AVA train/val/test images and normalized 0..1 scalar labels.
- Local evidence: `train_alamp.py` accepts CSV and `--target_col`; A-LAMP output is scalar 0..1; current AVA `mean_score` is 1..10.
- Precondition: Experiment 1 should be completed first, otherwise app-side A-LAMP scores remain confounded by preprocessing mismatch.
- Training command template after normalized CSV exists:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/train/train_alamp.py --train_csv data/processed/ava/train_unit.csv --val_csv data/processed/ava/val_unit.csv --target_col aesthetic_unit_score --batch_size 16 --epochs 10 --global_size 384 --patch_size 224 --num_patches 5 --out_dir outputs/alamp_ava_unit_20260506
```

- Export command: not verifiable from local evidence as a preset-specific command in the inspected files. The existing `exports/tflite/alamp_aadb_gpu.*` artifacts prove a previous A-LAMP TFLite export exists, but the inspected generic export preset list did not include A-LAMP.
- Expected output path: `outputs/alamp_ava_unit_20260506/`
- Required time estimate: not verifiable from local evidence.
- Metrics: SRCC, PLCC, MAE, MSE/RMSE, pairwise ranking agreement, top-k agreement, TFLite parity, patch-selection parity, and mobile latency.
- Success criterion: improvement over current AADB A-LAMP under the same preprocessing and evaluation protocol, without breaking TFLite parity.
- Rollback criterion: failed export, worse ranking/top-k metrics, unstable patch behavior, or app latency worse than acceptable target.

## Experiment 4. Test AVA Pretrain Plus AADB Fine-Tune

- Purpose: test whether larger AVA pretraining plus smaller AADB fine-tuning improves personal/mobile photo relevance.
- Required dataset: normalized AVA scalar manifests and existing AADB train/val CSVs.
- Local evidence: train scripts save `best.weights.h5` and `final_model.keras`, but inspected parser lines do not expose `--resume`, `--initial_weights`, or `--load_weights`.
- Command: not verifiable from local evidence without source changes or a verified manual weight-loading workflow.
- Expected output path: `outputs/rgnet_ava_pretrain_aadb_finetune_20260506/` and/or `outputs/alamp_ava_pretrain_aadb_finetune_20260506/`
- Required time estimate: not verifiable from local evidence.
- Metrics: same as Experiments 2 and 3, evaluated on held-out AADB/mobile-photo-like data and TFLite parity.
- Success criterion: beats both AADB-only and AVA-only variants on ranking/top-k metrics and does not worsen calibration.
- Rollback criterion: no gain over simpler AVA-only or AADB-only models, unstable score scale, or added implementation risk beyond the project deadline.

## Experiment 5. Consider MUSIQ TFLite Export Or MUSIQ Teacher Use

- Purpose: evaluate MUSIQ as an offline evaluator or teacher only after simpler deployment-safe paths are tested.
- Required dataset: AADB/KonIQ/PaQ/AVA depending on target label; exact MUSIQ paper dataset setup is not locally reproduced.
- Local evidence: MUSIQ-like code and checkpoint exist, but no MUSIQ TFLite export preset or Flutter usage was found.
- WSL inference command on one image:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python src/infer/predict_musiq.py --image_path <image_path> --model checkpoints/musiq_aadb_gpu/final_model.keras
```

- MUSIQ TFLite export command: not verifiable from local evidence. No `musiq` export preset exists in the inspected TFLite preset registry.
- Expected output path: `outputs/musiq_teacher_probe_20260506/`
- Required time estimate: not verifiable from local evidence.
- Metrics: agreement with ensemble/student targets, SRCC/PLCC/MAE, top-k agreement, teacher-student distillation gain, and only later TFLite parity/latency if export becomes available.
- Success criterion: MUSIQ improves offline teacher ranking or distillation targets without becoming an on-device dependency.
- Rollback criterion: weak agreement with current validated targets, no student improvement, failed export, or unacceptable mobile latency.
