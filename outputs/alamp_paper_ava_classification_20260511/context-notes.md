# A-LAMP Paper AVA v0 Context Notes

## 2026-05-11 Start

- This track is isolated from the practical app-oriented A-LAMP implementation.
- Do not modify Flutter, forWeights/, src/models/alamp.py, existing practical A-LAMP training code, RGNet files, or app assets.
- Required wording: "A-LAMP-paper-oriented approximation", "A-LAMP-style AVA classification model", and "A-LAMP-paper-AVA-v0".
- Forbidden positioning: official reproduction, same as the paper, or paper-faithful reproduction.
- Default label rule is paper_strict: label = 1 if mean_score > 5.0 else 0.
- v0_a is the required first candidate: shared VGG16 patch branch over five 224x224 patches with a binary classifier head.
- v0_b may add global/layout inputs, but exact object/global attribute graph and exact saliency-map pipeline are not implemented in v0.
- `./.venv_gpu/bin/python` py_compile passed for all three new source files.
- TensorFlow 2.20.0 currently reports no visible GPUs through `./.venv_gpu/bin/python`; mid-run must not be started as a long CPU job.
- Forward smoke passed for v0_a and v0_b with batch size 2 dummy tensors, output shape `[2, 1]`, finite predictions, and probabilities in `[0, 1]`.
- v0_a tiny smoke train passed with 64 train / 32 val samples, 1 epoch, batch size 2, random VGG16 weights, and save/load max abs diff `0.0`.
- v0_a tiny smoke eval passed with 64 test / 64 val samples and no skipped images.
- First optional v0_b smoke train attempt failed before training because CLI `--variant v0_b` inherited the config's v0_a `include_global_branch: false` setting. The isolated model builder was fixed so v0_b always includes global/layout inputs.
- v0_b tiny smoke train and eval passed after the model-builder fix, with 64 train / 32 val training samples and 64 test / 64 val eval samples.
- v0_a mid-run command was invoked with the requested 4096/1024, 3 epochs, batch size 4 settings, but it stopped before training because TensorFlow reported no visible GPU. The CPU override was not used because that would start a long-running job.
- Final validation passed: py_compile succeeded, summary JSON files parsed, and report/command-log files exist.

## 2026-05-11 v0_b_fixed XLA Failure Fix Start

- New task: fix `A-LAMP-paper-AVA-v0_b` immediate training failure.
- Exact failure reported by user: `XLA compilation requires a fixed tensor list size`, node `loop_body/strided_slice_1/pfor/while/Placeholder_0/accumulator`, during `tf2xla` conversion of `one_step_on_data`.
- User confirmed this is not OOM, GPU is visible, cuDNN loads, and `TF_XLA_FLAGS=--tf_xla_auto_jit=0` did not fix it.
- Scope remains isolated: do not touch Flutter, `forWeights/`, `src/models/alamp.py`, RGNet files, or practical A-LAMP files.
- Allowed source/config files for this fix are `src/models/alamp_paper_ava.py`, `src/train/train_alamp_paper_ava.py`, `src/eval/evaluate_alamp_paper_ava.py`, and `configs/paper_benchmarks/alamp_paper_ava_classification.yaml`.
- Preferred fix is to simplify v0_b to fixed-shape patch features, global VGG16 features, flattened layout features, dense fusion, and explicit `jit_compile=False`.
- Root cause found: the old `v0_b` path used dynamic per-patch context fusion for layout/global inputs, including `TimeDistributed` per-patch processing, repeated global context, and learned attention over the patch axis. On the GPU train step this matched the reported `pfor`/TensorList XLA failure in `one_step_on_data`.
- Implemented fixed `v0_b` path with fixed-shape `MergePatchBatch` / `RestorePatchBatch`, global VGG16 features, flattened layout features, dense fusion, and classifier output. `v0_a` retained its existing patch-attention behavior.
- Added explicit `jit_compile=False` in `src/train/train_alamp_paper_ava.py`; fixed mid-train summary recorded `jit_compile_false_supported: true` and `model_jit_compile: false`.
- Added specificity to `src/eval/evaluate_alamp_paper_ava.py`.
- Validation passed: py_compile, v0_b forward smoke, v0_b tiny smoke train, v0_b_fixed mid-train, and v0_b_fixed mid-eval.
- Default sandbox TensorFlow still reported no visible GPUs, but escalated mid-train and mid-eval created `/device:GPU:0` and loaded cuDNN.
- v0_b_fixed mid-train completed 3 epochs with 4095 usable train samples, 1024 val samples, and best val loss `0.5550218820571899`.
- v0_b_fixed test metrics on 1024 samples: accuracy `0.71875`, F1 `0.8283671036948749`, ROC-AUC `0.664537250995636`, AP `0.8180224299430847`, BCE `0.5673540559696448`, precision `0.7331223628691983`, recall `0.952054794520548`, specificity `0.13945578231292516`, confusion matrix `tn=41, fp=253, fn=35, tp=695`.
- v0_b_fixed val metrics on 1024 samples: accuracy `0.7255859375`, F1 `0.8300060496067755`, ROC-AUC `0.7049517035484314`, AP `0.8327690362930298`, BCE `0.5550215987279898`, precision `0.7368421052631579`, recall `0.9501385041551247`, specificity `0.18874172185430463`, confusion matrix `tn=57, fp=245, fn=36, tp=686`.
