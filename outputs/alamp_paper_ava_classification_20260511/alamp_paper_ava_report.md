# A-LAMP Paper-Oriented AVA Classification v0 Report

## 1. Current Fix Status
- Updated at: `2026-05-11T13:49:52.171336+09:00`
- Fixed target: `A-LAMP-paper-AVA-v0_b`
- Fixed train dir: `outputs/alamp_paper_ava_classification_20260511/mid_train/v0_b_fixed`
- Fixed eval dir: `outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_b_fixed`
- Full AVA training was not started.
- Flutter, `forWeights/`, `src/models/alamp.py`, and RGNet files were not modified.

## 2. Root Cause Found
The failing `v0_b` path used dynamic per-patch context fusion for layout/global inputs: `TimeDistributed` per-patch processing, repeated global context, and learned attention over the patch axis. On the GPU train step, that path lowered into `pfor`/TensorList work inside `one_step_on_data`, matching the reported node `loop_body/strided_slice_1/pfor/while/Placeholder_0/accumulator` and the XLA fixed tensor-list-size failure.

The fix removes the `v0_b` dynamic per-patch attention/fusion path. `v0_b_fixed` now uses fixed-shape patch features, a global VGG16 branch, flattened layout features, dense fusion, and a sigmoid classifier. `v0_a` keeps its existing patch-attention behavior.

## 3. Code Changes
- `src/models/alamp_paper_ava.py`: added fixed-shape `MergePatchBatch` and `RestorePatchBatch` layers for `v0_b` patch encoding.
- `src/models/alamp_paper_ava.py`: changed `v0_b` to patch summary + global VGG16 branch + flattened layout Dense branch + concatenation/classifier.
- `src/train/train_alamp_paper_ava.py`: added explicit `jit_compile=False` compile path and records compile metadata.
- `src/eval/evaluate_alamp_paper_ava.py`: added `specificity` to classification metrics.
- `configs/paper_benchmarks/alamp_paper_ava_classification.yaml`: added `v0_b_fixed` output directories.

## 4. Validation
- `py_compile`: passed.
- `v0_b_fixed` forward smoke: passed, output shape `[2, 1]`, finite predictions `True`, in `[0,1]` `True`.
- `v0_b_fixed` forward smoke confirmed `uses_time_distributed_layer=false`.
- `v0_b_fixed` tiny smoke train: passed, 64 train / 32 val samples, 1 epoch, best val loss `0.663689`.
- `jit_compile=False`: added and supported; mid-train recorded `model_jit_compile=False`.
- `v0_b_fixed` mid-train: passed, 4095 train / 1024 val usable samples, 3 epochs, best val loss `0.555022` at epoch `3`.
- `v0_b_fixed` mid-eval: passed, 1024 test / 1024 val samples.
- Note: default sandbox TensorFlow still showed no visible GPU, but escalated mid-train/eval created `/device:GPU:0` and loaded cuDNN.

## 5. v0_b_fixed Metrics
| Split | Accuracy | F1 | ROC-AUC | AP | BCE | Precision | Recall | Specificity | Confusion Matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| test | 0.718750 | 0.828367 | 0.664537 | 0.818022 | 0.567354 | 0.733122 | 0.952055 | 0.139456 | tn=41, fp=253, fn=35, tp=695 |
| val | 0.725586 | 0.830006 | 0.704952 | 0.832769 | 0.555022 | 0.736842 | 0.950139 | 0.188742 | tn=57, fp=245, fn=36, tp=686 |

## 6. Comparison Against v0_a
Test split:

| Metric | v0_a | v0_b_fixed | Delta |
|---|---:|---:|---:|
| accuracy | 0.723633 | 0.718750 | -0.004883 |
| F1 | 0.832841 | 0.828367 | -0.004474 |
| ROC-AUC | 0.667890 | 0.664537 | -0.003353 |
| AP | 0.819938 | 0.818022 | -0.001916 |
| BCE | 0.563631 | 0.567354 | 0.003723 |
| precision | 0.732087 | 0.733122 | 0.001035 |
| recall | 0.965753 | 0.952055 | -0.013699 |
| specificity | 0.122449 | 0.139456 | 0.017007 |
| confusion matrix | tn=36, fp=258, fn=25, tp=705 | tn=41, fp=253, fn=35, tp=695 | n/a |

Val split:

| Metric | v0_a | v0_b_fixed | Delta |
|---|---:|---:|---:|
| accuracy | 0.705078 | 0.725586 | 0.020508 |
| F1 | 0.820878 | 0.830006 | 0.009128 |
| ROC-AUC | 0.684839 | 0.704952 | 0.020113 |
| AP | 0.840112 | 0.832769 | -0.007343 |
| BCE | 0.559984 | 0.555022 | -0.004963 |
| precision | 0.717842 | 0.736842 | 0.019000 |
| recall | 0.958449 | 0.950139 | -0.008310 |
| specificity | 0.099338 | 0.188742 | 0.089404 |
| confusion matrix | tn=30, fp=272, fn=30, tp=692 | tn=57, fp=245, fn=36, tp=686 | n/a |

Interpretation: on the 1024-image test subset, `v0_b_fixed` is slightly below `v0_a` on accuracy, F1, ROC-AUC, AP, and BCE, while specificity and precision are slightly higher. On the 1024-image val subset, `v0_b_fixed` improves accuracy, F1, ROC-AUC, BCE, precision, and specificity, while recall and AP are lower.

## 7. Primary Artifacts
- Train summary: `outputs/alamp_paper_ava_classification_20260511/mid_train/v0_b_fixed/train_summary.json`
- Final model: `outputs/alamp_paper_ava_classification_20260511/mid_train/v0_b_fixed/final_model.keras`
- Best weights: `outputs/alamp_paper_ava_classification_20260511/mid_train/v0_b_fixed/best.weights.h5`
- Eval summary: `outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_b_fixed/evaluation_summary.json`
- Test predictions: `outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_b_fixed/test_predictions.csv`
- Val predictions: `outputs/alamp_paper_ava_classification_20260511/mid_eval/v0_b_fixed/val_predictions.csv`
