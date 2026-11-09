# Official Kaggle MUSIQ 100-image Sanity Benchmark Report

## 1. Executive Summary
- Final decision: `MUSIQ_BENCHMARK_FAILED`.
- Evaluated 300 images and wrote 897 predictions.
- Elapsed seconds: 259.1.
- Scope stayed benchmark-only: no training, TFLite export, Flutter edits, production scoring edits, distillation, or local repo-practical MUSIQ.

## 2. Inputs
- Input manifest: `/home/omen_pc1/photo_score_project/outputs/research_protocol_iqa_manifests_20260530/benchmark_all_research_protocol_iqa.csv`.
- Output directory: `/home/omen_pc1/photo_score_project/outputs/musiq_kaggle_sanity_20260601`.
- Seed: `20260601`.
- Sample target: `100` readable images per dataset.
- TensorFlow: `2.21.0`.
- TensorFlow Hub: `0.16.1`.
- setuptools: `80.10.2`.

## 3. Model Variants
### koniq-10k
- handle: `https://www.kaggle.com/models/google/musiq/TensorFlow2/koniq-10k/1`
- load_method: `tensorflow_hub_direct`
- loaded: `True`
- load_seconds: `3.24`
### spaq
- handle: `https://www.kaggle.com/models/google/musiq/TensorFlow2/spaq/1`
- load_method: `tensorflow_hub_direct`
- loaded: `True`
- load_seconds: `3.21`
### paq2piq
- handle: `https://www.kaggle.com/models/google/musiq/TensorFlow2/paq2piq/1`
- load_method: `tensorflow_hub_direct`
- loaded: `True`
- load_seconds: `3.05`

## 4. Dataset Sampling
- FLIVE: 100 images.
  - test: 4
  - train: 79
  - val: 17
- KonIQ-10k: 100 images.
  - test: 17
  - train: 83
- SPAQ: 100 images.
  - test: 9
  - train: 91

## 5. Raw Prediction Distributions
- koniq-10k / FLIVE: n=99, min=20.9172, max=78.9474, mean=46.0902, std=15.7917
- koniq-10k / KonIQ-10k: n=100, min=18.4128, max=79.0835, mean=61.5284, std=14.0118
- koniq-10k / SPAQ: n=100, min=12.8810, max=76.9882, mean=45.3901, std=20.9288
- spaq / FLIVE: n=99, min=17.6956, max=79.5341, mean=54.9682, std=13.5040
- spaq / KonIQ-10k: n=100, min=22.2393, max=79.9960, mean=62.1971, std=13.2504
- spaq / SPAQ: n=100, min=12.8729, max=80.9736, mean=48.1635, std=19.4008
- paq2piq / FLIVE: n=99, min=53.7524, max=80.1157, mean=72.7870, std=5.0488
- paq2piq / KonIQ-10k: n=100, min=51.0147, max=78.7185, mean=71.6378, std=5.8733
- paq2piq / SPAQ: n=100, min=41.1874, max=76.8992, mean=62.0896, std=10.9638

## 6. Dataset-wise Metrics
| Variant | Dataset | n | SRCC | PLCC | Aligned MAE | Failures |
|---|---:|---:|---:|---:|---:|---:|
| koniq-10k | FLIVE | 99 | 0.0794 | 0.1322 | 4.8735 | 1 |
| koniq-10k | KonIQ-10k | 100 | 0.9271 | 0.9547 | 2.2806 | 0 |
| koniq-10k | SPAQ | 100 | 0.8546 | 0.8552 | 8.2521 | 0 |
| spaq | FLIVE | 99 | 0.1770 | 0.2720 | 4.8489 | 1 |
| spaq | KonIQ-10k | 100 | 0.7501 | 0.7969 | 4.7930 | 0 |
| spaq | SPAQ | 100 | 0.9459 | 0.9441 | 5.4644 | 0 |
| paq2piq | FLIVE | 99 | 0.4720 | 0.5556 | 4.3566 | 1 |
| paq2piq | KonIQ-10k | 100 | 0.7998 | 0.8540 | 4.0036 | 0 |
| paq2piq | SPAQ | 100 | 0.8555 | 0.8419 | 9.0282 | 0 |

## 7. Comparison Against Production Baseline
| Dataset | Production SRCC | Best MUSIQ Variant | Best MUSIQ SRCC | Delta |
|---|---:|---|---:|---:|
| FLIVE | 0.574 | paq2piq | 0.4720 | -0.1020 |
| KonIQ-10k | 0.934 | koniq-10k | 0.9271 | -0.0069 |
| SPAQ | 0.873 | spaq | 0.9459 | 0.0729 |

## 8. Failure Cases
- inference | koniq-10k | flive | /home/omen_pc1/photo_score_project/data/raw/paq2piq/voc_emotic_ava/AVA__325065.jpg | Traceback (most recent call last):
  File "/home/omen_pc1/photo_score_project/scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py", line 498, in main
    prediction, call_method = predict_one(model, row["image_path"])
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py", line 206, in predict_one
    values = flatten_output(signature(**{input_name: image_bytes}))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1170, in __call__
    return self._call_impl(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1179, in _call_impl
    return self._call_with_structured_signature(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1263, in _call_with_structured_signature
    return self._call_flat(
           ^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/load.py", line 146, in _call_flat
    return super()._call_flat(args, captured_inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1322, in _call_flat
    return self._inference_function.call_preflattened(args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py", line 216, in call_preflattened
    flat_outputs = self.call_flat(*args)
                   ^^^^^^^^^^^^^^^^^^^^^
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:

Detected at node concat_4 defined at (most recent call last):
<stack traces unavailable>
ConcatOp : Dimension 1 in both shapes must be equal: shape[0] = [1,114,3072] vs. shape[1] = [1,342,1]
	 [[{{node concat_4}}]] [Op:__inference_signature_wrapper_25722]

- inference | spaq | flive | /home/omen_pc1/photo_score_project/data/raw/paq2piq/voc_emotic_ava/AVA__325065.jpg | Traceback (most recent call last):
  File "/home/omen_pc1/photo_score_project/scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py", line 498, in main
    prediction, call_method = predict_one(model, row["image_path"])
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py", line 206, in predict_one
    values = flatten_output(signature(**{input_name: image_bytes}))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1170, in __call__
    return self._call_impl(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1179, in _call_impl
    return self._call_with_structured_signature(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1263, in _call_with_structured_signature
    return self._call_flat(
           ^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/load.py", line 146, in _call_flat
    return super()._call_flat(args, captured_inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1322, in _call_flat
    return self._inference_function.call_preflattened(args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py", line 216, in call_preflattened
    flat_outputs = self.call_flat(*args)
                   ^^^^^^^^^^^^^^^^^^^^^
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:

Detected at node concat_4 defined at (most recent call last):
<stack traces unavailable>
ConcatOp : Dimension 1 in both shapes must be equal: shape[0] = [1,114,3072] vs. shape[1] = [1,342,1]
	 [[{{node concat_4}}]] [Op:__inference_signature_wrapper_52954]

- inference | paq2piq | flive | /home/omen_pc1/photo_score_project/data/raw/paq2piq/voc_emotic_ava/AVA__325065.jpg | Traceback (most recent call last):
  File "/home/omen_pc1/photo_score_project/scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py", line 498, in main
    prediction, call_method = predict_one(model, row["image_path"])
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/scripts/technical_iqa/run_musiq_kaggle_sanity_benchmark.py", line 206, in predict_one
    values = flatten_output(signature(**{input_name: image_bytes}))
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1170, in __call__
    return self._call_impl(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1179, in _call_impl
    return self._call_with_structured_signature(args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1263, in _call_with_structured_signature
    return self._call_flat(
           ^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/saved_model/load.py", line 146, in _call_flat
    return super()._call_flat(args, captured_inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py", line 1322, in _call_flat
    return self._inference_function.call_preflattened(args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/omen_pc1/photo_score_project/.venv_tf/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py", line 216, in call_preflattened
    flat_outputs = self.call_flat(*args)
                   ^^^^^^^^^^^^^^^^^^^^^
tensorflow.python.framework.errors_impl.InvalidArgumentError: Graph execution error:

Detected at node concat_4 defined at (most recent call last):
<stack traces unavailable>
ConcatOp : Dimension 1 in both shapes must be equal: shape[0] = [1,114,3072] vs. shape[1] = [1,342,1]
	 [[{{node concat_4}}]] [Op:__inference_signature_wrapper_80185]


## 9. Final Decision
MUSIQ_BENCHMARK_FAILED

## 10. Recommended Next Step
- Fix loading/inference/metric blockers and rerun this sanity benchmark only.
