# RGNet-v1-Cascaded-DenseASPP Report

This is a controlled paper-faithfulness ablation, not a full RGNet paper reproduction.

## Changed Files

- `src/models/rgnet_paper_v1.py`
- `src/train/train_rgnet_paper_v1_aadb.py`
- `src/eval/evaluate_rgnet_paper_v1_aadb.py`
- `configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml`

## Implementation

- Added `context_module_type` with default `parallel_aspp`.
- Added `V1CascadedDenseASPP` for `context_module_type=cascaded_denseaspp`.
- DenseASPP rates: `3,6,12,18`.
- DenseASPP growth rate: `64`.
- Each DenseASPP layer receives `concat(original_feature_map, previous_denseaspp_outputs)`.
- Each DenseASPP layer uses `Conv2D(64, 3, padding="same", dilation_rate=rate, use_bias=False)`, BatchNorm, and ReLU.
- Final DenseASPP output is `concat(original_feature_map, four_denseaspp_outputs)`.
- For the current DenseNet121 256x256 path, context channels are `1024 + 4 * 64 = 1280`.

## Parameter Counts

- Parallel ASPP default: `19,431,745` total / `19,345,025` trainable.
- Cascaded DenseASPP: `10,603,841` total / `10,519,681` trainable.

## Validation Commands

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m py_compile src/models/rgnet_paper_v1.py src/train/train_rgnet_paper_v1_aadb.py src/eval/evaluate_rgnet_paper_v1_aadb.py
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/dryrun_rgnet_v1_default_context_compat_20260525 --epochs 1 --batch_size 4 --max_train_samples 64 --max_val_samples 32
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/dryrun_rgnet_v1_cascaded_denseaspp_aadb_20260525 --epochs 1 --batch_size 4 --max_train_samples 64 --max_val_samples 32 --context_module_type cascaded_denseaspp --denseaspp_rates 3,6,12,18 --denseaspp_growth_rate 64 --use_context_batchnorm true --adjacency_type semantic --aggregation mean
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525 --epochs 5 --batch_size 4 --max_train_samples 1024 --max_val_samples 256 --context_module_type cascaded_denseaspp --denseaspp_rates 3,6,12,18 --denseaspp_growth_rate 64 --use_context_batchnorm true --adjacency_type semantic --aggregation mean
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.eval.evaluate_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --weights_path outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525/best.weights.h5 --output_dir outputs/rgnet_v1_cascaded_denseaspp_aadb_smallrun_20260525/eval_best_valtest256 --test_csv data/processed/aadb/test.csv --val_csv data/processed/aadb/val.csv --max_test_samples 256 --max_val_samples 256 --batch_size 8 --context_module_type cascaded_denseaspp --denseaspp_rates 3,6,12,18 --denseaspp_growth_rate 64 --use_context_batchnorm true --adjacency_type semantic
```

## Results

- `py_compile`: passed.
- Default compatibility smoke: passed with `context_module_type=parallel_aspp`.
- Cascaded DenseASPP smoke: passed, no NaNs, no OOM, save/load max absolute diff `0.0`.
- Cascaded DenseASPP smallrun: completed 5 epochs, no NaNs, no OOM.
- Best smallrun epoch: `3`.
- Best val loss: `0.0245169606`.
- Best val MAE: `0.1232288107`.
- Feature grid: `8x8`, `64` region nodes.
- Semantic adjacency row sums were normalized to approximately `1.0`.

## Bounded Evaluation

- Val 256: SRCC `0.5376994494`, PLCC `0.5570460283`, MAE `0.1232311428`.
- Test 256: SRCC `0.4923896118`, PLCC `0.5295091397`, MAE `0.1170761883`.

## Memory Notes

- Batch size 4 fit on the RTX 4070 SUPER 12GB path.
- TensorFlow allocated a GPU device with about 9513 MB available to the process.
- First training step triggered slow XLA compilation of about 2.4 minutes and allocator garbage collection.
- After first-step compilation, epochs ran quickly.

## Judgment

The cascaded DenseASPP implementation is stable enough for a full controlled AADB ablation. The bounded 1024/256 metrics are valid but do not prove improvement over the current full AADB SRCC baseline of `0.6819`.

Recommended full-run command:

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/rgnet_v1_cascaded_denseaspp_aadb_full_20260525 --epochs 20 --batch_size 4 --context_module_type cascaded_denseaspp --denseaspp_rates 3,6,12,18 --denseaspp_growth_rate 64 --use_context_batchnorm true --adjacency_type semantic --aggregation mean
```

