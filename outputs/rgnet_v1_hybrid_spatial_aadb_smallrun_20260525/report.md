# RGNet-v1-Hybrid-Spatial AADB Smallrun Report

This is a paper-oriented hybrid semantic-spatial adjacency experiment, not a full RGNet paper reproduction.

## Changed Behavior

- Default `adjacency_type=semantic` keeps the existing cosine-similarity softmax adjacency.
- Opt-in `adjacency_type=hybrid_spatial` computes `A = (1 - alpha) * A_semantic + alpha * A_spatial`, then row-normalizes.
- `A_spatial` is a row-normalized Gaussian prior over the post-ASPP spatial grid.

## Commands

```bash
python -m py_compile src/models/rgnet_paper_v1.py
python -m py_compile src/train/train_rgnet_paper_v1_aadb.py
python -m py_compile src/eval/evaluate_rgnet_paper_v1_aadb.py
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/dryrun_rgnet_v1_hybrid_spatial_default_compat_20260525 --epochs 1 --batch_size 4 --max_train_samples 64 --max_val_samples 32
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/dryrun_rgnet_v1_hybrid_spatial_20260525 --epochs 1 --batch_size 4 --max_train_samples 64 --max_val_samples 32 --adjacency_type hybrid_spatial --spatial_alpha 0.2 --spatial_sigma 0.25
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --out_dir outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525 --epochs 5 --batch_size 4 --max_train_samples 1024 --max_val_samples 256 --adjacency_type hybrid_spatial --spatial_alpha 0.2 --spatial_sigma 0.25
```

```bash
PYTHONPATH=. ./.venv_gpu/bin/python -m src.eval.evaluate_rgnet_paper_v1_aadb --config configs/paper_benchmarks/rgnet_v1_ablations/agg_mean.yaml --weights_path outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525/best.weights.h5 --output_dir outputs/rgnet_v1_hybrid_spatial_aadb_smallrun_20260525/eval_best_valtest256 --test_csv data/processed/aadb/test.csv --val_csv data/processed/aadb/val.csv --max_test_samples 256 --max_val_samples 256 --batch_size 8 --adjacency_type hybrid_spatial --spatial_alpha 0.2 --spatial_sigma 0.25
```

## Results

- `py_compile`: passed.
- Default semantic smoke: passed, 64 train / 32 val, no NaNs.
- Hybrid smoke: passed, 64 train / 32 val, no NaNs, save-load max absolute diff 0.0.
- Hybrid smallrun: completed 5 epochs, 1024 train / 256 val.
- Best smallrun epoch: 3.
- Best val loss: 0.0256827474.
- Best val MAE: 0.1269362718.
- Parameter count: 19,431,745 total / 19,345,025 trainable.
- Feature map node count: 64.
- Adjacency row sums: semantic, spatial, and final hybrid min/max were within approximately 1e-7 of 1.0.

## Bounded Evaluation

- Val subset, 256 samples: SRCC 0.5476465019, PLCC 0.5519604708, MAE 0.1269381642.
- Test subset, 256 samples: SRCC 0.5071672026, PLCC 0.5284526137, MAE 0.1175746620.

## Notes

- The sandboxed default smoke could not initialize CUDA and fell back to random DenseNet weights. The escalated hybrid smoke and smallrun used the RTX 4070 SUPER and cached DenseNet weights.
- Batch size 4 fit on the 12GB GPU but TensorFlow reported allocator garbage collection and a slow first-epoch XLA compile.
- The smallrun is stable enough to justify a full AADB experiment from a runtime perspective.
- The bounded SRCC/PLCC do not establish improvement over the current full AADB SRCC baseline of 0.6819.

