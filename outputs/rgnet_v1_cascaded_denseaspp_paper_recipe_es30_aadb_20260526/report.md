# RGNet-v1 Cascaded DenseASPP ES30 Ablation Report

This is an RGNet-paper-v1 approximation, not an official WACV RGNet reproduction.

## Scope

- Architecture was not modified.
- Flutter, TFLite/mobile assets, A-LAMP files, and unrelated dirty files were not modified.
- No commit was made.

## Preflight

Tracked unrelated dirty files present before the run:

- `configs/paper_benchmarks/alamp_multipatch_teacher_ava.yaml`
- `src/eval/evaluate_alamp_multipatch_teacher.py`
- `src/train/train_alamp_multipatch_teacher.py`
- `src/train/train_techiqa_guard.py`

These files were left untouched.

The first sandboxed training attempt failed before training because TensorFlow could not see the GPU. A first outside-sandbox start inherited `disable_early_stopping: true` from the config and was terminated before completing epoch 1. The corrected command explicitly passed `--disable_early_stopping false`.

## Exact Training Command

```bash
./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb \
  --config configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml \
  --out_dir outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526 \
  --epochs 30 \
  --batch_size 4 \
  --image_size 300 \
  --context_module_type cascaded_denseaspp \
  --denseaspp_rates 3,6,12,18 \
  --denseaspp_growth_rate 64 \
  --use_context_batchnorm true \
  --adjacency_type semantic \
  --aggregation lse \
  --lse_r 4 \
  --paper_recipe true \
  --random_horizontal_flip true \
  --random_scale_crop true \
  --scale_min 1.05 \
  --scale_max 1.25 \
  --crop_size 300 \
  --lr_schedule polynomial \
  --learning_rate 1e-4 \
  --poly_power 0.9 \
  --weight_decay 1e-5 \
  --early_stopping_patience 8 \
  --disable_early_stopping false
```

## Training Result

- Output: `outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526`
- Train samples: 7612
- Val samples: 846
- Epochs requested: 30
- Epochs completed: 14
- EarlyStopping: enabled
- Patience: 8
- Restore best weights: true
- Best epoch: 6
- Best val loss: 0.0227430612
- Best val MAE: 0.1203708574
- Total parameters: 10,603,841
- Trainable parameters: 10,519,681

## Exact Best-Checkpoint Evaluation Command

```bash
./.venv_gpu/bin/python -m src.eval.evaluate_rgnet_paper_v1_aadb \
  --config configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml \
  --weights_path outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/best.weights.h5 \
  --output_dir outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526 \
  --test_csv data/processed/aadb/test.csv \
  --batch_size 8
```

Best checkpoint test metrics:

- Eval output: `outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526`
- Test samples: 1000
- SRCC: 0.6532747076
- PLCC: 0.6641100023
- MAE: 0.1213828921
- RMSE: 0.1521848376
- MSE: 0.0231602248

## Exact Final-Model Evaluation Command

```bash
./.venv_gpu/bin/python -m src.eval.evaluate_rgnet_paper_v1_aadb \
  --config configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml \
  --model_path outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/final_model.keras \
  --output_dir outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/final_model \
  --test_csv data/processed/aadb/test.csv \
  --batch_size 8
```

Final model test metrics:

- Eval output: `outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_es30_aadb_20260526/final_model`
- Test samples: 1000
- SRCC: 0.6532700892
- PLCC: 0.6641100357
- MAE: 0.1213828772
- RMSE: 0.1521848253
- MSE: 0.0231602211

The final model is effectively identical to the best checkpoint because EarlyStopping restored the best weights before saving.

## Comparison

- Previous cascaded DenseASPP paper-recipe best SRCC: 0.6683026827
- ES30 best-checkpoint SRCC: 0.6532747076
- Current best RGNet AADB baseline SRCC: 0.6819
- Strong paper-near signal threshold: 0.70

The ES30 ablation did not improve over the previous paper-recipe run and did not beat the current RGNet baseline. It improved validation loss versus the previous run, but that did not transfer to AADB test SRCC.

## Recommendation

EarlyStopping alone is not the next winning change. The next controlled step should be:

A. `weight_decay=3e-5`

Reasoning: the run still shows overfitting, and weight decay is the smallest regularization-only change that preserves architecture. If that also fails to recover test SRCC above `0.6683`, the DenseASPP route should be deprioritized before trying larger architecture-combination experiments such as hybrid spatial adjacency.
