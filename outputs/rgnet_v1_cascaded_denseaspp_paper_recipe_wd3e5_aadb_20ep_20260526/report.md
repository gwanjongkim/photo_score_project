# RGNet-v1 Cascaded DenseASPP WD3e-5 Ablation Report

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

## Exact Training Command

```bash
./.venv_gpu/bin/python -m src.train.train_rgnet_paper_v1_aadb \
  --config configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml \
  --out_dir outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526 \
  --epochs 20 \
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
  --weight_decay 3e-5 \
  --disable_early_stopping true
```

## Training Result

- Output: `outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526`
- Train samples: 7612
- Val samples: 846
- Epochs requested: 20
- Epochs completed: 20
- EarlyStopping: disabled
- Weight decay: 0.00003
- Weight decay implementation: `optimizer_weight_decay`
- Best epoch: 8
- Best val loss: 0.0230989102
- Best val MAE: 0.1200934500
- Total parameters: 10,603,841
- Trainable parameters: 10,519,681

The final epoch overfit relative to the best checkpoint:

- Final val loss: 0.0267665926
- Final val MAE: 0.1292474866

## Exact Best-Checkpoint Evaluation Command

```bash
./.venv_gpu/bin/python -m src.eval.evaluate_rgnet_paper_v1_aadb \
  --config configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml \
  --weights_path outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/best.weights.h5 \
  --output_dir outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526 \
  --test_csv data/processed/aadb/test.csv \
  --batch_size 8
```

Best checkpoint test metrics:

- Eval output: `outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526`
- Test samples: 1000
- SRCC: 0.6518271602
- PLCC: 0.6592002461
- MAE: 0.1233198717
- RMSE: 0.1540849534
- MSE: 0.0237421729

## Exact Final-Model Evaluation Command

```bash
./.venv_gpu/bin/python -m src.eval.evaluate_rgnet_paper_v1_aadb \
  --config configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml \
  --model_path outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/final_model.keras \
  --output_dir outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/final_model \
  --test_csv data/processed/aadb/test.csv \
  --batch_size 8
```

Final model test metrics:

- Eval output: `outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_wd3e5_aadb_20ep_20260526/final_model`
- Test samples: 1000
- SRCC: 0.6517927630
- PLCC: 0.6602884616
- MAE: 0.1251951456
- RMSE: 0.1566265516
- MSE: 0.0245318767

## Comparison

- ES30 best-checkpoint SRCC: 0.6532747076
- Previous DenseASPP paper-recipe 20-epoch best SRCC: 0.6683026827
- WD3e-5 best-checkpoint SRCC: 0.6518271602
- Current best RGNet AADB baseline SRCC: 0.6819
- Strong paper-near signal threshold: 0.70

The stronger weight decay did not improve the result. It underperformed both ES30 and the previous DenseASPP paper-recipe run, and remains well below the current RGNet baseline.

## Recommendation

Pause the DenseASPP route. The last two controlled changes did not improve test SRCC:

- EarlyStopping improved validation loss but reduced test SRCC.
- `weight_decay=3e-5` did not improve validation-to-test transfer and reduced test SRCC further.

The best DenseASPP result remains the previous paper-recipe 20-epoch run at SRCC `0.6683026827`, still below the baseline `0.6819`. Continuing DenseASPP-only tuning is not justified without a new hypothesis stronger than small schedule or regularization changes.
