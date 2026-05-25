# RGNet-v1 Cascaded DenseASPP Paper-Recipe Report

This run is an RGNet-paper-v1 approximation, not an official WACV RGNet reproduction.

## Changed Files

- `src/train/train_rgnet_paper_v1_aadb.py`
  - Added opt-in paper-recipe controls for training-only random horizontal flip, random scale-crop, polynomial LR decay, optimizer weight decay, and early-stopping disable/patience handling.
- `src/eval/evaluate_rgnet_paper_v1_aadb.py`
  - Added `--aggregation` and `--lse_r` support for weights-only reconstruction.
- `configs/paper_benchmarks/rgnet_v1_ablations/cascaded_denseaspp_paper_recipe.yaml`
  - Added the 300x300 cascaded DenseASPP + LSE/r=4 paper-recipe config.

## Recipe Settings

- `image_size`: 300
- `context_module_type`: `cascaded_denseaspp`
- `denseaspp_rates`: `[3, 6, 12, 18]`
- `denseaspp_growth_rate`: 64
- `use_context_batchnorm`: true
- `adjacency_type`: `semantic`
- `graph_blocks`: 3
- `aggregation`: `lse`
- `lse_r`: 4.0
- `random_horizontal_flip`: true, training only
- `random_scale_crop`: true, training only
- `scale_min`: 1.05
- `scale_max`: 1.25
- `crop_size`: 300
- `lr_schedule`: `polynomial`
- `initial_learning_rate`: 0.0001
- `poly_power`: 0.9
- `weight_decay`: 0.00001
- `weight_decay_implementation`: `optimizer_weight_decay`
- `disable_early_stopping`: true

## Validation

`py_compile` passed for:

- `src/models/rgnet_paper_v1.py`
- `src/train/train_rgnet_paper_v1_aadb.py`
- `src/eval/evaluate_rgnet_paper_v1_aadb.py`

Default compatibility smoke passed:

- Output: `outputs/dryrun_rgnet_v1_default_recipe_compat_20260526`
- Default path remained `paper_recipe=false`, `random_scale_crop=false`, constant LR, no weight decay, parallel ASPP.
- Save/load parity: `max_abs_diff_vs_trained_forward_sample = 0.0`
- Eval smoke on 32 test / 32 val samples completed.

Paper-recipe cascaded DenseASPP smoke passed:

- Output: `outputs/dryrun_rgnet_v1_cascaded_denseaspp_paper_recipe_20260526`
- Batch size 4 at 300x300 fit on RTX 4070 SUPER 12GB.
- Feature grid: 9x9 = 81 nodes.
- Save/load parity: `max_abs_diff_vs_trained_forward_sample = 0.0`
- No NaNs or OOM.

## Controlled 20-Epoch Run

- Output: `outputs/rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526`
- Train samples: 7612
- Val samples: 846
- Epochs requested/completed: 20 / 20
- Batch size: 4
- Total parameters: 10,603,841
- Trainable parameters: 10,519,681
- Best epoch: 6
- Best val loss: 0.0232909694
- Best val MAE: 0.1220422015
- Last scheduled LR: 0.0000067464

The run completed, but train loss kept improving while validation loss stopped improving after epoch 6.

## Test Evaluation

Best checkpoint:

- Evaluation output: `outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526`
- Test samples: 1000
- SRCC: 0.6683026827
- PLCC: 0.6720532565
- MAE: 0.1206469983
- RMSE: 0.1510883039
- MSE: 0.0228276756

Final model:

- Evaluation output: `outputs/eval_rgnet_v1_cascaded_denseaspp_paper_recipe_aadb_20ep_20260526/final_model`
- Test samples: 1000
- SRCC: 0.6581795985
- PLCC: 0.6682071102
- MAE: 0.1239729822
- RMSE: 0.1556523935
- MSE: 0.0242276676

## Comparison

- Previous cascaded DenseASPP SRCC: 0.6049975569
- Paper-recipe cascaded DenseASPP best-checkpoint SRCC: 0.6683026827
- Current best RGNet AADB baseline SRCC: 0.6819
- AADB paper target SRCC: about 0.7104

The paper-style recipe improved cascaded DenseASPP by about +0.0633 SRCC over the previous cascaded DenseASPP run, but it still trails the current RGNet baseline by about -0.0136 SRCC.

## Recommendation

Do not run the 80-epoch version yet. The 20-epoch controlled run is stable and improved the DenseASPP path, but it overfits after epoch 6 and remains below the existing RGNet baseline. A better next step is another controlled ablation focused on regularization or training schedule before spending an 80-epoch run.
