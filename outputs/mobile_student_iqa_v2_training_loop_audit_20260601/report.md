# Mobile Student IQA v2 Training Loop Audit

## 1. Summary
- Audited `scripts/technical_iqa/train_mobile_student_iqa_v2.py`, `outputs/mobile_student_iqa_v2_20260601/training_history.csv`, and `outputs/mobile_student_iqa_v2_20260601/report.md`.
- The zero train metrics are not credible model performance. They are a training input pipeline artifact.
- Code fix applied only to `scripts/technical_iqa/train_mobile_student_iqa_v2.py`.
- No model tuning, Flutter changes, TFLite export, production scoring change, or training run was performed.

## 2. Evidence From Logs
- The saved report shows train steps per epoch `197` and validation steps `43`.
- The training history pattern shows nonzero train metrics on one epoch, then exactly zero train loss/MAE/RMSE on the next epoch.
- Validation metrics remain nonzero and repeat the previous values on zero-train epochs.
- The same alternating pattern appears in Stage A and Stage B, so it is not specific to fine-tuning or a newly unfrozen layer block.

## 3. Evidence From training_history.csv
- Stage A rows:
  - epoch 1: train loss `0.005350`, MAE `0.129947`, RMSE `0.167126`.
  - epoch 2: train loss `0.000000`, MAE `0.000000`, RMSE `0.000000`, while validation remains `0.003687/0.095637/0.125514`.
  - epoch 3: train metrics become nonzero again.
- Stage B rows:
  - phase epoch 1: train loss `0.002663`, MAE `0.074245`, RMSE `0.099158`.
  - phase epoch 2: train loss `0.000000`, MAE `0.000000`, RMSE `0.000000`, while validation remains `0.002841/0.078129/0.102037`.
  - The alternating zero/nonzero pattern continues through Stage B phase epoch 20.

## 4. Root Cause
- The script created finite `tf.data.Dataset.from_generator(...)` datasets.
- It also passed explicit `steps_per_epoch` and `validation_steps` to `model.fit`.
- With explicit steps, Keras expects the iterator to provide enough batches across epochs. The finite generator dataset was exhausted after one logical pass.
- On the next epoch, Keras had no train batches to consume, which produced zero train metrics instead of genuine computed metrics.
- Stage A and Stage B history merge did not create the zeros; it preserved the raw Keras history values.
- There was no evidence that callbacks were creating validation-only epochs or that missing values were filled with zero by the script.

## 5. Fix Plan
- Keep exact `steps_per_epoch = ceil(train_rows / batch_size)` and `validation_steps = ceil(val_rows / batch_size)` so progress bars remain known.
- Add `.repeat()` to train, validation, and prediction datasets that are consumed with explicit steps.
- Do not remove steps while using `from_generator`, because cardinality would remain unknown and progress would return to `Unknown`.
- Add `initial_epoch=0` to each independent `model.fit` call so Stage A and Stage B are clearly separate fits.
- Rename history columns to `phase`, `phase_epoch`, and `global_epoch`.
- Build history frames from raw Keras dictionaries and pad missing metric values with `NaN`, never zero.
- Add `--debug_history` to print raw Keras history keys and per-epoch values for Stage A and Stage B.

## 6. Safety Check
- The model architecture was not changed.
- The v2 output directory from the completed run was not overwritten.
- The fix does not tune hyperparameters.
- The fix does not export TFLite.
- The fix does not modify Flutter or production `technical_score`.
- A short verification run should be used before trusting a new full v2 report.

## 7. Next Smoke Command
```bash
./.venv_gpu/bin/python scripts/technical_iqa/train_mobile_student_iqa_v2.py \
  --manifest_csv outputs/student_training_manifest_design_20260601/student_training_manifest.csv \
  --out_dir outputs/mobile_student_iqa_v2_training_loop_verify_20260601 \
  --head_epochs 1 \
  --finetune_epochs 4 \
  --batch_size 32 \
  --max_train_samples 512 \
  --max_val_samples 256 \
  --max_test_samples 256 \
  --debug_history
```
