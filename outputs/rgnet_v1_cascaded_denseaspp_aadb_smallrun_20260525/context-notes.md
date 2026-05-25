# RGNet-v1-Cascaded-DenseASPP Context Notes

- This is a controlled paper-faithfulness ablation, not a full RGNet paper reproduction.
- The current parallel ASPP approximation must remain the default for existing configs and checkpoints.
- The new DenseASPP path should be selected only with `context_module_type=cascaded_denseaspp`.
- Unrelated dirty files were present before this work, including A-LAMP files, `src/train/train_techiqa_guard.py`, and many untracked experiment artifacts. They are intentionally out of scope.
- The previous reshape used `region_dim` as context channels. Cascaded DenseASPP outputs `backbone_channels + len(rates) * growth_rate`, so node reshaping now uses the static channel count from `context_map.shape[-1]`.
- Existing parallel ASPP still projects to `region_dim`, so default checkpoint/config compatibility should be preserved.
- Default compatibility smoke used `context_module_type=parallel_aspp`, context channels 256, and the same parameter count as the prior parallel path: 19,431,745 total / 19,345,025 trainable.
- TensorFlow again spent about 2 minutes compiling the first GPU train step with XLA and reported allocator garbage collection at batch size 4, but the run completed without NaNs or OOM.
- The first DenseASPP smoke produced exact save/load parity but warned that `V1CascadedDenseASPP` had no explicit `build()` method. Added a build method before rerunning validation.
- DenseASPP smoke after the build-method fix completed without the Keras build warning. It used context channels 1280 and had 10,603,841 total / 10,519,681 trainable params.
- Cascaded DenseASPP at batch size 4 still fits the 12GB GPU, with allocator garbage collection and a slow first-step XLA compile.
- The 1024/256 DenseASPP smallrun completed 5 epochs with best epoch 3, best val loss about 0.02452, and best val MAE about 0.12323.
- Bounded weights-only evaluation succeeded after removing the unsupported evaluator `--aggregation` flag. The config already supplies `aggregation: mean`.
- Bounded best-weight evaluation produced val SRCC about 0.5377 / PLCC about 0.5570 on 256 validation samples, and test-subset SRCC about 0.4924 / PLCC about 0.5295 on 256 test samples.
