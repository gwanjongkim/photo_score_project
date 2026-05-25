# RGNet-v1 Cascaded DenseASPP Paper-Recipe Context Notes

- This remains an RGNet-paper-v1 approximation, not an official WACV paper reproduction.
- Defaults must preserve the existing training path.
- Paper-recipe behavior is opt-in via flags/config.
- Pre-existing unrelated dirty files include A-LAMP files, `src/train/train_techiqa_guard.py`, and many untracked experiment artifacts. They are intentionally out of scope.
- Previous cascaded DenseASPP full result to beat: test SRCC `0.6049975569`.
- Current best RGNet AADB baseline to beat: SRCC `0.6819`.
- The evaluator now accepts `--aggregation` and `--lse_r` so weights-only reconstruction can match LSE/r=4 runs.
- Paper-recipe augmentation is training-only. Validation/evaluation remain deterministic resize paths.
- Optimizer weight decay uses Keras Adam `weight_decay` when the installed optimizer exposes that parameter; otherwise it records that decay was not applied.
- Default compatibility smoke passed with `paper_recipe=false`, `random_scale_crop=false`, constant LR, no weight decay, parallel ASPP, and exact save/load parity.
- Paper-recipe smoke passed at image size 300, batch size 4, LSE/r=4, random scale-crop/flip, polynomial LR, Adam optimizer weight decay, disabled early stopping, and exact save/load parity.
- Paper-recipe smoke feature grid was 9x9 = 81 nodes. Batch size 4 fit on RTX 4070 SUPER 12GB with allocator garbage collection and a slow first-step XLA compile.
- Controlled 20-epoch run completed all requested epochs. Best validation loss was epoch 6; later epochs showed overfitting despite LR decay.
- Best-checkpoint test SRCC was `0.6683026827`, improving over the previous cascaded DenseASPP SRCC `0.6049975569` but still below the current best RGNet baseline `0.6819`.
- Final-model test SRCC was `0.6581795985`, weaker than the best-validation checkpoint.
