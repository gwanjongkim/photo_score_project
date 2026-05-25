# Plan

Implement the class-weighted block5 fine-tuning experiment without changing the existing Multi-Patch teacher baseline defaults.

1. Add selective VGG16 backbone trainability so `--backbone_trainable true --unfreeze_from_layer block5` trains only block5 layers.
2. Add optional class-weight sample weights computed from the active training JSONL labels.
3. Add validation prediction diagnostics at threshold 0.5 for balanced accuracy, specificity, and predicted positive ratio.
4. Run compile checks, a 32/16 smoke run, then the requested 1024/256 two-epoch small run if smoke passes.
