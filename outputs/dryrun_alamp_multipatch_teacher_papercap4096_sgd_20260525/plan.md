# Plan

Implement and test the external A-LAMP-like SGD optimizer recipe for the A-LAMP Multi-Patch teacher paper-capacity 4096 variant.

1. Verify current optimizer behavior and add missing explicit SGD controls.
2. Preserve Adam as the default optimizer and keep existing default training behavior unchanged.
3. Record optimizer recipe fields in `train_summary.json`.
4. Run compile checks, default compatibility smoke, SGD smoke, then bounded 8192/2048 SGD midruns as requested.
