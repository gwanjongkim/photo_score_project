# Issue priority

## P0 — blocks defensible production claims

1. Resolve preprocessing range for KonIQ, FLIVE and practical RGNet. Training supplies `[0,1]` while default Keras MobileNetV2/EfficientNetV2 preprocessing generally expects `[0,255]`. Re-run same-image Python/TFLite/mobile parity with raw tensors recorded.
2. Update active runtime references so legacy NIMA cannot be confused with the fixed checkpoint/export. Prove byte hashes from fixed Keras → TFLite → shipped asset.
3. Name every deployed ensemble policy and normalize component scales explicitly. Remove ambiguous “technical ensemble” claims.

## P1 — blocks experimental validity

1. Publish immutable ground-truth prediction tables for every reported metric and independently recompute SRCC/PLCC/MAE/RMSE/accuracy.
2. Rebuild FLIVE evaluation on official image/patch splits; label random 80/10/10 results as non-paper protocol.
3. Run content-hash and perceptual near-duplicate audits across core and teacher/student splits.
4. Prove student target provenance and that test labels/predictions were not used for target generation or selection.
5. Resolve ICAA port architecture, head count/scales and checkpoint load with a deterministic forward pass.

## P2 — improves traceability and claim precision

1. Capture exact training commands, environment lockfiles, seeds, checkpoint-selection criteria and artifact hashes in a single lineage manifest.
2. Rename practical RGNet/A-LAMP/MUSIQ artifacts to include `inspired` or `adapted`.
3. Separate paper metrics, local in-domain metrics, cross-dataset metrics and conversion parity in reports.
4. Retire or clearly quarantine rejected and legacy assets from default configs.
