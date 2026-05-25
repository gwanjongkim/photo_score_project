# RGNet-v1 Cascaded DenseASPP Paper-Recipe Plan

1. Preserve default RGNet paper-v1 training behavior unless paper-recipe flags are explicitly passed.
2. Add training-only paper-style augmentation, polynomial LR decay, weight decay, and early-stopping controls.
3. Add a dedicated cascaded DenseASPP paper-recipe config and make weights-only evaluation reconstruct aggregation/LSE settings.
4. Verify with compile checks, default smoke, paper-recipe smoke, and one controlled 20-epoch run if smoke succeeds.

